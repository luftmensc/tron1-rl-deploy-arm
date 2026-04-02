#!/usr/bin/env python3
"""
Arm planner for the asynchronous RL EE target interface.

This node keeps the MoveIt planning pipeline, FK EE path extraction, and RViz
visualization, but it does not execute the planned path as an open-loop
time-based stream.

Why:
  - `/EEPose_cmd_rc` only updates the controller's accumulated `rc_ee_cmd`.
  - The RL policy consumes that accumulated target inside `computeObservation()`
    on the policy decimation schedule, not when the ROS message arrives.
  - The arm response is therefore asynchronous and limited by the learned
    policy, the observation history, and joint torque/stiffness limits.

Execution strategy in this node:
  1. Plan to a named SRDF pose with MoveIt.
  2. Convert the joint trajectory to an EE path with FK.
  3. Densify the EE path into small collision-free subgoals.
  4. Use joint-state FK feedback to supervise those subgoals.
  5. Continuously publish absolute EE targets on `/EEPose_cmd_abs`
     and advance to the next waypoint when the arm catches up.

Controller-side orientation slots are not ROS RPY order:
  msg[3] -> rc_ee_cmd.ee_rpy[0] -> yaw offset
  msg[4] -> rc_ee_cmd.ee_rpy[1] -> roll offset
  msg[5] -> rc_ee_cmd.ee_rpy[2] -> pitch offset
"""

import copy
import sys
import threading

import moveit_commander
import numpy as np
import rospy
import tf.transformations as tf_trans

from geometry_msgs.msg import Point, Pose, PoseStamped
from moveit_msgs.msg import RobotState as RobotStateMsg
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray


class ArmPlannerNode:
    """MoveIt arm planner with feedback-gated RC EE target execution."""

    EE_INIT_POS = np.array([0.446, 0.0, 0.241], dtype=float)
    EE_INIT_RPY = np.array([-np.pi / 2, 0.0, -np.pi / 2], dtype=float)

    POS_DELTA_LIMITS = np.array([
        [0.0 - 0.446, 0.8 - 0.446],
        [-0.5, 0.5],
        [-0.3 - 0.241, 0.5 - 0.241],
    ], dtype=float)

    ARM_JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
    FK_FRAME = "base_Link"
    EE_LINK = "link6"

    def __init__(self):
        rospy.init_node("arm_planner_node")

        self.joint_state_topic = rospy.get_param(
            "~joint_state_topic", "/pointfoot_hw/joint_states"
        )
        self.control_rate = max(1.0, float(rospy.get_param("~control_rate", 10.0)))
        self.path_segment_length = max(
            1e-4, float(rospy.get_param("~path_segment_length", 0.03))
        )
        self.target_advance_speed = max(
            1e-4, float(rospy.get_param("~target_advance_speed", 0.05))
        )
        self.max_target_lead = max(
            1e-3, float(rospy.get_param("~max_target_lead", 0.10))
        )
        self.stall_warning_sec = max(
            0.1, float(rospy.get_param("~stall_warning_sec", 8.0))
        )
        self.log_interval_sec = max(
            0.1, float(rospy.get_param("~log_interval_sec", 0.5))
        )
        self.settle_time = max(0.0, float(rospy.get_param("~settle_time", 2.0)))
        self.gripper_state = float(rospy.get_param("~gripper_state", 0.0))

        # Oscillation detection & cursor retreat parameters
        self.smoothing_alpha = max(
            0.01, min(1.0, float(rospy.get_param("~smoothing_alpha", 0.3)))
        )
        self.oscillation_window = max(
            3, int(rospy.get_param("~oscillation_window", 10))
        )
        self.oscillation_threshold = max(
            1e-3, float(rospy.get_param("~oscillation_threshold", 0.04))
        )
        self.retreat_threshold = max(
            1e-3, float(rospy.get_param("~retreat_threshold", 0.15))
        )
        self.retreat_speed_fraction = max(
            0.1, float(rospy.get_param("~retreat_speed_fraction", 0.5))
        )

        moveit_commander.roscpp_initialize(sys.argv)
        self.move_group = moveit_commander.MoveGroupCommander("arm")
        self.move_group.set_planning_time(5.0)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)

        self.available_named_targets = self.move_group.get_named_targets()
        rospy.loginfo("Available named poses: %s", self.available_named_targets)

        self.current_joint_state = None
        self.js_lock = threading.Lock()
        self.executing = False
        self.cancel_requested = False
        self.execution_done = threading.Event()
        self.execution_done.set()  # not executing initially

        # Controller feedback state (from /rc_ee_cmd_state)
        self.rc_ee_cmd_state = None
        self.rc_ee_cmd_lock = threading.Lock()

        rospy.loginfo("Waiting for /compute_fk service...")
        rospy.wait_for_service("/compute_fk", timeout=30.0)
        self.fk_client = rospy.ServiceProxy("/compute_fk", GetPositionFK)
        rospy.loginfo("/compute_fk service available")

        self.ee_cmd_pub = rospy.Publisher(
            "/EEPose_cmd_abs", Float32MultiArray, queue_size=10
        )
        self.marker_pub = rospy.Publisher(
            "/arm_planner/ee_trajectory_markers",
            MarkerArray,
            queue_size=10,
            latch=True,
        )
        self.status_pub = rospy.Publisher("/arm_planner/status", String, queue_size=10)
        self.progress_pub = rospy.Publisher(
            "/arm_planner/progress", Float32MultiArray, queue_size=10
        )
        self.ee_pose_debug_pub = rospy.Publisher(
            "/arm_planner/ee_pose_debug", Float32MultiArray, queue_size=10
        )

        self.js_sub = rospy.Subscriber(
            self.joint_state_topic, JointState, self._joint_states_cb
        )
        self.goal_sub = rospy.Subscriber(
            "/arm_planner/goal_pose", String, self._goal_pose_cb
        )
        self.goal_cartesian_sub = rospy.Subscriber(
            "/arm_planner/goal_pose_cartesian",
            Float32MultiArray,
            self._goal_pose_cartesian_cb,
        )
        self.goal_joints_sub = rospy.Subscriber(
            "/arm_planner/goal_joint_angles",
            Float32MultiArray,
            self._goal_joint_angles_cb,
        )
        self.stop_sub = rospy.Subscriber(
            "/arm_planner/stop", String, self._stop_cb
        )
        self.rc_ee_cmd_sub = rospy.Subscriber(
            "/rc_ee_cmd_state", Float32MultiArray, self._rc_ee_cmd_cb
        )

        self._wait_for_joint_states()

        self.ee_debug_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self._ee_pose_debug_cb
        )

        rospy.loginfo("ArmPlannerNode ready on /EEPose_cmd_abs")
        rospy.loginfo(
            "  joint_state_topic=%s control_rate=%.1fHz segment=%.3fm",
            self.joint_state_topic,
            self.control_rate,
            self.path_segment_length,
        )

    @staticmethod
    def _fmt_vector(vector):
        return "[{:.3f}, {:.3f}, {:.3f}]".format(vector[0], vector[1], vector[2])

    @staticmethod
    def _angle_wrap(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _wrap_rpy_delta(self, delta):
        return np.array([self._angle_wrap(value) for value in delta], dtype=float)

    def _ee_pose_debug_cb(self, _event):
        ee_pose = self._get_actual_ee_pose()
        if ee_pose is None:
            return
        msg = Float32MultiArray()
        msg.data = [
            float(ee_pose["position"][0]),
            float(ee_pose["position"][1]),
            float(ee_pose["position"][2]),
            float(ee_pose["rpy"][0]),
            float(ee_pose["rpy"][1]),
            float(ee_pose["rpy"][2]),
        ]
        self.ee_pose_debug_pub.publish(msg)

    def _wait_for_joint_states(self):
        rospy.loginfo("Waiting for joint states on %s...", self.joint_state_topic)
        timeout_time = rospy.Time.now() + rospy.Duration(10.0)

        while (
            self.current_joint_state is None
            and rospy.Time.now() < timeout_time
            and not rospy.is_shutdown()
        ):
            rospy.sleep(0.1)

        if self.current_joint_state is None:
            rospy.logwarn("No joint states received within timeout")
            return

        rospy.loginfo("Joint states received")

    def _joint_states_cb(self, msg):
        with self.js_lock:
            self.current_joint_state = msg

    def _rc_ee_cmd_cb(self, msg):
        if len(msg.data) < 7:
            return
        with self.rc_ee_cmd_lock:
            self.rc_ee_cmd_state = {
                "pos_offsets": np.array(msg.data[0:3], dtype=float),
                "rpy_offsets": np.array(msg.data[3:6], dtype=float),
                "arm_hold_still": msg.data[6] > 0.5,
            }

    def _get_controller_cmd_state(self):
        with self.rc_ee_cmd_lock:
            if self.rc_ee_cmd_state is None:
                return None
            return copy.deepcopy(self.rc_ee_cmd_state)

    def _abort_execution(self, reason="preempted"):
        """Abort any running trajectory execution and wait for it to finish."""
        if not self.executing:
            return
        rospy.loginfo("Aborting execution: %s", reason)
        self.cancel_requested = True
        self.execution_done.wait(timeout=5.0)
        if self.executing:
            rospy.logwarn("Execution did not stop within timeout")

    def _goal_pose_cb(self, msg):
        pose_name = msg.data.strip()
        if not pose_name:
            self._publish_status("error", "Empty goal pose name")
            return

        if self.executing:
            self._abort_execution("new named pose goal: {}".format(pose_name))

        rospy.loginfo("Received goal pose: %s", pose_name)
        worker = threading.Thread(target=self._plan_and_execute, args=(pose_name,))
        worker.daemon = True
        worker.start()

    def _goal_pose_cartesian_cb(self, msg):
        """Callback for Cartesian goal: Float32MultiArray [x, y, z, roll, pitch, yaw] in base_Link frame."""
        if len(msg.data) < 6:
            self._publish_status("error", "Cartesian goal needs 6 values: [x, y, z, roll, pitch, yaw]")
            rospy.logerr("Cartesian goal needs 6 values, got %d", len(msg.data))
            return

        x, y, z = float(msg.data[0]), float(msg.data[1]), float(msg.data[2])
        roll, pitch, yaw = float(msg.data[3]), float(msg.data[4]), float(msg.data[5])

        if self.executing:
            self._abort_execution("new Cartesian goal")

        rospy.loginfo(
            "Received Cartesian goal: pos=[%.3f, %.3f, %.3f] rpy=[%.3f, %.3f, %.3f]",
            x, y, z, roll, pitch, yaw,
        )
        worker = threading.Thread(
            target=self._plan_and_execute_cartesian,
            args=(x, y, z, roll, pitch, yaw),
        )
        worker.daemon = True
        worker.start()

    def _goal_joint_angles_cb(self, msg):
        """Callback for joint-angle goal: Float32MultiArray [J1..J6] in radians."""
        if len(msg.data) < len(self.ARM_JOINT_NAMES):
            self._publish_status(
                "error",
                "Joint angle goal needs {} values ({}), got {}".format(
                    len(self.ARM_JOINT_NAMES),
                    ", ".join(self.ARM_JOINT_NAMES),
                    len(msg.data),
                ),
            )
            return

        joint_values = [float(v) for v in msg.data[: len(self.ARM_JOINT_NAMES)]]

        if self.executing:
            self._abort_execution("new joint angle goal")

        rospy.loginfo(
            "Received joint angle goal: %s",
            [round(v, 4) for v in joint_values],
        )
        worker = threading.Thread(
            target=self._plan_and_execute_joints, args=(joint_values,)
        )
        worker.daemon = True
        worker.start()

    def _stop_cb(self, _msg):
        """Stop current execution immediately without starting a new plan."""
        self._abort_execution("stop requested")
        self._publish_status("stopped", "Execution stopped by user")

    def _get_arm_positions(self):
        with self.js_lock:
            if self.current_joint_state is None:
                return None
            joint_state = self.current_joint_state

        position_map = dict(zip(joint_state.name, joint_state.position))
        arm_positions = []
        for joint_name in self.ARM_JOINT_NAMES:
            if joint_name not in position_map:
                rospy.logwarn_throttle(
                    5.0, "Joint %s not present in joint states", joint_name
                )
                return None
            arm_positions.append(position_map[joint_name])
        return np.array(arm_positions, dtype=float)

    def _get_full_joint_state_copy(self):
        with self.js_lock:
            if self.current_joint_state is None:
                return None
            return copy.deepcopy(self.current_joint_state)

    def _compute_fk(self, arm_joint_positions):
        request = GetPositionFKRequest()
        request.header.frame_id = self.FK_FRAME
        request.header.stamp = rospy.Time(0)
        request.fk_link_names = [self.EE_LINK]

        full_joint_state = self._get_full_joint_state_copy()
        if full_joint_state is not None:
            request.robot_state.joint_state.name = list(full_joint_state.name)
            positions = list(full_joint_state.position)
            joint_names = list(full_joint_state.name)
            for index, joint_name in enumerate(self.ARM_JOINT_NAMES):
                if joint_name in joint_names:
                    positions[joint_names.index(joint_name)] = float(
                        arm_joint_positions[index]
                    )
            request.robot_state.joint_state.position = positions
        else:
            request.robot_state.joint_state.name = list(self.ARM_JOINT_NAMES)
            request.robot_state.joint_state.position = [
                float(value) for value in arm_joint_positions
            ]

        try:
            response = self.fk_client(request)
        except rospy.ServiceException as exc:
            rospy.logwarn_throttle(2.0, "FK service call failed: %s", str(exc))
            return None

        if response.error_code.val != 1:
            rospy.logwarn_throttle(2.0, "FK error code: %d", response.error_code.val)
            return None

        pose = response.pose_stamped[0].pose
        position = np.array(
            [pose.position.x, pose.position.y, pose.position.z], dtype=float
        )
        quaternion = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rpy = np.array(
            tf_trans.euler_from_quaternion(quaternion, axes="sxyz"), dtype=float
        )
        return position, rpy

    def _get_actual_ee_pose(self):
        arm_positions = self._get_arm_positions()
        if arm_positions is None:
            return None

        fk_result = self._compute_fk(arm_positions)
        if fk_result is None:
            return None

        position, rpy = fk_result
        return {"position": position, "rpy": rpy}

    def _clip_position_target(self, pos_target):
        return np.clip(
            pos_target,
            self.POS_DELTA_LIMITS[:, 0],
            self.POS_DELTA_LIMITS[:, 1],
        )

    def _ee_pose_to_controller_target(self, position, rpy):
        pos_target = self._clip_position_target(position - self.EE_INIT_POS)

        # Controller order is [yaw_offset, roll_offset, pitch_offset].
        ctrl_rpy_target = np.array(
            [
                rpy[2] - self.EE_INIT_RPY[2],
                rpy[0] - self.EE_INIT_RPY[0],
                rpy[1] - self.EE_INIT_RPY[1],
            ],
            dtype=float,
        )

        return pos_target, ctrl_rpy_target

    def _plan_and_execute(self, pose_name):
        self.executing = True
        self.cancel_requested = False
        self.execution_done.clear()

        try:
            self._publish_status("planning", pose_name)

            full_joint_state = self._get_full_joint_state_copy()
            if full_joint_state is None:
                self._publish_status("error", "No joint states available")
                return

            start_state = RobotStateMsg()
            start_state.joint_state = full_joint_state
            self.move_group.set_start_state(start_state)

            try:
                self.move_group.set_named_target(pose_name)
            except Exception as exc:
                message = "Unknown pose '{}'. Available: {}".format(
                    pose_name, self.available_named_targets
                )
                self._publish_status("error", message)
                rospy.logerr(
                    "Unknown pose '%s': %s. Available: %s",
                    pose_name,
                    exc,
                    self.available_named_targets,
                )
                return

            rospy.loginfo("Planning to named pose '%s'...", pose_name)
            plan_result = self.move_group.plan()
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                plan = plan_result
                success = bool(plan.joint_trajectory.points)

            if not success or not plan.joint_trajectory.points:
                self._publish_status("error", "MoveIt planning failed")
                rospy.logerr("Planning to '%s' failed", pose_name)
                return

            joint_waypoints = len(plan.joint_trajectory.points)
            duration = plan.joint_trajectory.points[-1].time_from_start.to_sec()
            rospy.loginfo(
                "Plan OK for '%s': %d joint waypoints, %.2fs nominal duration",
                pose_name,
                joint_waypoints,
                duration,
            )

            self._publish_status("computing_fk", "Computing EE trajectory")
            ee_trajectory = self._compute_fk_trajectory(plan)
            if not ee_trajectory:
                self._publish_status("error", "FK computation failed for all waypoints")
                return

            dense_trajectory = self._densify_trajectory(ee_trajectory)
            self._publish_trajectory_markers(dense_trajectory)

            rospy.loginfo(
                "EE trajectory for '%s': raw=%d dense=%d start=%s goal=%s",
                pose_name,
                len(ee_trajectory),
                len(dense_trajectory),
                self._fmt_vector(dense_trajectory[0]["position"]),
                self._fmt_vector(dense_trajectory[-1]["position"]),
            )

            self._publish_status("executing", pose_name)
            self._execute_trajectory(pose_name, dense_trajectory)

            if self.cancel_requested:
                self._publish_status("cancelled", pose_name)
            else:
                self._publish_status("done", pose_name)
        except Exception as exc:
            rospy.logerr("Error in plan_and_execute: %s", str(exc))
            self._publish_status("error", str(exc))
        finally:
            self.executing = False
            self.execution_done.set()

    def _plan_and_execute_joints(self, joint_values):
        """Plan and execute to explicit joint angles [J1..J6] in radians."""
        self.executing = True
        self.cancel_requested = False
        self.execution_done.clear()
        label = "joints({})".format(
            ",".join("{:.3f}".format(v) for v in joint_values)
        )

        try:
            self._publish_status("planning", label)

            full_joint_state = self._get_full_joint_state_copy()
            if full_joint_state is None:
                self._publish_status("error", "No joint states available")
                return

            start_state = RobotStateMsg()
            start_state.joint_state = full_joint_state
            self.move_group.set_start_state(start_state)

            target_dict = dict(zip(self.ARM_JOINT_NAMES, joint_values))
            self.move_group.set_joint_value_target(target_dict)

            rospy.loginfo("Planning to joint angles %s ...", label)
            plan_result = self.move_group.plan()
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                plan = plan_result
                success = bool(plan.joint_trajectory.points)

            if not success or not plan.joint_trajectory.points:
                self._publish_status("error", "MoveIt planning failed for joint angle goal")
                rospy.logerr("Planning to joint angles %s failed", label)
                return

            joint_waypoints = len(plan.joint_trajectory.points)
            duration = plan.joint_trajectory.points[-1].time_from_start.to_sec()
            rospy.loginfo(
                "Plan OK for '%s': %d joint waypoints, %.2fs nominal duration",
                label, joint_waypoints, duration,
            )

            self._publish_status("computing_fk", "Computing EE trajectory")
            ee_trajectory = self._compute_fk_trajectory(plan)
            if not ee_trajectory:
                self._publish_status("error", "FK computation failed for all waypoints")
                return

            dense_trajectory = self._densify_trajectory(ee_trajectory)
            self._publish_trajectory_markers(dense_trajectory)

            rospy.loginfo(
                "EE trajectory for '%s': raw=%d dense=%d start=%s goal=%s",
                label, len(ee_trajectory), len(dense_trajectory),
                self._fmt_vector(dense_trajectory[0]["position"]),
                self._fmt_vector(dense_trajectory[-1]["position"]),
            )

            self._publish_status("executing", label)
            self._execute_trajectory(label, dense_trajectory)

            if self.cancel_requested:
                self._publish_status("cancelled", label)
            else:
                self._publish_status("done", label)
        except Exception as exc:
            rospy.logerr("Error in plan_and_execute_joints: %s", str(exc))
            self._publish_status("error", str(exc))
        finally:
            self.executing = False
            self.execution_done.set()

    def _plan_and_execute_cartesian(self, x, y, z, roll, pitch, yaw):
        """Plan and execute to a Cartesian pose [x,y,z,roll,pitch,yaw] in base_Link frame."""
        self.executing = True
        self.cancel_requested = False
        self.execution_done.clear()
        label = "cartesian({:.3f},{:.3f},{:.3f})".format(x, y, z)

        try:
            self._publish_status("planning", label)

            full_joint_state = self._get_full_joint_state_copy()
            if full_joint_state is None:
                self._publish_status("error", "No joint states available")
                return

            start_state = RobotStateMsg()
            start_state.joint_state = full_joint_state
            self.move_group.set_start_state(start_state)

            # Build a PoseStamped target in the FK_FRAME (base_Link)
            quat = tf_trans.quaternion_from_euler(roll, pitch, yaw, axes="sxyz")
            target_pose = Pose()
            target_pose.position.x = x
            target_pose.position.y = y
            target_pose.position.z = z
            target_pose.orientation.x = quat[0]
            target_pose.orientation.y = quat[1]
            target_pose.orientation.z = quat[2]
            target_pose.orientation.w = quat[3]

            self.move_group.set_pose_target(target_pose, self.EE_LINK)

            rospy.loginfo("Planning to Cartesian pose %s ...", label)
            plan_result = self.move_group.plan()
            if isinstance(plan_result, tuple):
                success = plan_result[0]
                plan = plan_result[1]
            else:
                plan = plan_result
                success = bool(plan.joint_trajectory.points)

            # Clear the pose target after planning
            self.move_group.clear_pose_targets()

            if not success or not plan.joint_trajectory.points:
                self._publish_status("error", "MoveIt planning failed for Cartesian goal")
                rospy.logerr("Planning to Cartesian pose %s failed", label)
                return

            joint_waypoints = len(plan.joint_trajectory.points)
            duration = plan.joint_trajectory.points[-1].time_from_start.to_sec()
            rospy.loginfo(
                "Plan OK for '%s': %d joint waypoints, %.2fs nominal duration",
                label, joint_waypoints, duration,
            )

            self._publish_status("computing_fk", "Computing EE trajectory")
            ee_trajectory = self._compute_fk_trajectory(plan)
            if not ee_trajectory:
                self._publish_status("error", "FK computation failed for all waypoints")
                return

            dense_trajectory = self._densify_trajectory(ee_trajectory)
            self._publish_trajectory_markers(dense_trajectory)

            rospy.loginfo(
                "EE trajectory for '%s': raw=%d dense=%d start=%s goal=%s",
                label, len(ee_trajectory), len(dense_trajectory),
                self._fmt_vector(dense_trajectory[0]["position"]),
                self._fmt_vector(dense_trajectory[-1]["position"]),
            )

            self._publish_status("executing", label)
            self._execute_trajectory(label, dense_trajectory)

            if self.cancel_requested:
                self._publish_status("cancelled", label)
            else:
                self._publish_status("done", label)
        except Exception as exc:
            rospy.logerr("Error in plan_and_execute_cartesian: %s", str(exc))
            self._publish_status("error", str(exc))
        finally:
            self.executing = False
            self.execution_done.set()

    def _compute_fk_trajectory(self, robot_trajectory):
        joint_trajectory = robot_trajectory.joint_trajectory
        if not joint_trajectory.points:
            return []

        trajectory_joint_indices = {
            name: index for index, name in enumerate(joint_trajectory.joint_names)
        }

        ee_trajectory = []
        for point in joint_trajectory.points:
            arm_positions = np.zeros(len(self.ARM_JOINT_NAMES), dtype=float)
            for index, joint_name in enumerate(self.ARM_JOINT_NAMES):
                if joint_name in trajectory_joint_indices:
                    arm_positions[index] = point.positions[
                        trajectory_joint_indices[joint_name]
                    ]
                else:
                    current_positions = self._get_arm_positions()
                    if current_positions is not None:
                        arm_positions[index] = current_positions[index]

            fk_result = self._compute_fk(arm_positions)
            if fk_result is None:
                rospy.logwarn(
                    "FK failed for waypoint at t=%.3f, skipping",
                    point.time_from_start.to_sec(),
                )
                continue

            position, rpy = fk_result
            ee_trajectory.append(
                {
                    "time": point.time_from_start.to_sec(),
                    "position": position.copy(),
                    "rpy": rpy.copy(),
                }
            )

        return ee_trajectory

    def _interpolate_waypoint(self, waypoint_a, waypoint_b, alpha):
        position = (
            (1.0 - alpha) * waypoint_a["position"] + alpha * waypoint_b["position"]
        )
        rpy = waypoint_a["rpy"] + alpha * self._wrap_rpy_delta(
            waypoint_b["rpy"] - waypoint_a["rpy"]
        )
        return {"position": position, "rpy": rpy}

    def _densify_trajectory(self, ee_trajectory):
        if len(ee_trajectory) < 2:
            return ee_trajectory

        dense_trajectory = [ee_trajectory[0]]
        for index in range(len(ee_trajectory) - 1):
            waypoint_a = ee_trajectory[index]
            waypoint_b = ee_trajectory[index + 1]
            segment_pos_dist = np.linalg.norm(
                waypoint_b["position"] - waypoint_a["position"]
            )
            segment_rpy_dist = np.linalg.norm(
                self._wrap_rpy_delta(waypoint_b["rpy"] - waypoint_a["rpy"])
            )

            if segment_pos_dist < 1e-6 and segment_rpy_dist < 1e-4:
                continue

            steps = max(1, int(np.ceil(segment_pos_dist / self.path_segment_length)))
            for step in range(1, steps + 1):
                alpha = step / float(steps)
                interpolated = self._interpolate_waypoint(waypoint_a, waypoint_b, alpha)
                dense_trajectory.append(
                    {
                        "time": waypoint_a["time"]
                        + alpha * (waypoint_b["time"] - waypoint_a["time"]),
                        "position": interpolated["position"],
                        "rpy": interpolated["rpy"],
                    }
                )

        return dense_trajectory

    def _pose_error(self, actual_pose, target_pose):
        pos_error = np.linalg.norm(target_pose["position"] - actual_pose["position"])

        _, actual_ctrl_rpy = self._ee_pose_to_controller_target(
            actual_pose["position"], actual_pose["rpy"]
        )
        _, target_ctrl_rpy = self._ee_pose_to_controller_target(
            target_pose["position"], target_pose["rpy"]
        )
        rpy_error = np.linalg.norm(
            self._wrap_rpy_delta(target_ctrl_rpy - actual_ctrl_rpy)
        )
        return pos_error, rpy_error

    def _send_absolute_target(self, desired_pose):
        """Publish an absolute EE target on /EEPose_cmd_abs."""
        pos_target, ctrl_rpy_target = self._ee_pose_to_controller_target(
            desired_pose["position"], desired_pose["rpy"]
        )

        msg = Float32MultiArray()
        msg.data = [
            float(pos_target[0]),
            float(pos_target[1]),
            float(pos_target[2]),
            float(ctrl_rpy_target[0]),
            float(ctrl_rpy_target[1]),
            float(ctrl_rpy_target[2]),
            float(self.gripper_state),
        ]
        self.ee_cmd_pub.publish(msg)

    def _compute_arc_lengths(self, ee_trajectory):
        """Return cumulative arc-length array for the trajectory."""
        arc = [0.0]
        for i in range(1, len(ee_trajectory)):
            seg = np.linalg.norm(
                ee_trajectory[i]["position"] - ee_trajectory[i - 1]["position"]
            )
            arc.append(arc[-1] + seg)
        return arc

    def _interpolate_at_arc_length(self, ee_trajectory, arc_lengths, distance):
        """Interpolate the EE pose at a given arc-length along the trajectory."""
        if distance <= 0.0:
            return ee_trajectory[0]
        if distance >= arc_lengths[-1]:
            return ee_trajectory[-1]
        for i in range(1, len(arc_lengths)):
            if arc_lengths[i] >= distance:
                seg_len = arc_lengths[i] - arc_lengths[i - 1]
                alpha = (
                    (distance - arc_lengths[i - 1]) / seg_len
                    if seg_len > 1e-8
                    else 0.0
                )
                return self._interpolate_waypoint(
                    ee_trajectory[i - 1], ee_trajectory[i], alpha
                )
        return ee_trajectory[-1]

    def _execute_trajectory(self, pose_name, ee_trajectory):
        """Execute trajectory with a smoothly advancing target ("carrot").

        The cursor advances along the dense trajectory at
        ``target_advance_speed`` m/s, modulated by tracking quality:

        - **Normal**: speed_scale proportional to closeness.
        - **Oscillation**: cursor freezes until arm stabilises.
        - **Too far behind**: cursor retreats toward the arm.

        Controller feedback from ``/rc_ee_cmd_state`` is used to verify
        command delivery and detect armHoldStill_ activation.
        """
        if not ee_trajectory:
            return

        rate = rospy.Rate(self.control_rate)
        dt = 1.0 / self.control_rate

        arc_lengths = self._compute_arc_lengths(ee_trajectory)
        total_length = arc_lengths[-1]

        if total_length < 1e-6:
            rospy.loginfo("Trajectory too short (%.4fm), skipping", total_length)
            return

        cursor = 0.0
        advance_per_tick = self.target_advance_speed * dt

        # EMA-smoothed position for lead computation
        smoothed_pos = None

        # Rolling buffer for oscillation detection
        pos_history = []

        last_log_time = rospy.Time(0)
        last_cursor_for_stall = 0.0
        last_cursor_progress_time = rospy.Time.now()
        stall_warning_count = 0
        retreat_count = 0
        oscillation_count = 0
        publish_count = 0

        rospy.loginfo(
            "Executing '%s': waypoints=%d length=%.3fm "
            "speed=%.3fm/s max_lead=%.3fm retreat_thr=%.3fm osc_thr=%.3fm",
            pose_name,
            len(ee_trajectory),
            total_length,
            self.target_advance_speed,
            self.max_target_lead,
            self.retreat_threshold,
            self.oscillation_threshold,
        )

        while (
            not rospy.is_shutdown()
            and not self.cancel_requested
            and cursor < total_length
        ):
            actual_pose = self._get_actual_ee_pose()
            if actual_pose is None:
                rospy.logwarn_throttle(
                    2.0, "Execution waiting: could not compute actual EE pose from FK"
                )
                rate.sleep()
                continue

            actual_pos = actual_pose["position"]

            # --- EMA smoothing ---
            if smoothed_pos is None:
                smoothed_pos = actual_pos.copy()
            else:
                alpha = self.smoothing_alpha
                smoothed_pos = alpha * actual_pos + (1.0 - alpha) * smoothed_pos

            # --- Oscillation detection ---
            pos_history.append(actual_pos.copy())
            if len(pos_history) > self.oscillation_window:
                pos_history.pop(0)

            is_oscillating = False
            if len(pos_history) >= self.oscillation_window:
                pos_arr = np.array(pos_history)
                pos_stddev = np.mean(np.std(pos_arr, axis=0))
                is_oscillating = pos_stddev > self.oscillation_threshold

            # Current target at cursor position
            target_pose = self._interpolate_at_arc_length(
                ee_trajectory, arc_lengths, cursor
            )

            # Lead: use smoothed position for stability
            raw_lead = np.linalg.norm(target_pose["position"] - actual_pos)
            lead = np.linalg.norm(target_pose["position"] - smoothed_pos)

            # --- State-aware cursor advancement ---
            if is_oscillating:
                speed_scale = 0.0
                oscillation_count += 1
                cursor_mode = "FREEZE_OSC"
            elif lead > self.retreat_threshold:
                speed_scale = -self.retreat_speed_fraction
                retreat_count += 1
                cursor_mode = "RETREAT"
            elif lead > self.max_target_lead:
                speed_scale = 0.0
                cursor_mode = "FREEZE_LEAD"
            else:
                speed_scale = max(0.1, 1.0 - lead / self.max_target_lead)
                cursor_mode = "ADVANCE"

            cursor = max(0.0, min(cursor + advance_per_tick * speed_scale, total_length))

            # Recompute target at updated cursor
            target_pose = self._interpolate_at_arc_length(
                ee_trajectory, arc_lengths, cursor
            )

            # Publish absolute target (keeps controller's armHoldStill_ false)
            self._send_absolute_target(target_pose)
            publish_count += 1

            # --- Controller feedback verification ---
            ctrl_state = self._get_controller_cmd_state()
            cmd_mismatch = False
            if ctrl_state is not None:
                if ctrl_state["arm_hold_still"]:
                    rospy.logwarn_throttle(
                        2.0,
                        "armHoldStill=true during execution! "
                        "Commands may not be arriving fast enough.",
                    )
                intended_pos, _ = self._ee_pose_to_controller_target(
                    target_pose["position"], target_pose["rpy"]
                )
                cmd_diff = np.linalg.norm(intended_pos - ctrl_state["pos_offsets"])
                if cmd_diff > 0.01:
                    cmd_mismatch = True
                    rospy.logwarn_throttle(
                        2.0,
                        "Command mismatch: intended=%s ctrl=%s diff=%.4f "
                        "(possible clamping)",
                        self._fmt_vector(intended_pos),
                        self._fmt_vector(ctrl_state["pos_offsets"]),
                        cmd_diff,
                    )

            # Progress & stall tracking
            progress_frac = cursor / total_length
            self._publish_progress(progress_frac)

            now = rospy.Time.now()
            if cursor - last_cursor_for_stall > 0.005:
                last_cursor_for_stall = cursor
                last_cursor_progress_time = now

            stall_seconds = (now - last_cursor_progress_time).to_sec()
            if stall_seconds >= self.stall_warning_sec:
                stall_warning_count += 1
                rospy.logwarn(
                    "STALL pose=%s progress=%.1f%% cursor=%.3fm lead=%.3fm "
                    "mode=%s no_advance_for=%.1fs",
                    pose_name,
                    progress_frac * 100.0,
                    cursor,
                    lead,
                    cursor_mode,
                    stall_seconds,
                )
                last_cursor_progress_time = now

            # Periodic tracking log (enhanced)
            if (
                last_log_time == rospy.Time(0)
                or (now - last_log_time).to_sec() >= self.log_interval_sec
            ):
                rospy.loginfo(
                    "TRACK pose=%s progress=%.1f%% cursor=%.3fm/%.3fm "
                    "lead=%.3fm raw_lead=%.3fm spd=%.2f mode=%s "
                    "osc=%s actual=%s target=%s",
                    pose_name,
                    progress_frac * 100.0,
                    cursor,
                    total_length,
                    lead,
                    raw_lead,
                    speed_scale,
                    cursor_mode,
                    "T" if is_oscillating else "F",
                    self._fmt_vector(actual_pos),
                    self._fmt_vector(target_pose["position"]),
                )
                last_log_time = now

            rate.sleep()

        if self.cancel_requested:
            rospy.loginfo(
                "Execution cancelled for '%s' at %.1f%%",
                pose_name,
                (cursor / total_length) * 100.0 if total_length > 0 else 0.0,
            )
            return

        # Settle: keep publishing final target so controller stays stiff.
        settle_cycles = int(self.control_rate * self.settle_time)
        final_target = ee_trajectory[-1]
        final_pos_error = None
        final_rpy_error = None

        for _ in range(settle_cycles):
            if rospy.is_shutdown() or self.cancel_requested:
                break
            self._send_absolute_target(final_target)
            actual_pose = self._get_actual_ee_pose()
            if actual_pose is not None:
                final_pos_error, final_rpy_error = self._pose_error(
                    actual_pose, final_target
                )
            rate.sleep()

        rospy.loginfo(
            "Execution summary pose=%s published=%d stall_warnings=%d "
            "retreats=%d oscillations=%d "
            "final_pos_err=%s final_rpy_err=%s",
            pose_name,
            publish_count,
            stall_warning_count,
            retreat_count,
            oscillation_count,
            "n/a" if final_pos_error is None else "{:.3f}".format(final_pos_error),
            "n/a" if final_rpy_error is None else "{:.3f}".format(final_rpy_error),
        )

    def _publish_trajectory_markers(self, ee_trajectory):
        if not ee_trajectory:
            return

        marker_array = MarkerArray()
        now = rospy.Time.now()

        path_marker = Marker()
        path_marker.header.frame_id = self.FK_FRAME
        path_marker.header.stamp = now
        path_marker.ns = "arm_planner_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.005
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.2
        path_marker.color.a = 0.9
        path_marker.pose.orientation.w = 1.0
        for waypoint in ee_trajectory:
            point = Point()
            point.x = waypoint["position"][0]
            point.y = waypoint["position"][1]
            point.z = waypoint["position"][2]
            path_marker.points.append(point)
        marker_array.markers.append(path_marker)

        for index, waypoint in enumerate(ee_trajectory):
            marker = Marker()
            marker.header.frame_id = self.FK_FRAME
            marker.header.stamp = now
            marker.ns = "arm_planner_waypoints"
            marker.id = index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = waypoint["position"][0]
            marker.pose.position.y = waypoint["position"][1]
            marker.pose.position.z = waypoint["position"][2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.008
            marker.scale.y = 0.008
            marker.scale.z = 0.008
            color_scale = index / max(1, len(ee_trajectory) - 1)
            marker.color.r = 1.0 - color_scale * 0.8
            marker.color.g = 0.5 + color_scale * 0.5
            marker.color.b = color_scale
            marker.color.a = 1.0
            marker_array.markers.append(marker)

        start_marker = Marker()
        start_marker.header.frame_id = self.FK_FRAME
        start_marker.header.stamp = now
        start_marker.ns = "arm_planner_endpoints"
        start_marker.id = 0
        start_marker.type = Marker.SPHERE
        start_marker.action = Marker.ADD
        start_marker.pose.position.x = ee_trajectory[0]["position"][0]
        start_marker.pose.position.y = ee_trajectory[0]["position"][1]
        start_marker.pose.position.z = ee_trajectory[0]["position"][2]
        start_marker.pose.orientation.w = 1.0
        start_marker.scale.x = 0.02
        start_marker.scale.y = 0.02
        start_marker.scale.z = 0.02
        start_marker.color.g = 1.0
        start_marker.color.a = 1.0
        marker_array.markers.append(start_marker)

        goal_marker = Marker()
        goal_marker.header.frame_id = self.FK_FRAME
        goal_marker.header.stamp = now
        goal_marker.ns = "arm_planner_endpoints"
        goal_marker.id = 1
        goal_marker.type = Marker.SPHERE
        goal_marker.action = Marker.ADD
        goal_marker.pose.position.x = ee_trajectory[-1]["position"][0]
        goal_marker.pose.position.y = ee_trajectory[-1]["position"][1]
        goal_marker.pose.position.z = ee_trajectory[-1]["position"][2]
        goal_marker.pose.orientation.w = 1.0
        goal_marker.scale.x = 0.02
        goal_marker.scale.y = 0.02
        goal_marker.scale.z = 0.02
        goal_marker.color.r = 1.0
        goal_marker.color.a = 1.0
        marker_array.markers.append(goal_marker)

        label_marker = Marker()
        label_marker.header.frame_id = self.FK_FRAME
        label_marker.header.stamp = now
        label_marker.ns = "arm_planner_label"
        label_marker.id = 0
        label_marker.type = Marker.TEXT_VIEW_FACING
        label_marker.action = Marker.ADD
        label_marker.pose.position.x = ee_trajectory[-1]["position"][0]
        label_marker.pose.position.y = ee_trajectory[-1]["position"][1]
        label_marker.pose.position.z = ee_trajectory[-1]["position"][2] + 0.05
        label_marker.pose.orientation.w = 1.0
        label_marker.scale.z = 0.025
        label_marker.color.r = 1.0
        label_marker.color.g = 1.0
        label_marker.color.b = 1.0
        label_marker.color.a = 1.0
        label_marker.text = "Goal"
        marker_array.markers.append(label_marker)

        self.marker_pub.publish(marker_array)
        rospy.loginfo("Published %d trajectory markers", len(marker_array.markers))

    def _publish_status(self, status, detail=""):
        msg = String()
        msg.data = "{}: {}".format(status, detail) if detail else status
        self.status_pub.publish(msg)

    def _publish_progress(self, fraction):
        msg = Float32MultiArray()
        msg.data = [float(np.clip(fraction, 0.0, 1.0))]
        self.progress_pub.publish(msg)


def main():
    try:
        ArmPlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()


if __name__ == "__main__":
    main()
