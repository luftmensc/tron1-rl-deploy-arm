# arm_planner

Collision-aware arm trajectory planner for the TRON1 bipedal robot with a 6-DOF arm. This package bridges MoveIt motion planning with the RL-based whole-body controller by converting planned joint trajectories into end-effector (EE) position targets that the RL policy can track.

## How It Works

The robot's arm is not controlled through traditional joint-position streaming. Instead, an RL policy running inside `SolefootController` reads an **absolute EE pose command** (`/EEPose_cmd_abs`) as part of its observation vector and outputs joint torques to reach that target. The arm response is asynchronous — it depends on the learned policy, joint stiffness/damping, and torque limits.

The `arm_planner_node` handles:

1. **Planning** — Uses MoveIt (`move_group`, CHOMP/OMPL) to find a collision-free joint trajectory to the goal.
2. **FK conversion** — Converts the joint trajectory into an EE path using the `/compute_fk` service.
3. **Densification** — Interpolates the EE path into fine sub-goal waypoints.
4. **Feedback-gated execution** — Publishes EE targets on `/EEPose_cmd_abs` and advances along the path only as the arm catches up (using FK feedback from live joint states). This prevents the target from running too far ahead of the physical arm.

## Dependencies

- `tron_moveit` — MoveIt configuration (URDF, SRDF, kinematics, planning pipelines)
- `robot_hw` — Hardware interface that publishes joint states and runs the RL controller
- MoveIt (`moveit_commander`, `moveit_ros_planning_interface`)

## Launch

```bash
# Source the workspace
source devel/setup.bash

# Make sure the robot HW is running (sim or real)
# Sim:  roslaunch robot_hw pointfoot_hw_sim.launch
# Real: roslaunch robot_hw pointfoot_hw.launch

# Launch the arm planner (includes MoveIt move_group)
roslaunch arm_planner arm_planner.launch
```

Launch arguments:

| Argument | Default | Description |
|---|---|---|
| `launch_move_group` | `true` | Set `false` if move_group is already running |
| `joint_state_topic` | `/pointfoot_hw/joint_states` | Joint state topic from the robot |
| `launch_rviz` | `false` | Launch RViz for visualization |
| `control_rate` | `10.0` | EE target publish rate (Hz) |
| `target_advance_speed` | `0.05` | Cursor speed along the EE path (m/s) |
| `max_target_lead` | `0.10` | Max distance target can lead the arm (m) |
| `settle_time` | `0.5` | Time to keep publishing the final target after arrival (s) |
| `gripper_state` | `0.0` | Gripper command value sent with each EE target |

## Topics

### Input (subscribe to these to send commands)

| Topic | Type | Description |
|---|---|---|
| `/arm_planner/goal_pose` | `std_msgs/String` | Named pose from SRDF (e.g. `start`, `aware`, `up`) |
| `/arm_planner/goal_pose_cartesian` | `std_msgs/Float32MultiArray` | Cartesian EE goal `[x, y, z, roll, pitch, yaw]` in `base_Link` frame |
| `/arm_planner/goal_joint_angles` | `std_msgs/Float32MultiArray` | Joint angles `[J1, J2, J3, J4, J5, J6]` in radians |
| `/arm_planner/stop` | `std_msgs/String` | Stop current execution immediately (any data) |

All three goal topics are **preemptive**: sending a new goal while the arm is moving will immediately abort the current trajectory and start planning/executing the new one.

### Output (published by the node)

| Topic | Type | Description |
|---|---|---|
| `/EEPose_cmd_abs` | `std_msgs/Float32MultiArray` | Absolute EE target sent to the RL controller `[x_off, y_off, z_off, yaw_off, roll_off, pitch_off, gripper]` |
| `/arm_planner/status` | `std_msgs/String` | Execution status (`planning`, `executing`, `done`, `cancelled`, `stopped`, `error`) |
| `/arm_planner/progress` | `std_msgs/Float32MultiArray` | Execution progress `[0.0 .. 1.0]` |
| `/arm_planner/ee_trajectory_markers` | `visualization_msgs/MarkerArray` | RViz trajectory visualization (latched) |
| `/arm_planner/ee_pose_debug` | `std_msgs/Float32MultiArray` | Current EE pose from FK `[x, y, z, roll, pitch, yaw]` |

### Robot interface topics (used internally)

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/pointfoot_hw/joint_states` | `sensor_msgs/JointState` | Subscribe | Live joint positions for FK feedback |
| `/rc_ee_cmd_state` | `std_msgs/Float32MultiArray` | Subscribe | Controller's current EE command state (optional) |
| `/compute_fk` | `moveit_msgs/GetPositionFK` | Service | Forward kinematics from MoveIt |

## Usage Examples

### Named pose (from SRDF)

```bash
rostopic pub /arm_planner/goal_pose std_msgs/String "data: 'start'"
```

Available named poses are defined in the SRDF (`tron_moveit/config/point_foot.srdf`): `aware`, `initial`, `up`, `front_reach`, `start`, `first`, `second`.

### Cartesian EE pose

Send `[x, y, z, roll, pitch, yaw]` in the `base_Link` frame:

```bash
rostopic pub /arm_planner/goal_pose_cartesian std_msgs/Float32MultiArray \
  "data: [0.4, 0.0, 0.3, -1.5708, 0.0, -1.5708]"
```

Note: When using the CHOMP planner (default), the node internally uses `set_pose_target()` which requires Cartesian goal support. If CHOMP rejects it with "Only joint-space goals are supported", switch to OMPL or use the IK-first approach.

### Joint angles

Send `[J1, J2, J3, J4, J5, J6]` in radians:

```bash
rostopic pub /arm_planner/goal_joint_angles std_msgs/Float32MultiArray \
  "data: [-0.111, -1.021, 0.311, 1.286, 0.812, -1.406]"
```

### Stop execution

```bash
rostopic pub /arm_planner/stop std_msgs/String "data: ''"
```

### Preemption

Sending any new goal while the arm is moving will automatically abort the current trajectory and start the new one:

```bash
# Start moving to 'start' pose
rostopic pub /arm_planner/goal_pose std_msgs/String "data: 'start'"

# While it's moving, send a different goal — previous one is aborted
rostopic pub /arm_planner/goal_pose std_msgs/String "data: 'aware'"
```

## Tuning

### Arm movement speed (planner side)

These parameters control how fast the EE target cursor advances along the planned path:

| Parameter | Default | Effect |
|---|---|---|
| `target_advance_speed` | `0.05` m/s | Lower = slower arm movement |
| `max_target_lead` | `0.10` m | Lower = target stays closer to actual arm position |

### Arm response (RL controller side)

These are in `robot_controllers/config/pointfoot/SF_TRON1A/params.yaml` and affect how aggressively the RL policy drives the arm joints:

| Parameter | Default | Effect |
|---|---|---|
| `arm_j123_stiffness` | `18` | Kp for J1-J3. Lower = softer/slower |
| `arm_j123_damping` | `1` | Kd for J1-J3. Higher = more velocity damping |
| `arm_j456_stiffness` | `4` | Kp for J4-J6 |
| `arm_j456_damping` | `0.5` | Kd for J4-J6 |
