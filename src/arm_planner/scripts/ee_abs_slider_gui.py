#!/usr/bin/env python3
"""
GUI node with 6 sliders (x, y, z, r, p, y) that publishes absolute
EE commands on /EEPose_cmd_abs as Float32MultiArray.

Supports two modes toggled by a button:
  - Always Publish (default): publishes at a fixed rate (25 Hz)
  - Publish on Change: only publishes when a slider value changes
"""

import rospy
from std_msgs.msg import Float32MultiArray
import tkinter as tk


class EEAbsSliderGUI:
    def __init__(self):
        rospy.init_node("ee_abs_slider_gui", anonymous=True)

        self.topic = rospy.get_param("~topic", "/EEPose_cmd_abs")
        self.rate_hz = rospy.get_param("~rate", 25.0)

        import math
        self.init_vals = [
            rospy.get_param("~init_x",     0.446),
            rospy.get_param("~init_y",     0.0),
            rospy.get_param("~init_z",     0.241),
            rospy.get_param("~init_roll",  -math.pi / 2),
            rospy.get_param("~init_pitch", 0.0),
            rospy.get_param("~init_yaw",   -math.pi / 2),
        ]

        self.xyz_min = rospy.get_param("~xyz_min", -1.0)
        self.xyz_max = rospy.get_param("~xyz_max",  1.0)
        self.rpy_min = rospy.get_param("~rpy_min", -3.14)
        self.rpy_max = rospy.get_param("~rpy_max",  3.14)

        self.pub = rospy.Publisher(self.topic, Float32MultiArray, queue_size=10)

        self.root = tk.Tk()
        self.root.title("EE Abs Publisher")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        slider_cfg = [
            ("X",     self.xyz_min, self.xyz_max),
            ("Y",     self.xyz_min, self.xyz_max),
            ("Z",     self.xyz_min, self.xyz_max),
            ("Yaw",  self.rpy_min, self.rpy_max),
            ("Pitch", self.rpy_min, self.rpy_max),
            ("Roll",   self.rpy_min, self.rpy_max),
        ]

        self.sliders = []
        for i, (label, lo, hi) in enumerate(slider_cfg):
            frame = tk.Frame(self.root)
            frame.pack(fill=tk.X, padx=10, pady=2)

            tk.Label(frame, text=label, width=6, anchor="w").pack(side=tk.LEFT)

            slider = tk.Scale(
                frame, from_=lo, to=hi,
                resolution=0.001, orient=tk.HORIZONTAL, length=350,
                command=lambda val, idx=i: self._on_slider_change(idx, val),
            )
            slider.set(self.init_vals[i])
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sliders.append(slider)

            init_v = self.init_vals[i]
            btn = tk.Button(frame, text="reset", width=5,
                            command=lambda s=slider, v=init_v: s.set(v))
            btn.pack(side=tk.LEFT, padx=(4, 0))

        # --- control buttons ---
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=6)

        self._always_publish = True
        self._mode_btn = tk.Button(
            btn_frame, text="Mode: Always Publish", width=22,
            command=self._toggle_mode,
        )
        self._mode_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(btn_frame, text="Reset All", command=self._reset_all).pack(side=tk.LEFT, padx=4)

        self._running = True
        self._last_values = list(self.init_vals)

    # --- helpers ---

    def _get_msg(self):
        msg = Float32MultiArray()
        # publish offset from init pose (what the controller expects)
        vals = [s.get() - self.init_vals[i] for i, s in enumerate(self.sliders)]
        msg.data = vals + [0.0, 0.0]
        return msg

    def _on_slider_change(self, idx, val):
        if not self._always_publish:
            current = [s.get() for s in self.sliders]
            if current != self._last_values:
                self._last_values = list(current)
                self.pub.publish(self._get_msg())

    def _toggle_mode(self):
        self._always_publish = not self._always_publish
        if self._always_publish:
            self._mode_btn.config(text="Mode: Always Publish")
        else:
            self._mode_btn.config(text="Mode: On Change")
            self._last_values = [s.get() for s in self.sliders]

    def _reset_all(self):
        for s, v in zip(self.sliders, self.init_vals):
            s.set(v)

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def run(self):
        period_ms = max(1, int(1000.0 / self.rate_hz))

        def tick():
            if rospy.is_shutdown() or not self._running:
                self.root.destroy()
                return

            if self._always_publish:
                self.pub.publish(self._get_msg())

            self.root.after(period_ms, tick)

        self.root.after(period_ms, tick)
        self.root.mainloop()


if __name__ == "__main__":
    try:
        gui = EEAbsSliderGUI()
        gui.run()
    except rospy.ROSInterruptException:
        pass
