#!/usr/bin/env python3
"""
GUI node with 6 sliders (x, y, z, r, p, y) that continuously publishes
delta EE commands on /EEPose_cmd_rc as Float32MultiArray.
"""

import rospy
from std_msgs.msg import Float32MultiArray
import tkinter as tk


class EERCSliderGUI:
    def __init__(self):
        rospy.init_node("ee_rc_slider_gui", anonymous=True)

        self.topic = rospy.get_param("~topic", "/EEPose_cmd_rc")
        self.rate_hz = rospy.get_param("~rate", 25.0)

        self.xyz_min = rospy.get_param("~xyz_min", -0.01)
        self.xyz_max = rospy.get_param("~xyz_max",  0.01)
        self.rpy_min = rospy.get_param("~rpy_min", -0.05)
        self.rpy_max = rospy.get_param("~rpy_max",  0.05)

        self.pub = rospy.Publisher(self.topic, Float32MultiArray, queue_size=10)

        self.root = tk.Tk()
        self.root.title("EE RC Delta Publisher")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        slider_cfg = [
            ("X", self.xyz_min, self.xyz_max),
            ("Y", self.xyz_min, self.xyz_max),
            ("Z", self.xyz_min, self.xyz_max),
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
            )
            slider.set(0.0)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.sliders.append(slider)

            btn = tk.Button(frame, text="0", width=3,
                            command=lambda s=slider: s.set(0.0))
            btn.pack(side=tk.LEFT, padx=(4, 0))

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="Reset All", command=self._reset_all).pack()

        self._running = True

    def _reset_all(self):
        for s in self.sliders:
            s.set(0.0)

    def _on_close(self):
        self._running = False
        self.root.destroy()

    def run(self):
        period_ms = max(1, int(1000.0 / self.rate_hz))

        def tick():
            if rospy.is_shutdown() or not self._running:
                self.root.destroy()
                return

            msg = Float32MultiArray()
            msg.data = [s.get() for s in self.sliders] + [0.0, 0.0]
            self.pub.publish(msg)

            self.root.after(period_ms, tick)

        self.root.after(period_ms, tick)
        self.root.mainloop()


if __name__ == "__main__":
    try:
        gui = EERCSliderGUI()
        gui.run()
    except rospy.ROSInterruptException:
        pass
