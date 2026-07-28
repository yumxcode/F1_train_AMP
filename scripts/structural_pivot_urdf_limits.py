"""Structural pivot (iter-23, §9): enforce real X1 joint limits in the physics simulation.

ROOT CAUSE of dof_viol: the URDF has placeholder ±π joint limits. Isaac Gym's physics
engine enforces these wide limits, allowing joints to go far beyond the real X1's
conservative range. The env's _apply_manual_joint_limits only sets dof_pos_limits for
the REWARD penalty and ACTION clamp — it does NOT change the physics engine's joint
limits. So PD overshoot can push dof_pos beyond the real X1 limits without the physics
engine stopping it.

FIX: copy the URDF, replace all joint limits with the real X1 values, and point the
AMP env config at the modified URDF. This makes the physics engine HARD-STOP joints
at the real limits — dof_pos cannot exceed them by definition, so dof_viol=0.

This is a STRUCTURAL change (physics constraint), not a reward/algorithm tweak.
"""
import xml.etree.ElementTree as ET
import shutil
import os

SRC = "resources/robots/x1/urdf/x1.urdf"
DST = "resources/robots/x1/urdf/x1_amp_limits.urdf"

# Real X1 joint limits (from motion_lib.py X1_JOINT_LIMITS, same for both legs)
X1_LIMITS = {
    "left_hip_pitch_joint": (-0.90, 0.70),
    "left_hip_roll_joint": (-0.35, 0.35),
    "left_hip_yaw_joint": (-0.45, 0.45),
    "left_knee_pitch_joint": (0.00, 1.45),
    "left_ankle_pitch_joint": (-0.55, 0.45),
    "left_ankle_roll_joint": (-0.30, 0.30),
    "right_hip_pitch_joint": (-0.90, 0.70),
    "right_hip_roll_joint": (-0.35, 0.35),
    "right_hip_yaw_joint": (-0.45, 0.45),
    "right_knee_pitch_joint": (0.00, 1.45),
    "right_ankle_pitch_joint": (-0.55, 0.45),
    "right_ankle_roll_joint": (-0.30, 0.30),
}

tree = ET.parse(SRC)
root = tree.getroot()
replaced = 0
for joint in root.iter("joint"):
    name = joint.get("name", "")
    if name in X1_LIMITS:
        lim = joint.find("limit")
        if lim is not None:
            lo, hi = X1_LIMITS[name]
            old_lo, old_hi = lim.get("lower", "?"), lim.get("upper", "?")
            lim.set("lower", str(lo))
            lim.set("upper", str(hi))
            replaced += 1
            print(f"  {name}: [{old_lo}, {old_hi}] -> [{lo}, {hi}]")

tree.write(DST)
print(f"\nWrote {DST} with {replaced}/12 joint limits replaced.")
print(f"Point x1_amp_config asset.file at this URDF for physics-enforced limits.")
