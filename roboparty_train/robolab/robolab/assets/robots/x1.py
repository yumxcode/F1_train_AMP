# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

X1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/x1/urdf/x1.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            # legs – slight squat for stable standing
            "left_hip_pitch_joint": -0.1,
            "right_hip_pitch_joint": -0.1,
            "left_knee_pitch_joint": 0.2,
            "right_knee_pitch_joint": 0.2,
            "left_ankle_pitch_joint": -0.1,
            "right_ankle_pitch_joint": -0.1,
            # arms – natural slightly-forward pose
            "left_shoulder_pitch_joint": 0.3,
            "right_shoulder_pitch_joint": 0.3,
            "left_elbow_pitch_joint": 0.3,
            "right_elbow_pitch_joint": 0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "waist_legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                "lumbar_yaw_joint",
                "lumbar_roll_joint",
                "lumbar_pitch_joint",
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim=180.0,
            velocity_limit_sim=27.0,
            stiffness={
                "lumbar_yaw_joint": 120.0,
                "lumbar_roll_joint": 120.0,
                "lumbar_pitch_joint": 150.0,
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 150.0,
                ".*_knee_pitch_joint": 150.0,
            },
            damping={
                "lumbar_yaw_joint": 4.0,
                "lumbar_roll_joint": 4.0,
                "lumbar_pitch_joint": 5.0,
                ".*_hip_yaw_joint": 4.0,
                ".*_hip_roll_joint": 4.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_pitch_joint": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=14.0,
            stiffness=60.0,
            damping=2.5,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=14.0,
            stiffness=25.0,
            damping=1.5,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=14.0,
            stiffness={
                ".*_elbow_pitch_joint": 20.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.0,
                ".*_elbow_yaw_joint": 1.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "wrists": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wrist_pitch_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=12.0,
            velocity_limit_sim=3.0,
            stiffness=10.0,
            damping=0.5,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


X1_LINKS = [
    "base_link",
    "lumbar_yaw_link",
    "lumbar_roll_link",
    "lumbar_pitch_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "left_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
    "right_wrist_pitch_link",
    "right_wrist_roll_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]
