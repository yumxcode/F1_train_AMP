# Implementation Checklist — F1 Algorithm Improvement v2

## Phase 1A: 步态参数与脚踝约束 ✅

| # | 修改项 | 文件 | 状态 |
|---|--------|------|------|
| 1 | cycle_time: 0.64 → 1.0 | `x1_dh_stand_config.py` | ✅ |
| 2 | target_feet_height: 0.04 → 0.06 | `x1_dh_stand_config.py` | ✅ |
| 3 | target_feet_height_max: 0.07 → 0.10 | `x1_dh_stand_config.py` | ✅ |
| 4 | toe_scuff_height: 0.035 → 0.04 | `x1_dh_stand_config.py` | ✅ |
| 5 | final_swing_joint_delta_pos: 脚踝(4,5,10,11)置零 | `x1_dh_stand_config.py` | ✅ |
| 6 | 新增 _reward_ankle_motion 奖励函数 | `x1_dh_stand_env.py` | ✅ |
| 7 | 新增 ankle_motion=-0.10, ankle_torques=-0.03 scales | `x1_dh_stand_config.py` | ✅ |

## Phase 1B: 奖励权重重构 ✅

| # | 修改项 | 文件 | 状态 |
|---|--------|------|------|
| 8 | tracking_lin_vel: 2.5 → 3.0 | `x1_dh_stand_config.py` | ✅ |
| 9 | tracking_ang_vel: 0.6 → 1.0 | `x1_dh_stand_config.py` | ✅ |
| 10 | ref_joint_pos: 2.2 → 1.2 | `x1_dh_stand_config.py` | ✅ |
| 11 | action_smoothness: -0.01 → -0.15 | `x1_dh_stand_config.py` | ✅ |
| 12 | torques: -8e-9 → -5e-8 | `x1_dh_stand_config.py` | ✅ |
| 13 | dof_acc: -1e-7 → -1e-6 | `x1_dh_stand_config.py` | ✅ |
| 14 | collision: -1.0 → -5.0 | `x1_dh_stand_config.py` | ✅ |
| 15 | foot_slip: -0.25 → -0.40 | `x1_dh_stand_config.py` | ✅ |
| 16 | feet_clearance: 0.35 → 0.50 | `x1_dh_stand_config.py` | ✅ |
| 17 | feet_contact_forces: -0.01 → -0.03 | `x1_dh_stand_config.py` | ✅ |
| 18 | toe_scuff: -0.8 → -1.0 | `x1_dh_stand_config.py` | ✅ |
| 19 | track_vel_hard: 0.8 → 0.5 | `x1_dh_stand_config.py` | ✅ |
| 20 | feet_air_time: 1.2 → 1.0 | `x1_dh_stand_config.py` | ✅ |
| 21 | default_joint_pos: 1.0 → 0.8 | `x1_dh_stand_config.py` | ✅ |

## Phase 2: 延迟与幅值衰减 DR ✅

| # | 修改项 | 范围 | 状态 |
|---|--------|------|------|
| 22 | dof_lag_timesteps_range: [0,12] → [5,50] | `config.py` | ✅ |
| 23 | randomize_dof_lag_timesteps_perstep: False → True | `config.py` | ✅ |
| 24 | add_dof_pos_vel_lag: False → True | `config.py` | ✅ |
| 25 | dof_pos_lag_timesteps_range: [7,25] → [5,50] | `config.py` | ✅ |
| 26 | dof_vel_lag_timesteps_range: [7,25] → [5,50] | `config.py` | ✅ |
| 27 | add_imu_lag: False → True | `config.py` | ✅ |
| 28 | imu_lag_timesteps_range: [0,8] → [3,20] | `config.py` | ✅ |
| 29 | stiffness_multiplier_range: [0.85,1.15] → [0.6,1.3] | `config.py` | ✅ |
| 30 | torque_multiplier_range: [0.9,1.1] → [0.5,1.2] | `config.py` | ✅ |
| 31 | joint_coulomb_range: [0.1,0.9] → [0.05,1.5] | `config.py` | ✅ |
| 32 | 新增 add_actuator_magnitude_saturation = True | `config.py` | ✅ |
| 33 | 新增 add_actuator_dynamics = True | `config.py` | ✅ |
| 34 | step() 中实现幅值衰减（速度相关衰减 + 随机抖动） | `x1_dh_stand_env.py` | ✅ |
| 35 | step() 中实现一阶低通滤波（10-50ms 时间常数） | `x1_dh_stand_env.py` | ✅ |
| 36 | init 中初始化 filtered_actions 缓冲 | `x1_dh_stand_env.py` | ✅ |
| 37 | reset_idx 中重置 filtered_actions | `x1_dh_stand_env.py` | ✅ |

## Phase 4: 训练参数调优 ✅

| # | 修改项 | 状态 |
|---|--------|------|
| 38 | learning_rate: 1e-5 → 3e-5 | ✅ |
| 39 | num_learning_epochs: 2 → 5 | ✅ |
| 40 | entropy_coef: 0.001 → 0.002 | ✅ |
| 41 | gamma: 0.994 → 0.99 | ✅ |
| 42 | lam: 0.9 → 0.95 | ✅ |
| 43 | max_iterations: 20000 → 30000 | ✅ |

---

## 语法验证

| 文件 | 结果 |
|------|------|
| `x1_dh_stand_config.py` | ✅ Python AST 通过 |
| `x1_dh_stand_env.py` | ✅ Python AST 通过 |
