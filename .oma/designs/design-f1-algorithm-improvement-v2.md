# Design: F1 (智元X1) 运动控制算法改进方案 v2

**Status**: DRAFT  
**Author**: Meta-Agent  
**Date**: 2026-05-27  
**Based on**: `agibot_x1_train` baseline (Isaac Gym + PPO + DH asymmetric actor-critic)

---

## 1. 问题定义

### 1.1 已验证的核心问题

| # | 问题 | 来源 | 严重程度 |
|---|------|------|----------|
| P1 | **20-50ms 执行延迟** — step response 测试确认延迟 20-50ms，仿真未覆盖此范围（当前 DR delay 仅 [0,12] timesteps = 0-12ms） | 经验库 exp_mpnriyrz | 🔴 阻塞 |
| P2 | **高速下关节模组响应不足** — act RMS 仅 des 的 15-20%，高速时幅值衰减更严重，sim 中无幅值衰减建模 | 经验库 exp_mpnriyrz | 🔴 阻塞 |
| P3 | **脚踝动作过多** — swing 参考轨迹驱动 ankle_pitch(±0.16rad)，但实机脚踝执行器响应弱，应尽量减少 | 代码分析 | 🟡 高 |
| P4 | **奖励函数过于密集** — 25 项奖励 + only_positive_rewards，部分奖励项相互冲突（ref_joint_pos vs tracking_lin_vel） | 代码分析 | 🟡 中 |

### 1.2 设计约束

| 约束 | 值 | 理由 |
|------|-----|------|
| 步态周期 | **1.0 s** | 用户要求，降低步频有利于实机执行器跟随 |
| 抬腿高度 | 需设计适合的离地高度 | 用户要求，兼顾"不绊倒"和"能量效率" |
| 延迟覆盖 | **50ms 左右** | 已验证真实延迟 20-50ms，DR 需覆盖到该范围 |
| 脚踝动作 | **尽量少** | 脚踝执行器响应弱，减少不必要的 ankle 动作 |

### 1.3 设计目标

| 指标 | 当前(baseline 实机) | 目标(实机) | 测量方式 |
|------|--------------------|------------|----------|
| 前向速度范围 | ~0.4 m/s (不稳定) | 0.2 - 0.8 m/s | 实机 5m 直线测试 |
| 延迟容忍 | 未建模 → 实机相位滞后 | 50ms 延迟下不显著降质 | step response 事件锁定 |
| 脚踝动作 RMS | ankle_pitch ~0.08rad (摆动期) | < 0.03rad (摆动期) | 关节轨迹分析 |
| 跌倒率 | 频繁 (诊断中) | < 10% / 100步 | 实机测试 |
| 速度跟踪 RMSE | 未量化 | < 0.15 m/s | 命令 vs. 实测 |

---

## 2. 设计方案

### 2.1 Phase 1: 步态参数调整 & 脚踝约束（基线修改）

#### 2.1.1 步态周期改为 1.0s

```python
# x1_dh_stand_config.py
class rewards:
    cycle_time = 1.0   # 0.64 → 1.0
```

**影响分析**：
- 步频从 1.56 Hz 降到 1.0 Hz
- 单步时间从 ~0.32s 增加到 ~0.5s
- 每个 swing 阶段有更多时间完成抬腿→前摆→落地，减轻执行器跟踪负担
- 最大速度理论上受限（step = 0.5s × stride），但实机执行器跟不上当前高速，降速反而稳定

#### 2.1.2 抬腿高度设计

```python
# x1_dh_stand_config.py
target_feet_height = 0.06       # 0.04 → 0.06 略提高，给执行器足够余量
target_feet_height_max = 0.10   # 0.07 → 0.10 放宽上限
toe_scuff_height = 0.04         # 0.035 → 0.04 轻微提高防拖脚
```

**设计理由**：
- 步态周期 1.0s → 摆动时间 ~0.5s → 抬腿时间充裕
- 执行器响应不足（幅值 15-20%）→ 留出余量：期望 0.06m，即便衰减到 0.01-0.012m 仍可避免拖脚
- 当前 0.04m 在幅值衰减下可能只剩 ~0.006-0.008m，接近拖脚

#### 2.1.3 脚踝动作抑制

**策略 A — 修改 swing 参考轨迹，脚踝 delta 置零**（简单直接）：

```python
# final_swing_joint_delta_pos 中脚踝关节设为 0
# 索引 4,5 = left ankle_pitch, left ankle_roll
# 索引 10,11 = right ankle_pitch, right ankle_roll
final_swing_joint_delta_pos = [0.25, 0.05, -0.11, 0.35, 0.0, 0.0,  # left, ankle=0
                               -0.25, -0.05, 0.11, 0.35, 0.0, 0.0] # right, ankle=0
```

**理由**：
- 脚踝在摆动期的主要功能是离地 clearance，由髋膝抬腿实现
- 脚踝 pitch 的主动摆动在小腿前摆时可通过重力实现自然落下
- 去除脚踝参考轨迹后，策略仍可通过 RL 自行学到必要的微调，但不会被迫跟踪不必要的大幅度脚踝动作

**策略 B — 增加脚踝动作惩罚（补充）**：

```python
class scales:
    ankle_torques = -0.05   # 新增：显式惩罚脚踝力矩使用
    ankle_motion = -0.1     # 新增：惩罚脚踝关节位移
```

**实现**：
```python
def _reward_ankle_motion(self):
    """惩罚脚踝关节的主动运动，鼓励自然下垂。"""
    ankle_idx = [4, 5, 10, 11]
    ankle_pos = self.dof_pos[:, ankle_idx]
    ankle_ref = self.default_dof_pos[:, ankle_idx]  # 默认姿态作为参考
    motion = torch.sum(torch.abs(ankle_pos - ankle_ref), dim=1)
    return motion
```

### 2.2 Phase 2: 延迟与幅值衰减域随机化（核心 sim-to-real）

#### 2.2.1 延迟随机化 — 覆盖 50ms

```python
# x1_dh_stand_config.py
class domain_rand:
    # --- 关键修改：扩大延迟到 50ms ---
    add_dof_lag = True
    dof_lag_timesteps_range = [5, 50]    # [0,12] → [5,50] 覆盖5-50ms
    randomize_lag_timesteps_perstep = True   # 每步随机变化，模拟不确定响应
    
    add_dof_pos_vel_lag = True            # 启用 pos/vel 解耦延迟
    dof_pos_lag_timesteps_range = [5, 50] # 位置信号延迟 5-50ms
    dof_vel_lag_timesteps_range = [5, 50] # 速度信号延迟 5-50ms
    randomize_dof_lag_timesteps_perstep = True
    
    add_imu_lag = True                    # 启用 IMU 延迟
    imu_lag_timesteps_range = [3, 20]     # IMU 3-20ms 延迟
```

**延迟模型说明**：
- sim DT = 1ms，decimation = 10 → 策略频率 50Hz（每 10 sim steps 一个 action）
- `dof_lag_timesteps_range = [5, 50]` 表示位置信号延迟 5-50 个 sim steps = **5-50ms**
- `randomize_lag_timesteps_perstep = True` 使每步延迟随机变化，模拟电机不定响应时间
- 解耦 pos/vel 延迟模拟编码器+速度估计的不同处理路径

#### 2.2.2 幅值衰减模型（针对 P2）

新建 `add_actuator_magnitude_saturation` 标志：

```python
# x1_dh_stand_env.py step() 中
if self.cfg.domain_rand.add_actuator_magnitude_saturation:
    # 模拟执行器幅值响应不足：高速时衰减更严重
    cmd_scale = torch.norm(self.commands[:, :2], dim=1, keepdim=True)
    # 速度越高，衰减比例越低：0.5-0.95 随速度变化
    scale_factor = 0.95 - 0.35 * torch.clamp(cmd_scale / 1.2, 0.0, 1.0)
    # 每步随机抖动 ±0.05
    noise = torch_rand_float(-0.05, 0.05, (self.num_envs, 1), device=self.device)
    act_scale = torch.clamp(scale_factor + noise, 0.5, 1.0)
    actions = actions * act_scale
```

此外，扩大扭矩与 PD 增益随机化范围：

```python
randomize_gains = True
stiffness_multiplier_range = [0.6, 1.3]    # [0.85, 1.15] → 扩大
damping_multiplier_range = [0.6, 1.3]      # [0.85, 1.15] → 扩大

randomize_torque = True
torque_multiplier_range = [0.5, 1.2]       # [0.9, 1.1] → 大幅扩大，模拟电机响应不一致
```

#### 2.2.3 一阶低通滤波器模拟电机动态

```python
# step() 中执行动作前
if self.cfg.domain_rand.add_actuator_dynamics:
    # 时间常数 10-50ms，对应电机响应慢
    time_const = torch_rand_float(0.01, 0.05, (self.num_envs, 1), device=self.device)
    alpha = 1.0 / (time_const / self.dt + 1.0)  # 滤波器系数
    self.filtered_actions = (1 - alpha) * self.filtered_actions + alpha * actions
    actions = self.filtered_actions
```

### 2.3 Phase 3: 奖励函数重构

#### 2.3.1 奖励调整总表

| 奖励项 | 原权重 | 新权重 | 说明 |
|--------|--------|--------|------|
| `tracking_lin_vel` | 2.5 | **3.0** | 主要任务，提高权重 |
| `tracking_ang_vel` | 0.6 | **1.0** | 方向稳定性 |
| `ref_joint_pos` | 2.2 | **1.2** | ⬇大幅降低，给策略自由度尤其脚踝 |
| `stand_still` | 2.5 | 2.5 | 维持 |
| `low_speed` | 1.0 | 1.0 | 维持 |
| `default_joint_pos` | 1.0 | 0.8 | 略降 |
| `orientation` | 1.0 | 1.0 | 维持 |
| `feet_air_time` | 1.2 | 1.0 | 略降（周期变长，air time 增加自然） |
| `track_vel_hard` | 0.8 | 0.5 | 降低冗余的硬惩罚 |
| `action_smoothness` | -0.01 | **-0.15** | ⬆大幅提高，抑制高频抖动 |
| `torques` | -8e-9 | **-5e-8** | ⬆提高，降低能耗 |
| `dof_vel` | -2e-8 | -5e-8 | 配合平滑化 |
| `dof_acc` | -1e-7 | **-1e-6** | ⬆惩罚加加速度，平滑运动 |
| `feet_contact_forces` | -0.01 | -0.03 | 减少冲击 |
| `collision` | -1.0 | -5.0 | 严厉惩罚碰撞 |
| `feet_clearance` | 0.35 | 0.50 | 提高确保抬腿够高 |
| `foot_slip` | -0.25 | -0.40 | 严惩打滑 |
| **`ankle_torques`** (已有) | 未配权重 | **-0.03** | 🔑新增配权重：惩罚脚踝力矩 |
| **`ankle_motion`** (新增) | — | **-0.10** | 🔑新增：惩罚脚踝位移 |
| **`toe_scuff`** | -0.8 | -1.0 | 提高防拖脚 |

#### 2.3.2 `only_positive_rewards` 决策

**维持 True**（与 v1 不同）。

理由：
- 周期改为 1.0s 后，步态更慢，负奖励截断的影响变小
- 负奖励截断有助于训练稳定性，尤其当奖励项多且有冲突时
- 方案中已大幅提高 `action_smoothness`、`torques`、`dof_acc` 等负项权重，即使截断为 0，梯度中通过 PPO 的 advantage 计算仍能传递惩罚信号
- 实机测试前保持稳定比追求理论正确更重要

### 2.4 Phase 4: 网络架构与训练优化（可选）

#### 2.4.1 状态估计器扩展

当前输出 3 维线速度 (vx, vy, vz)。扩展为 6 维 (vx, vy, vz, ωx, ωy, ωz)：

- 实机 IMU 提供角速度直接测量，但 state estimator 的估计可作为冗余
- 帮助策略更好感知 yaw 漂移，对 `tracking_ang_vel` 奖励有辅助作用

#### 2.4.2 训练参数调整

| 参数 | 当前 | 修改 | 理由 |
|------|------|------|------|
| `learning_rate` | 1e-5 | **3e-5** | 略微提高，加快收敛 |
| `num_learning_epochs` | 2 | **5** | 更多 epoch 提高数据效率 |
| `entropy_coef` | 0.001 | 0.002 | 鼓励探索，避免局部最优 |
| `max_iterations` | 20000 | **30000** | 周期变长 + 更复杂 DR，需要更多迭代 |
| `desired_kl` | 0.01 | 0.02 | 允许更大更新步长 |

---

## 3. 消融实验计划

### 3.1 实验矩阵

| Exp ID | 修改内容 | 阶段 | 核心变量 |
|--------|----------|------|----------|
| **A1** | Baseline (原始代码) | — | cycle=0.64s, DR default |
| **B1** | 步态周期 1.0s + 抬腿高度调整 | Phase 1 | cycle_time, target_feet_height |
| **B2** | + 脚踝 delta=0 + ankle_motion 惩罚 | Phase 1 | final_swing_joint_delta_pos, scales |
| **B3** | + 奖励重构 (权重调整) | Phase 3 | scales 表 |
| **C1** | + 延迟 DR 覆盖 50ms | Phase 2 | dof_lag_timesteps_range |
| **C2** | + 幅值衰减 + 一阶滤波 | Phase 2 | add_actuator_magnitude_saturation |
| **C3** | = B3 + C2 (完整 Phase 1-3) | ALL | 综合 |
| **D1** | + 训练优化 (lr/epoch/kl) | Phase 4 | 训练参数 |
| **D2** | + 状态估计器扩展 | Phase 4 | state_estimator |

### 3.2 评估协议

所有实验在 sim 内评估以下指标（N=3 seeds）：

| 指标 | 计算方式 | 目标(C3) |
|------|----------|----------|
| 速度跟踪 RMSE | `mean|cmd_vel - base_lin_vel|` | < 0.12 m/s |
| 成功率 | 不跌倒完成 24s | > 95% |
| 脚踝动作 RMS | ankle_pitch 关节 mean|pos - default| | < 0.03 rad |
| yaw 漂移 | 每 5m 累积 yaw 变化 | < 5° |
| 能耗 | `mean(torque × dof_vel)dt` | baseline 的 90% 以内 |

### 3.3 sim2real 验证

Phase 1-3 完成后，进入实机验证：

1. **step response 测试**：发送阶跃动作，测量实机延迟是否在 DR 覆盖范围内（需 < 50ms）
2. **平坦地面慢速行走**：0.2 m/s, 100 步，记录跌倒率
3. **速度扫描**：0.2 → 0.4 → 0.6 → 0.8 m/s，各走 5m
4. **脚踝关节分析**：对比 baseline 与新版策略的 ankle 动作 RMS

---

## 4. 关键设计与实施细节

### 4.1 关节索引总表

```
索引: 0              1              2              3              4              5
关节: left_hip_pitch left_hip_roll  left_hip_yaw   left_knee_pitch left_ankle_pitch left_ankle_roll
动作: 抬腿/放腿      外展/内收      旋转           伸/屈           脚尖上翘/下压   脚踝内外翻

索引: 6              7              8              9              10             11
关节: right_hip_pitch right_hip_roll right_hip_yaw  right_knee_pitch right_ankle_pitch right_ankle_roll
```

**脚踝关节 = [4, 5, 10, 11]**

### 4.2 奖励函数修改要点

**修改 final_swing_joint_delta_pos**：
```python
# B2 修改：脚踝 delta 置零
final_swing_joint_delta_pos = [
    0.25, 0.05, -0.11, 0.35,   # left 髋+膝：保持
    0.0,  0.0,                   # left 踝：置零
   -0.25, -0.05, 0.11, 0.35,   # right 髋+膝：保持
    0.0,  0.0                    # right 踝：置零
]
```

**新增 _reward_ankle_motion**：
```python
def _reward_ankle_motion(self):
    ankle_idx = [4, 5, 10, 11]
    # 直接用关节位置相对 default 的偏移
    diff = self.dof_pos[:, ankle_idx] - self.default_dof_pos[:, ankle_idx]
    return torch.sum(torch.abs(diff), dim=1)
```

### 4.3 DR 修改要点

**信号延迟**（核心覆盖 50ms）：
```python
dof_lag_timesteps_range = [5, 50]       # 5-50ms 覆盖
dof_pos_lag_timesteps_range = [5, 50]   # 解耦 pos 延迟
dof_vel_lag_timesteps_range = [5, 50]   # 解耦 vel 延迟
```

**幅值衰减**（新增）：
```python
add_actuator_magnitude_saturation = True
```
实现为 `step()` 中 `actions *= scale_factor`，scale_factor 范围为 [0.5, 1.0] 且与命令速度负相关。

**一阶低通滤波**（新增）：
```python
add_actuator_dynamics = True
```
实现为 `actions = (1-α)·filtered + α·actions`，时间常数 10-50ms。

---

## 5. 风险评估

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 步态周期 1.0s 导致最大速度下降 | 高 | 中 | 目标是 0.2-0.8m/s，1.0s 周期理论上限 ~1.2m/s（0.6m stride），够用 |
| 脚踝 delta=0 导致摆动期脚踝失控 | 低 | 中 | 策略仍可学习主动控制，仅移除参考轨迹，不是锁定脚踝 |
| DR 太强导致训练不收敛 | 中 | 高 | 分阶段引入：先加延迟，确认收敛后再加幅值衰减和滤波 |
| 训练需要更多迭代 (30000) | 中 | 低 | 计算成本线性增加，可接受 |

---

## 6. 实施路线图

| 步骤 | 内容 | 产出 | 预计时间 |
|------|------|------|----------|
| 1 | 修改 cycle_time, target_feet_height, toe_scuff_height | `x1_dh_stand_config.py` | 5 min |
| 2 | 修改 final_swing_joint_delta_pos（脚踝置零） | `x1_dh_stand_config.py` | 5 min |
| 3 | 新增 _reward_ankle_motion + scales 调整 | `x1_dh_stand_env.py` + `config` | 15 min |
| 4 | 增强延迟 DR（扩大到 50ms + 解耦 pos/vel + IMU） | `x1_dh_stand_config.py` | 10 min |
| 5 | 新增幅值衰减 + 一阶低通滤波 | `x1_dh_stand_env.py` | 20 min |
| 6 | 奖励权重表更新 | `x1_dh_stand_config.py` | 10 min |
| 7 | 训练参数调优 | `x1_dh_stand_config.py` | 10 min |
| 8 | 训练 B3/C1/C2/C3 实验并评估 | 各 1-2 天 | 3-5 天 |
| 9 | 状态估计器扩展（可选） | `actor_critic_dh.py` | 1 天 |

**总计代码修改时间**：~1 小时  
**总计训练验证时间**：~5 天
