# Design: F1 (智元X1) 运动控制算法改进方案 v1

**Status**: DRAFT  
**Author**: Meta-Agent  
**Date**: 2026-05-27  
**Based on**: `agibot_x1_train` baseline (Isaac Gym + PPO + DH asymmetric actor-critic)

---

## 1. 问题定义

### 1.1 已知问题

| # | 问题 | 来源 | 严重程度 |
|---|------|------|----------|
| P1 | **电机响应不一致** — R52/R86 系列执行器动态特性在实机上不稳定 | Hardware Profile | 🔴 阻塞 |
| P2 | **左 hip_roll 策略输出不合理** — 摆动期 84% 帧要求外展而非内收，导致 yaw drift -36° | 经验库 exp_mpnriyrz | 🔴 阻塞 |
| P3 | **20-50ms 执行延迟** — 实际延迟使仿真中训练的 policy 在实机上相位滞后 | 经验库 exp_mpnriyrz | 🟡 高 |
| P4 | **act RMS 仅 des 的 15-20%** — 执行器幅值响应严重不足，方向匹配率仅 ~34% | 经验库 exp_mpnriyrz | 🟡 高 |
| P5 | **奖励函数过于密集且冲突** — 25 项奖励 (+only_positive)，ref_joint_pos 与 tracking_lin_vel 在高速时冲突 | 代码分析 | 🟡 中 |
| P6 | **CNN 长时历史压缩瓶颈** — 66帧×47d→64d 经 2 层 Conv1D，信息丢失严重 | 代码分析 | 🟢 中 |
| P7 | **步态周期固定 0.64s** — 无法自适应速度变化 | 代码分析 | 🟢 低 |

### 1.2 设计目标

| 指标 | 当前(baseline sim) | 目标(实机) | 测量方式 |
|------|--------------------|------------|----------|
| 前向速度范围 | 0.4 - 1.2 m/s | 0.2 - 1.0 m/s | 实机 5m 直线测试 |
| yaw drift per 5m | -36° (有问题) | < ±5° | IMU 积分 |
| 速度跟踪 RMSE | — | < 0.15 m/s | 命令 vs. 实测 |
| 跌倒率（平坦地面） | — | < 5% / 100步 | 实机测试 |
| 步态对称性 | 左 hip_roll 故障 | 左右对称 | 关节轨迹分析 |

---

## 2. 设计方案

### 2.1 Phase 1: 奖励重构与执行器随机化增强（短期，1-2周）

#### 2.1.1 奖励函数重构

**原则**：减少奖励项数量，消除冲突，负信号必须可感知。

```
修改前: 25项奖励 + only_positive_rewards=True + action_smoothness=-0.01
修改后: 约15项核心奖励 + only_positive_rewards=False + action_smoothness=-0.1
```

**具体调整**：

| 奖励项 | 原权重 | 新权重 | 理由 |
|--------|--------|--------|------|
| `tracking_lin_vel` | 2.5 | 3.0 | 主要任务，提升优先级 |
| `tracking_ang_vel` | 0.6 | 1.0 | yaw drift 问题的针对性修复 |
| `ref_joint_pos` | 2.2 | 1.5 | 降低刚性轨迹约束，给策略自由度 |
| `stand_still` | 2.5 | 2.0 | 略降以平衡 |
| `action_smoothness` | -0.01 | **-0.15** | 🔑大幅提升，抑制高频抖动 |
| `torques` | -8e-9 | **-5e-8** | 🔑提升，抑制不必要力矩 |
| `dof_vel` | -2e-8 | -5e-8 | 配合平滑化 |
| `dof_acc` | -1e-7 | -5e-7 | 加速度惩罚（jerk隐式抑制） |
| `feet_contact_forces` | -0.01 | -0.03 | 减少冲击 |
| `collision` | -1.0 | -5.0 | 更严厉地惩罚非预期碰撞 |
| **新增: `hip_roll_symmetry`** | — | **-0.5** | 🔑针对P2：对称性损失 |
| **新增: `ankle_torques`** | (exists, 未配权重) | **-0.02** | 踝关节力矩效率 |

**`_reward_hip_roll_symmetry` 实现**：
```python
def _reward_hip_roll_symmetry(self):
    """惩罚左右 hip_roll 的不对称。针对 P2：左腿摆动期应内收而非外展。"""
    left_hip_roll = self.dof_pos[:, 1]  # left_hip_roll_joint
    right_hip_roll = self.dof_pos[:, 7] # right_hip_roll_joint
    # 期望左右 hip_roll 在摆动期对称
    asymmetry = torch.abs(left_hip_roll + right_hip_roll)
    # 仅在摆动期惩罚（stance脚承重时允许不对称）
    swing_mask = 1 - self._get_stance_mask()
    left_swing = swing_mask[:, 0]
    right_swing = swing_mask[:, 1]
    # 左hip_roll摆动期应内收(负), 右hip_roll摆动期应外展(正)
    left_err = torch.abs(left_hip_roll - (-right_hip_roll)) * left_swing
    right_err = torch.abs(right_hip_roll - (-left_hip_roll)) * right_swing
    return left_err + right_err
```

**移除 `only_positive_rewards` 的影响**：
- 风险：训练初期可能不稳定（负奖励导致过早终止）
- 缓解：同时降低 `_reward_termination` = -0（当前为 0，维持不变），确保终止本身不被惩罚
- 实际效果：策略能学会"什么动作不好"，长期收益大于初期波动

#### 2.1.2 执行器延迟随机化增强

**针对 P1（电机响应不一致）+ P3（20-50ms 延迟）+ P4（幅值不足）**：

| 参数 | 当前 | 修改后 | 说明 |
|------|------|--------|------|
| `add_dof_lag` | True | True | 保持不变 |
| `dof_lag_timesteps_range` | [0, 12] | **[3, 25]** | 扩大覆盖 3-25ms 延迟 |
| `randomize_lag_timesteps_perstep` | False | **True** | 每步随机延迟，模拟不确定响应 |
| `add_dof_pos_vel_lag` | False | **True** | 位置和速度延迟解耦 |
| `dof_pos_lag_timesteps_range` | [7, 25] | **[5, 30]** | 扩大范围 |
| `dof_vel_lag_timesteps_range` | [7, 25] | **[5, 30]** | 同步扩大 |
| `add_imu_lag` | False | **True** | 启用 IMU 延迟，模拟传感器滞后 |
| `imu_lag_timesteps_range` | [0, 8] | **[2, 15]** | 扩大 |
| `randomize_gains` | [0.85, 1.15] | **[0.7, 1.3]** | 扩大 PD 增益随机化范围 |
| `torque_multiplier_range` | [0.9, 1.1] | **[0.6, 1.2]** | 模拟"电机响应不一致" |
| `joint_coulomb_range` | [0.1, 0.9] | **[0.05, 1.5]** | 扩大库仑摩擦范围 |
| `joint_friction_range` | [0.01, 1.15] | [0.01, 1.5] | 略扩大 |

**关键新增** — **幅值衰减模型**（针对 P4：act 仅 des 的 15-20%）：

```python
# x1_dh_stand_env.py 的 step() 方法内
if self.cfg.domain_rand.randomize_torque:
    # 模拟执行器输出幅值不足
    act_scale = torch_rand_float(0.6, 1.0, (self.num_envs, 1), device=self.device)
    actions = actions * act_scale
```

#### 2.1.3 执行器响应模型

针对 P1（电机响应不一致）的更精确建模：

```python
# 在 step() 中引入一阶低通滤波模拟电机动态
if self.cfg.domain_rand.add_actuator_dynamics:
    tau = torch_rand_float(0.01, 0.05, (self.num_envs, 1), device=self.device)  # 10-50ms 时间常数
    self.filtered_actions = (1 - 1/(tau/self.dt)) * self.filtered_actions + (1/(tau/self.dt)) * actions
    actions = self.filtered_actions
```

### 2.2 Phase 2: 自适应步态参数（中期，2-3周）

#### 2.2.1 速度自适应步频

将固定的 cycle_time 0.64s 改为速度自适应：

```python
# 根据当前命令速度动态调整步态周期
cmd_speed = torch.norm(self.commands[:, :2], dim=1)
# 低速(0.2m/s): cycle=0.8s, 高速(1.0m/s): cycle=0.5s
cycle_time = 0.80 - 0.30 * (cmd_speed / self.command_ranges["lin_vel_x"][1])
cycle_time = torch.clamp(cycle_time, 0.45, 0.90)
```

#### 2.2.2 步幅奖励与速度联动

当前 `stride_length_target = 0.36` 固定。改为：

```python
stride_target = 0.20 + 0.40 * torch.clamp(cmd_speed / self.command_ranges["lin_vel_x"][1], min=0.0, max=1.0)
```

#### 2.2.3 离地高度自适应

```python
# 高速时降低足部离地高度（能量效率），低速时提高（越障能力）
target_feet_height = 0.03 + 0.04 * (1.0 - cmd_speed / self.command_ranges["lin_vel_x"][1])
```

### 2.3 Phase 3: 网络架构升级（中期，3-4周）

#### 2.3.1 长期历史编码器：Transformer 替代 CNN

| 组件 | 当前 (CNN) | 改进 (Transformer) |
|------|------------|--------------------|
| 编码器类型 | 2层 Conv1D (k=6,4) | 2层 TransformerEncoder |
| 输出维度 | 64 | 64 |
| 参数量 | ~2.5M | ~3.2M（可控） |
| 关键优势 | 局部时序模式 | **全局注意力+长程依赖** |
| 推理成本 | 低 | 中等（~1.2×） |

**架构图**：
```
观测 (66帧 × 47d) → Linear(47→64) + PositionalEncoding 
  → TransformerEncoder(L=2, H=4, d_model=64) 
    → [CLS] token 取均值 → Linear(64→64)
      → concat(短期历史编码, 状态估计器输出) → Actor MLP
```

#### 2.3.2 状态估计器扩展

当前输出 3 维线速度 (vx, vy, vz)。

| 输出 | 当前 | 扩展后 | 用途 |
|------|------|--------|------|
| `est_lin_vel` | (vx, vy, vz) ✓ | (vx, vy, vz) ✓ | 速度反馈 |
| `est_ang_vel` | — | (ωx, ωy, ωz) ✚ | 角速度估计，辅助 yaw 控制 |
| `est_foot_contact` | — | (left, right) ✚ | 触地检测备份 |

#### 2.3.3 Actor 网络增容

```
当前: [512, 256, 128]
改进: [1024, 512, 256]
```
增加容量约 2 倍，提升策略表达能力。参数量约 50 万→120 万。

### 2.4 Phase 4: 训练策略优化

| 参数 | 当前 | 修改 | 理由 |
|------|------|------|------|
| `learning_rate` | 1e-5 | **5e-5** | 当前过低，20000次迭代可能未充分收敛 |
| `num_learning_epochs` | 2 | **5** | 更多 epoch 提高数据效率 |
| `entropy_coef` | 0.001 | 0.002 | 适度鼓励探索，防止过早陷入局部最优 |
| `gamma` | 0.994 | 0.99 | 略降低折扣，更关注近期奖励 |
| `lam` | 0.9 | 0.95 | 略提高 GAE lambda |
| `desired_kl` | 0.01 | 0.02 | 允许更大的策略更新步长 |

---

## 3. 消融实验计划

### 3.1 实验矩阵

| Exp ID | 变量 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | 预期效果 |
|--------|------|---------|---------|---------|---------|----------|
| A1 | baseline | ✗ | ✗ | ✗ | ✗ | 当前策略 |
| B1 | 奖励重构 | ✓ | ✗ | ✗ | ✗ | 跟踪改善 +5% |
| B2 | DR 增强 | ✓ | ✗ | ✗ | ✗ | sim2real 鲁棒 +10% |
| B3 | B1+B2 | ✓ | ✗ | ✗ | ✗ | 综合提升 |
| C1 | +自适应步态 | ✓ | ✓ | ✗ | ✗ | 速度范围扩大 |
| D1 | +Transformer | ✓ | ✓ | ✓ | ✗ | 复杂地形改善 |
| D2 | +训练策略 | ✓ | ✓ | ✓ | ✓ | 最终版本 |

### 3.2 评估指标（sim 内）

| 指标 | 计算方式 | 目标(B3) |
|------|----------|----------|
| 速度跟踪 RMSE | `mean\|cmd_vel - base_lin_vel\|` | < 0.12 m/s |
| yaw 跟踪 RMSE | `mean\|cmd_yaw_rate - base_ang_vel_z\|` | < 0.15 rad/s |
| 能耗 | `mean(torque·dof_vel)dt` | baseline 的 90% |
| 成功率 | 不跌倒情况下完成 20s 轨迹 | > 95% |
| 对称性指标 | left/right hip_roll 轨迹互相关 | > 0.85 |

---

## 4. 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 奖励重构导致训练不稳定 | 中 | 高 | 逐步引入，先改权重再移除 only_positive |
| DR 过度使仿真过于困难 | 中 | 中 | 分阶段扩大范围，监控 DR 成功率 |
| Transformer 推理速度不达标 | 低 | 高 | 前向传播基准测试；准备 LSTM fallback |
| 训练成本增加（参数多+epoch多） | 高 | 中 | 保持 4096 envs；监控 FPS |
| 训练 20000 次迭代不足 | 中 | 中 | 监控收敛曲线，准备 max_iter=30000 |

---

## 5. 实施计划

| 步骤 | 内容 | 产出 | 预计时间 |
|------|------|------|----------|
| 1 | Phase 1 奖励修改 | `x1_dh_stand_config.py` 更新 | 1 天 |
| 2 | Phase 1 DR 增强 | `x1_dh_stand_config.py` + `_step` 修改 | 1 天 |
| 3 | 训练 B1/B2/B3 并评估 | 实验数据 | 1-2 天(含训练) |
| 4 | Phase 2 自适应步态 | `x1_dh_stand_env.py` 修改 | 1 天 |
| 5 | 训练 C1 并评估 | 实验数据 | 1 天 |
| 6 | Phase 3 Transformer + SE 扩展 | `actor_critic_dh.py` 新架构 | 2-3 天 |
| 7 | 训练 D1 并评估 | 实验数据 | 1-2 天 |
| 8 | Phase 4 训练策略调优 | `dh_ppo.py` 参数 | 0.5 天 |
| 9 | 最终训练 D2 | 最终策略 | 2 天 |

---

## 6. 参考文献

- [Legged Gym](https://github.com/leggedrobotics/legged_gym) — 基础架构
- [Humanoid-Gym: Zero-Shot Sim2Real Transfer](https://arxiv.org/abs/2404.05695) — 奖励与 DR 设计参考
- [RSL RL](https://github.com/leggedrobotics/rsl_rl) — PPO 实现
- 经验库 `exp_mpnriyrz_5c53fece` — X1 真机诊断与延迟分析
