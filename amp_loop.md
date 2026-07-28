# F1 AMP 行走模仿学习长期自主研究 Loop 需求

目标：将 `F1_train_AMP` 从当前普通 PPO 行走项目改造成真正可训练、可评估、可导出的 AMP（Adversarial Motion Priors）项目；完成 F1 行走动作重定向并生成可靠的专家运动数据；最后训练 F1，使其能够以参考步态稳定行走。该任务是长周期自主研究任务，必须按“AMP 工程化 → F1 行走重定向 → AMP 训练达标”的顺序推进，建立可恢复、可追溯、可重复迭代的工作流。

参考实现与数据流程：

- Roboparty 训练说明：<https://github.com/Roboparty/roboto_origin/blob/main/modules/roboparty_train/README_CN.md>
- Roboparty GMR：<https://github.com/Roboparty/GMR>
- roboparty_train参考项目只用于理解 AMP 组成、动作数据准备、关节重排、训练/测试和 Sim2Sim 流程；必须结合 F1 的 URDF/MJCF、关节定义、控制频率和现有工程结构实现，禁止直接照搬 RPO/Atom01 的关节索引或参数。
- GMR重定向参考项目是重定向的参照实现，请全面复现GMR F1适配版，必须结合 F1 的 URDF/MJCF，不得简化IK等，确保重定向结果准确。

请按以下协议设计图。

## 1. 强制阶段与门禁

图必须显式维护 `phase`，取值至少包括：

- `amp_conversion`：把现有工程改造成 AMP 项目；
- `motion_retargeting`：把参考行走动作重定向到 F1；
- `amp_training`：使用通过验收的 F1 专家动作训练；
- `completed`：全部训练验收通过。

阶段顺序是 hard constraint：

1. `amp_conversion` 未通过 Gate A，不得进入重定向或正式训练；
2. `motion_retargeting` 未通过 Gate B，不得把数据用于正式 AMP 训练；
3. 只有 `amp_training` 通过 Gate C 才能进入 `completed`；
4. 禁止通过修改状态文件、降低验收阈值或跳过失败检查跨越门禁。

每个阶段允许独立迭代和 structural pivot。切换阶段时必须写入阶段验收证据、代码版本、配置版本和数据版本。

## 2. 状态目录

任务状态放在 `{taskDir}/state/`：

- `task_spec.md`：总目标、三个阶段的里程碑、固定评测集、数值阈值和成功标准；
- `progress.json`：至少包含 `iteration`、`phase`、`phase_status`、`status`、`stale_count`、`total_findings`、`code_revision`、`dataset_revision`、`updated_at`；
- `findings.jsonl`：append-only，记录每轮新增 findings，且标记所属阶段、实验/数据版本和证据路径；
- `directions_tried.json`：按阶段记录已尝试方向、假设、结果和是否允许重试；
- `iteration_log.jsonl`：append-only，记录每轮摘要、判断、门禁结果和路由；
- `amp_contract.json`：AMP observation、expert transition、policy transition、discriminator、replay buffer、style reward、task reward、归一化和 checkpoint/export 的接口契约；
- `retarget_manifest.json`：源动作、许可证/来源、F1 模型版本、关节映射、坐标系、单位、四元数约定、帧率、裁剪范围、输出文件及校验统计；
- `evaluation.json`：固定评测协议、各 checkpoint 指标、seed 级结果和最终验收结论。

日志放在 `{taskDir}/logs/`：

- `work.jsonl`；
- `orchestrator.jsonl`；
- 远端训练任务 ID、监控时间点、停止原因和指标拉取记录。

大体积动作数据和训练产物不要写进 JSON state；state 中只保存稳定的项目内相对路径、哈希和摘要。

## 3. 启动审计

首次运行必须先审计当前仓库，不能从项目名推断它已经是 F1 或 AMP：

- 核对真实机器人身份、URDF/MJCF、自由度、actuated joint、关节上下限、默认姿态、关节顺序、body/link 名称、足端定义和碰撞体；
- 核对训练环境、task 注册、actor/critic、runner、PPO、rollout storage、train/play/export 和 Sim2Sim 数据流；
- 搜索并列出现有 AMP、motion loader、discriminator、expert replay buffer、style reward、retarget 工具和专家动作数据；
- 记录当前仓库中遗留的 X1 命名或假设，判断哪些仅是命名问题，哪些会导致 F1 资产、观测、动作或关节映射错误；
- 生成 baseline：现有普通 PPO 是否能启动、短程训练、play 和 Sim2Sim；保存命令、版本和输出证据。

若真实 F1 模型、参考行走源动作或其合法可用来源缺失，将其记录为明确的 blocking precondition，不得用 X1/RPO 数据冒充 F1 结果。

## 4. 阶段一：AMP 工程化（Gate A）

以当前项目为基线实现完整 AMP 闭环，至少包括：

- 定义 AMP observation 和 transition，明确包含哪些 root、关节、末端、速度和历史帧特征；expert 与 policy 数据必须具有完全一致的维度、顺序、单位、坐标系和时间间隔；
- 专家动作 loader：读取重定向数据，支持 clip 边界、时间采样/插值、循环动作、合法性检查和确定性复现；
- policy transition 收集和 replay buffer；
- discriminator 网络、优化器、expert/policy minibatch、梯度惩罚或参考实现所需的正则项；
- discriminator 输出到 style reward 的变换、数值裁剪/归一化，以及 style reward 与 task reward 的组合；
- 在 PPO/runner/storage/config/checkpoint 中接入 AMP 训练状态，resume 后不得丢失 discriminator、optimizer、normalizer 或必要 buffer 状态；
- 独立的 F1 AMP task 注册，以及对应 train、play、checkpoint、JIT/ONNX 导出入口；
- 指标记录至少覆盖 policy/task/style reward、discriminator expert/policy loss、accuracy/logit、gradient penalty、PPO loss、episode length、fall/termination；
- 保留最小可用的普通 PPO baseline，方便做回归和 AMP 增益对照。

Gate A 必须同时满足：

- 静态检查、import 和配置解析通过；
- 用小数据、小环境数完成一次 bounded smoke train，AMP 各模块确实执行且参数发生有限更新；
- expert/policy transition schema 自动比对通过，无 NaN/Inf、空数据集、越界索引或错误 broadcasting；
- checkpoint save/resume 和 play 至少各验证一次；
- AMP task 不是普通 PPO task 的改名，训练日志中可看到独立的 discriminator 与 style reward 证据；
- 形成 `amp_contract.json`，并将验证命令、日志和结果写入 findings。

## 5. 阶段二：F1 行走重定向（Gate B）

行走重定向是 AMP 的关键输入，必须作为独立阶段优化，不得把未经验证的数据直接送入训练。

重定向流程按照参照GMR项目流程包括：

- 选择来源明确、允许使用、包含稳定周期行走的参考动作；保存原始数据哈希、fps、clip 范围和左右脚接触信息；
- 按 F1 URDF/MJCF 与训练环境实际使用顺序建立显式 joint mapping，再转换为 AMP loader 所需格式；不可依赖字典插入顺序或硬编码他机索引；
- 生成可视化视频、数值报告和最终专家动作 manifest；原始、重定向中间结果、关节重排后训练数据要版本化区分。

每个候选 clip 都必须检查：

- joint name 一一对应，维度/顺序与 F1 task 一致，无缺失、重复或静默截断；
- 关节位置/速度不越过 F1 可接受范围，root 高度、姿态和运动方向合理，无 NaN/Inf 和突跳；
- 左右脚接触时序与参考动作相符，支撑脚滑移、穿地、悬空、膝部反折和自碰撞在固定阈值内；
- 首尾帧用于循环时位置、姿态、速度和接触连续；不满足循环条件的 clip 不得强制循环；
- 在 MuJoCo 或等价运动学回放中完整播放并人工/自动复核，而不是只确认文件可读取。

Gate B 必须同时满足：

- 数据通过 schema、单位、坐标系、关节映射、范围、连续性和接触检查；
- 至少一个参考行走 clip 通过完整可视化回放和定量验收；
- AMP loader 读取该数据后的统计与重定向输出一致；
- 不合格 clip 被隔离并记录原因，不得混入专家数据集。

注：
重定向参考动作：/Users/yumx/code/robot_x/X1/Retargeting_X1/BMLrub_stageii/rub058/0008_normal_walk4_stageii.npz

## 6. 阶段三：AMP 训练与最终指标（Gate C）

训练目标是 F1 能在策略闭环中按照参考步态稳定行走，而不是只获得更高总 reward 或让 discriminator loss 看起来正常。首次进入本阶段时，必须在 `task_spec.md` 中冻结固定评测集、命令范围、地形、domain randomization、seed 和数值阈值；后续不得为了宣告成功而降低阈值。若项目已有正式指标，以项目指标为准；否则至少采用下面的验收框架并给出明确数值。

最终评测至少覆盖以下四组指标：

1. 模仿质量：AMP/style reward、参考与策略的关节姿态/速度误差、root 高度与姿态误差、步频/周期误差、左右脚接触相位一致性；
2. 行走任务：前向速度跟踪误差、横向漂移、yaw 漂移、目标距离/位移完成率；
3. 稳定性：episode 完成率、跌倒率、连续无摔倒时长、base roll/pitch、足端滑移、异常双腾空/双支撑比例；
4. 安全与可部署性：关节限位、速度/扭矩饱和率、非足端碰撞、自碰撞、NaN/Inf，以及 Isaac Gym → MuJoCo Sim2Sim 回归。

最低验收协议：

- 使用未参与调参的固定 seed 集合，至少 5 个 seeds；汇报均值、标准差和最差 seed，禁止只选最好视频/checkpoint；
- nominal 环境中参考命令下连续稳定行走至少 60 秒，无跌倒、无 NaN/Inf、无关节硬限位违规；
- 固定批量评测中 episode 成功率至少 95%、跌倒率不高于 5%；
- 速度、步态相位、足滑、姿态和模仿误差的具体阈值必须在首次正式训练前依据 F1 尺寸、控制频率和参考 clip 写入 `task_spec.md`；
- 在受控 domain randomization 和 MuJoCo Sim2Sim 中重复评测；若未达到预先冻结的 Sim2Sim 阈值，不得标记 completed；
- 最终 checkpoint 必须可 resume、可 play、可导出，且导出策略的 observation/action 契约与训练一致。
- F1关节限位可稍微放宽，不作为严格要求（但不要出现频繁、过量的关节动作超限）

判定训练改善时采用固定评测结果，不以训练窗口内的 reward 峰值、单次视频或 discriminator accuracy 单独作结论。

## 7. 每轮 loop

图中包含这些阶段感知节点：

- `load_state`：读取 task spec、phase、progress、directions、findings 摘要、AMP contract、retarget manifest 和 evaluation；
- `audit_preconditions`：检查当前阶段的代码、数据、CLI、凭据、远端资源和前一 Gate 证据；
- `choose_direction`：只选择当前 phase 内与历史不同、可证伪的方向，写清假设、改动面和预期指标；
- `research_design_execute`：按当前 phase 执行 AMP 实现、重定向实验或训练实验；一次只改变一个主要假设，保留可比较 baseline；
- `monitor_remote_job`：监控 Gradmotion 任务并拉取曲线与产物；
- `extract_findings`：将实现、数据检查、训练和评测结果整理为结构化 findings；
- `semantic_eval`：判断是否真正产生新证据、结果是否改善、当前 Gate 是否满足；
- `reduce_progress`：确定性 code 节点，根据 semantic_eval 更新 stale_count/status，不允许 agent 自由改写阈值；
- `state_writer`：原子更新 replace 文件，append-only 更新 findings 和 iteration log，保证 phase、代码、数据、实验相互可追溯；
- `gate_check`：确定性检查当前阶段全部必要证据，只有通过才推进 phase；
- `route_by_status`：根据 phase、Gate 结果和 status 路由下一步。

## 8. stale、路由与完成规则

每个 phase 独立维护有效进展；切换 phase 后 `stale_count` 清零，但保留完整历史。

- 0 个 new findings，或固定评测结果显著变差 → `stale_count + 1`；
- 有可复现的新 finding 且结果改善 → `stale_count` 清零或降低；
- `stale_count >= 2` → `status = pivot_required`；
- `stale_count >= 4` → `status = attention_required`；
- 其他为 `healthy` 或 `stale`；
- 执行/数据损坏/外部任务失败且无法在本轮恢复 → `error`。

路由规则：

- `healthy` 且当前 Gate 未通过：留在当前 phase 进入下一轮；
- 当前 Gate 通过：按 `amp_conversion → motion_retargeting → amp_training → completed` 推进；
- `stale`：留在当前 phase 并选择多样化方向；
- `pivot_required`：进入当前 phase 对应的 `structural_pivot`；
- `attention_required`：写报告后停止，不向用户提问；
- `error`：写错误状态和 iteration log 后停止；
- `completed`：写最终验收报告、最佳 checkpoint、数据/配置/代码版本和复现命令后优雅退出。

## 9. Structural pivot

当 `stale_count >= 2` 时，不得只调学习率、reward scale 或网络宽度，必须改变结构性约束或研究框架。例如：

- AMP 工程化阶段：重新核对 expert/policy feature contract、时间表示、reward 变换、buffer 分布或 discriminator 正则；
- 重定向阶段：改变 keypoint mapping、IK 约束、尺度/offset、接触约束、clip 选择或数据源；
- 训练阶段：重新检查数据覆盖、style/task reward 冲突、curriculum、初始化分布、控制频率或评测假设；
- 从相反假设出发，补充消融实验或更换证据类型；
- 调研高置信度的一手论文/官方实现并记录与当前实现的结构差异。

Pivot 后必须更新 `directions_tried.json`，写明为什么旧方向失效以及新方向如何区分于已尝试方案。

## 10. 远端训练与恢复

- 训练在远端 Gradmotion 执行；启动前记录代码 revision、配置、数据 hash、seed、命令和 task ID；
- 建议约每 30 分钟检查一次结果。进入平台期且无明确改善趋势时及时终止并进入 `extract_findings`，不要仅为等待 max iterations 消耗资源；
- 快速 smoke train 与正式长训练分开，未通过本地/小规模检查不得启动高成本长训练；
- Gradmotion 账号无余额时，先执行 `account-pool remove <当前 id>`，再执行 `account-pool get` 获取有额度账号并有界重试；换号事件写入日志，但不改变研究结论；
- 网络、队列、余额和任务失败都必须有有限重试、退避和恢复路径，不得形成忙等循环。

## 11. 边界与禁止事项

- 所有节点、训练、轮询、重试、phase 迭代和总 loop 都必须有合理 bounds，并有优雅退出路径；
- 不得将“代码能 import”“数据能 load”“reward 上升”或“视频看起来会动”单独视为阶段完成；
- 不得复用错误机器人关节顺序、坐标系或物理参数；不得静默 drop/补零不匹配关节；
- 不得在同一实验同时大改数据、算法、奖励和控制参数后声称归因成立；
- findings 必须绑定可读取的日志、指标、视频、数据报告或代码 diff；无证据的主观描述不算 new finding；
- attention_required、error 和 completed 都必须写完 state/log/report 后停止，不依赖撞上限退出，也不触发用户交互。
