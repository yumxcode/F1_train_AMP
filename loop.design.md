# Loop Blueprint

Goal: 将 F1_train_AMP 从当前普通 PPO 行走项目改造成真正可训练、可评估、可导出的 AMP（Adversarial Motion Priors）项目；完成 F1 行走动作重定向并生成可靠的专家运动数据；最终训练 F1 使其能按参考步态稳定行走。该长周期任务按 amp_conversion → motion_retargeting → amp_training → completed 的门禁顺序推进，建立可恢复、可追溯、可重复迭代的工作流。

## Intent and constraints

构建一个以三阶段门禁为骨架、可恢复且可追溯的长周期研究 loop：每轮先装载已冻结的 task spec 与各阶段 state，审计当前 phase 的代码/数据/远端/前置 Gate 证据，再选择一个与历史不同且可证伪的研究方向并执行（amp_conversion 写 AMP 闭环代码、motion_retargeting 做 F1 重定向实验、amp_training 跑远端训练与固定评测），随后把结果整理为带证据的 findings，做语义级判定（是否产生新证据/是否改善/Gate 是否满足），再由确定性节点更新 stale_count 与 status、原子写 state、确定性 gate 检查，并按 status 路由（继续迭代/结构 pivot/停止/进入下一阶段）。所有阶段切换、版本与证据都写入 state，使任意一轮可被重建与审计；不预先固化可执行拓扑、节点 ID 或路由外键。

| ID | Kind | Strength | Statement | Source |
| --- | --- | --- | --- | --- |
| C1 | goal | hard | 把现有普通 PPO 行走工程改造成完整 AMP 闭环（可训练/可评估/可导出），完成 F1 行走动作重定向生成专家运动数据，并训练 F1 按参考步态稳定行走。 | amp_loop.md:标题/目标 行3 |
| C2 | goal | hard | 阶段顺序为硬约束：amp_conversion 未过 Gate A 不得进入重定向或正式训练；motion_retargeting 未过 Gate B 不得把数据用于正式 AMP 训练；仅 amp_training 过 Gate C 才进入 completed。 | amp_loop.md:§1 强制阶段与门禁 行15-27 |
| C3 | deterministic_rule | hard | 禁止通过修改状态文件、降低验收阈值或跳过失败检查跨越门禁。 | amp_loop.md:§1 行27 |
| C4 | success_criteria | hard | Gate A 须同时满足：静态检查/import/配置解析通过；小数据小环境数完成一次 bounded smoke train 且 AMP 各模块执行、参数有限更新；expert/policy transition schema 自动比对通过且无 NaN/Inf/空集/越界/错误 broadcasting；checkpoint save/resume 与 play 至少各验证一次；训练日志可见独立 discriminator 与 style reward 证据；形成 amp_contract.json 并把命令/日志/结果写入 findings。 | amp_loop.md:§4 Gate A 行78-85 |
| C5 | success_criteria | hard | Gate B 须同时满足：retarget_manifest.json 完整且能从源数据确定性重建训练数据；数据通过 schema/单位/坐标系/关节映射/范围/连续性/接触检查；至少一个参考行走 clip 通过完整可视化回放与定量验收；AMP loader 读取后统计与重定向输出一致；不合格 clip 被隔离并记录原因，不混入专家数据集。 | amp_loop.md:§5 Gate B 行109-115 |
| C6 | success_criteria | hard | Gate C 须同时满足：首次进入 amp_training 前在 task_spec.md 冻结固定评测集/命令范围/地形/DR/seed/数值阈值且后续不得为宣告成功而降低；用未参与调参的固定 ≥5 seeds 评测并报均值/标准差/最差 seed；nominal 环境参考命令下连续稳定行走 ≥60s 无跌倒/NaN/Inf/关节硬限位；批量评测 episode 成功率 ≥95% 且跌倒率 ≤5%；受控 DR 与 MuJoCo Sim2Sim 重复评测且达到冻结的 Sim2Sim 阈值；最终 checkpoint 可 resume/play/导出且导出 obs/action 契约与训练一致。 | amp_loop.md:§6 行122-138 |
| C7 | workspace_protocol | hard | 任务状态置于 {taskDir}/state/：task_spec.md、progress.json（含 iteration/phase/phase_status/status/stale_count/total_findings/code_revision/dataset_revision/updated_at）、findings.jsonl（append-only）、directions_tried.json、iteration_log.jsonl（append-only）、amp_contract.json、retarget_manifest.json、evaluation.json；日志置于 {taskDir}/logs/：work.jsonl、orchestrator.jsonl、远端任务 ID/监控点/停止原因/指标拉取记录。大体积动作数据与训练产物不写入 JSON state，state 仅存稳定的项目内相对路径、哈希与摘要。 | amp_loop.md:§2 状态目录 行32-50 |
| C8 | workspace_protocol | hard | 切换阶段时必须写入阶段验收证据、代码版本、配置版本与数据版本，保证 phase/代码/数据/实验相互可追溯。 | amp_loop.md:§1 行29 |
| C9 | ownership | hard | reduce_progress 为确定性 code 节点，是唯一可据 semantic_eval 更新 stale_count 与 status 的环节，且不允许 agent 自由改写阈值；state_writer 以原子 replace 更新可替换文件、以 append-only 更新 findings 与 iteration log。 | amp_loop.md:§7 行153-155 |
| C10 | ownership | hard | gate_check 为确定性检查当前阶段全部必要证据，只有通过才推进 phase 的唯一环节。 | amp_loop.md:§7 行155 |
| C11 | deterministic_rule | hard | 首次运行必须先审计当前仓库，不得从项目名推断它已是 F1 或 AMP：核对真实机器人身份、URDF/MJCF、自由度、actuated joint、关节上下限、默认姿态、关节顺序、body/link 名、足端定义与碰撞体；核对训练环境、task 注册、actor/critic、runner、PPO、rollout storage、train/play/export 与 Sim2Sim 数据流；搜索并列出现有 AMP/motion loader/discriminator/expert replay buffer/style reward/retarget 工具与专家动作数据；记录 X1 命名遗留并判断哪些仅为命名问题、哪些会导致 F1 资产/观测/动作/关节映射错误；生成普通 PPO baseline（能否启动/短程训练/play/Sim2Sim，保存命令、版本与输出证据）。 | amp_loop.md:§3 启动审计 行52-61 |
| C12 | failure_boundary | hard | 若真实 F1 模型、参考行走源动作或其合法可用来源缺失，记录为明确 blocking precondition，不得用 X1/RPO 数据冒充 F1 结果。 | amp_loop.md:§3 行62 |
| C13 | capability | hard | 重定向参考源为外部文件 /Users/yumx/code/robot_x/X1/Retargeting_X1/BMLrub_stageii/rub058/0008_normal_walk4_stageii.npz；其位于本项目根之外，沙箱不可访问，须作为启动前置条件拷贝/放置到项目内某数据目录后再读取，并记录原始哈希/fps/clip 范围/左右脚接触信息。 | amp_loop.md:§5 注 行118, amp_loop.md:§5 行93 |
| C14 | deterministic_rule | hard | 重定向流程须：基于 F1 模型建立 T-pose/初始姿态、human-to-F1 keypoint/link 对应、尺度、位置/旋转 offset 与 IK 约束；明确 world/base 坐标、up axis、长度/角度单位、四元数顺序与朝向约定；输出 (root_translation, root_rotation, joint_positions)；按 F1 URDF/MJCF 与训练环境实际使用顺序建立显式 joint mapping 再转为 AMP loader 格式，不依赖字典插入顺序或他机硬编码索引；由相邻帧与真实 dt 计算速度、处理首尾/循环/插值/滤波，禁止错误 fps 放大速度；原始、重定向中间结果、关节重排后训练数据须版本化区分，并产出可视化视频、数值报告与专家动作 manifest。 | amp_loop.md:§5 行93-99 |
| C15 | deterministic_rule | hard | 每个候选 clip 须检查：joint name 一一对应、维度/顺序与 F1 task 一致无缺失/重复/静默截断；关节位置/速度不越 F1 范围、root 高度/姿态/方向合理无 NaN/Inf/突跳；左右脚接触时序与参考相符且支撑脚滑移/穿地/悬空/膝反折/自碰撞在固定阈值内；首尾帧循环时位置/姿态/速度/接触连续，不满足循环条件者不得强制循环；在 MuJoCo 或等价运动学回放中完整播放并人工/自动复核，而非仅确认文件可读。 | amp_loop.md:§5 行101-107 |
| C16 | goal | hard | AMP 工程化须实现完整闭环：AMP observation/transition（含 root、关节、末端、速度与历史帧特征，expert 与 policy 维度/顺序/单位/坐标系/时间间隔完全一致）；专家动作 loader（clip 边界/时间采样插值/循环/合法性检查/确定性复现）；policy transition 收集与 replay buffer；discriminator 网络/优化器/expert-policy minibatch/梯度惩罚或正则；discriminator 输出到 style reward 的变换与数值裁剪归一化及与 task reward 的组合；在 PPO/runner/storage/config/checkpoint 接入 AMP 训练状态且 resume 后不丢 discriminator/optimizer/normalizer/必要 buffer；独立的 F1 AMP task 注册及 train/play/checkpoint/JIT-ONNX 导出入口；指标覆盖 policy/task/style reward、disc expert/policy loss、accuracy/logit、gradient penalty、PPO loss、episode length、fall/termination；保留最小可用的普通 PPO baseline 作回归与 AMP 增益对照。 | amp_loop.md:§4 行66-76 |
| C17 | success_criteria | hard | 最终评测至少覆盖四组指标：(1) 模仿质量（AMP/style reward、参考与策略的关节姿态/速度误差、root 高度与姿态误差、步频/周期误差、左右脚接触相位一致性）；(2) 行走任务（前向速度跟踪误差、横向漂移、yaw 漂移、目标距离/位移完成率）；(3) 稳定性（episode 完成率、跌倒率、连续无摔倒时长、base roll/pitch、足端滑移、异常双腾空/双支撑比例）；(4) 安全与可部署性（关节限位、速度/扭矩饱和率、非足端碰撞、自碰撞、NaN/Inf，及 Isaac Gym → MuJoCo Sim2Sim 回归）。 | amp_loop.md:§6 行124-129 |
| C18 | deterministic_rule | hard | 判定训练改善时采用固定评测结果，不以训练窗口内的 reward 峰值、单次视频或 discriminator accuracy 单独作结论。 | amp_loop.md:§6 行140 |
| C19 | event | hard | 训练在远端 Gradmotion 执行；启动前记录代码 revision、配置、数据 hash、seed、命令与 task ID；建议约每 30 分钟检查一次结果；进入平台期且无明确改善趋势时及时终止并进入 extract_findings，不仅为等 max iterations 消耗资源；快速 smoke train 与正式长训练分开，未通过本地/小规模检查不得启动高成本长训练。 | amp_loop.md:§10 行192-195 |
| C20 | recovery | hard | Gradmotion 账号无余额时，先 account-pool remove <当前 id> 再 account-pool get 获取有额度账号并有界重试；换号事件写入日志但不改变研究结论；网络、队列、余额与任务失败都必须有有限重试、退避与恢复路径，不得形成忙等循环。 | amp_loop.md:§10 行196-197 |
| C21 | deterministic_rule | hard | stale 规则：0 个 new findings 或固定评测显著变差 → stale_count+1；有可复现新 finding 且结果改善 → 清零或降低；stale_count≥2 → status=pivot_required；stale_count≥4 → status=attention_required；其余为 healthy 或 stale；执行/数据损坏/外部任务失败且本轮无法恢复 → error。每个 phase 独立维护进展，切换 phase 后 stale_count 清零但保留完整历史。 | amp_loop.md:§8 行159-167 |
| C22 | deterministic_rule | hard | 路由规则：healthy 且当前 Gate 未通过 → 留在当前 phase 进下一轮；Gate 通过 → 按 amp_conversion→motion_retargeting→amp_training→completed 推进；stale → 留当前 phase 并选多样化方向；pivot_required → 进当前 phase 对应 structural_pivot；attention_required → 写报告后停止且不向用户提问；error → 写错误状态与 iteration log 后停止；completed → 写最终验收报告、最佳 checkpoint、数据/配置/代码版本与复现命令后优雅退出。 | amp_loop.md:§8 路由规则 行169-177 |
| C23 | recovery | hard | stale_count≥2 时不得只调学习率/reward scale/网络宽度，必须改变结构性约束或研究框架（如重核 feature contract/时间表示/reward 变换/buffer 分布/discriminator 正则；改 keypoint mapping/IK/尺度/接触/clip/数据源；重查数据覆盖/style-task reward 冲突/curriculum/初始化/控制频率/评测假设；从相反假设补充消融或换证据类型；调研一手论文/官方实现并记录结构差异）；pivot 后必须更新 directions_tried.json 写明旧方向失效原因与新方向如何区分于已尝试方案。 | amp_loop.md:§9 行181-189 |
| C24 | budget | hard | 所有节点、训练、轮询、重试、phase 迭代与总 loop 都必须有合理 bounds 并有优雅退出路径；attention_required/error/completed 都必须在写完 state/log/report 后停止，不依赖撞上限退出，也不触发用户交互。 | amp_loop.md:§11 行201 与 行206 |
| C25 | failure_boundary | hard | 禁止将“代码能 import / 数据能 load / reward 上升 / 视频看起来会动”单独视为阶段完成。 | amp_loop.md:§11 行202 |
| C26 | failure_boundary | hard | 不得复用错误机器人关节顺序/坐标系/物理参数，不得静默 drop 或补零不匹配关节；不得在同一实验同时大改数据、算法、奖励与控制参数后声称归因成立；findings 必须绑定可读取的日志/指标/视频/数据报告/代码 diff，无证据的主观描述不算 new finding。 | amp_loop.md:§11 行203-205 |
| C27 | deterministic_rule | hard | 每轮 loop 须含阶段感知职责：读取 task spec/phase/progress/directions/findings 摘要/AMP contract/retarget manifest/evaluation；审计当前阶段前置条件（代码/数据/CLI/凭据/远端资源/前一 Gate 证据）；选择当前 phase 内与历史不同且可证伪的方向（写清假设/改动面/预期指标）；按 phase 执行实现/重定向/训练实验且一次只改一个主要假设并保留可比较 baseline；监控远端任务并拉取曲线与产物；将结果整理为结构化 findings；判断是否产生新证据、是否改善、当前 Gate 是否满足；确定性更新 stale_count/status；原子写 state 并 append findings/iteration log；确定性 gate 检查；按 phase/Gate/status 路由。 | amp_loop.md:§7 每轮 loop 行144-156 |

## Success criteria

- amp_conversion 通过 Gate A：存在可独立训练、play、checkpoint resume、JIT/ONNX 导出的 F1 AMP task（非改名普通 PPO），smoke train 中 discriminator 与 style reward 真实执行且 expert/policy schema 自动比对通过，amp_contract.json 与验证证据写入 findings。
- motion_retargeting 通过 Gate B：retarget_manifest.json 可从源数据确定性重建训练数据，至少一个 F1 专家行走 clip 通过 schema/单位/坐标系/关节映射/范围/连续性/接触检查与 MuJoCo 回放复核，AMP loader 读后统计与重定向输出一致，不合格 clip 被隔离。
- amp_training 通过 Gate C：用 ≥5 个未参与调参 seed 的固定评测证明 F1 在 nominal 命令下连续稳定行走 ≥60s、批量成功率 ≥95% 且跌倒率 ≤5%、Sim2Sim 达到冻结阈值，最终 checkpoint 可 resume/play/导出且导出契约与训练一致。
- 进入 completed 时已写最终验收报告、最佳 checkpoint、代码/配置/数据版本与复现命令，并可优雅退出。
- attention_required 与 error 终态均在写完 state/log/report 后停止，且不触发用户交互、不依赖撞上限退出。

## Workspace

- 项目根为 {taskDir}（/Users/yumx/code/robot_x/X1/F1_train_AMP）；state/ 与 logs/ 目录当前不存在，由 loop 首轮自建。
- state/ 下原子 replace 维护：task_spec.md、progress.json、directions_tried.json、amp_contract.json、retarget_manifest.json、evaluation.json；append-only 维护：findings.jsonl、iteration_log.jsonl。
- logs/ 下 append-only 维护：work.jsonl、orchestrator.jsonl，以及远端 Gradmotion task ID/监控时间点/停止原因/指标拉取记录与 account-pool 换号事件。
- 大体积专家动作数据、重定向中间结果、关节重排后训练数据、checkpoint 与导出产物不写入 JSON state，state 仅存其项目内相对路径、哈希与摘要；这些产物写入项目内新建的数据/产物目录并按原始/中间/训练版本化区分。
- amp_conversion 阶段直接读写现有 humanoid 包：新增 AMP 模块（observation/transition、expert loader、replay buffer、discriminator、style reward）、在 humanoid/envs/__init__.py 注册独立的 F1 AMP task、新增 F1 config/env、并在 humanoid/algo/ppo（dh_ppo、dh_on_policy_runner、actor_critic_dh、rollout_storage）与 scripts（train/play/sim2sim/export_policy_dh/export_onnx_dh）接入 AMP 训练状态与 resume 完整性；同时保留普通 PPO baseline。
- loop 只读现有资产作为审计与重定向依据：resources/robots/x1/urdf/x1.urdf、resources/robots/x1/mjcf/*.xml、resources/robots/x1/meshes/*.STL、train_f1_v2.sh、monitor_train.sh；这些文件路径已确认存在。
- 外部重定向参考源 /Users/yumx/code/robot_x/X1/Retargeting_X1/BMLrub_stageii/rub058/0008_normal_walk4_stageii.npz 位于项目根之外、沙箱不可读写，必须作为启动前置条件拷贝/放置到项目内某数据目录后才能被重定向读取，并记录其原始哈希/fps/clip 范围/接触信息。
- 禁止向项目根之外的任何路径写入；禁止复用错误关节顺序/坐标系/物理参数或静默 drop/补零不匹配关节。

## Lanes

- 需要一条连续对话的厚研究 lane：在 amp_conversion 连贯地实现并自测 AMP 闭环代码、在 motion_retargeting 连贯地做重定向与回放复核、在 amp_training 连贯地设计训练/评测实验；该 lane 拥有对 humanoid 包、专家数据与产物目录的写权限，但一次只改一个主要假设并保留可比较 baseline。
- 需要一条被串行化的确定性状态控制职责：依据语义判定更新 stale_count/status、做 gate 检查、原子 replace state 文件、append findings 与 iteration log、并按 phase/Gate/status 路由；其写权限被约束在 state/ 与 logs/ 且不允许自由改写阈值或越权推进 phase。
- 需要一条独立权限的远端交互职责：发起/轮询/停止 Gradmotion 训练、拉取曲线与产物、在余额不足时 account-pool 换号；该职责持有网络与账号凭据并以有界重试/退避运行，不与研究结论耦合。
- 首次运行的审计职责需在同一连续上下文中核对机器人身份与现有工程结构、并产出 baseline，避免中途丢失对 URDF/MJCF/关节定义/AMP 缺口的判断。

## Control

- phase 取值 amp_conversion/motion_retargeting/amp_training/completed，顺序为硬约束，仅当 gate_check 确认当前阶段全部必要证据齐备且通过时才推进到下一阶段。
- stale_count 与 status 由确定性 reduce 逻辑依据语义判定更新（0 新发现或固定评测显著变差 +1、可复现改善清零/降低、≥2 pivot_required、≥4 attention_required、不可恢复失败 error），agent 不得自由改写阈值；切换 phase 后 stale_count 清零但保留完整历史。
- 路由完全由 status 与 Gate 结果决定：healthy+未过门留当前 phase、过门则推进、stale 选多样化方向、pivot_required 进当前 phase 结构 pivot、attention_required 写报告后停止、error 写状态/log 后停止、completed 写最终报告后优雅退出。
- stale_count≥2 触发 structural_pivot：必须改变结构性约束或研究框架而非只调学习率/reward scale/网络宽度，且 pivot 后更新 directions_tried.json 记录旧方向失效原因与新方向区分度。
- 远端训练按有界节奏运行：启动前记录 revision/配置/数据 hash/seed/命令/task ID，约每 30 分钟轮询一次，进入平台期无改善趋势则及时终止并进入 findings 抽取；smoke train 通过本地/小规模检查后方可启动高成本长训练。
- 余额/网络/队列/任务失败均走有限重试、退避与恢复（account-pool remove→get 有界重试），不得形成忙等循环；换号仅写日志不改变研究结论。
- 所有节点、轮询、重试、phase 迭代与总 loop 都有 bounds 与优雅退出路径；attention_required/error/completed 必须写完 state/log/report 后停止，不依赖撞上限、不触发用户交互。
- 首次运行须先审计仓库（不因项目名推断 F1/AMP）、产出普通 PPO baseline；真实 F1 模型或合法参考源缺失时记为 blocking precondition，不得用 X1/RPO 冒充 F1。
- 训练改善判定只采用固定评测结果，不接受“能 import/能 load/reward 上升/视频会动”作为阶段完成证据，也不接受无证据绑定的主观 finding。

## Assumptions

- {taskDir} 即项目根 /Users/yumx/code/robot_x/X1/F1_train_AMP，state/ 与 logs/ 置于其下。
- 现有 humanoid 包（envs/algo/utils/scripts + resources/robots/x1 资产）是 AMP 工程化的代码基线，loop 在其上扩展而非重写。
- 需求引用的 Roboparty 训练说明与 GMR 仓库仅用于理解 AMP 组成/数据准备/关节重排/训练测试/Sim2Sim 流程，属参考资料，不要求作为项目内资产存在。
- 预存的 team/ 目录（activity/decisions/units/modules/goals/board 等）与本 loop 的 state 协议无关，loop 不向其写入。
- Gradmotion 远端训练 CLI 与 account-pool 工具在运行环境中可用。
- 进入 amp_training 前若 repo 现有 X1 资产经审计确认为 F1 真实资产（仅命名遗留），则可直接使用；若审计判定 F1 资产缺失或与 X1 不同，则需由外部提供 F1 URDF/MJCF/关节定义作为 blocking precondition。

## Capability gaps

- 外部重定向参考源 0008_normal_walk4_stageii.npz 位于项目根之外、沙箱不可读写，须作为启动前置条件被拷贝/放置到项目内数据目录；若无法获取则按 blocking precondition 处理，motion_retargeting 不得启动。
- 真实 F1 模型资产（URDF/MJCF、自由度、actuated joint、关节上下限、默认姿态、关节顺序、足端/碰撞体）须经首轮审计与现有 X1 资产比对确认；若 F1 与 X1 不一致且 F1 资产缺失，需外部提供，否则为 blocking precondition。
- 本仓库当前无任何 AMP/motion loader/discriminator/expert replay buffer/style reward/retarget 工具实现（envs/__init__.py 仅注册 x1_dh_stand），AMP 闭环与重定向工具链需在 amp_conversion/motion_retargeting 阶段由零自建。
- 固定评测的数值阈值（速度/步态相位/足滑/姿态/模仿误差/Sim2Sim）须在首次进入 amp_training 前依据 F1 尺寸、控制频率与参考 clip 写入 task_spec.md 并冻结；该阈值集合当前不存在，需 loop 在该阶段产出。
