# Loop Blueprint

Goal: 将 F1_train_AMP 从当前普通 PPO 行走项目改造成真正可训练、可评估、可导出的 AMP 项目；完成 F1 行走动作重定向并生成可靠的专家运动数据；最终训练 F1 使其能以参考步态稳定行走。按 amp_conversion → motion_retargeting → amp_training → completed 顺序推进，建立可恢复、可追溯、可重复迭代的工作流。

## Intent and constraints

一个长周期自主研究 Loop，在单个项目工作树内按三阶段硬性递进推进：先把以 X1 命名的现有工程审计并改造为完整 AMP 闭环（Gate A），再参照 GMR 流程在项目内建立 F1 行走重定向并产出经验收的专家运动数据（Gate B），最后在远端 Gradmotion 上训练并用固定评测协议判定是否达到稳定行走标准（Gate C）。每轮 Loop 先加载状态与历史 findings，在当前阶段选择一个可证伪的新方向，执行 AMP 实现/重定向实验/训练实验，监控远端任务并拉取结果，将结果结构化为 findings，由 Agent 做语义判断，再由确定性节点更新 stale_count/status、持久化状态、检查门禁、路由下一步。三阶段各自维护独立的 stale_count 与路由状态，不共用。

| ID | Kind | Strength | Statement | Source |
| --- | --- | --- | --- | --- |
| C1 | goal | hard | 将现有项目改造为完整可训练、可评估、可导出的 AMP 项目；完成 F1 行走动作重定向并生成可靠专家运动数据；最终训练 F1 使其按参考步态稳定行走。 | amp_loop.md:L3 (目标) |
| C2 | deterministic_rule | hard | phase 取值至少为 amp_conversion、motion_retargeting、amp_training、completed；阶段顺序为硬性递进：amp_conversion 未过 Gate A 不得进入重定向或正式训练；motion_retargeting 未过 Gate B 不得用于正式 AMP 训练；只有 amp_training 通过 Gate C 才能进入 completed。 | amp_loop.md:§1 L16-L28 |
| C3 | deterministic_rule | hard | 上层维护 A_healthy、B_healthy、C_healthy，默认值均为 false；只有 X_healthy==true 时才允许进入下一阶段。 | amp_loop.md:§8 L156-L157 |
| C4 | deterministic_rule | hard | 禁止通过修改状态文件、降低验收阈值或跳过失败检查来跨越门禁。 | amp_loop.md:§1 L28 |
| C5 | deterministic_rule | hard | Gate A 路由：维护 stale_countA 默认 0。执行 AMP 改造并提交验证——通过则 A_healthy=true；不通过则 stale_countA+1；stale_countA>=10 时 error 且整体停止；执行/数据损坏/外部任务失败且无法在本轮恢复时 error 且整体停止。 | amp_loop.md:§8 L160-L166 |
| C6 | deterministic_rule | hard | Gate B 路由：维护 stale_countB 默认 0。参照 GMR 项目流程建立或改造本地 GMR F1 项目并提交验证——通过则 B_healthy=true；不通过则 stale_countB+1；stale_countB>=10 时 error 且整体停止；执行/数据损坏/外部任务失败且无法在本轮恢复时 error 且整体停止。 | amp_loop.md:§8 L170-L176 |
| C7 | deterministic_rule | hard | Gate C 状态转换：维护 stale_countC 默认 0、status_C 默认 normal。stale_countC>=2 时 status_C=pivot_stateC；stale_countC>=4 时 status_C=attention_required；执行/数据损坏/外部任务失败且无法在本轮恢复时 status_C=error。 | amp_loop.md:§8 L180-L184 |
| C8 | deterministic_rule | hard | Gate C normal 模式行为：选择一个方向进行 AMP 训练并查看结果——通过则 GateC 结束且 C_healthy=true；不通过/无效/变差则 stale_countC+1；有效果或有效则 stale_countC-1；stale_countC 最小值为 0。 | amp_loop.md:§8 L186-L191 |
| C9 | deterministic_rule | hard | Gate C pivot_stateC 模式行为：从相反假设出发补充消融实验或更换证据类型；调研高置信度一手论文/官方实现并记录与当前实现的结构差异；重新设计完后 stale_countC=0 且 status_C=normal；必须更新 directions_tried.json 写明旧方向失效原因及新方向如何区分于已尝试方案。 | amp_loop.md:§8 L192-L196 |
| C10 | deterministic_rule | hard | Gate C attention_required 模式行为：写报告后停止，整体停止。Gate C error 模式行为：整体停止。 | amp_loop.md:§8 L197-L200 |
| C11 | workspace_protocol | hard | 任务状态放在 {taskDir}/state/ 下，包含：task_spec.md（总目标、里程碑、固定评测集、数值阈值、成功标准）、progress.json（至少含 iteration/phase/phase_status/total_findings/code_revision/dataset_revision/updated_at）、findings.jsonl、directions_tried.json、iteration_log.jsonl、evaluation.json。 | amp_loop.md:§2 L34-L41 |
| C12 | workspace_protocol | hard | findings.jsonl 和 iteration_log.jsonl 为 append-only；progress.json/directions_tried.json/task_spec.md/evaluation.json 为原子 replace。日志放在 {taskDir}/logs/ 下，包含 work.jsonl、orchestrator.jsonl 以及远端训练任务 ID、监控时间点、停止原因和指标拉取记录。大体积动作数据和训练产物不写进 JSON state，state 中只保存项目内相对路径、哈希和摘要。 | amp_loop.md:§2 L38-L49 |
| C13 | workspace_protocol | hard | 切换阶段时必须写入阶段验收证据、代码版本、配置版本和数据版本。每个阶段允许独立迭代和 structural pivot。 | amp_loop.md:§1 L30 |
| C14 | workspace_protocol | hard | 首次进入 amp_training 阶段时，必须在 task_spec.md 中冻结固定评测集、命令范围、地形、domain randomization、seed 和数值阈值；后续不得为了宣告成功而降低阈值。 | amp_loop.md:§6 L116 |
| C15 | workspace_protocol | hard | Gate A 通过时形成 amp_contract.json，并将验证命令、日志和结果写入 findings。 | amp_loop.md:§4 L84 |
| C16 | deterministic_rule | hard | 评测使用未参与调参的固定 seed 集合，至少 5 个 seeds；汇报均值、标准差和最差 seed，禁止只选最好视频/checkpoint。判定训练改善时采用固定评测结果，不以训练窗口内 reward 峰值、单次视频或 discriminator accuracy 单独作结论。 | amp_loop.md:§6 L127, L135 |
| C17 | workspace_protocol | hard | 一次只改变一个主要假设，保留可比较 baseline。 | amp_loop.md:§7 L143 |
| C18 | ownership | hard | reduce_progress 为确定性 code 节点，根据 semantic_eval 更新 stale_count/status，不允许 agent 自由改写阈值。 | amp_loop.md:§7 L147 |
| C19 | ownership | hard | state_writer 负责原子更新 replace 文件、append-only 更新 findings 和 iteration log，保证 phase、代码、数据、实验相互可追溯；Agent 不直接写 state 文件，所有持久化提交经 state_writer 串行完成。 | amp_loop.md:§7 L148 |
| C20 | ownership | hard | gate_check 为确定性检查当前阶段全部必要证据，只有全部通过才推进 phase；route_by_status 根据 phase、Gate 结果和 status 路由下一步。二者均为确定性节点，不被 Agent 意志覆盖。 | amp_loop.md:§7 L149-L150 |
| C21 | terminal_obligation | hard | attention_required、error 和 completed 都必须写完 state/log/report 后停止，不依赖撞上限退出，也不触发用户交互。 | amp_loop.md:§10 L218 |
| C22 | failure_boundary | hard | 所有节点、训练、轮询、重试、phase 迭代和总 loop 都必须有合理 bounds，并有优雅退出路径。 | amp_loop.md:§10 L214 |
| C23 | recovery | hard | 网络、队列、余额和任务失败都必须有有限重试、退避和恢复路径，不得形成忙等循环。Gradmotion 账号无余额时先执行 account-pool remove <当前 id> 再执行 account-pool get 获取有额度账号并有界重试；换号事件写入日志但不改变研究结论。快速 smoke train 与正式长训练分开，未通过本地/小规模检查不得启动高成本长训练。 | amp_loop.md:§9 L208-L210, §9 L209 |
| C24 | timer | hard | 远端训练建议约每 30 分钟检查一次结果；进入平台期且无明确改善趋势时及时终止并进入 extract_findings，不要仅为等待 max iterations 消耗资源。 | amp_loop.md:§9 L207 |
| C25 | other | hard | 首次运行必须先审计当前仓库，不能从项目名推断它已经是 F1 或 AMP：核对真实机器人身份、URDF/MJCF、自由度、actuated joint、关节上下限、默认姿态、关节顺序、body/link 名称、足端定义和碰撞体；核对训练环境、task 注册、actor/critic、runner、PPO、rollout storage、train/play/export 和 Sim2Sim 数据流；搜索并列出现有 AMP/motion loader/discriminator/expert replay buffer/style reward/retarget 工具和专家动作数据；记录当前仓库中遗留的 X1 命名或假设并判断哪些仅是命名问题、哪些会导致 F1 资产/观测/动作/关节映射错误；生成 baseline（现有 PPO 是否能启动、短程训练、play 和 Sim2Sim）。若真实 F1 模型、参考行走源动作或其合法可用来源缺失，记录为 blocking precondition，不得用 X1/RPO 数据冒充 F1 结果。 | amp_loop.md:§3 L53-L61 |
| C26 | other | hard | 禁止将'代码能 import''数据能 load''reward 上升'或'视频看起来会动'单独视为阶段完成。不得复用错误机器人关节顺序、坐标系或物理参数；不得静默 drop/补零不匹配关节。不得在同一实验同时大改数据、算法、奖励和控制参数后声称归因成立。 | amp_loop.md:§10 L215-L217 |
| C27 | success_criteria | hard | Gate A（amp_conversion）须同时满足：静态检查/import/配置解析通过；用小数据小环境数完成 bounded smoke train 且 AMP 各模块确实执行且参数有限更新；expert/policy transition schema 自动比对通过，无 NaN/Inf/空数据集/越界索引/错误 broadcasting；checkpoint save/resume 和 play 至少各验证一次；训练日志中可见独立的 discriminator 与 style reward 证据（AMP task 不是普通 PPO 改名）；形成 amp_contract.json 并将验证命令/日志/结果写入 findings。 | amp_loop.md:§4 L77-L84 |
| C28 | success_criteria | hard | Gate B（motion_retargeting）须同时满足：数据通过 schema/单位/坐标系/关节映射/范围/连续性和接触检查；至少一个参考行走 clip 通过完整可视化回放和定量验收；AMP loader 读取该数据后的统计与重定向输出一致；不合格 clip 被隔离并记录原因，不得混入专家数据集。重定向须按参照 GMR 项目流程及算法做 F1 适配，结合 F1 URDF/MJCF，不得简化 IK。每个候选 clip 须检查 joint name 一一对应且维度/顺序与 F1 task 一致、root 高度/姿态/方向合理无 NaN/Inf 突跳、左右脚接触时序与参考相符且支撑脚滑移/穿地/悬空/膝部反折/自碰撞在固定阈值内、首尾帧循环连续性（不满足不强制循环）、在 MuJoCo 或等价运动学回放中完整播放并复核。原始/重定向中间结果/关节重排后训练数据须版本化区分。 | amp_loop.md:§5 L88-L109 |
| C29 | success_criteria | hard | Gate C（amp_training）须满足：nominal 环境中参考命令下连续稳定行走至少 60 秒，无跌倒、无 NaN/Inf、无关节硬限位违规（F1 关节限位可稍微放宽，不作为严格要求，但不允许频繁过量超限）；固定批量评测中 episode 成功率至少 95%、跌倒率不高于 5%；在受控 domain randomization 和 MuJoCo Sim2Sim 中重复评测且达到预先冻结的 Sim2Sim 阈值；最终 checkpoint 可 resume、可 play、可导出，且导出策略的 observation/action 契约与训练一致。评测覆盖模仿质量、行走任务、稳定性、安全与可部署性四组指标，使用至少 5 个固定 seed。速度/步态相位/足滑/姿态/模仿误差的具体阈值必须在首次正式训练前写入 task_spec.md。 | amp_loop.md:§6 L118-L136 |
| C30 | capability | hard | GMR 重定向参照实现（https://github.com/Roboparty/GMR）必须被克隆或复现到项目内部某个目录（项目外不可写），作为本地 GMR F1 适配项目的基础；Roboparty train README 仅用于理解 AMP 组成/数据准备/关节重排/训练测试/Sim2Sim 流程。 | amp_loop.md:§5 L90, L6-L10 |
| C31 | capability | hard | Gradmotion 远端训练平台须可用：gm CLI 可调用、account-pool 可获取有额度账号；参考行走源动作（/Users/yumx/code/robot_x/X1/Retargeting_X1/BMLrub_stageii/rub058/0008_normal_walk4_stageii.npz，已复制为项目内 data/0008_normal_walk4_stageii.npz）须可用。 | amp_loop.md:§5 L112, §9 L205-L210 |
| C32 | workspace_protocol | hard | 远端训练启动前必须记录代码 revision、配置、数据 hash、seed、命令和 task ID 到 logs；进入 attention_required/error/completed 终态时必须写完 state、log 和 report 后才停止。 | amp_loop.md:§9 L206, §10 L218 |

## Success criteria

- Gate A：smoke train 日志中出现独立 discriminator 与 style reward 指标；expert/policy transition schema 比对 PASS；checkpoint save/resume/play 各验证一次；amp_contract.json 生成并写入 findings。
- Gate B：至少一个参考行走 clip 通过 MuJoCo 回放和定量验收（关节映射/范围/连续性/接触/循环性）；AMP loader 读取统计与重定向输出一致；不合格 clip 被隔离。
- Gate C：nominal 连续行走 ≥60s 无跌倒/NaN/Inf；批量评测 episode 成功率 ≥95%、跌倒率 ≤5%；Sim2Sim 达冻结阈值；checkpoint 可 resume/play/export 且契约一致；至少 5 个固定 seed 汇报均值/标准差/最差。
- A_healthy、B_healthy、C_healthy 依次置 true 后进入 completed，终态写完 state/log/report。
- 训练改善判定仅基于固定评测结果，不以 reward 峰值/单视频/discriminator accuracy 单独作结论。

## Workspace

- 项目根 /Users/yumx/code/robot_x/X1/F1_train_AMP 为唯一可写工作树；taskDir 即项目根。
- state/ 目录由 loop 首轮自建，含 task_spec.md、progress.json、findings.jsonl、directions_tried.json、iteration_log.jsonl、evaluation.json；findings.jsonl 和 iteration_log.jsonl 为 append-only，其余为原子 replace。
- logs/ 目录由 loop 首轮自建（已在 .gitignore），含 work.jsonl、orchestrator.jsonl 及远端任务 ID/监控时间点/停止原因/指标拉取记录。
- Agent 读：现有 humanoid/**/*.py（含 humanoid/algo/amp/ 的 discriminator、motion_lib、replay_buffer、amp_on_policy_runner 及 humanoid/envs/x1/ 下的 AMP/PPO config/env）、resources/robots/x1/urdf/*.urdf、resources/robots/x1/mjcf/**/*.xml、data/0008_normal_walk4_stageii.npz、monitor_train.sh、各 scripts（train_amp/smoke_amp/test_amp_contract/eval_amp/sim2sim/export 等）、state/ 全部文件、项目内克隆的 GMR 代码。
- Agent 写：humanoid/**/*.py 的 AMP/重定向/训练代码修改、state/ 下经 state_writer 串行提交的全部文件、amp_contract.json、重定向产物（可视化视频、数值报告、manifest、data/retarget_gmr/ 下重排后训练数据）、项目内 GMR F1 适配代码、训练 checkpoint 与导出文件（.jit/.onnx）。
- 外部路径 /Users/yumx/code/robot_x/X1/Retargeting_X1/... 仅供只读参考；GMR 仓库（GitHub）必须克隆到项目内子目录后才能在其上做改造与写入。Gradmotion 实验产物经 .oma/experiments/（gitignored）或 state 中相对路径/哈希引用，不写入 JSON state。
- 大体积数据（npz/checkpoint/视频）不进 JSON state，state 中只保存项目内相对路径、哈希和摘要。

## Lanes

- Agent 研究/执行 Lane（厚 Agent）：load_state → choose_direction → research_design_execute → monitor_remote_job → extract_findings → semantic_eval，在单条连续会话中完成；持有全项目上下文与历史 findings 以做研究判断；可提议任意代码/数据/训练改动。
- 确定性控制 Lane（代码节点）：reduce_progress、state_writer、gate_check、route_by_status 与 Agent Lane 分离，串行执行；reduce_progress 机械推导 stale_count/status，gate_check 机械检查证据全集，state_writer 独占 state 文件原子/追加写入，route_by_status 机械路由。Agent 无权直接修改阈值或越过 gate_check 推进 phase。
- 远端训练监控：monitor_remote_job 以约 30 min 间隔轮询 Gradmotion（gm CLI），拉取曲线/产物；可经 account-pool 换号有界重试；进入平台期或异常时终止并交回 extract_findings。

## Control

- phase 取值 amp_conversion/motion_retargeting/amp_training/completed；三阶段硬性递进，只有 X_healthy==true 才允许进下一阶段。
- Gate A/B 路由：pass→X_healthy=true；fail→stale_count+1；stale_count>=10→error→整体停止；不可恢复执行/数据/外部失败→error→整体停止。
- Gate C 路由由 stale_countC 阈值驱动：>=2→pivot_stateC；>=4→attention_required；不可恢复失败→error。normal 模式 pass→C_healthy=true/Done，fail/worse→stale+1，effective→stale-1(min 0)。pivot 完成后 stale=0+normal 且必须更新 directions_tried。attention_required/error→写 report+整体停止。
- reduce_progress/gate_check/route_by_status/state_writer 为确定性节点，Agent 不可越过或改写其判定。
- 所有节点、训练、轮询、重试、phase 迭代、总 loop 均有有限 bounds 和优雅退出路径；不依赖撞限退出。
- attention_required、error、completed 三类终态必须写完 state/log/report 后停止，不触发用户交互。
- 首次进入 amp_training 前在 task_spec.md 冻结评测集/命令/地形/DR/seed/阈值，后续不得降低。
- 一次实验只改一个主要假设；不得同时大改数据/算法/奖励/控制参数后声称归因成立。
- 首次运行先做启动审计（C25），审计未完成不得进入任何 Gate 验证。

## Assumptions

- taskDir 即项目根 /Users/yumx/code/robot_x/X1/F1_train_AMP。
- 项目中全链路使用 X1 命名（URDF/envs/configs/scripts/meshes），真实机器人身份为 F1 还是 X1 须由启动审计判定；审计结论将决定后续资产/观测/关节映射是否需要修正，但不改变三阶段路由结构。
- data/0008_normal_walk4_stageii.npz 已在项目内（与外部参考路径同名），视为参考动作的工作副本；外部源路径供只读核对。
- 项目已有 AMP 代码骨架与先前迭代痕迹（config 注释显示 iter-15~24），但其是否真正通过 Gate A 须由审计和 Gate A 验证判定，不预设已通过。
- Gradmotion 的 gm CLI 和 account-pool 机制在执行环境中可用（monitor_train.sh 已使用 gm task info/logs）。

## Capability gaps

- GMR 参照实现（https://github.com/Roboparty/GMR）尚未克隆到项目内；须在 Gate B 前克隆或复现到项目子目录后才能进行重定向改造。
- data/retarget_gmr/x1_walk_retargeted.npz（config 中引用的 Gate B 重定向产物）尚不存在，须由 Gate B 产出。
- state/ 和 logs/ 目录尚不存在，须由 loop 首轮自建。
- Gradmotion 远端训练的实际可用性（账号余额、队列、网络）须在运行时验证；若不可用须作为 blocking precondition 记录。
