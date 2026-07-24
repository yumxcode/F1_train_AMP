# Loop Semantic Review

Accepted: yes

## intent_constraints

Status: pass

- 约束台账 C1-C27 覆盖来源全部 hard constraints：三阶段门禁顺序（C2/C3）、Gate A/B/C 验收标准（C4/C5/C6）、状态协议含 code_revision/dataset_revision（C7/C8）、确定性 reduce/writer/gate_check（C9/C10）、启动审计（C11）、F1 缺失 blocking（C12）、外部 npz 前置条件（C13）、重定向流程（C14/C15）、AMP 闭环工程化（C16）、四组评测指标（C17）、固定评测判定（C18）、远端训练（C19）、换号恢复（C20）、stale 规则（C21）、路由规则（C22）、结构 pivot（C23）、bounds 与终态（C24）、失败边界（C25/C26）、每轮职责（C27）。Constraint Ledger 中 27 条 hard constraint 全部完整保留，来源原文核对一致，未被改写。  
  Sources: amp_loop.md:行3, amp_loop.md:行15-27, amp_loop.md:行78-85, amp_loop.md:行109-115, amp_loop.md:行122-138  
  Blueprint: Blueprint goal, Blueprint successCriteria, Blueprint control  
  Graph: /state/phase, /state/status, /nodes/research/prompt, /transitions/3, /transitions/2
- C21 stale 阈值映射正确：when 读取更新前 $state.stale_count，attention_route stale_count>=3（+1 后=4 对应 C21 的 >=4）、pivot_route stale_count>=1（+1 后=2 对应 C21 的 >=2）、stale_route stale_count<1（+1 后=1 对应 stale）。C22 路由全部闭环：completed→done、attention→停止、error→停止、其余→research 继续。  
  Sources: amp_loop.md:行159-167, amp_loop.md:行169-177  
  Blueprint: Blueprint control stale_count/status/routing  
  Graph: /transitions/5, /transitions/6, /transitions/7, /transitions/8, /transitions/9, /transitions/10, /transitions/11

## workspace_contract

Status: pass

- control lane 独占 state/ 和 logs/ 全部文件（atomic_replace/append_only 逐文件精确声明），research lane deny state/。writer prompt 要求 progress.json 包含 C7 行36 要求的全部字段（iteration/phase/phase_status/status/stale_count/total_findings/code_revision/dataset_revision/updated_at），且 code_revision/dataset_revision 从 $input 读取不得省略。amp_loop.md 行36 确认 progress.json 必须包含这两个版本字段。  
  Sources: amp_loop.md:行32-50  
  Blueprint: Blueprint workspace state/ and logs/ files, Blueprint workspace read-only assets  
  Graph: /lanes/control/workspace/write, /lanes/research/workspace/write, /lanes/research/workspace/deny, /lanes/control/workspace/deny
- research lane 拥有 humanoid/data/scripts 三个 owned 写路径。prompt 明确要求'写入大体积数据到 data/ 下按原始/中间/训练版本化区分，不要写进 JSON state'。项目根 .git 存在且被两条 lane deny，无 scm:'git'，符合来源不要求 git commit/push 的事实。外部 npz 路径不在 prompt 中作为写目标，仅作为 blocking precondition。  
  Sources: amp_loop.md:行50  
  Blueprint: Blueprint workspace: 大体积数据仅存路径/哈希/摘要, Blueprint workspace: 外部 npz 须拷贝到项目内  
  Graph: /lanes/research/workspace/write, /nodes/research/prompt

## lane_ownership

Status: pass

- Blueprint 描述 4 个逻辑职责但声明'不预先固化可执行拓扑'。Graph 实现为 2 Lane（research 持久 + control 持久），将远端交互与首次审计合并到 research Agent 中：research prompt 包含 Gradmotion CLI 调用（bash 工具）、30 分钟轮询（timerPolicy maxDelayMs=1800000）、account-pool 换号有界重试逻辑；bootstrap_research（when bootstrapped==false）处理首次审计。research lane 拥有 humanoid/data/scripts 写权限覆盖全部代码实现与产物写入需求。control lane 写权限仅限 state/ 与 logs/。两条 lane 写路径不重叠。  
  Sources: amp_loop.md:行144-156, amp_loop.md:行192-197  
  Blueprint: Blueprint lanes: 厚研究 lane、确定性状态控制 lane、远端交互职责、首次审计职责  
  Graph: /lanes/research, /lanes/control, /nodes/research, /nodes/writer, /transitions/0
- 唯一 writer 模式完整实现：research 全部 8 条 success Transition（bootstrap_research 到 healthy_route）均路由到 writer，无绕过。writer 的 4 条 success 出边按 $state.status 路由到 terminal 或回 research。research 的 failure/exhausted 直接到 terminal（基础设施级故障，Agent 无法写入时合理绕过）。writer prompt 明确禁止自行计算阈值或越权推进 phase。  
  Sources: amp_loop.md:行153-155  
  Blueprint: Blueprint lanes: state_writer 是唯一文件 writer  
  Graph: /nodes/writer, /transitions/0, /transitions/1, /transitions/2, /transitions/3, /transitions/4, /transitions/5, /transitions/6, /transitions/7, /transitions/8, /transitions/9, /transitions/10, /transitions/11

## control_flow

Status: pass

- 确定性路由真值表完整：research 出边 priority 200→190→185→180→170→160→150→default，覆盖 bootstrapped×outcome×gate_passed×gate_complete×stale_count 全部有效组合且互斥。Transition Reducer（set/increment/add）是唯一更新 stale_count/status/phase/gate_passed/total_findings/iteration/bootstrapped 的环节。writer 出边 completed(210)→attention(200)→error(190)→default(research) 按 $state.status 闭环。writer_error Transition 确保错误状态写完后停止而非命中 default 继续循环。终态 done/failed/attention/exhausted 均有界。maxTotalActivations=200、maxWallTimeMs=172800000、maxCostUsd=200 提供总量上限。  
  Sources: amp_loop.md:行159-177, amp_loop.md:行201-206  
  Blueprint: Blueprint control: 确定性路由、stale 规则、终态义务  
  Graph: /transitions/0, /transitions/1, /transitions/2, /transitions/3, /transitions/4, /transitions/5, /transitions/6, /transitions/7, /transitions/8, /transitions/9, /transitions/10, /transitions/11, /transitions/12, /transitions/13, /transitions/14, /transitions/15
- Writer 读取 $state 提交后值写 progress.json：每条 research→writer Transition 的 target inputs 使用 $state.* 引用读取 Reducer 更新后的 stale_count/total_findings/iteration，以及与 Reducer 一致的 literal status。code_revision/dataset_revision 从 $output 绑定传递。gate_passed 从 $output.gate_passed 绑定到 writer inputs，不依赖 $state.gate_passed 的陈旧值。  
  Sources: amp_loop.md:行29, amp_loop.md:行36  
  Blueprint: Blueprint control: 阶段切换时写入代码/配置/数据版本  
  Graph: /nodes/writer/inputs, /transitions/2/updates, /transitions/3/updates, /nodes/research/outputSchema/properties/code_revision, /nodes/research/outputSchema/properties/dataset_revision

## capability_resolution

Status: pass

- research Agent 工具集（read_file/write_file/edit_file/append_file/bash/glob/grep/web_fetch）覆盖 AMP 代码实现、重定向脚本编写、Gradmotion CLI 调用、account-pool 换号全部需求。timerPolicy allowHardPark=true/maxDelayMs=1800000/maxParks=48 实现 C19 的 30 分钟轮询与有界长等待。writer Agent 工具集（read_file/write_file/edit_file/append_file）覆盖原子 replace 与 append 写入。capabilityGaps 中全部 4 项（外部 npz、F1 资产身份、AMP 工具链从零自建、评测阈值冻结）在 prompt 或 preconditions 中有对应处理。  
  Sources: amp_loop.md:行66-76, amp_loop.md:行93-99, amp_loop.md:行192-197  
  Blueprint: Blueprint capabilityGaps, Blueprint assumptions: Gradmotion CLI/account-pool 可用  
  Graph: /nodes/research/tools, /nodes/research/timerPolicy, /nodes/writer/tools, /lanes/research/workspace/write, /lanes/control/workspace/write
- 机械 Lint 提示 git-without-capability 核验：research prompt 中 code_revision 字段定义为'当前 git commit hash 或代码版本标识'，是只读获取 commit hash 用于可追溯性（C8），不包含 git add/commit/push 指令。项目根 .git 存在（已确认 .git/config）。owned 前缀（humanoid/data/scripts）下均无嵌套 .git。该 lint 为误报，无实际 git 写操作依赖。  
  Sources: amp_loop.md:行118  
  Blueprint: Blueprint capabilityGaps: 外部 npz 须拷贝到项目内  
  Graph: /nodes/research/prompt, /lanes/research/workspace/write

## runtime_preconditions

Status: pass

- 前置条件清单核验：amp_loop.md（项目根存在，blocking）✓；resources/robots/x1/urdf/x1.urdf（glob 确认存在，blocking）✓；data/ 目录（项目中不存在但为 owned 写路径，首次 write_file 自动创建）✓；data/0008_normal_walk4_stageii.npz（项目中不存在，外部路径沙箱不可访问，正确列为 blocking file precondition，缺失则 motion_retargeting 不得启动）✓；python command（blocking）✓；gradmotion credential（non-blocking，amp_training 阶段才需要）✓；F1-asset-identity decision（首轮审计做出，non-blocking）✓。  
  Sources: amp_loop.md:行52-62, amp_loop.md:行118  
  Blueprint: Blueprint workspace: 资产路径已确认存在, Blueprint capabilityGaps: 外部 npz 为 blocking precondition  
  Graph: /nodes/research/prompt, /lanes/research/workspace/read, /preconditions
- state/ 与 logs/ 目录项目中不存在，由 control lane 首次 write_file/edit_file/append_file 自动创建父目录。research lane read 列表含 scripts（项目根无此目录），但 scripts 为 owned 写路径将由 Agent 首次写入时创建。humanoid/scripts/（现有脚本所在位置）被 humanoid owned 写权限覆盖。所有 prompt 中声明的写目标均在项目内（data/、state/、logs/、humanoid/、scripts/），无项目外写入。  
  Sources: amp_loop.md:行33-50, amp_loop.md:行192-197  
  Blueprint: Blueprint workspace: state/ logs/ 由首轮自建  
  Graph: /lanes/control/workspace/write, /nodes/research/workspace/read
