# Loop Semantic Review

Accepted: yes

## intent_constraints

Status: pass

- C1 (goal) 在 work Agent prompt 中完整陈述三阶段目标与成功标准，属于 agent 落点，已交底。C2 (阶段硬性递进) 通过三条确定性 advance transitions (advance_B/B_healthy→motion_retargeting, advance_C/B_healthy→amp_training, advance_done/C_healthy→completed) 实现，无任何 transition 可在对应 health flag 为 false 时推进 phase。  
  Sources: amp_loop.md:L3, amp_loop.md:L16-L28  
  Blueprint: Blueprint.goal, Blueprint.control[0]  
  Graph: /state/phase, /transitions/14, /transitions/15, /transitions/16
- C3 (X_healthy 默认 false，仅 true 时进入下一阶段) 与 C4 (禁止篡改状态/降低阈值/跳过检查跨门禁) 均已实现：三个 health flags 初始值 false，gate pass transitions 同时要求所有验证 flag 为 true，Agent 无权直接写 state 文件（writer lane 独占），且 prompt 规则 2 明确禁止降低阈值。  
  Sources: amp_loop.md:L28, amp_loop.md:L156-L157  
  Blueprint: Blueprint.control[0-1]  
  Graph: /transitions/3/when, /transitions/5/when, /transitions/7/when, /state/A_healthy, /state/B_healthy, /state/C_healthy

## workspace_contract

Status: pass

- C11/C12 workspace protocol: writer lane 对 state/ 下 6 个文件声明了精确的 write mode（findings.jsonl 和 iteration_log.jsonl 为 append_only，progress.json/directions_tried.json/task_spec.md/evaluation.json 为 atomic_replace），amp_contract.json (atomic_replace) 位于项目根，logs/ 下 work.jsonl 和 orchestrator.jsonl 为 append_only。commit prompt action 1 写入 progress.json 含 C11 要求的全部字段（iteration, phase, phase_status, total_findings, code_revision, dataset_revision, updated_at）。实地验证 state/ 和 logs/ 目录尚不存在，由 commit prompt action 9 在首轮创建。  
  Sources: amp_loop.md:L34-L49  
  Blueprint: Blueprint.workspace[1-2]  
  Graph: /lanes/writer/workspace/write, /nodes/commit/prompt
- 大体积数据（npz/checkpoint/视频）不进 JSON state，仅保存相对路径/哈希/摘要。work lane 的 owned writes 为 humanoid/scripts/data/gmr_f1/.oma，writer lane 的 writes 全部限定在 state/、logs/、amp_contract.json，无写路径冲突或越权。  
  Sources: amp_loop.md:L49  
  Blueprint: Blueprint.workspace[6]  
  Graph: /lanes/writer/workspace, /lanes/work/workspace

## lane_ownership

Status: pass

- C18/C19/C20 ownership: reduce_progress → work→commit transition 的 builtin reducers（increment/decrement/set on stale_count 和 status），Agent 仅输出语义 outcome；state_writer → commit Agent 节点独占 state/logs 写入；gate_check → gate pass transitions 的确定性 when 条件；route_by_status → commit 侧优先级有序 transitions。work prompt 规则 1 明确禁止 Agent 写 state/logs。两 lane 均为 persistent、maxConcurrency=1，强相关生命周期（研究→评估→提交）在 work lane 单条会话内完成。  
  Sources: amp_loop.md:L147-L150  
  Blueprint: Blueprint.lanes[0-1]  
  Graph: /lanes/work, /lanes/writer, /nodes/work/prompt

## control_flow

Status: pass

- Gate A/B stale_stop (idx 18/19) 在 stale_count>=10 时设 error=true 并 commit→commit 自循环，确保 commit 再次执行写入终态 report 后 stop_error (idx 12, priority 1000) 捕获 error==true→failed。Gate C: attention (idx 20, stale>=4) priority 930 > pivot (idx 21, stale>=2&&normal) priority 920，保证 attention 优先。所有终态路径（error/completed/attention）均经 commit 写入后才进入 terminal。自循环恰好执行一次（第二轮 commit 中 stop_error 优先级 1000 > gateA_stale_stop 950），有界。  
  Sources: amp_loop.md:L160-L200, amp_loop.md:L218  
  Blueprint: Blueprint.control[1-5]  
  Graph: /transitions/18, /transitions/19, /transitions/20, /transitions/21, /transitions/12, /transitions/13
- Gate C normal: pass→C_healthy (idx 7), fail→stale+1 (idx 9), better→stale-1 min-0 guard (idx 11, when stale_countC>0)。pivot_done→stale=0+normal (idx 8)。pivot_fail→stale+1 (idx 10)。stale>=4→attention→report→error→stop。所有路径有界：stale 上限触发终态，maxTotalActivations=300 为顶层 bounds。  
  Sources: amp_loop.md:L186-L196, amp_loop.md:L214  
  Blueprint: Blueprint.control[2], Blueprint.control[4]  
  Graph: /transitions/7, /transitions/9, /transitions/11, /transitions/8, /transitions/10, /limits

## capability_resolution

Status: pass

- Gate A/B/C 的 hard gate 条件分别要求 gA_smoke∧gA_schema∧gA_ckpt∧gA_contract、gB_replay∧gB_loader∧gB_isolation、gC_walk∧gC_eval∧gC_sim2sim∧gC_export 全部同时为 true。这些 flag 由 work Agent 基于具体证据设置，仅当全部满足时确定性 transition 才推进。work Agent tools 含 bash（执行 gm CLI）、read_file/write_file/edit_file/glob/grep/web_fetch/skill。C27/C28/C29 (agent success_criteria) 已在 work prompt 各 phase 行为段中完整交底。  
  Sources: amp_loop.md:L77-L84, amp_loop.md:L88-L109, amp_loop.md:L118-L136  
  Blueprint: Blueprint.intent  
  Graph: /transitions/3/when, /transitions/5/when, /transitions/7/when, /nodes/work/prompt
- C30 (GMR 克隆到项目内) 和 C31 (gm CLI + 参考动作 npz) 为 human capability 落点：work lane owned write on gmr_f1/ 允许克隆，gm CLI 通过 bash 工具调用，data/0008_normal_walk4_stageii.npz 实地确认存在。gmr_f1/ 不存在已列为 blocking directory precondition。  
  Sources: amp_loop.md:L90, amp_loop.md:L112, amp_loop.md:L205-L210  
  Blueprint: Blueprint.capabilityGaps  
  Graph: /lanes/work/workspace/write, /lanes/work/tools

## runtime_preconditions

Status: pass

- work prompt 首行动要求读取 state/progress.json, state/findings.jsonl, state/directions_tried.json, state/evaluation.json 和 amp_loop.md。实地验证：amp_loop.md 存在 ✓；state/ 下文件尚不存在，由 commit Agent 首轮创建（prompt actions 2-6, 9）；data/0008_normal_walk4_stageii.npz 存在 ✓；humanoid/**/*.py 含 AMP discriminator/motion_lib/replay_buffer 等存在 ✓；resources/robots/x1/urdf/*.urdf 存在 ✓；monitor_train.sh 存在 ✓。gmr_f1/ 不存在，已列为 blocking directory precondition ✓；.oma/ 不存在，work lane owned write 允许 Agent/Gradmotion 创建，.gitignore 已覆盖 .oma/experiments/ ✓；gm CLI 已列为 blocking command precondition ✓。  
  Sources: amp_loop.md:L34-L49, amp_loop.md:L53-L61, amp_loop.md:L112  
  Blueprint: Blueprint.capabilityGaps, Blueprint.assumptions  
  Graph: /nodes/work/inputs, /nodes/work/prompt
- C24 timer (~30 min poll) 与 C23 recovery (account-pool 换号有界重试) 均通过 work Agent timerPolicy (maxDelayMs=3600000=30min, maxParks=48) 和 prompt 中的 REMOTE TRAINING PROTOCOL 实现。gm CLI 为外部依赖已列为 precondition。两条 decision preconditions (C25-audit-findings, C14-eval-protocol-freeze) 正确覆盖了运行时可能需要人工介入的场景。  
  Sources: amp_loop.md:L207, amp_loop.md:L209  
  Blueprint: Blueprint.lanes[2]  
  Graph: /nodes/work/timerPolicy
