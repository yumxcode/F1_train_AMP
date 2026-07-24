## 关键 Lowering 决策

### Lane/节点架构（未变）
- **2 Lane**：`research`（持久化，拥有 humanoid/scripts/data 的写权限）和 `control`（持久化，拥有 state/ 和 logs/ 的全部 write）。
- **2 Agent 节点**：`research`（厚 Agent）和 `writer`（薄 Agent，唯一文件 writer）。
- **4 Terminal 节点**：done、failed、attention、exhausted。

### 诊断修复（本轮变更）

#### 1. progress.json 补全 code_revision 和 dataset_revision（C7）
- reviewer 发现 writer prompt 中 progress.json 字段列表缺少 C7（amp_loop.md 行36）要求的 code_revision 和 dataset_revision。
- **修复**：
  - research Agent outputSchema 新增 `code_revision`（string）和 `dataset_revision`（string）字段，Agent 每轮输出当前 git commit hash 和数据版本标识。
  - 所有 research→writer Transition 的 target inputs 绑定 `code_revision: {$output.code_revision}` 和 `dataset_revision: {$output.dataset_revision}`。
  - writer Agent inputs 新增这两个字段（`$input.code_revision`、`$input.dataset_revision`）。
  - writer prompt 的 progress.json 字段列表新增 code_revision 和 dataset_revision，明确从输入读取不得省略。

#### 2. 补全 writer error→failed 终态路由（C22/C24）
- reviewer 发现 error_route set status='error' 后，writer 命中 default writer_continue 返回 research 继续运行，违反 C22 'error 须写完后停止' 和 C24 的优雅退出要求。
- **修复**：新增 `writer_error` Transition（priority 190，when: `$state.status == 'error'`，target: failed）。插入在 writer_attention（200）和 writer_continue（default）之间。现在 writer 的 4 条 success 出边完整覆盖所有终态：completed→done(210), attention_required→attention(200), error→failed(190), default→research。

### 确定性路由真值与阈值（未变，仅 transition 索引因新增 writer_error 而后移）
- research 出边：error(190) > gate_complete(185) > gate_advance(180) > attention(170) > pivot(160) > stale(150) > healthy(default)。
- writer 出边：completed→done(210) > attention→attention(200) > error→failed(190) > default→research。
- stale 阈值（when 读取更新前 $state）：attention>=3, pivot>=1, stale<1。

### Workspace 路径与 Owner（未变）
- state/ 和 logs/ 全部文件由 control lane 独占，逐文件精确 atomic_replace / append_only。
- research lane deny state/，拥有 humanoid/data/scripts owned 写路径。
- 无 scm:'git'。

### 外部能力缺口与前置条件（未变）

### 预算（未变）

### 人工审查点（未变）
- 无用户交互；attention/error/completed 写完 state/log/report 后自动停止。