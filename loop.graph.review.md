## Key Lowering Decisions

### Lane/Node Architecture
Two lanes, two agent nodes, two terminals:
- **work lane** (persistent): Thick research/execution Agent with owned write on humanoid/, scripts/, data/, gmr_f1/, .oma/. Reads all project files including state/ and logs/ for context but cannot write them.
- **writer lane** (persistent): Serial state_writer Agent that exclusively owns all state/ and logs/ file writes (9 file-level write rules: 5 atomic_replace, 4 append_only). No code/data write access.

The source's named deterministic nodes (reduce_progress, state_writer, gate_check, route_by_status) are **not** separate physical nodes. They map to:
- **reduce_progress** → work→commit transition updates (builtin/increment, decrement, set reducers on stale_count and status fields)
- **state_writer** → the commit Agent node
- **gate_check** → deterministic when conditions on gate pass transitions (all gA_*/gB_*/gC_* flags required simultaneously)
- **route_by_status** → priority-ordered commit→work/terminal transitions reading post-update $state

No identity/gate/status intermediate nodes were added. The work Agent's success edges directly update counters+status and route to commit; commit's success edges route by phase/health/thresholds.

### Fixes Applied in This Revision
1. **C11 (progress.json schema)**: Commit prompt action 1 now writes ALL required fields: iteration, phase, phase_status (derived from error/phase/status_C/health flags), total_findings (counted from findings.jsonl), code_revision and dataset_revision (from commit_payload or carried forward from existing progress.json). Work Agent prompt updated to always output code_revision/dataset_revision in commit_payload.
2. **C21 (terminal write-before-stop for stale exhaustions)**: gateA_stale_stop (idx 18) and gateB_stale_stop (idx 19) now route commit→commit (self-loop) setting error=true, instead of commit→failed directly. This ensures the commit node runs again to persist the final error state, stop reason in logs, and report before stop_error (idx 12, priority 1000) catches error==true and routes to failed terminal.

### Deterministic Thresholds (when reads pre-update $state)
- Gate A/B: stale_count >= 10 checked post-increment on commit side; gateA/B_stale_stop now routes to commit self-loop (ensures final write), then stop_error catches error
- Gate C pivot: stale_countC >= 2 && status_C=='normal' on commit side
- Gate C attention: stale_countC >= 4 on commit side (reachable via pivot failures incrementing counter)
- Gate C better decrement: guarded by when stale_countC > 0 (prevents negative)
- Phase advance: X_healthy == true checked on commit side, phase set by advance transition

### Workspace Protocol
- State files: 6 in state/ (2 append_only, 4 atomic_replace) + amp_contract.json (atomic_replace at project root)
- Log files: 2 in logs/ (both append_only)
- Bulk data (npz/checkpoint/video) stays out of JSON state; only relative paths/hashes in progress.json
- progress.json always includes: iteration, phase, phase_status, total_findings, code_revision, dataset_revision, updated_at (C11 compliance)
- No scm:'git' declared: source requirements do not mandate git commit/push; code changes are in-place

### Remote Training
- Agent timer hard park (maxDelayMs=3600000, maxParks=48) for ~30 min Gradmotion polling
- No Wait node inserted (would split activation and lose single-activation timing semantics)
- gm CLI invoked via bash tool; account-pool rotation for quota

### Budget
- Work: 50 turns/$25/10min per segment; lifetime 5000 turns/$500/7 days
- Commit: 15 turns/$3/5min per segment
- Graph: 300 total activations, $600, 7 days wall time, 1 live activation (serial)

### Capability Gaps (human must address before/during runtime)
1. **gmr_f1/ directory**: Must be created by cloning GMR before Gate B. Declared as blocking precondition.
2. **gm CLI**: Must be installed and callable. Declared as blocking precondition.
3. **F1 robot identity**: Audit (first activation) determines if assets are X1-named-but-F1 or genuinely X1. If missing, loop stops with error.
4. **Eval protocol freeze**: First amp_training entry requires human-aware decisions on threshold values.

### Assumptions
- taskDir = project root /Users/yumx/code/robot_x/X1/F1_train_AMP
- data/0008_normal_walk4_stageii.npz already exists in project (copied from external source)
- Project uses X1 naming throughout; real F1 vs X1 identity determined by audit
- Existing AMP code skeleton exists but Gate A pass is not assumed
- .oma/ directory for Gradmotion experiment outputs (gitignored)
- No git commit/push required by the workflow (no scm lane)