# Experience Index
*Last updated: 2026-07-31 08:19 | Total: 5 entries*

## general (5)
- [exp_ms8o89y7_46c3ecd6] **完全被支配的 AMP 判别器(acc=1.0/style_r≈0)被当 PASS：style_r 仅被 exp-floor 掩盖而非修复** | ✗ 判别器 acc=1.0、style_r≈0 全程恒定证明 AMP 判别器被完全支配，style_r 只被 exp-flo | confidence: observed | tags: amp, discriminator-domination, style-reward-collapse, gate-c
- [exp_ms8lw71j_f203a051] **上下文重置后未执行 load_state 导致阶段回退：重做已验收的 Gate B 重定向、遗弃在飞训练任务** | ✗ 上下文重置后未 load_state，导致回退重做已验收 Gate B 并遗弃在飞 gradmotion 训练；应恢复  | confidence: observed | tags: context-reset, load-state, phase-regression, amp-loop
- [exp_ms8kbeue_9b0550d4] **重定向产物实为"过程式步态合成"+自带验证 pass:false 仍欲放行** | ✗ 重定向产物方法错（合成非重定向）且验证 pass:false，须先修方法+IK 并以 pass:true 过严格穿地/滑 | confidence: observed | tags: motion-retargeting, gait-synthesis-vs-retarget, amp-reference, validation-gate
- [exp_ms8jtr5s_905964dc] **IK 重定向关节名拼写错致约束静默丢失：输出"格式合法但无步态"** | ✓ 重定向输出"格式合法但关节冻结/无步态"是 IK 约束静默丢失的典型；校验须补关节活动范围、左右对称、双脚接触交替，不能 | confidence: observed | tags: retargeting, ik, silent-failure, gate-b
- [exp_ms8j8wbh_f314afce] **Vendored pylibs 路径未加入 sys.path 致 mink/mujoco 静默导入失败、重定向无产物** | ✗ 重定向脚本因 PYLIBS 未入 sys.path 导致 mink/mujoco 导入失败、30 轮无产物；须修路径并以 | confidence: observed | tags: retargeting, python-import, vendored-libs, execution-blocked

## Quick Search
`experience_search domain=<domain> tags=<tag1,tag2> keyword=<word>`
`experience_load id=<id>` — load full entry with report