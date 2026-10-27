# Condition CE identity audit (2026-10-27 Asia/Shanghai)

## Verdict
**Partial:** forecast CE 对条件路径 **有可微连通**（错 sector/size 可以改变 logits→CE），但 CE **不显式惩罚错身份**；最优解可退化为近常数偏置 → Full−None 可升、Full−Shuffled 贴平是 **预期**。

## Mechanism (paths)
- Inject: `model/kronos.py` `_add_context` → `γ * (sector_emb + size_mlp)` broadcast 到全时间步；`context_layer=10` / `n_layers=12` 在 layer 10 前注入。
- Train loss (base clean): `KRONOS_PREDICTOR_LOSS_MODE=forecast` + `history_weight=0.02`；`USE_BETA_V21=0` → **仅 token CE**，无 sector/size 正确性辅助项。
- Optimizer: `sector_emb` / `size_mlp` / `condition_scale` 在 condition 组，`requires_grad=True`；条件路径无 stop-grad（仅 stats `.detach()`）。
- Ablation: val-only Full/None/Shuffled forecast CE；训练不反传 shuffled 对比。

## Why B looks like bias
抬条件 LR 放大「有条件 vs 无」的共享偏置 μ；若各 sector 行塌向 μ，则 shuffle 仍≈μ → Full−Shuffled 微弱且远未可用。

## Next (no kernel)
1. Offline collapse metric on smoke ckpt: sector_emb row pairwise cosine / ‖row−mean‖.
2. Smallest identity-sensitive term: train-time `L += λ * relu(CE_full − CE_shuffled + m)`（或 asof→sector CE），再谈是否并 Bootstrap。
