# sector_emb row-collapse probe (2026-10-27 Asia/Shanghai)

## Verdict
**未塌缩（sector_emb）**；**近常数偏置主要在 size_mlp**（B 更明显）。

- `sector_emb` 行间几乎正交，**不是**行塌向公共 μ。
- A+C 与 B 的 `sector_emb` Frobenius 仍 ≈ **5.387 ≈ Normal(0,0.02) 初值** → 行业表几乎没被训动（解释此前「钉死 ~5.387」：不是冻/塌缩，是步长下几乎未离 init）。
- B **没有**更强的 sector 行塌缩；B 把 `size_mlp` 放大成更大的近常向量，抬 Full−None，但 shuffle 行业在激活上仍有 Δ（||s_i−s_j||·γ≈20），Full−Shuffled 贴平仍更支持「CE 无身份压力」（见 `20261027_condition_ce_identity_audit.md`），而非 sector 行塌缩。

## Weights used (Kaggle Output → local)
| run | kernel | path |
|-----|--------|------|
| A+C+randperm Seg5 | `user281434/kronos-base-clean-0-1-ac-randperm-smoke` | `scratch/sector_emb_collapse_probe/ac_randperm/.../checkpoints/last_model/model.safetensors` |
| B lr10x Seg5 | `user281434/kronos-base-clean-0-1-acb-lr10x-smoke` | `scratch/sector_emb_collapse_probe/acb_lr10x/.../checkpoints/last_model/model.safetensors` |
| A+C γ25 roll (optional) | `user281434/kronos-base-clean-0-1-ac-gamma25-smoke` | `scratch/sector_emb_collapse_probe/ac_gamma25/.../checkpoints/last_model/model.safetensors` |

Config: `num_sectors=86` → emb `(87, 832)`；指标主报 **real rows 0..85**（排除 `unknown_sector_id=86`）。`condition_last_layer_init=normal` std=0.02，`condition_scale`≈25。

No kernel push. Bootstrap untouched. No train-path / aux-loss edits.

## sector_emb metrics (real 0..85)

| metric | A+C randperm | B lr10x | A+C γ25 roll |
|--------|-------------:|--------:|-------------:|
| shape | [86, 832] | [86, 832] | [86, 832] |
| mean pairwise cos (off-diag) | 0.000787 | 0.00827 | 0.000787 |
| mean ‖row−mean‖ / mean ‖row‖ | 0.9938 | 0.9900 | 0.9938 |
| frac cos(row,mean)>0.95 | 0.000 | 0.000 | 0.000 |
| frac cos(row,mean)>0.99 | 0.000 | 0.000 | 0.000 |
| mean cos(row,mean) | 0.1114 | 0.1407 | 0.1114 |
| top singular share (row-centered) | 0.0200 | 0.0211 | 0.0200 |
| mean ‖row‖ | 0.5774 | 0.5838 | 0.5774 |
| ‖W‖_F (full 87×832) | 5.3870 | 5.4462 | 5.3870 |
| γ (`condition_scale`) | 25.0063 | 25.0580 | 25.0063 |

Init reference: E[‖W‖_F]≈`0.02·√(87·832)`≈**5.381**；A+C/γ25 实测 **5.387**，B **5.446**（略动）。

A+C randperm ≈ A+C γ25 roll（消融方式不影响表权重几何）。

**B vs A+C (sector):** B 的 pairwise cos 略升（0.0008→0.008）、mean cos-to-mean 略升（0.11→0.14），仍远未塌缩；**不能**用「B 更塌缩」解释对照。

## size_mlp (cheap probe; known=1, pct∈[0,1] grid)

| metric | A+C | B |
|--------|----:|--:|
| mean pairwise cos of outs across pct | 0.9812 | 0.9991 |
| frac cos(out, mean_out)>0.99 | 0.619 | 1.000 |
| mean ‖size_out‖ | ~0.88 | ~5.13 |
| final bias ‖b₂‖ | 0.5819 | 0.7749 |
| rms var across pct / rms mean | 0.1467 | 0.1105 |

After γ≈25: A+C size 贡献 ~22 vs sector ~14；**B size ~128 vs sector ~15**（size 主导）。B 的 size 输出跨分位方向几乎不变（cos≈0.999）→ **近常数偏置 μ 主要来自 size_mlp**。

## Does this support「近常数偏置 μ」?
**部分支持，但机制要改写：**
1. ❌ `sector_emb` 行塌缩 → μ：**否**（行未塌，几乎仍是 init 随机方向）。
2. ✅ `size_mlp` → 近常 μ：**是**（A+C 已偏常，B 放大且更常）→ 解释 Full−None 抬升。
3. Full−Shuffled 贴平：shuffle 行业在激活上仍有大幅 Δ（γ·‖s_i−s_j‖≈20），故 **不能**归因于 sector 塌缩；与 CE 身份审计一致——损失不惩罚错行业。

## Next (no kernel until Elon nod)
- Aux / identity-sensitive term 仍待点头；本探测 **不**支持「先修塌缩再 aux」，因行业表根本没学起来也没塌掉。
- 若动代码：优先让 CE（或 aux）对错 sector 有可分压力；单纯再抬 LR 只会继续放大 size 常偏置。

## Artifacts
- Metrics JSON: `scratch/sector_emb_collapse_probe/collapse_metrics.json`
- Probe time: Asia/Shanghai (box-local)
