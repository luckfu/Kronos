# GPU Alpha Wave 2B（G27–G28）事前登记（2026-11-19）

> **严格防泄漏铁律**：
> 1. 所有特征投影（PCA）、OOF Ridge 锚点、微调训练与验证选型严格截止 **`2026-07-02`**（`train_data.pkl <= 2026-04-30` + `val_data.pkl <= 2026-07-02`）；
> 2. 所有超参、早停 Epoch 与 `breadth_ma20` 软门控参数仅在 **`2026-05-18 .. 2026-07-02`** 未见面验证段选型；
> 3. 所有开发期样本外评测统一对齐 **`2026-07-17 .. 2026-10-23`（65 交易日、`335,312` 行）**；
> 4. **严禁触碰 `2026-10-24+` 封存验收窗或用其调参**；
> 5. 生产线上维持 `deploy/oracle-kronos/head_val_alpha1e4.npy` 不变。

---

## 1. Wave 2B 拟填补的两大核心空白（G27 & G28）

| 实验 | Kaggle 账号 / Slug | 核心科学问题与方法设计 |
|---|---|---|
| **G27** | `user281434`<br>`kronos-alpha-g27-nonlinear-moe-lgbm` | **从纯线性岭回归跨越到非线性 / 门控双专家（MoE）**：<br>在冻结 `PT-Small L=45 [L3,L5,L6_pre,L7_last,L7_m5] (2,560d)` + `Kronos-base L=45 [L5,L8,L10] (2,496d)` 上：<br>1. 仅用 `<= 2026-04-30` 计算 PCA-128 基与 2 折 OOF 线性锚点（`pts_all / pts_bull / pts_bear / base_all / base_bull / base_bear`）+ 4 个因果市场特征，构成 `138d` 无泄漏输入；<br>2. 检验 **因果市场宽度门控双专家 MoE**（`breadth_ma20` Sigmoid 软切换强/弱市 Ridge 专家，`g27_moe_breadth_{pts,base,joint}`）；<br>3. 检验 **4 路非线性/高阶目标头**：`ResidualAlphaMLP (Cosine-IC)`、`ResidualAlphaMLP (Top-K ListNet)`、`BreadthFiLMMLP`、`ICIR-Linear`，以及 **`LightGBM` 非线性树模型（`74d`）**。 |
| **G28** | `smmt315`<br>`kronos-alpha-g28-l45-gentle-ft` | **黄金短窗 `L=45` 极温和顶层微调 + 微调后隐状态闭式 Ridge**：<br>此前微调（G2/G4）仅做过 `L=60/90/120`，从未在已验证最优的短窗 `L=45` 上专门微调：<br>1. **Arm 0 (`g28_pts_l45_ft`)**：冻结 `NeoQuasar/Kronos-small` 底层 `0..4`，仅以极低学习率（`1.5e-5`）+ 强 KL 蒸馏锚（`0.08`）+ 强权重 L2-SP 锚（`2000.0`）微调顶层 `5..6`；<br>2. **Arm 1 (`g28_c2_l45_ft`)**：同步在 `L=45` 上对 `C2 (Small-0.1-Cosine-C2-Best)` 做同等温和顶层微调；<br>3. **两路产出一并评测**：端到端排序头（`*_e2e`）以及微调后隐状态 `[L3, L5, L6_pre, L7_last, L7_m5]` 闭式岭回归头（`*_ridge_{l7,ml,pool}`）。 |

---

## 2. 配套脚本

- G27：[`g27_nonlinear_moe_lgbm_entry.py`](../gpu_alpha/g27_nonlinear_moe_lgbm_entry.py) / [`push_g27.py`](../gpu_alpha/push_g27.py)
- G28：[`g28_l45_gentle_ft_entry.py`](../gpu_alpha/g28_l45_gentle_ft_entry.py) / [`push_g28.py`](../gpu_alpha/push_g28.py)
