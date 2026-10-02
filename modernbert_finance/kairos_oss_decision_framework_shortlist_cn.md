# OSS 决策框架短名单（mfe10 决策轴）

日期：2026-10-02 10:33:12 CST（北京时间）。  
范围：为 `y=1{max(high[T+1:T+10])/close[T]-1 >= 0.10}` 找可改编开源框架；闸门 `Δ logloss vs prior ≤ -0.04` 或明确决策效用。  
前置硬结论：ModernBERT/序列短烟死；Ranking 不作产品；日线 Base+xsection 表格仅 `val_temporal` 过闸、`train→val` 主协议 FAIL（U/V/W）。  
禁止：碰 TPU WIP `beta_v21_c1*`；同轴再加长表格烟测；发明不存在的仓库。

## 一句话

**先烟测 Microsoft Qlib（LGBModel `loss:binary` + 自定义 mfe10 标签 + 时序 segments）**；用 Δ logloss（非 IC/TopK）同时报 `train→val` 与 `val_temporal`。这是相对失败轴的**新特征/工作流轴**（Alpha158/可扩展 handler），不是再训 ModernBERT，也不是 ranking 产品。

## 排名短名单（最多 5）

| # | 名称 | Repo | 为何贴合 mfe10 决策 | 改编成本 | vs 失败轴风险 |
|---:|---|---|---|---|---|
| 1 | **Microsoft Qlib** | https://github.com/microsoft/qlib | 原生 A 股 `cn_data`；handler 可自定义标签；`LGBModel` 支持 `loss: binary`；`DatasetH` 时序 train/valid/test；可直接输出概率做 Δ logloss | **低–中**：预计算/表达式注入 mfe10；关 IC/Topk 主闸，换成 logloss；扩到 Alpha158 相对 26 维 Base+xsection | **主风险=默认 workflow 是 ranking/Topk**——必须禁用作产品判据。特征轴新，但若仍只用极薄截面特征，可能复刻 train→val FAIL |
| 2 | **Time-Series-Library**（TimesNet/PatchTST 等） | https://github.com/thuml/Time-Series-Library | 内置 **classification** 任务；易挂二分类头与自定义标签；活跃 | **中**：需自写 A 股日线适配与 walk-forward；标签外置 | **高复刻序列失败轴**（I/J/K/P ModernBERT 已否证短预算序列）。仅当特征/窗口/校准与 Kairos 序列烟测明显不同才值得 |
| 3 | **pytorch-forecasting TFT** | https://github.com/sktime/pytorch-forecasting | 多步概率预测；可用 high 路径分位数/样本近似 `P(MFE≥10%)`；天然 walk-forward | **中–高**：无 A 股一等公民；预测→决策需校准层与 logloss 闸 | 间接决策；算力重；归纳偏置偏序列/路径，接近已失败序列轴 |
| 4 | **Chronos** | https://github.com/amazon-science/chronos-forecasting | 概率预测 FM；可从样本路径估触及概率；活跃 OSS | **高**：非决策原生；A 股面板/截面弱；fine-tune≠校准分类 | 零样本/预测轴，难直接过 Δ logloss 闸；勿当第一烟测 |
| 5 | **TradeMaster** | https://github.com/TradeMaster-NTU/TradeMaster | 有多市场数据与评估管线；可改 reward≈mfe10 触及 | **高**：RL 目标≠校准概率；logloss 闸别扭 | 产品形态偏离决策概率；易漂向交易仿真而非过闸证据 |

## 明确排除 / 降级（已核实）

| 名称 | 原因 |
|---|---|
| **LARA**（论文） | 概念极贴「有利可图样本」；**未找到可靠官方金融实现仓库**（同名 GitHub 多为无关项目）→ 不发明 |
| **AlphaCare** | `llSourcell/AlphaCare` 为医疗 DL，非股票决策框架 |
| **StockNet** | https://github.com/yumoxu/stocknet-code — 二分类涨跌但依赖推文+美股，陈旧，A 股适配差 |
| **MASTER** | https://github.com/SJTU-Quant/MASTER — A 股友好，但默认收益/截面排序产品，易滑回 ranking |
| **AlphaGen / CSAlpha** | 因子挖掘/RankIC 导向，与「决策 only」冲突 |
| **Moirai (uni2ts)** | https://github.com/SalesforceAIResearch/uni2ts — 可用作 Chronos 同级备选，但同属预测 FM，不进前五主推 |
| **本仓 Kronos FM / ModernBERT** | 序列决策短烟已死；勿重启同配方 |
| **FinRL** | 活跃但 RL 管线；与 Δ logloss 闸不匹配，低于 TradeMaster 列入价值 |

## 推荐：第一个本地 / Kaggle 烟测

**选 Qlib。**

最小协议建议：

1. 标签：预计算 `y = 1{mfe10 ≥ 0.10}`（未来 10 日 high 路径），注入 handler/label，勿用默认 `Ref($close,-2)/Ref($close,-1)-1`。  
2. 模型：`qlib.contrib.model.gbdt.LGBModel`，`loss: binary`（文档确认支持 mse/binary）。  
3. 特征：先 Alpha158（或现有日线特征进 Qlib handler）——相对失败的 26 维 Base+xsection 是**新证据轴**。  
4. 切分：显式时序 `train/valid/test`；**同时**报  
   - 主协议近似：较早窗 → 全 val（对齐 U/V/W train→val）  
   - 次协议：近端 `val_temporal`  
5. 闸门：**仅** `Δ logloss vs 常数先验 ≤ -0.04`（或明确决策效用）；**禁止** Rank IC / Topk 作产品成功标准。  
6. 不做：Ranking Phase；ModernBERT；TPU `beta_v21_c1*`；同表格 Logistic/enet 加长。

若 Qlib 烟测主协议仍 FAIL 而 temporal 仍过：与 W 硬结论同构 → 下一轴应换标签定义或特征族，而非再加树深度。

## 与宪章对齐

见 `kairos_decision_only_handoff_cn.md`：产品 = 决策标签 only；W HARD-CONCLUDE 后需**新证据轴**。本短名单以 Qlib 特征/工作流轴开烟，符合「非同轴空转」。

## Phase X 烟测结果（2026-10-02 10:38:47 CST）

- 主协议 train→val：LGB+Alpha158 Δ≈**−0.025819** → **FAIL**（闸 −0.04）
- 次协议 val_temporal：Δ≈**−0.035231** → **FAIL**
- 决策：`FAIL_HARD_CONCLUDE_QLIB_ALPHA158_AXIS`；不加长 confirm
- 详见：`kairos_phase_x_qlib_alpha158_decision_results_cn.md`

## Phase Y 标签轴结果（2026-10-02 11:04:47 CST）

- CS top 五分位：主协议 Δ≈**−0.031380 FAIL**；temporal −0.041619 PASS  
- 备选 soft `mfe10≥0.08`：主协议 Δ≈**−0.021921 FAIL**（更差）  
- 决策：`FAIL_HARD_REPORT_LABEL_AXIS`；停标签族空转；不自动开 #2 TSLib
