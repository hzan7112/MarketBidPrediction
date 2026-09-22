# bidprediction 阶段性说明：输入特征、策略转移画像与模板分类模型

> 当前基准日期：2026-09-21  
> 项目：MarketBidPrediction  
> 阶段：第三阶段 `scripts/bidprediction`  
> 当前任务：在已完成主体策略画像和报价模板库的基础上，构建无泄漏预测输入，完成模板预测，并进入“模板 + 曲线参数”预测阶段。

---

## 1. 当前预测框架

第三阶段不再直接对完整报价曲线做高维预测，而采用“模板 + 参数”的分解方式：

$$
\text{主体历史与市场状态}
\rightarrow
\text{报价模板}
\rightarrow
\text{模板条件下的曲线参数}
\rightarrow
\text{完整报价曲线}
$$

当前模板库由 12 个 shape 模板 `T00~T11` 和 1 个平价模板 `FLAT` 构成，共 13 类。

当前用于预测建模的特征空间统一记为：

$$
X_{i,t}
=
[
Z^{base}_{i,t},
Z^{transition}_{i,t},
M_t,
U_{i,t},
H_{i,t}
]
$$

其中：

- `Z_base`：主体既有策略画像，26 维；
- `Z_transition` / `Z_tr`：新增策略转移画像，26 维；
- `M`：市场环境特征，8 维；
- `U`：主体/机组运行状态代理特征，12 维；
- `H`：主体直接历史报价状态，9 维。

因此当前分类实验使用的完整候选输入共：

$$
26+26+8+12+9=81
$$

维。

需要强调：`Z_tr`、`M`、`U`、`H` 的语义不同，不应简单混为一个“历史特征集合”。

- `Z_base` 描述主体长期/短期报价策略属性；
- `Z_tr` 描述主体历史策略切换行为；
- `M` 描述外部市场环境；
- `U` 描述机组历史可观测运行边界与成本状态；
- `H` 保存最直接的上一历史报价状态。

当前 prediction dataset 中还保留 calendar/readiness 等辅助字段，但它们不属于当前已锁定的模板分类输入主体。

---

## 2. `Z_base`：既有主体策略画像（26 维）

`Z_base` 由第一阶段 `bidprofile` 最终确定的 9 个长期特征、9 个短期特征和 8 个 Break 特征组成。

### 2.1 LT：长期策略特征（9 维）

| 特征 | 含义 |
|---|---|
| `lt_bid_level` | 主体长期报价价格水平 |
| `lt_adjustment_magnitude` | 长期报价调整幅度 |
| `lt_strategy_persistence` | 长期策略持续性/稳定性 |
| `lt_quantity_hhi` | 报价电量在各段之间的集中程度 |
| `lt_effective_segment_count` | 长期有效报价段数 |
| `lt_flat_curve_rate` | 长期平价曲线出现比例 |
| `lt_tail_uplift_ratio` | 曲线尾部价格抬升程度 |
| `lt_curve_bend_ratio` | 报价曲线弯曲程度 |
| `lt_shape_variability` | 历史报价曲线形状波动程度 |

LT 特征采用滚动历史构造。当前 prediction dataset 使用 180 天历史窗口，并要求至少 30 个有效历史活跃日后才视为 LT-ready。

### 2.2 ST：短期状态特征（9 维）

| 特征 | 含义 |
|---|---|
| `st_bid_level_z` | 当前短期报价水平相对主体自身历史的标准化偏移 |
| `st_adjustment_bias_z` | 当前短期调整方向偏置 |
| `st_adjustment_magnitude_z` | 当前短期调整幅度 |
| `st_quantity_hhi_z` | 当前电量段集中度相对历史的变化 |
| `st_effective_segment_count_z` | 当前有效段数相对历史的变化 |
| `st_flat_curve_rate_z` | 当前平价倾向相对历史的变化 |
| `st_tail_uplift_ratio_z` | 当前尾部抬升相对历史的变化 |
| `st_curve_bend_ratio_z` | 当前曲线弯曲程度相对历史的变化 |
| `st_shape_shift` | 当前报价曲线形态相对历史基准的整体偏移 |

ST 特征用于表达“主体当前是否偏离其长期策略状态”，而不是替代 LT 特征。

### 2.3 Break：结构突变特征（8 维）

| 特征 | 含义 |
|---|---|
| `break_bid_level` | 报价水平是否出现显著结构变化 |
| `break_adjustment_bias` | 调整方向是否发生明显变化 |
| `break_adjustment_magnitude` | 调整幅度是否发生明显变化 |
| `break_quantity_hhi` | 报量集中程度是否发生明显变化 |
| `break_effective_segment_count` | 有效报价段数是否发生明显变化 |
| `break_flat_curve_rate` | 平价报价倾向是否发生明显变化 |
| `break_tail_uplift_ratio` | 尾部抬升策略是否发生明显变化 |
| `break_curve_bend_ratio` | 曲线弯曲结构是否发生明显变化 |

因此：

$$
Z_{base}
=
[
LT_9,\ ST_9,\ Break_8
]
$$

共 26 维。

---

## 3. `Z_transition`：新增策略转移画像（26 维）

模板分类实验表明，`Z_base` 能较好描述“主体通常采用什么类型的报价策略”，但对“主体什么时候会从当前模板切换到另一个模板”的描述不足。

因此在第三阶段新增：

```text
scripts/bidprediction/02b_build_transition_features.py
```

生成：

```text
transition_strategy_profile_<year>.csv
transition_strategy_feature_schema_<year>.csv
```

### 3.1 构造原则

`Z_tr` 是 participant-day 级特征。对于目标日 $D$，所有转移统计严格只使用：

$$
d < D
$$

的历史模板记录。

因此不存在将目标日真实模板泄漏到输入的问题。

`Z_tr` 不包含市场负荷、系统容量、停运、机组能力等外部因素。它只回答：

> 该主体过去的模板选择和模板转移具有怎样的动态规律？

市场等外部因素仍独立放在 `M` 和 `U` 中。

### 3.2 具体 26 个特征

| 特征 | 含义 |
|---|---|
| `tr_hist_active_days` | 截至目标日前累计有效历史活跃日数 |
| `tr_switch_rate_7d` | 最近 7 个历史活跃日的模板切换率 |
| `tr_switch_rate_30d` | 最近 30 个历史活跃日的模板切换率 |
| `tr_switch_rate_90d` | 最近 90 个历史活跃日的模板切换率 |
| `tr_switch_count_30d` | 最近 30 日切换次数 |
| `tr_switch_count_90d` | 最近 90 日切换次数 |
| `tr_ever_switched` | 历史上是否曾发生模板切换 |
| `tr_dwell_active_days` | 当前模板已连续维持的活跃日数 |
| `tr_days_since_last_switch` | 距离最近一次模板切换的活跃日距离 |
| `tr_dominant_share_7d` | 最近 7 日主导模板占比 |
| `tr_dominant_share_30d` | 最近 30 日主导模板占比 |
| `tr_template_entropy_7d` | 最近 7 日模板分布熵 |
| `tr_template_entropy_30d` | 最近 30 日模板分布熵 |
| `tr_unique_template_count_7d` | 最近 7 日出现过的不同模板数量 |
| `tr_unique_template_count_30d` | 最近 30 日出现过的不同模板数量 |
| `tr_lag1_daily_dominant_share` | 前一历史日主导模板占比 |
| `tr_lag1_daily_template_entropy` | 前一历史日模板分布熵 |
| `tr_lag1_daily_unique_template_count` | 前一历史日模板种类数 |
| `tr_daily_purity_mean_7d` | 最近 7 日单日模板纯度均值 |
| `tr_daily_purity_mean_30d` | 最近 30 日单日模板纯度均值 |
| `tr_daily_entropy_mean_7d` | 最近 7 日单日模板熵均值 |
| `tr_daily_entropy_mean_30d` | 最近 30 日单日模板熵均值 |
| `tr_transition_dest_concentration_90d` | 最近 90 日所有切换目的模板的集中度 |
| `tr_origin_switch_rate_90d` | 当前 origin 模板在过去 90 日的历史切出概率 |
| `tr_origin_dest_concentration_90d` | 当前 origin 条件下目的模板的集中度 |
| `tr_reversion_rate_90d` | 最近 90 日发生 A→B 后重新回到 A 的比例 |

构建结果：

- Participant-days：216,103；
- Transition-ready：211,607；
- Ready share：97.92%；
- 历史日级模板切换率约 4.08%。

`tr_transition_dest_concentration_90d`、`tr_origin_dest_concentration_90d`、`tr_reversion_rate_90d`、`tr_days_since_last_switch` 等属于事件条件统计。对于长期未发生切换的主体，其缺失具有明确语义，因此不强制填 0，而是在模型训练时采用 TRAIN-only 缺失值处理。

### 3.3 `Z_tr` 的实验结论

`Z_tr` 对普通 13 类模板直接分类的提升不稳定，但对 switch/no-switch 判断具有非常明显的增益。

例如测试集 Random Forest：

```text
Z_base:
Switch PR-AUC = 0.1813
Switch F1     = 0.2742

Z_base + Z_tr:
Switch PR-AUC = 0.4352
Switch F1     = 0.4974
```

因此当前结论是：

> `Z_tr` 不是 `Z_base` 的简单替代，而是专门用于描述主体策略转移风险的动态画像。

---

## 4. `M`：市场环境特征（8 维）

当前实际可用的市场环境输入主要来自 PJM 的负荷预测、计划停运预测和日前容量信息。

| 特征 | 含义 |
|---|---|
| `mkt_load_forecast_mw` | 预测时点可获得的目标时段系统负荷预测 |
| `mkt_load_forecast_age_h` | 当前采用的负荷预测信息距离预测截点的时间年龄 |
| `mkt_forecast_outage_rto_mw` | RTO 范围预测发电停运容量 |
| `mkt_forecast_outage_west_mw` | West 区域预测发电停运容量 |
| `mkt_forecast_outage_other_mw` | 其他区域预测发电停运容量 |
| `mkt_prev_day_eco_max_mw` | 前一日可获得的系统经济最大容量统计 |
| `mkt_prev_day_emerg_max_mw` | 前一日可获得的系统应急最大容量统计 |
| `mkt_prev_day_total_committed_mw` | 前一日系统总承诺容量 |

市场特征严格按照预测截点构造。当前采用：

```text
D-1 11:00 EPT
```

作为目标日 $D$ 的预测信息截点，不能使用截点之后发布的数据。

`M` 描述的是全市场共同环境，而不是某个主体的个体属性。

---

## 5. `U`：机组运行状态代理特征（12 维）

`U` 来源于历史 energy market offer 中可观测的机组运行参数。为避免使用目标时段真实报价附带的运行参数，全部采用严格历史 lag 值。

当前 12 维为：

| 特征 | 含义 |
|---|---|
| `unit_lag1_no_load_cost` | 上一历史状态的空载成本 |
| `unit_lag1_cold_start_cost` | 冷启动成本 |
| `unit_lag1_inter_start_cost` | 中间状态启动成本 |
| `unit_lag1_hot_start_cost` | 热启动成本 |
| `unit_lag1_max_daily_starts` | 最大日启动次数 |
| `unit_lag1_min_runtime` | 最小运行时间 |
| `unit_lag1_max_ecomax` | 历史最大 EcoMax |
| `unit_lag1_min_ecomax` | 历史最小 EcoMax |
| `unit_lag1_avg_ecomax` | 历史平均 EcoMax |
| `unit_lag1_max_ecomin` | 历史最大 EcoMin |
| `unit_lag1_min_ecomin` | 历史最小 EcoMin |
| `unit_lag1_avg_ecomin` | 历史平均 EcoMin |

`U` 的作用主要是表达主体当前可能受到的物理能力边界和运行成本约束。

在后续曲线参数预测中，`U` 特别可能对 `q_anchor_mw`、`q_span_mw` 等电量尺度参数产生直接影响。

---

## 6. `H`：主体直接历史报价状态（9 维）

`H` 与 `Z_base/Z_tr` 的区别是：`H` 不做高层画像抽象，而直接保留上一可用同槽位历史报价的状态。

当前模型使用 9 个 `participant_history` 输入：

| 特征 | 含义 |
|---|---|
| `hist_days_since_prev_same_slot` | 当前样本距上一可用同槽位历史报价的天数 |
| `hist_lag1_template_id` | 上一可用同槽位历史报价模板 |
| `hist_lag1_curve_mode` | 上一历史报价的 curve mode |
| `hist_lag1_effective_segment_count` | 上一历史报价有效段数 |
| `hist_lag1_breakpoint_count` | 上一历史报价 breakpoint 数量 |
| `hist_lag1_q_anchor_mw` | 上一历史报价的电量起点 |
| `hist_lag1_q_span_mw` | 上一历史报价的电量跨度 |
| `hist_lag1_p_anchor` | 上一历史报价的价格起点 |
| `hist_lag1_p_span` | 上一历史报价的价格跨度 |

数据集中另有：

```text
hist_prev_available_flag
```

用于表示是否存在可用上一历史样本，但当前分类实验把它作为 readiness/availability 标记排除，不计入上述 H=9。

其中最重要的历史变量是：

```text
hist_lag1_template_id
```

因为实际数据表现出非常强的模板持续性。

---

## 7. 当前各输入组的关系

当前完整特征结构可以理解为：

```text
Z_base
├─ LT：长期报价习惯
├─ ST：短期偏离状态
└─ Break：结构突变状态

Z_tr
├─ 最近切换频率
├─ 当前模板驻留时间
├─ 模板多样性/熵
├─ origin 条件切换概率
├─ destination 集中程度
└─ reversion 行为

M
├─ 系统负荷预测
├─ 预测信息新鲜度
├─ 系统发电停运
└─ 日前系统容量

U
├─ 启停成本
├─ 最小运行约束
└─ EcoMax / EcoMin 能力边界

H
├─ lag1 模板
├─ lag1 curve mode / 段结构
└─ lag1 电量/价格尺度参数
```

因此：

$$
Z_{base}
$$

主要解决“这个主体通常是什么策略”；

$$
Z_{tr}
$$

主要解决“这个主体当前是否处于策略切换状态”；

$$
M,U
$$

描述当前外部环境和物理运行条件；

$$
H
$$

提供最直接的上一历史报价状态。

---

## 8. 模板分类实验

### 8.1 数据划分

采用严格时间外切分：

```text
Train <= 2025-10-02
Validation = 2025-10-03 .. 2025-11-16
Test >= 2025-11-17
```

测试集不会参与模型选择和阈值选择。

### 8.2 直接 13 类分类

对以下模型进行了统一比较：

- Logistic Regression；
- Decision Tree；
- Random Forest；
- LightGBM。

同时比较不同输入：

```text
Z_base
Z_base + Z_tr
Z_base + M + U
Z_base + Z_tr + M + U
Z_base + H
Z_base + Z_tr + H
Z_base + M + U + H
Z_base + Z_tr + M + U + H
```

直接分类能够学习主体模板偏好，但无法超过模板历史持续性基线。

测试集上较好的直接分类结果之一为：

```text
LightGBM + Z_base + Z_tr + M + U + H
Accuracy = 0.9311
Macro-F1 = 0.8370
```

而 lag1 baseline 为：

```text
hist_lag1_template_id
Accuracy = 0.9517
Balanced Accuracy = 0.8933
Macro-F1 = 0.8934
```

因此最终不采用直接 13 类分类器作为主模板预测器。

---

## 9. 新增 switch 模型

由于模板高度持续，直接预测 13 类会被大量“不切换”样本主导，因此进一步将模板预测拆解为：

```text
Layer 1：是否发生模板切换？
Layer 2：若切换，切到哪个 destination？
```

定义：

$$
switch_t
=
\mathbb{I}(T_t \ne T_{t-1})
$$

2025 数据中的实际 switch 比例约为：

```text
Train: 4.63%
Validation: 3.54%
Test: 4.83%
```

这意味着模板预测首先是一个高度不平衡的“持续/切换”问题。

新增 `Z_tr` 后，switch detector 的识别能力明显增强，因此策略转移画像得到保留。

---

## 10. Conditional destination 实验

针对真实 switch 样本，进一步测试：

$$
P(T_t=j \mid T_{t-1}=i, switch_t=1, Z,M,U,H)
$$

比较：

1. `OriginPrior`：仅用训练期 origin→destination 转移矩阵；
2. `GlobalOriginMasked`：共享全局分类器 + origin 候选模板约束；
3. `ConditionalOriginModel`：每个 origin 单独训练 destination 分类器。

实验结果表明：

- origin→destination 转移矩阵本身不足以完成预测；
- 按 origin 分成 13 个独立分类器并没有改善时间外泛化；
- 共享的 `GlobalOriginMasked` 整体优于 per-origin 子模型；
- destination 模型存在明显时间漂移，Validation 明显好于 Test。

因此当前不采用 13 个独立 origin destination 模型。

---

## 11. 最终敲定的模板预测方案

### 11.1 主模板预测规则

当前正式主模型采用：

```text
hist_lag1_template_id
```

即：

$$
\hat T_{i,t}=T_{i,t-1}
$$

其严格时间外 Test 结果为：

| 指标 | Test |
|---|---:|
| Accuracy | **0.9517** |
| Balanced Accuracy | **0.8933** |
| Macro-F1 | **0.8934** |

这不是简单因为模型没有训练，而是经过 LR / DT / RF / LightGBM 直接分类、switch 分层模型和 conditional destination 模型完整对比后得到的最终结果。

历史模板持续性是当前数据中最稳定、最可泛化的模板预测信号。

### 11.2 辅助策略切换风险模型

保留一个独立的策略切换风险分类器：

```text
Model:
RandomForest

Input:
Z_base + Z_tr + H

Feature count:
26 + 26 + 9 = 61
```

最终 `03d_final_template_predictor.py` 中测试结果：

| 指标 | Test |
|---|---:|
| Switch PR-AUC | **0.4421** |
| Switch F1 | **0.4830** |
| Switch Recall | **0.5369** |
| Switch Precision | **0.4389** |

该模型不直接替代主模板，而输出：

$$
P(switch_t=1)
$$

作为主体当前“策略发生变化”的风险状态。

### 11.3 高置信度 hierarchical override

最终还测试了：

```text
默认：
    template = hist_lag1_template_id

当：
    P(switch) >= 0.95

则：
    使用 RF GlobalOriginMasked destination 结果覆盖 lag1
```

Destination 模型：

```text
RandomForest
+ Z_base
+ M
+ U
+ H_context
+ explicit origin
+ TRAIN-only origin→destination candidate mask
```

其中 `H_context` 去掉 `hist_lag1_template_id`，因为 origin 已显式作为条件输入。

最终 Test：

| 指标 | Lag1 | Hierarchical |
|---|---:|---:|
| Accuracy | 0.9517 | **0.9518** |
| Balanced Accuracy | 0.8933 | **0.8934** |
| Macro-F1 | 0.8934 | **0.8934** |

Test override rate 只有约 `0.0110%`，仅纠正约 70 个 switch 样本，虽然没有破坏 persistence 样本，但相对 lag1 的总体提升仅约 0.01 个百分点。

因此：

> 高置信度 hierarchical override 保留为实验结果和辅助机制，但不作为当前主模板预测器。

---

## 12. 当前最终结论

当前模板预测阶段的模型定义正式固定为：

```text
主模板预测
    hist_lag1_template_id

Test Accuracy
    95.17%

Test Macro-F1
    0.8934
```

同时保留：

```text
策略切换风险
    RandomForest
    Input = Z_base + Z_tr + H

Test PR-AUC
    0.4421

Test F1
    0.4830
```

因此当前完整建模逻辑不是“用一个复杂分类器直接覆盖所有模板”，而是：

$$
\boxed{
\text{稳定模板状态：依赖 lag1 persistence}
}
$$

$$
\boxed{
\text{策略变化状态：由 }Z_{tr}\text{ + RF 提供 switch risk}
}
$$

这一结构同时保留了高预测准确率、较好的可解释性和较低工程复杂度。

---

## 13. 下一阶段

模板预测部分完成后，进入“模板条件下的曲线参数预测”。

当前计划预测：

```text
q_anchor_mw
q_span_mw
p_anchor
p_span
```

已建立：

```text
scripts/bidprediction/04a_build_curve_parameter_dataset.py
```

后续流程为：

```text
04a_build_curve_parameter_dataset.py
    ↓
构建严格无泄漏的参数回归数据集
    ↓
04b_train_curve_parameter_models.py
    ↓
分别预测 q_anchor / q_span / p_anchor / p_span
    ↓
04c_evaluate_reconstructed_curve.py
    ↓
模板 + 参数恢复完整报价曲线
```

后续评估必须同时区分：

```text
Oracle-template：
真实模板 + 预测参数

Full-pipeline：
预测模板 + 预测参数
```

从而判断最终曲线误差究竟来自模板选择还是参数回归。

---

## 14. 当前阶段一句话总结

截至当前，第三阶段已形成完整且无泄漏的预测输入体系：

$$
\boxed{
X=[Z_{base}, Z_{tr}, M, U, H]
}
$$

其中 `Z_tr` 已验证能够显著增强模板切换风险识别；最终模板预测以历史模板持续性 `hist_lag1_template_id` 为主，测试准确率为 **95.17%**，Random Forest switch detector 作为辅助动态状态模型保留，下一步进入报价曲线参数预测。
