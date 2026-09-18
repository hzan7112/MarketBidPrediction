# BidTemplate 验证报告

**项目：** `MarketBidPrediction / scripts/bidtemplate`  
**数据：** PJM 2025 Energy Market Offers  
**验证范围：** `03_visualize_curve_templates.py`、`04_validate_template_representatives.py`、`06_build_profile_template_relation_dataset.py`、`07_validate_profile_template_relation.py`

本报告集中回答三个问题：当前 `12 shape + FLAT` 模板是否具有稳定、可解释的实际报价含义；KMeans centroid 是否因 block/sloped 混合而产生明显失真；第一阶段 `9 LT + 9 ST + 8 Break` 是否与模板选择存在足够强的结构关联。

---

## 1. 模板占比与典型形态验证（03）

最终模板库由 12 个非平价 shape 模板和 1 个 `FLAT` 模板组成，共覆盖 `5,119,107` 条有效历史报价。

| 模板 | 样本数 | 占全部有效报价 | 占 Shape |
|---|---:|---:|---:|
| T00 | 998,071 | 19.50% | 21.95% |
| T01 | 210,813 | 4.12% | 4.64% |
| T02 | 205,870 | 4.02% | 4.53% |
| T03 | 172,223 | 3.36% | 3.79% |
| T04 | 109,128 | 2.13% | 2.40% |
| T05 | 150,426 | 2.94% | 3.31% |
| T06 | 124,961 | 2.44% | 2.75% |
| T07 | 94,227 | 1.84% | 2.07% |
| T08 | 259,662 | 5.07% | 5.71% |
| T09 | 135,255 | 2.64% | 2.97% |
| T10 | 2,036,624 | 39.78% | 44.78% |
| T11 | 50,576 | 0.99% | 1.11% |
| FLAT | 571,271 | 11.16% | - |

分布呈明显长尾结构。T10 是最大的趋势簇，T00 次之，FLAT 独立占 11.16%。其余模板覆盖不同位置和不同程度的中后段抬升、尾部抬升、平滑增长和前段快速抬升等趋势。

03 的可视化表明，12 个模板之间并非简单的幅值差异，主要区别体现在价格抬升发生在归一化容量轴的不同位置以及抬升的集中程度。例如 T00 主要表现为极末端抬升，T10 接近均匀增长，T11 则表现为前段快速抬升后进入高位。

![模板占比与典型形态](../../data/processed/bidtemplate/2025/template_library/figures/template_gallery.png)

这里的曲线应理解为**趋势模板**。KMeans centroid 是同一类历史 shape 的均值中心，不能直接等同于任意一条原始 PJM 报价曲线。

---

## 2. Block / Sloped 真实结构验证（04）

对 4,547,836 条 shape 样本重新扫描后，整体报价模式为：

| 模式 | 样本数 | Shape 占比 |
|---|---:|---:|
| Block | 1,602,056 | 35.23% |
| Sloped | 2,945,780 | 64.77% |

不同趋势模板的 mode 构成差异明显：

| 模板 | Block | Sloped | 主要特征 |
|---|---:|---:|---|
| T00 | 85.88% | 14.12% | 以 block 为主，主要价格变化集中在末端 |
| T01 | 54.58% | 45.42% | 两种形式均较常见 |
| T03 | 21.49% | 78.51% | sloped 为主 |
| T05 | 2.33% | 97.67% | 几乎完全为 sloped |
| T06 | 69.15% | 30.85% | block 为主，但 centroid 较平滑 |
| T08 | 46.28% | 53.72% | 两种形式接近均衡 |
| T09 | 56.81% | 43.19% | block 略占多数 |
| T10 | 8.78% | 91.22% | 近线性趋势主要来源于真实 sloped 报价 |
| T11 | 9.00% | 91.00% | 以 sloped 为主 |

这一验证首先确认了 T10 的近线性现象并非纯粹由 KMeans 平均制造。T10 中 91.22% 的样本本身被 PJM 标记为 sloped，其真实代表样本也与 centroid 高度接近，因此“近线性增长”具有实际数据基础。需要注意，PJM sloped 报价仍是多个报价 knot 之间的分段线性连接，不表示原始报价仅由一根单直线构成。

同时，T06、T09 等模板说明 centroid 与原始表达结构不能混为一谈。一个以 block 为主的簇，其均值中心依然可能表现为平滑曲线。因此 04 最终采用每个模板同时展示：

```text
KMeans centroid
真实 block 代表样本
真实 sloped 代表样本
block/sloped 构成比例
```

![Centroid 与真实报价模式对照](../../data/processed/bidtemplate/2025/template_library/validation/centroid_vs_mode_representatives.png)

由此得到的方法结论是：**无需把 block/sloped 再拆成不同趋势模板。** 当前 `T00~T11` 继续用于描述报价的宏观归一化趋势，`curve_mode、segment_count、breakpoints` 作为后续结构参数单独保存。这样可以在保持模板数量紧凑的同时保留真实报价的分段形式。

---

## 3. 画像与模板的对齐数据验证（06）

06 将第二阶段 template assignment 按 `participant_id + local_date` 聚合，并与第一阶段正式的 `9 LT + 9 ST + 8 Break` 对齐。这里未使用 bidprofile 中任何辅助聚类标签。

构建结果为：

| 指标 | 数量 |
|---|---:|
| 主体数 | 652 |
| LT-ready 主体 | 652 |
| 主体日 | 216,103 |
| Profile-ready 主体日 | 174,948 |
| Profile-ready 覆盖率 | 80.96% |
| 主导模板占比中位数 | 1.000 |
| 归一化模板熵中位数 | ≈ 0 |

`dominant_template_share` 中位数为 1.000，且归一化熵中位数接近 0，说明至少一半主体日在日内具有非常稳定的模板选择。即同一主体当天多个时段虽然实际 MW 和价格尺度可能变化，但归一化报价趋势通常保持在同一个模板中。

这一结果与第一阶段 ST 的主体日时间尺度是匹配的，也支持后续使用“主体日画像 → 当日主要模板”作为轻量级模板选择问题。

06 最终得到两层关系数据：

```text
主体级：
9 LT → 全年 template usage share / dominant template

主体日级：
9 LT + 9 ST + 8 Break
→ 当日 template usage share / dominant template
```

因此 06 已验证第一阶段画像与第二阶段模板在主体和时间索引上能够稳定连接。

---

## 4. 画像—模板关联强度验证（07）

07 使用三类互补方法验证关联：单变量 Spearman、Break 条件分布变化，以及多变量分类诊断。

### LT 与长期模板偏好

9 个 LT 特征对模板使用比例的最大绝对 Spearman 相关如下：

| LT 特征 | 最强关联模板 | ρ |
|---|---|---:|
| `lt_flat_curve_rate` | FLAT | +0.734 |
| `lt_curve_bend_ratio` | T00 | +0.675 |
| `lt_tail_uplift_ratio` | T00 | +0.590 |
| `lt_shape_variability` | T02 | +0.571 |
| `lt_effective_segment_count` | FLAT | -0.562 |
| `lt_quantity_hhi` | T08 | -0.416 |
| `lt_bid_level` | T10 | +0.317 |
| `lt_adjustment_magnitude` | T10 | +0.308 |
| `lt_strategy_persistence` | T07 | -0.232 |

结果与画像定义具有较强一致性。例如长期平价报价比例直接对应 FLAT 偏好，尾部抬升与曲线弯折特征明显对应 T00，说明 LT 能够解释主体长期选择何种报价趋势。

### ST 与短期模板变化

ST 单变量最大相关整体弱于 LT：

| ST 特征 | 最强关联模板 | ρ |
|---|---|---:|
| `st_shape_shift` | T02 | +0.269 |
| `st_tail_uplift_ratio_z` | T05 | -0.088 |
| `st_curve_bend_ratio_z` | T01 | +0.060 |
| `st_quantity_hhi_z` | T05 | -0.046 |
| `st_flat_curve_rate_z` | T11 | +0.041 |
| `st_bid_level_z` | T05 | +0.037 |

单变量相关较弱并不意味着 ST 无效。后续多变量诊断显示 ST 单独仍能显著区分模板，说明短期状态更多表现为多变量组合关系，而非单一变量与单一模板之间的强线性或单调关系。

### Break 与模板分布变化

Break 特征本身发生频率较低，但部分 Break 出现时模板分布变化较大。使用 total variation distance 衡量 `break=0` 与 `break=1` 两组的模板分布差异：

| Break 特征 | Break rate | TVD |
|---|---:|---:|
| `break_flat_curve_rate` | 0.71% | 0.398 |
| `break_effective_segment_count` | 4.97% | 0.304 |
| `break_adjustment_magnitude` | 1.78% | 0.230 |
| `break_adjustment_bias` | 1.71% | 0.224 |
| `break_curve_bend_ratio` | 4.29% | 0.204 |
| `break_tail_uplift_ratio` | 4.41% | 0.190 |
| `break_bid_level` | 4.15% | 0.176 |
| `break_quantity_hhi` | 4.83% | 0.116 |

因此 Break 更适合作为**低频策略切换信号**，不适合作为独立的模板类别判别主体。

### 多变量关系诊断

07 使用 `HistGradientBoostingClassifier` 对当日 dominant template 做关系诊断。该模型只用于检验特征中是否包含足够的模板区分信息，并非最终报价预测模型。

时间留出采用前段日期训练、后段日期测试，测试起点为 `2025-11-02`：

| 特征组 | Accuracy | Balanced Acc. | Macro-F1 | Majority baseline |
|---|---:|---:|---:|---:|
| LT_9 | 0.835 | 0.631 | 0.643 | 0.387 |
| ST_9 | 0.665 | 0.323 | 0.346 | 0.387 |
| Break_8 | 0.393 | 0.085 | 0.060 | 0.387 |
| STBreak_17 | 0.663 | 0.319 | 0.343 | 0.387 |
| **Full_26** | **0.877** | **0.727** | **0.732** | **0.387** |

在完全未见过的 20% 主体上：

| 特征组 | Accuracy | Balanced Acc. | Macro-F1 | Majority baseline |
|---|---:|---:|---:|---:|
| LT_9 | 0.830 | 0.540 | 0.549 | 0.355 |
| ST_9 | 0.702 | 0.443 | 0.475 | 0.355 |
| Break_8 | 0.369 | 0.086 | 0.059 | 0.355 |
| STBreak_17 | 0.704 | 0.440 | 0.474 | 0.355 |
| **Full_26** | **0.887** | **0.676** | **0.692** | **0.355** |

相比 LT_9，Full_26 在时间留出上将 Macro-F1 从 `0.643` 提升到 `0.732`，在未见主体上从 `0.549` 提升到 `0.692`。这说明：

- LT 是模板选择的主要稳定信息来源；
- ST 提供了额外的时变区分信息；
- Break 单独预测能力很弱，但能指示低频策略分布切换；
- 三类特征组合后，对模板的区分能力明显强于任何单组特征。

![画像-模板关系诊断](../../data/processed/bidtemplate/2025/profile_template_relation/validation/relation_model_metrics.png)

---

## 5. 综合结论与限制

四组验证共同支持当前 `bidtemplate` 设计：

1. `T00~T11 + FLAT` 能以较少模板覆盖 511 万条有效 PJM 报价，并形成清晰的长期趋势类别；
2. `block/sloped` 在同一趋势模板中可以并存，KMeans centroid 应解释为趋势中心，真实分段形式由结构参数单独描述；
3. 主体日的模板选择具有较强稳定性，能够与主体日级 ST/Break 画像自然对齐；
4. LT 与长期模板偏好存在明显关联，Full_26 对模板具有很强的多变量区分能力，且在未见主体留出中仍保持较高指标；
5. 因此第一阶段主体画像与第二阶段趋势模板之间的连接是成立的，当前模板库可以作为后续 `bidprediction` 的预测目标之一。

同时必须保留一个方法学限制：当前 `lt_*` 特征由 2025 全年历史汇总得到，因此 07 的分类结果属于**回顾性关联验证**，不能直接解释成未来模板预测精度。LT 中包含了测试日期之后的信息，且部分 LT 特征本身来自报价曲线结构，与模板存在定义层面的联系。

因此本报告能够支持的结论是：

> `9 LT + 9 ST + 8 Break` 与 `T00~T11 + FLAT` 之间存在显著、稳定且具有可解释性的统计结构关联，足以支持后续预测模块将主体画像作为模板选择的重要输入。

真正的预测性能必须在 `bidprediction` 子项目中重新构造仅使用预测时点之前历史数据的滚动 LT/ST/Break，并结合当期市场特征进行无泄漏时间外推验证。
