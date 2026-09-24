# bidprediction 阶段性技术报告

> 更新：2026-09-24。已整合 04 实验台账、代码复核与 05 实际结果。当前正式基线为 **Raw Curve Persistence：VAL WAPE 6.0881%，TEST WAPE 8.1139%**；下一步验证“历史原始曲线 + 低维曲线变化量”。

项目使用已完成的 26 维主体画像（9 LT + 9 ST + 8 Break）、市场环境、机组状态和历史报价预测未来曲线。Stage 2 的 13 类模板保留用于策略解释；预测主线已从 **Template + θ** 转向 **连续曲线表示**。时间划分沿用 Train ≤ 2025-10-02、VAL 2025-10-03～11-16、TEST ≥ 2025-11-17；PCA 与标准化仅在 Train 拟合，模型选择使用 VAL，TEST 用于最终评价。以下曲线指统一数量网格上的 21 点表示。

## 04：Template + θ 的方法、实验与瓶颈

04 的思路是先预测模板状态，再预测少量连续参数，最后重构曲线。最新版本采用“未触发模板切换则沿用历史 θ，触发后预测目标模板与条件 θ”的分支结构。

| 环节 | 方法 |
|---|---|
| **04a：构造参数数据** | `04a` 合并预测特征、曲线标签与转移画像；`04a2` 提取模板调整参数及 5 段数量占比；`04a2b` 排除数量占位值；`04a3_build_stable_theta_continuous_dataset_fixed.py` 转为 Stable-θ，并构造同主体、同槽位的历史参数和变化特征。最终 θ 包含价格起点、slope、tail uplift、数量起点、数量跨度及 `q1…q5`。 |
| **04b：预测条件 θ** | `04b_train_conditional_theta_model_v3.py` 只用真实模板切换行训练；输入画像、市场、机组、历史及起始/目标模板，使用 9 个独立 ExtraTrees 回归器预测价格参数、数量起点、log 数量跨度和 4 个数量占比 ALR 参数。训练条件为真实目标模板，部署条件为预测目标模板。 |
| **04c：重构与校准** | `04c_calibrate_conditional_theta_pipeline_v3.py` 按数量占比变形模板横轴，再用三个价格锚点恢复价格；在真实 MW 轴上评价。VAL 按完整曲线 WAPE 选择切换阈值，TEST 固定该阈值。v3 最终选出 0.93。 |

下表汇总[04 实验总记录](MarketBidPrediction_实验与结论总记录_2026-09-24.md)中的主要尝试，并补入本地 v3 运行结果。除注明外，WAPE 均为 TEST 曲线误差；子集结果仅与同一子集的对照比较。

| 实验 | 关键结果 | 结论 / 状态 |
|---|---|---|
| 初始 13 类模板分类 | Lag-1 accuracy 96.46%；LT 特征 RF 90.08%，LR / DT 更低。 | 历史模板惯性强，转向切换检测与目标模板预测。 |
| 直接 21 点预测 | 早期 full-pipeline WAPE 约 25.82%。 | 已停止该版本的直接预测方案。 |
| Stable-θ Persistence / Ridge / ExtraTrees | Persistence WAPE **11.1957%**；Ridge、ExtraTrees 均更差。 | Persistence 成为 04 阶段基线。 |
| 全样本 θ residual + 全局 λ | VAL 最优 λ=0，非零修正使误差变大。 | 废弃。 |
| Binary gate residual | WAPE 11.195703% → 11.195650%，几乎没有改善。 | 退化为 Persistence，废弃。 |
| Continuous gate / Oracle α | 已设计，未正式运行。 | 复杂度与当前目标不符，停止。 |
| Residual + stacking | VAL 最优 w=0；TEST residual 11.3600%，stacked 仍为 11.1957%。 | 统一 residual 未提供稳定收益，停止调全局权重。 |
| θ 可预测性诊断 | 模板切换时参数 MAE 放大约 4～20 倍。 | 切换行变化较大，但不能据此认定未切换行无需更新。 |
| Switch-θ v1 | TEST 仅预测 70 条切换，recall 0.2279%。 | 局部收益无法覆盖整体；约 95% accuracy 不足以评价切换预测。 |
| 切换阈值诊断 | VAL / TEST ROC-AUC 0.9368 / 0.9469；原阈值 0.95 极保守。 | 排序能力仍在；阈值须结合下游曲线误差选择。 |
| 真切换训练 + WAPE 校准 v2 | VAL 选 0.93；TEST full WAPE 11.1957% → 11.1885%。 | 高置信度子集有收益，整体改善极小。 |
| 旧版误差归因 | TEST 预测目标 + 预测 θ 为 55.68%；换成真实 θ 后为 8.27%；双 Oracle 为 5.46%。 | θ 是主要误差源之一；旧 θ 仍以预测目标为条件，不能严格隔离目标模板错误。 |
| Conditional-θ v3 | 真切换行：Persistence **45.5060%**，真实目标模板 + 预测 θ **49.8314%**，预测目标模板 + 预测 θ **53.9584%**；full WAPE **11.1890%**。 | **当前方案不可行，已停止**；即使目标模板正确，θ 回归仍未超过 Persistence。 |

**04 的瓶颈同时存在于表示和预测。** 模板在归一化物理数量坐标上学习，重构时却按样本 `q1…q5` 再变形；Stable-θ 的“70%锚点”实际位于 `q1+q2+q3+0.5q4`，可能接近数量末端，将尾部跳价转成很大的 slope。代码复核中，即使用真实模板与真实 θ，WAPE 仍为 **5.4577%**；同一模板改用固定物理坐标的三个真实价格锚点后为 **1.7197%**。这是表示能力诊断，尚不是预测收益。

另一方面，模板不变仍可能发生价格整体平移、价差或数量尺度变化；TEST 未切换行贡献了 Persistence **87.87% 的总绝对价格误差**。v3 的价格参数预测又未超过历史值，迫使阈值保持很高，最终仅 274 / 633,913 行进入预测分支。由此停止当前 Template + θ 预测路线，保留模板的解释用途。详细证据见[重构诊断](../../results/bidprediction_code_audit_20260924/diagnosis.md)与[v3 指标](../../data/processed/bidprediction/2025/conditional_theta_operational_calibration_v3/test_reconstruction_metrics.csv)。

## 05：连续曲线 PCA、预测结果与下一步

05 将每条报价直接表示为 `V=[P₀,…,P₂₀,q_anchor,log(q_span)]`，共 23 维，经标准化和 PCA 压缩为连续 latent 状态 z；价格水平、形状和数量范围都在同一表示中，不依赖模板类别或 θ。当前 05a～05d 已完成的是 **绝对 latent 状态预测**。

| 脚本 | 方法与作用 |
|---|---|
| [05a_build_curve_latent_representation.py](05a_build_curve_latent_representation.py) | 在 498,650 条训练样本上拟合 StandardScaler 和 PCA，按训练累计解释方差 ≥99.5% 选出 **8 维 latent**，实际解释方差 99.6223%。零价差曲线直接由价格起点恢复，允许其归一化 shape 缺失。 |
| [05b_build_latent_forecasting_dataset.py](05b_build_latent_forecasting_dataset.py) | 构造同主体、同槽位的 latent lag1/2/7、7/30 期统计及变化特征；84 个 context 特征 + 80 个历史 latent 特征，共 **164 维输入**。上一历史原始曲线另存，仅用于 Persistence 对照。 |
| [05c_train_latent_regressors.py](05c_train_latent_regressors.py) | 比较 latent Persistence、Ridge、Spline-GAM（样条加性项 + Ridge 正则）与多输出 Random Forest，目标均为未来绝对 z。共享训练样本 293,788 行，GAM 使用 150,000 行。 |
| [05d_reconstruct_latent_forecasts.py](05d_reconstruct_latent_forecasts.py) | 将预测 z 反变换为价格和数量范围，在真实 MW 轴上插值计算曲线 WAPE；同时评价原始曲线 Persistence、latent Persistence 与真实 latent 解码的 Oracle。VAL 选择 latent 候选，TEST 固定选择。 |

下表统一采用 05d 口径：VAL 637,524 行、TEST 633,753 行。`latent_oracle` 使用当期真实 latent，只衡量表示误差；“—”表示未列入 05d 最终 TEST 曲线评估。

| 方法 | VAL WAPE | TEST WAPE | 判断 |
|---|---:|---:|---|
| **Raw Curve Persistence** | **6.0881%** | **8.1139%** | 当前最强、正式预测基线。 |
| Latent Persistence | 7.6223% | 9.5548% | PCA 压缩再解码会损失部分历史曲线信息。 |
| Ridge：绝对 latent 回归 | 10.9980% | — | 未超过 Persistence，停止当前预测方式。 |
| Spline-GAM：绝对 latent 回归 | 12.3729% | — | 未超过 Persistence，停止当前预测方式。 |
| Random Forest：绝对 latent 回归 | 14.8184% | — | 未超过 Persistence，停止当前预测方式。 |
| Latent Oracle（真实状态重构） | 2.2318% | 2.1936% | 8 维表示保留了大部分曲线信息；不代表预测精度。 |

05d 的 `selected_model=latent_persistence` 指它在 latent 候选中最好；原始曲线 Persistence 单列为对照，实际表现更好。05a 单独的表示评价在均匀数量网格上进行，数值与上表的物理 MW 轴评价不同。结果文件见[VAL 指标](../../data/processed/bidprediction/2025/latent_curve_forecast_evaluation/validation_curve_metrics.csv)与[TEST 指标](../../data/processed/bidprediction/2025/latent_curve_forecast_evaluation/test_curve_metrics.csv)。

**05 已验证连续低维表示可用，但当前绝对 latent 回归没有预测增益。** 三个回归器在全部 8 个 latent 维度上的 VAL MAE 均差于 Persistence；结果与报价的强惯性、重新估计绝对状态时产生不必要偏移相符。GAM 的平均 latent RMSE 略低（0.3501 vs 0.3606），仅提示可能改善少数大误差，仍需分组验证。原始曲线 Persistence 的 8.11% 也优于 04 的约 11.20%，支持旧参数化存在信息损失；两阶段有效样本略有不同，不把差值全部视为单项消融收益。

**下一步只验证“上一历史原始曲线 + 低维曲线变化量”，尚未得到结果。** 对 `ΔVₜ=Vₜ−V_hist` 单独拟合 PCA，再用 Ridge、Spline-GAM、RF 预测变化系数，解码后得到 `V̂ₜ=V_hist+ΔV̂ₜ`。这样保留历史原始曲线，仅预测连续曲线空间的调整量；与旧 θ residual 的区别是预测坐标不再受模板边界和 q-share 重参数化影响，但是否有效仍须实验验证。零变化基线应显式设置 `ΔV̂=0`；中心化 PCA 的 latent 零向量通常解码为平均变化，不能直接当作零修正。后续必须在同口径 VAL 选定方案，再用 TEST 检验是否稳定优于 **8.1139%**，不再以 04 的约 11.20% 作为主要过关线。
