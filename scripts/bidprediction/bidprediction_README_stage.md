# bidprediction 阶段性技术报告

> 更新：2026-09-26。已整合 04～10 实验结果。最终路线冻结为 **83 个 feature-only 特征 → RF → 8D absolute latent → 完整报价曲线**，使用全部有效曲线，不做模板或 Family 路由。TEST WAPE **28.85%**，显著优于同输入约束下的均值、Ridge、GAM，但明显差于使用完整历史曲线的 Persistence **8.11%**。技术路线定稿，精度边界保留；停止模型、PCA 维数及特殊路由的继续筛选。

项目围绕主体画像、策略转移、市场环境和机组状态预测未来报价曲线；05～07 另使用历史曲线信息，08 移除直接历史曲线输入。预测主线已从 **Template + θ** 转向 **连续曲线表示**；Stage 2 的 13 类模板保留用于预测曲线的策略解释，当前不再作为回归器的路由标签。时间划分沿用 Train ≤ 2025-10-02、VAL 2025-10-03～11-16、TEST ≥ 2025-11-17；PCA 与标准化仅在 Train 拟合，模型选择使用 VAL，TEST 用于最终评价。以下曲线指统一数量网格上的 21 点表示。

## 04：Template + θ 的方法、实验与瓶颈

04 的思路是先预测模板状态，再预测少量连续参数，最后重构曲线。最新版本采用“未触发模板切换则沿用历史 θ，触发后预测目标模板与条件 θ”的分支结构。

| 环节 | 方法 |
|---|---|
| **04a：构造参数数据** | `04a` 合并预测特征、曲线标签与转移画像；`04a2` 提取模板调整参数及 5 段数量占比；`04a2b` 排除数量占位值；`04a3_build_stable_theta_continuous_dataset_fixed.py` 转为 Stable-θ，并构造同主体、同槽位的历史参数和变化特征。最终 θ 包含价格起点、slope、tail uplift、数量起点、数量跨度及 `q1…q5`。 |
| **04b：预测条件 θ** | `04b_train_conditional_theta_model_v3.py` 只用真实模板切换行训练；输入画像、市场、机组、历史及起始/目标模板，使用 9 个独立 ExtraTrees 回归器预测价格参数、数量起点、log 数量跨度和 4 个数量占比 ALR 参数。训练条件为真实目标模板，部署条件为预测目标模板。 |
| **04c：重构与校准** | `04c_calibrate_conditional_theta_pipeline_v3.py` 按数量占比变形模板横轴，再用三个价格锚点恢复价格；在真实 MW 轴上评价。VAL 按完整曲线 WAPE 选择切换阈值，TEST 固定该阈值。v3 最终选出 0.93。 |

下表汇总 04 阶段的主要尝试，并补入本地 v3 运行结果。除注明外，WAPE 均为 TEST 曲线误差；子集结果仅与同一子集的对照比较。

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

## 05：连续曲线 PCA 与绝对状态预测

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

05 后续提出的“上一历史原始曲线 + 低维曲线变化量”已在 06a～06c 实施；下节记录实际结果。

## 06：曲线变化量 PCA 与增量预测

06 保留同主体、同槽位的上一条原始曲线 `V_hist`，以 `ΔVₜ=Vₜ−V_hist` 为目标单独拟合 PCA；回归器预测变化系数 `Δz`，再解码为 `V̂ₜ=V_hist+ΔV̂ₜ`。`zero_change` 明确设定 `ΔV̂=0`，因此与 Raw Curve Persistence 完全一致；中心化 PCA 的 latent 零向量并不等于零变化。训练、VAL、TEST 的时间边界与 05 相同，PCA 仅在 Train 拟合，曲线 WAPE 仅在 VAL 选择模型，TEST 固定该选择。

| 脚本 | 方法与作用 |
|---|---|
| [06a_build_curve_change_latent_dataset.py](06a_build_curve_change_latent_dataset.py) | 用 486,701 条训练样本对 23 维曲线变化向量拟合 StandardScaler + PCA；按训练累计解释方差 ≥99.5% 选择 **16 维**，实际为 **99.5385%**。在原有 164 个输入特征上增加 160 个历史变化特征，共 **324 维**。 |
| [06b_train_curve_change_regressors.py](06b_train_curve_change_regressors.py) | 比较显式零变化、Ridge、Spline-GAM 和 Random Forest；回归器预测未来 `Δz`，共享训练样本 293,788 行，GAM 使用 150,000 行。输出 VAL/TEST 预测，不在 latent 空间直接选最终模型。 |
| [06c_reconstruct_curve_change_forecasts.py](06c_reconstruct_curve_change_forecasts.py) | 反变换预测 `Δz`，加回历史原始曲线，在真实 MW 轴上评价完整曲线；VAL 以价格 WAPE 选择模型，再仅对冻结的胜出模型做 TEST 评价。真实 `Δz` 的 oracle 仅衡量表示损失。 |
| [06d_diagnose_curve_change_predictability.py](06d_diagnose_curve_change_predictability.py) | 使用已有预测，按 TRAIN 真实变化幅度分位点固定区间，诊断稳定样本与大调整样本的修正收益，同时检查修正幅度、方向和时间分布；不重训或重新选择模型。 |

06c 与 05d 使用相同的有效样本：VAL 637,524 行、TEST 633,753 行。“—”表示该回归器未被 VAL 选中，因此没有最终 TEST 曲线评价。

| 方法 | VAL 曲线 WAPE | TEST 曲线 WAPE | 判断 |
|---|---:|---:|---|
| **zero_change / Raw Curve Persistence** | **6.0881%** | **8.1139%** | VAL 胜出；06 的最终 TEST 结果与原始曲线基线相同。 |
| Ridge：预测 `Δz` | 10.7015% | — | 大幅差于零变化。 |
| Spline-GAM：预测 `Δz` | 10.5796% | — | 大幅差于零变化。 |
| Random Forest：预测 `Δz` | 8.4764% | — | 回归器中最好，仍差于零变化。 |
| `delta_latent_oracle`：真实 `Δz` 重构 | 0.0790% | **0.0797%** | 16 维变化表示的误差下限；不是可部署预测。 |

**表示层已基本排除为当前瓶颈。** 16 维 `ΔCurve PCA` 在 TEST 上以真实变化系数重构，曲线 WAPE 仅 **0.0797%**，远低于 8.1139% 的零变化预测误差。因此这次失败主要发生在变化系数的预测，而不是变化量压缩或解码。详见[06a 维度指标](../../data/processed/bidprediction/2025/curve_change_latent_dataset/delta_latent_dimension_metrics.csv)与[06c 曲线指标](../../data/processed/bidprediction/2025/curve_change_forecast_evaluation/validation_curve_metrics.csv)。

**全样本增量回归尚无收益。** Ridge、GAM、RF 的 VAL 曲线 WAPE 均差于 `zero_change`，故按预定选择规则没有启用学习修正。GAM 的变化系数平均 RMSE 在 VAL 为 **1.1749**，优于零变化的 **1.2337**；TEST 诊断为 **1.5553** 对 **1.6035**，但其 MAE 更高，VAL 曲线 WAPE 也更差。详见[06b 变化系数指标](../../data/processed/bidprediction/2025/curve_change_regression_models/val_delta_latent_metrics.csv)与[06c TEST 指标](../../data/processed/bidprediction/2025/curve_change_forecast_evaluation/test_curve_metrics.csv)。

06d 的[分组诊断](../../data/processed/bidprediction/2025/curve_change_predictability_diagnostics/model_benefit_by_change_bin.csv)进一步发现：VAL / TEST 的 Q00–Q95 各区间中，GAM、RF 均差于零变化；Q95–Q99、Q99–Q100 则出现收益。这里的区间由 TRAIN 分位点固定，不等于 VAL / TEST 各自恰好前 95% 或后 5%。该现象支持“多数稳定样本被不必要修正、大调整样本可能有可预测信息”，但真实变化幅度只用于事后诊断，预测时仍需用历史可得信息识别状态。07 随后检验报价族混合是否是这一问题的主要原因。

## 07：Macro-B 同类报价内的变化量预测

07 将常规阶梯 / 温和递增报价定义为 **Macro-B 行为族**，在该子集内重做 06 的变化量 PCA 与回归，检验减少报价族异质性后模型能否超过 Persistence。筛选要求当期和上一期曲线均为 Macro-B，因此这是使用真实当期曲线的 **Oracle 子集诊断**，并非已经可部署的报价族识别方案。

| 脚本 | 方法与作用 |
|---|---|
| [07a_filter_macro_b_curves.py](07a_filter_macro_b_curves.py) | 筛选正价格、非平价、近似单调且至少有两处显著上升的曲线；最大单步涨幅占比、粗糙度和尾部涨幅占比的上限仅由 TRAIN 确定。保留前后两期均满足条件的 Train 1,327,125 行、VAL 293,108 行、TEST 289,495 行。 |
| [07b_build_macro_b_curve_change_latent.py](07b_build_macro_b_curve_change_latent.py) | 在 468,057 条训练样本上重拟合 ΔCurve PCA，选择 **8 维**，累计解释方差 **99.5568%**；164 个原有输入 + 80 个 Macro-B 历史变化特征，共 **244 维**。 |
| [07c_train_macro_b_curve_change_regressors.py](07c_train_macro_b_curve_change_regressors.py) | 比较 zero-change、Ridge、Spline-GAM 和 RF，预测该族内的未来变化系数；共享训练样本 288,820 行，GAM 使用 150,000 行。 |
| [07d_evaluate_macro_b_curve_change_forecasts.py](07d_evaluate_macro_b_curve_change_forecasts.py) | 将预测变化加回上一原始曲线，在真实 MW 轴评价；VAL 选择结果仍是 **zero_change**。另报告全部 TEST 模型用于诊断，不据 TEST 重新选模型。 |

| 方法 | VAL 曲线 WAPE | TEST 曲线 WAPE | TEST latent 平均 RMSE |
|---|---:|---:|---:|
| **zero_change（近似 Persistence）** | **4.6002%** | **6.4155%** | 1.8992 |
| Ridge | 7.4380% | 10.0887% | 1.8706 |
| Spline-GAM | 6.7569% | 8.0736% | **1.8401** |
| Random Forest | 7.2830% | 9.9640% | 1.9976 |
| `delta_latent_oracle`：真实变化重构 | 0.1657% | **0.1493%** | — |

07 的 `zero_change` 将零变化编码后经截断 PCA 解码，存在微小投影残差，并非严格复制历史曲线。08 在同一子集直接复制历史曲线的参考 WAPE 为 **4.5992% / 6.4145%**；上表保留 07 原始运行结果。

**报价族异质性影响误差水平，但不足以解释回归失败。** 筛选后 Persistence 的 TEST WAPE 从全样本 8.11% 降至子集 6.42%，表明这组保留样本更适合历史延续预测；这是评价样本改变，不能记为模型带来的提升。三种回归器在族内仍均差于 Persistence，而真实变化系数解码的误差仅 0.1493%，已基本排除 PCA 表示不足和“仅仅因为不同报价族混合”的解释。瓶颈仍是现有特征到未来变化量的稳定映射。

**latent 误差改善没有转化为曲线收益。** Ridge、GAM 的 TEST latent RMSE 低于 zero-change，最终曲线 WAPE 却更高，说明优化 Δlatent 的平方误差不能保证改善曲线 WAPE。与此同时，Persistence 的 VAL curve MAE 的 P50 / P90 仅 **0.77 / 5.80**，TEST 为 **1.07 / 13.51**，大量样本沿用历史已足够准确。当前优先检查修正适用条件，继续调 PCA 维数、RF 深度或 Ridge α 尚无充分依据。数据见[07d VAL 指标](../../data/processed/bidprediction/2025/macro_b_curve_change_evaluation/validation_curve_metrics.csv)、[07d TEST 指标](../../data/processed/bidprediction/2025/macro_b_curve_change_evaluation/test_curve_metrics.csv)及[07c latent 指标](../../data/processed/bidprediction/2025/macro_b_curve_change_regression_models/test_latent_metrics.csv)。

## 08：无直接历史曲线输入的绝对报价预测与分组诊断

08 回到 **Profile + Transition + Market + Unit + Calendar → 绝对曲线 latent → 报价曲线**。核心结论是：**不直接输入历史报价曲线的路线具有明确预测信息，但当前将整个 Macro-B 作为统一回归问题，精度与跨时段泛化仍不足。** 08a～08d 验证预测能力，08e 进一步检验现有模板能否提供有效分组。

模型输入不含历史原始曲线、历史 latent、Δlatent 或 θ / 模板惯性字段；画像和转移统计仍包含历史行为信息，因此不等于完全无历史信息。样本沿用 07 的 Macro-B Oracle 筛选子集，结论目前仅适用于该子集。

| 脚本 | 方法与作用 |
|---|---|
| [08a_build_macro_b_absolute_curve_representation.py](08a_build_macro_b_absolute_curve_representation.py) | 对 `V=[P₀,…,P₂₀,q_anchor,log(q_span)]` 拟合 StandardScaler + PCA；使用 468,057 条训练样本，选出 **5 维绝对 latent**，解释方差 **99.5944%**。 |
| [08b_build_macro_b_feature_only_dataset.py](08b_build_macro_b_feature_only_dataset.py) | 移除直接历史报价字段并加入日历周期特征，最终 **83 维输入**。历史原始曲线仅另存为评价参考，不进入回归器。 |
| [08c_train_macro_b_feature_only_regressors.py](08c_train_macro_b_feature_only_regressors.py) | 比较 `train_mean`、Ridge、Spline-GAM 和 RF；均值基线对所有样本输出训练抽样的平均 latent。共享训练样本 288,820 行，GAM 使用 150,000 行。 |
| [08d_evaluate_macro_b_feature_only_forecasts.py](08d_evaluate_macro_b_feature_only_forecasts.py) | 解码后在真实 MW 轴计算曲线 WAPE；VAL 选中 **RF**，TEST 固定该选择。Persistence 使用额外的完整历史曲线，仅作跨任务参考，不参与本任务选模。 |
| [08e_diagnose_true_template_conditioning.py](08e_diagnose_true_template_conditioning.py) | 用真实模板对已有预测分组，比较全局均值与 TRAIN 模板均值，并计算类内 absolute latent 离散度；不重训回归器。 |

08d 评价样本为 VAL 293,108 行、TEST 289,495 行；Oracle 使用当期真实 latent，仅衡量表示误差。

| 方法 | VAL 曲线 WAPE | TEST 曲线 WAPE | VAL / TEST latent RMSE |
|---|---:|---:|---:|
| `train_mean`：无特征均值基线 | 80.35% | **75.22%** | 2.563 / 2.743 |
| Ridge | 35.13% | 40.70% | 1.187 / 1.445 |
| Spline-GAM | 34.85% | 34.36% | 1.096 / 1.259 |
| **Random Forest（VAL 选中）** | **17.74%** | **26.85%** | **0.717 / 1.158** |
| 5 维绝对 latent Oracle | 2.04% | **1.50%** | — |
| Raw Curve Persistence（额外历史信息参考） | 4.60% | 6.41% | — |

**特征本身具有较强预测信息，原研究方向应继续保留。** TEST 上，均值基线、Ridge、GAM、RF 的 WAPE 依次为 **75.22%、40.70%、34.36%、26.85%**；RF 相对无特征基线改善 **64.3%**。这为“画像 + 转移 + 市场 + 机组 + 日历 → 报价曲线”的预测关系提供了实验支持。26.85% 尚不足以满足精度目标，但不能据此否定不直接输入历史曲线的路线。

**PCA 不是当前主要瓶颈，问题集中在 X → Z_curve。** 5 维真实 latent 解码的 TEST Oracle WAPE 仅 **1.50%**，远低于 RF 的 **26.85%**，说明表示已保留大部分曲线信息。08a 已验证更高维可降低表示误差，后续可比较 5 / 8 / 10 维的预测收益，但现有证据不支持将主体误差归因于维数不足。结果见[08d VAL 指标](../../data/processed/bidprediction/2025/macro_b_feature_only_evaluation/validation_curve_metrics.csv)、[TEST 指标](../../data/processed/bidprediction/2025/macro_b_feature_only_evaluation/test_curve_metrics.csv)与[08a 维度对照](../../data/processed/bidprediction/2025/macro_b_absolute_curve_representation/representation_metrics.csv)。

**Macro-B 内的全局回归仍不够准确，值得检验更有效的分组。** RF 的 WAPE 从 VAL **17.74%** 升至 TEST **26.85%**，latent RMSE 从 **0.717** 升至 **1.158**，跨时段表现明显恶化。条件分布 `P(Z_curve | X)` 内仍有多个子机制是值得验证的解释，支持尝试“先判断报价类型，再按类回归”；但仅凭这些指标还不能排除时间分布变化或模型拟合不足。08e 随后检验了现有 13 个模板，修正了这个设想中的分类依据。

### 08e：现有 shape 模板不足以划分绝对报价预测问题

08e 仅在 TRAIN 计算各模板的平均 absolute latent；VAL / TEST 假设已知真实模板，使用对应均值重构曲线，形成 `oracle_template_mean`，模板映射覆盖率均为 100%。离散度在 TRAIN 计算，为各模板内 latent 相对类均值的 RMSE 与全局 RMSE 之比，再按训练样本数加权；它不是方差解释率。

| 诊断 | 关键结果 | 结论 |
|---|---|---|
| 加权类内 / 全局 latent 离散度 | **0.9165**，仅降低约 **8.35%** | 知道真实模板后，绝对曲线目标仍然分散。 |
| TEST 均值基线 | 全局均值 **75.22%** → 真实模板均值 **72.78%**，相对改善仅 **3.24%**；全局 RF 为 **26.85%** | 模板本身对绝对报价主要差异的解释力有限。 |
| 最大模板 T10 | 占 TEST **74.61%**；模板均值 / RF / Persistence WAPE 为 **66.26% / 27.14% / 6.17%**；TRAIN 离散度比 **1.027** | 最大组内部没有比全局更集中，是直接按模板拆分的主要障碍。 |
| T06 | TRAIN 离散度比 **1.559** | 该组绝对 latent 比全局更分散。 |
| 部分较小模板 | T04 / T05 / T03 / T07 / T01 / T08 离散度比分别为 **0.267 / 0.252 / 0.281 / 0.388 / 0.417 / 0.464** | 局部分组确有聚合作用，但覆盖有限，不能据此推断整体回归收益。 |

**当前停止“直接按现有 13 个模板各训练一套回归器”的方案。** 现有模板主要聚类归一化 `shape_v00～20`，描述曲线弯曲和抬升位置；08 的目标还包含绝对价格水平、价格跨度及数量位置 / 跨度。同样的归一化形状可对应完全不同的绝对报价，因此 shape 相似不足以保证回归目标同质。08e 不支持直接拆分模型，但尚未实测分模板回归，不能将其写成“分模型已经失败”。证据见[整体均值对照](../../data/processed/bidprediction/2025/macro_b_true_template_diagnostics/overall_oracle_template_metrics.csv)、[模板诊断汇总](../../data/processed/bidprediction/2025/macro_b_true_template_diagnostics/template_diagnostic_summary.csv)与[TRAIN 离散度](../../data/processed/bidprediction/2025/macro_b_true_template_diagnostics/template_latent_dispersion.csv)。

08e 因此提出以 absolute latent 构建 **Prediction Family**，让分组同时反映价格水平、价差、数量尺度和形状，再检验“特征预测 Family + 类内回归”。该方向已在 09a～09e 实施，实际结果如下。

## 09：Prediction Family 有效，硬路由抵消了类内回归收益

09 沿用 08 的 **83 个 feature-only 输入、5 维 absolute latent 和 Macro-B 样本**，将“分族是否有效”“Family 能否预测”“完整流程是否改善”分开验证。VAL / TEST 仍为 293,108 / 289,495 行；真实 Family 由当期真实 latent 分配，仅用于训练标签与 Oracle 诊断。完整硬路由使用特征预测 Family，但评价范围仍是 Macro-B Oracle 筛选子集。

| 脚本 | 方法与结果 |
|---|---|
| [09a_diagnose_absolute_prediction_families.py](09a_diagnose_absolute_prediction_families.py) | 在 468,057 条 TRAIN 样本的 absolute latent 上拟合 MiniBatchKMeans，比较 **K=2～8**；输出族内离散度、族规模和 Oracle 族均值误差，不自动选择 K。后续采用 **K=5**：TEST Oracle 族均值 WAPE **24.79%**，相对全局 TRAIN 离散度比 **0.5057**，各族占比约 **8.69%～27.80%**。 |
| [09b_train_prediction_family_classifier.py](09b_train_prediction_family_classifier.py) | 用 83 个特征预测 K=5 Family，比较 LR、DT、RF；共享训练抽样 288,820 行，按 VAL Macro-F1 选择 **RF**。 |
| [09c_train_oracle_family_regressors.py](09c_train_oracle_family_regressors.py) | 按真实 Family 分别训练 Ridge、GAM、RF，并比较 family mean；按各族 VAL 曲线 WAPE 选择专家，**F00～F04 全部选中 RF**。TEST 各族 RF 也均优于其余候选。 |
| [09d_evaluate_full_family_routing_pipeline.py](09d_evaluate_full_family_routing_pipeline.py) | 固定分类器与专家，比较 Global RF、真实 Family 路由、预测 Family 硬路由，并按路由是否正确及具体错分方向统计曲线误差。 |
| [09e_summarize_prediction_family_experiment.py](09e_summarize_prediction_family_experiment.py) | 汇总分族收益、专家收益、路由损失和最终净收益；将当前主要瓶颈定位到 **Family 预测及错误专家选择**。 |

**09 阶段的模型选择结论：分类采用 RF，DT 仅作轻量基线；回归采用 RF。** 08 全局 RF 已优于 GAM / Ridge，09c 五个族内的 VAL 与 TEST 对照再次支持 RF。分类中 RF 与 DT 的 TEST 表现接近，RF 按 VAL 规则胜出。10 最终保留全局 RF 回归，Family 分类器不进入最终模型。

| Family 分类器 | VAL Macro-F1 | TEST Accuracy | TEST Macro-F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.7992 | 0.5424 | 0.5267 |
| Decision Tree | 0.8882 | 0.7695 | 0.7796 |
| **Random Forest** | **0.9060** | **0.7708** | **0.7807** |

| 路由与曲线指标 | VAL | TEST |
|---|---:|---:|
| Global RF WAPE | 17.7358% | **26.8493%** |
| Oracle Family + Family RF WAPE | **10.3703%** | **17.0198%** |
| Predicted Family + Family RF WAPE（硬路由） | 15.0995% | **26.8266%** |
| Oracle 专家相对 Global RF 的收益 | 7.3654 个百分点 | **9.8294 个百分点** |
| 预测路由相对 Oracle 路由的损失 | 4.7292 个百分点 | **9.8068 个百分点** |
| 硬路由相对 Global RF 的净收益 | 2.6362 个百分点 | **0.0227 个百分点** |
| 路由正确子集 WAPE | 9.9602% | **15.9644%** |
| 路由错误子集 WAPE | 67.7764% | **62.9417%** |

**分族 + 类内 RF 有明显 Oracle 收益，硬路由几乎将其全部抵消。** 真实 Family 已知时，TEST 从 26.85% 降至 17.02%；换成预测 Family 后，9.81 个百分点的路由损失几乎吃掉 9.83 个百分点的专家收益。硬路由只比 Global RF 低 0.0227 个百分点，不能据此宣称取得稳定提升。正确与错误路由子集的 WAPE 为 15.96% / 62.94%，说明选错专家的代价很高；两个子集的 WAPE 不能直接按样本比例平均。这是 09 阶段提出优先降低路由损失的依据，最终是否保留该路线由 10 定稿结论决定。证据见[09b 分类指标](../../data/processed/bidprediction/2025/prediction_family_classifier/classifier_metrics.csv)、[09c 类内 TEST 指标](../../data/processed/bidprediction/2025/oracle_family_regressors/test_oracle_family_metrics.csv)与[09e 误差分解](../../data/processed/bidprediction/2025/prediction_family_experiment_summary/experiment_decomposition.csv)。

**09 同时暴露了时间泛化问题。** RF 分类 Macro-F1 从 VAL 0.906 降至 TEST 0.781，Oracle 专家 WAPE 也从 10.37% 升至 17.02%，提示分类与类内回归均有跨时段退化，尚需诊断是否由时间漂移引起。因此不能把全部预测误差归于路由；能明确归因的是，预测路由损失抵消了大部分分族收益。

09 曾提出通过软路由、置信度回退及 Family 稳定性诊断降低路由损失。**10 定稿后，Prediction Family 作为探索实验保留，不再推进或重跑 09f 等特殊路由，也不再将其作为最终框架。**

## 10：全量 feature-only RF 定稿与能力边界

**10 定稿的是技术路线，28.85% 不代表已达到高精度报价预测。** 最终输入为主体策略画像、策略转移状态、市场环境、机组状态及时段信息；历史报价用于形成画像和转移统计，完整历史曲线不直接进入模型。“全量”指现有 `latent_forecasting_dataset` 中全部有效曲线，未做 Macro-B、模板或 Family 筛选：Train **2,867,152** 行、VAL **638,052** 行、TEST **633,985** 行。

| 脚本 | 最终方法 |
|---|---|
| [10a_build_final_feature_only_dataset.py](10a_build_final_feature_only_dataset.py) | 从 84 个基础特征中移除历史间隔 `hist_days_since_prev_same_slot`，冻结 **83 个输入**。禁止直接历史曲线、历史 latent、ΔCurve、θ 和目标曲线字段；上一原始曲线只存为评价参考。 |
| [10b_build_final_absolute_curve_latent.py](10b_build_final_absolute_curve_latent.py) | 对 23 维绝对曲线向量在 TRAIN 拟合 StandardScaler + PCA，拟合样本 498,650 行；**固定 8 维**，解释方差 **99.6242%**，不依据 VAL / TEST 重选维数。 |
| [10c_train_final_curve_regressor.py](10c_train_final_curve_regressor.py) | **RF 为预先冻结的主模型**，均值、Ridge、GAM 为对照；共享训练抽样 300,000 行，GAM 使用 150,000 行，不用 TEST 重新选模。 |
| [10d_evaluate_final_bid_prediction.py](10d_evaluate_final_bid_prediction.py) | 在真实 MW 轴评价价格轨迹，补充数量边界、五个等数量段中点、归一化 shape 与模板一致性；Oracle 衡量表示损失，Persistence 仅作额外历史信息参考。 |

| 方法 | VAL 曲线 WAPE | TEST 曲线 WAPE | 定位 |
|---|---:|---:|---|
| Train mean | 104.57% | 95.23% | 无特征均值基线。 |
| Ridge | 50.73% | 66.70% | 线性对照。 |
| Spline-GAM | 47.38% | 45.92% | 加性非线性对照。 |
| **Random Forest** | **21.16%** | **28.85%** | 最终 feature-only 模型。 |
| 8D representation Oracle | 2.23% | **2.19%** | 当期真实 latent 解码，仅衡量表示误差。 |
| Raw Curve Persistence | 6.09% | **8.11%** | 沿用同主体、同槽位上一历史完整曲线，信息条件更强。 |

RF 的 VAL / TEST 有效曲线覆盖率均为 **100%**；Persistence 的 TEST 覆盖率为 **99.9634%**（633,753 行），两者评价样本略有差异。结果见[10d VAL 指标](../../data/processed/bidprediction/2025/final_bid_prediction_evaluation/val_final_metrics.csv)、[TEST 指标](../../data/processed/bidprediction/2025/final_bid_prediction_evaluation/test_final_metrics.csv)与[10b 表示指标](../../data/processed/bidprediction/2025/final_absolute_curve_dataset/representation_metrics.csv)。

**RF 与 8D 表示可以冻结，主要瓶颈是 X → Z_abs。** RF 显著优于均值、Ridge 和 GAM；Oracle 2.19% 与 RF 28.85% 相差约 **26.66 个百分点**，说明当前主要误差来自未来绝对曲线状态的预测，PCA 压缩和解码不是主要瓶颈。Macro-B RF 的 26.85% 到全量 RF 的 28.85% 仅增加约 2 个百分点，支持该路线在完整有效报价总体上仍有预测能力；但两次实验的特征清单、PCA 维数和训练抽样也不同，不能把差值全部归因于取消筛选，也不能据此认定两组难度相同。

| RF 最终评价维度 | TEST 结果 | 解释 |
|---|---|---|
| **价格轨迹（主指标）** | **MAE 31.02；RMSE 64.64；WAPE 28.85%；sMAPE 42.67%**；MAPE 330.19% | 主要预测瓶颈。MAPE 对低绝对价格敏感，代码虽排除阈值以下价格点，仍不适合作为主要结论。 |
| **数量边界** | `q_anchor` MAE **12.57 MW** / WAPE **15.72%**；`q_span` **22.41 MW / 12.99%**；`q_max` **23.40 MW / 9.27%** | 相对误差低于价格轨迹，数量边界具有较好的可预测性。 |
| **五段中点价格 MAE** | **[24.12, 25.37, 27.10, 31.61, 46.37]** | 误差向高出力尾段上升，第五段最难预测；这些是等数量段中点，不是旧 θ 的五个断点。 |
| 21 点表示节点 MAE | **26.95** | 表示节点上的直接误差；`breakpoint_proxy` 不是原始报价断点的逐点恢复指标。 |
| Normalized shape MAE | **0.840** | 细粒度形态复现仍不稳定；归一化对小价差敏感，作为辅助诊断。 |
| Template consistency | Accuracy **0.376**；非 FLAT **0.429** | 仅用于策略解释的一致性检查，不作为主验收目标或预测路由依据。 |

**能力边界集中在价格轨迹、尾部和时间泛化。** 尾段误差在 absolute latent 路线中依然显著，说明它并非仅由旧 θ 参数化造成；但这些指标本身不能进一步区分特征信息不足和模型拟合误差。VAL WAPE **21.16% → TEST 28.85%**，增加 **7.69 个百分点**、相对恶化约 **36.3%**，明确暴露跨时段泛化不足，与 09 的退化现象一致；行为漂移是合理解释，尚不能仅凭两个时间段的指标确认原因。

**最终表述：** 在不直接输入历史报价曲线、仅依赖主体画像、策略转移、市场、机组及时段特征的条件下，RF 能够显著学习报价行为，较均值、线性和 GAM 对照大幅降低误差；但绝对价格与细粒度形态预测仍有限，精度明显低于使用完整历史曲线的 Persistence。评价优先级为 **价格 WAPE / MAE / RMSE → 数量边界 → 分段中点误差 → shape MAE → Template 辅助解释**。技术路线定稿，不再继续模型、latent 维数或特殊 routing 筛选。

## 阶段结论：04～10 的路线收敛

| 阶段 | 核心结果 | 路线判断 |
|---|---|---|
| **04** | Template + θ 在表示与参数预测两端均有瓶颈。 | **停止旧参数化预测，模板保留为解释体系。** |
| **05 / 06** | 历史曲线预测力很强，绝对状态回归与 ΔCurve 回归均未超过 Persistence。 | **Persistence 优势显著，当前回归没有整体增益。** |
| **07** | 筛选常规阶梯型 Macro-B 后，Persistence 更强，回归仍未胜出。 | **问题不能简单归因于异质曲线混合。** |
| **08** | 去掉直接历史曲线输入，TEST WAPE 从无特征均值基线 **75.22%** 降至 RF **26.85%**。 | **证明主体画像、市场、机组等特征具有预测能力。** |
| **08e** | 旧 13 类 Template 未能充分降低 absolute curve 异质性：类内离散度仅降低 **8.35%**。 | **旧模板不适合作为当前预测路由，保留用于策略解释。** |
| **09a～09e** | K=5 真实 Family + RF 将 TEST WAPE 降至 **17.02%**，预测 Family 硬路由却为 **26.83%**，几乎回到 Global RF 的 **26.85%**。 | **分族有 Oracle 收益，路由损失抵消收益；保留为探索实验。** |
| **10** | 全部有效曲线 RF TEST WAPE **28.85%**，优于同输入对照；Oracle **2.19%**、Persistence **8.11%**。 | **冻结全局 feature-only RF + 8D absolute latent，明确价格与时间泛化的能力边界。** |

**最终技术路线：**

```text
历史报价 → 策略画像 Z / 策略转移 Z_tr
Z + Z_tr + Market + Unit + Calendar（83 个特征）
    → Random Forest → 8D Absolute Latent → 完整报价曲线
    → Stage 2 Template（仅作行为解释）
```

**Prediction Family、Macro-B、θ、ΔCurve 作为路线探索与消融记录，不进入最终模型。** Stage 2 Template 不参与预测路由；历史模板转移的汇总统计仍可属于 `Z_tr`。Persistence 保留为额外历史信息的强参考基线。停止新的模型横向筛选，不再推进 09f 或其他特殊 routing；定稿结论是路线及研究边界明确，不能表述为已获得高精度主体报价预测模型。
