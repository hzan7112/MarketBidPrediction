# bidprediction 阶段性技术报告

> 项目：MarketBidPrediction  
> 子项目：`scripts/bidprediction`  
> 当前阶段：无泄漏预测数据集 + 模板预测 + 报价曲线回归可行性验证  
> 当前基准：2025 年 PJM 数据，最终价格侧采用 21 点直接回归，数量侧采用轻量树模型回归

---

## 1. 子项目目标

`bidprediction` 的任务不是重新定义市场主体策略画像或重新聚类报价模板，而是在前两个阶段结果已经冻结的基础上，建立从预测时刻可获得的信息到未来报价曲线的映射。

整体输入输出关系为：

```text
主体长期策略画像 Z_base
+ 策略转移状态 Z_tr
+ 市场环境 M
+ 机组状态/成本代理 U
+ 当前预测模板条件
        ↓
模板预测 + 连续曲线预测
        ↓
未来报价曲线
(q_0, p_0), ..., (q_20, p_20)
```

当前阶段主要验证三个问题：

1. 主体报价模板能否提前预测；
2. 已知/预测模板后，报价曲线的价格水平和形状能否预测；
3. 报价曲线的数量范围和分段位置能否预测。

当前目标是验证“轻量、可解释、可工程实现”的方案是否可行，而不是追求深度学习模型或极限预测精度。

---

## 2. 与前两个阶段的关系

### 2.1 Stage 1：`bidprofile`

第一阶段已经形成固定的主体策略画像体系：

- 9 个长期策略特征（LT）；
- 9 个短期策略状态特征（ST）；
- 8 个 Break/结构性特征。

共 26 个基础策略特征。

LT 特征：

```text
lt_bid_level
lt_adjustment_magnitude
lt_strategy_persistence
lt_quantity_hhi
lt_effective_segment_count
lt_flat_curve_rate
lt_tail_uplift_ratio
lt_curve_bend_ratio
lt_shape_variability
```

ST 特征：

```text
st_bid_level_z
st_adjustment_bias_z
st_adjustment_magnitude_z
st_quantity_hhi_z
st_effective_segment_count_z
st_flat_curve_rate_z
st_tail_uplift_ratio_z
st_curve_bend_ratio_z
st_shape_shift
```

Break 特征：

```text
break_bid_level
break_adjustment_bias
break_adjustment_magnitude
break_quantity_hhi
break_effective_segment_count
break_flat_curve_rate
break_tail_uplift_ratio
break_curve_bend_ratio
```

这些特征描述的是主体长期报价风格、近期偏移和策略结构，而不是直接保存历史报价曲线。

### 2.2 Stage 2：`bidtemplate`

第二阶段将不同原始段数的 PJM 报价曲线统一表示到 21 点归一化电量网格：

```text
x = 0, 0.05, ..., 1.00
```

对非平价曲线：

```text
shape(x) = [P(x) - P(0)] / [P(1) - P(0)]
```

同时保留：

```text
q_anchor_mw
q_span_mw
p_anchor
p_span
```

从而可以恢复报价曲线。

平价曲线单独作为 `FLAT` 模板族。

当前模板体系共 13 类：

```text
T00 ~ T11
FLAT
```

因此 `bidprediction` 不再重新聚类模板，而是把模板 ID 作为未来报价行为的一部分进行预测。

---

## 3. 无泄漏预测数据集

主要脚本：

```text
01_build_prediction_dataset.py
02_validate_prediction_dataset.py
04a3_freeze_modeling_dataset.py
```

### 3.1 预测时间边界

当前任务按 D 日报价预测构造数据。

信息截止时间：

```text
D-1 11:00 EPT
```

只允许使用截止时刻之前可以获得的信息。

长期策略画像采用滚动历史窗口：

```text
LT lookback = 180 天
LT minimum active history = 30 天
```

### 3.2 当前预测可用率

2025 年预测数据：

```text
原始有效报价曲线：5,119,107
Profile-ready：4,153,493
Market-ready：5,119,107
Prediction-ready：4,153,493
Prediction-ready share：81.14%
```

清洗异常数量标签后：

```text
theta-ready：4,139,189
```

### 3.3 时间划分

冻结数据集使用严格时间划分：

```text
Train：<= 2025-10-02
Validation：2025-10-03 ~ 2025-11-16
Test：>= 2025-11-17
```

最终固定训练样本：

```text
300,000 rows
```

模型选择只使用 Train + Validation。

Test 只用于最终评价。

泄漏审计结果：

```text
Leakage violations = 0
```

---

## 4. 最终回归输入

当前连续曲线预测固定使用：

```text
Z_base + Z_tr + M + U
```

其中：

| 特征组 | 维数 | 含义 |
|---|---:|---|
| `Z_base` | 26 | 当前主体策略画像：9 LT + 9 ST + 8 Break |
| `Z_tr` | 26 | 策略转移/变化状态特征，由冻结 schema 中 `transition_strategy_profile` 读取 |
| `M` | 8 | 市场环境特征 |
| `U` | 12 | 机组运行状态与成本代理特征 |
| 合计 | 72 | 模型基础数值输入 |

市场环境 `M` 主要包括：

- 系统负荷预测；
- 负荷预测信息年龄；
- RTO / West / Other outage 状态；
- 前一日经济最大出力、紧急最大出力、总承诺容量等市场状态。

机组状态 `U` 为冻结 schema 中 `unit_state_proxy` 的 12 个特征，主要反映：

- 机组经济最大/最小出力统计；
- 运行状态和启停状态；
- 机组成本代理信息。

具体列名以：

```text
data/processed/bidprediction/<year>/frozen_modeling_dataset/feature_schema.csv
```

为唯一基准，模型按 `feature_group` 自动读取，不针对 2025 PJM 数据手工写主体或模板特例。

### 4.1 模板条件输入

连续曲线模型还加入当前模板的 13 维 one-hot 条件：

```text
cond_template_T00
...
cond_template_T11
cond_template_FLAT
```

因此回归模型实际输入维数为：

```text
72 + 13 = 85
```

### 4.2 不使用的直接历史报价状态

连续曲线回归不使用 `participant_history` 组中的直接历史报价状态 `H`。

即连续回归不是简单复制：

```text
上一时刻价格
上一条完整报价曲线
上一条曲线的原始 MW/BID 点
```

当前冻结的模板预测器本身仍使用上一期模板状态作为模板切换的 origin state；这一部分属于模板状态转移模型，而不是连续价格/数量回归输入。

---

## 5. 模板预测结构

当前最终模板预测器采用分层结构，而不是直接对 13 类模板做一次性硬分类。

逻辑为：

```text
上一期模板 origin
        ↓
Switch Model
判断本期是否发生模板切换
        ↓
若不切换：沿用 origin
若切换：
        ↓
Destination Model
在允许的目标模板中预测新模板
```

### 5.1 Switch Model

输入：

```text
switch_features
```

经过缺失值填补后输出：

```text
P(switch = 1)
```

使用验证阶段冻结的层级阈值：

```text
hierarchy_threshold
```

决定是否发生模板切换。

### 5.2 Destination Model

输入：

```text
destination_features
+ origin template
```

输出 13 类目标模板概率。

随后根据训练集观测到的模板转移关系：

```text
allowed_destinations_train
```

屏蔽不允许的 destination，再选择概率最大的模板。

### 5.3 当前结果

TEST：

```text
Template accuracy = 0.951665
                  = 95.17%
```

从 `oracle_template_model` 与 `full_pipeline` 的误差差异看，模板分类误差已经不是当前主要瓶颈。

---

# 6. 连续报价曲线预测：当前最终结构

## 6.1 路线演化

可行性验证阶段依次尝试过：

```text
模板 + p_base/alpha/beta + q 参数
        ↓
HGB / RF 分组参数回归
        ↓
p_base 加权回归
        ↓
模板 + PCA residual 低维价格参数
        ↓
直接预测 21 点价格
```

主要结论：

- `p_base/alpha/beta` 可以表示曲线，但参数之间存在明显补偿关系，不容易稳定预测；
- PCA residual 能把表示误差压到很低，但 PCA 系数预测后总体价格精度仍有限；
- 因此最终阶段不再强迫价格曲线经过少量人工 θ 参数，而直接预测统一 21 点价格。

当前最终方案：

```text
价格：ExtraTrees → 21 个价格残差
数量尺度：RandomForest
数量分段：RandomForest
```

---

# 7. 价格回归模型

## 7.1 预测对象

Stage2 中每条报价曲线都已经转换到：

```text
x_j = j / 20, j = 0,...,20
```

对应真实价格：

```text
P_0, P_1, ..., P_20
```

不直接预测绝对价格，而以长期主体报价水平 `lt_bid_level` 为基准：

$$
r_j=P_j-\mathrm{lt\_bid\_level}
$$

因此每个训练样本的价格目标是 21 维向量：

$$
\mathbf r=[r_0,r_1,\ldots,r_{20}]
$$

训练集对每一个位置分别计算：

$$
z_j=\frac{r_j-\mu_j}{\sigma_j}
$$

最终回归目标为：

```text
z_0 ... z_20
```

共 21 个连续输出。

## 7.2 最终价格模型：ExtraTreesRegressor

当前验证集自动选择：

```text
ExtraTreesRegressor
```

而不是 Ridge、RandomForest 或 Ridge + Tree hybrid。

模型形式为多输出树集成：

```text
X ∈ R^85
        ↓
ExtraTrees
        ↓
[z_0, z_1, ..., z_20]
```

每棵随机树在训练样本和随机特征切分的基础上学习多维价格残差映射，最终对所有树的预测取集成平均。

当前候选模型包括：

```text
Ridge
RandomForest
ExtraTrees
Ridge + RandomForest residual
Ridge + ExtraTrees residual
Ridge + weighted ExtraTrees residual
```

但当前 Validation 最优仍为纯：

```text
ExtraTrees
```

说明继续叠加线性外推模块没有带来验证收益。

## 7.3 价格反变换

模型输出标准化残差：

```text
zhat_j
```

恢复：

$$
\hat r_j=\hat z_j\sigma_j+\mu_j
$$

最终价格：

$$
\hat P_j=\mathrm{lt\_bid\_level}+\hat r_j
$$

因此最终直接得到：

```text
P_hat_0 ... P_hat_20
```

无需再次通过 `p_base/alpha/beta` 或 PCA 系数恢复。

---

# 8. 数量尺度回归

当前数量尺度包括：

```text
q_base
q_span
```

其中：

```text
q_base = 报价曲线起始 MW
q_span = 最大 MW - 起始 MW
q_max  = q_base + q_span
```

## 8.1 尺度归一化

Validation 最终选择：

```text
unit_lag1_avg_ecomax
```

作为机组容量尺度代理。

定义：

$$
s_q=\mathrm{unit\_lag1\_avg\_ecomax}
$$

训练目标：

$$
y_{base}
=
\operatorname{signed\_log1p}
\left(
\frac{q_{base}}{s_q}
\right)
$$

$$
y_{span}
=
\log(1+\frac{q_{span}}{s_q})
$$

然后再做训练集标准化。

这种处理的目的是让不同容量机组在统一相对尺度下学习数量行为，而不是直接让模型拟合绝对 MW。

## 8.2 模型

当前最终选择：

```text
RandomForestRegressor
```

输入：

```text
85-dimensional conditional feature vector
```

输出：

```text
2-dimensional latent vector
```

对应：

```text
q_base latent
q_span latent
```

反变换后重新乘容量尺度 `s_q`，恢复 MW。

---

# 9. 分段位置回归

当前保留 5 个归一化数量段占比：

```text
q1, q2, q3, q4, q5
```

满足：

$$
q_k>0
$$

以及：

$$
\sum_{k=1}^{5}q_k=1
$$

为了保证模型输出天然满足 simplex 约束，不直接预测 5 个独立比例，而采用 additive log-ratio：

$$
l_1=\log(q_1/q_5)
$$

$$
l_2=\log(q_2/q_5)
$$

$$
l_3=\log(q_3/q_5)
$$

$$
l_4=\log(q_4/q_5)
$$

因此模型只回归：

```text
l1, l2, l3, l4
```

当前模型：

```text
RandomForestRegressor
```

预测完成后追加：

```text
l5 = 0
```

再通过 softmax 恢复：

```text
q1...q5
```

从而严格满足：

```text
qk > 0
sum(qk) = 1
```

累计断点为：

```text
b1 = q1
b2 = q1 + q2
b3 = q1 + q2 + q3
b4 = q1 + q2 + q3 + q4
```

物理 MW 断点为：

$$
Q_{b_k}
=
q_{base}
+
q_{span}b_k
$$

---

# 10. 最终报价曲线生成

最终预测得到：

```text
predicted template
P_hat_0 ... P_hat_20
q_base_hat
q_span_hat
q1_hat ... q5_hat
```

21 点物理电量坐标：

$$
\hat Q_j
=
\hat q_{base}
+
\hat q_{span}
\frac{j}{20}
$$

最终报价曲线：

$$
(\hat Q_0,\hat P_0),
(\hat Q_1,\hat P_1),
...,
(\hat Q_{20},\hat P_{20})
$$

`q1...q5` 不用于改变 21 点均匀物理电量网格，而用于评价和描述报价曲线内部的分段位置/数量结构。

---

# 11. 当前最终 TEST 结果

当前 full pipeline：

```text
Eligible coverage = 633,913 / 633,985
                  = 99.99%

Template accuracy = 95.1665%
```

## 11.1 价格

```text
MAE   = 28.109767 $/MWh
RMSE  = 57.816386 $/MWh
sMAPE = 43.4082%
WAPE  = 26.1463%
```

普通 MAPE：

```text
332.6209%
```

不作为主要结果指标。

原因是 PJM 报价中存在：

```text
0
接近 0
负价格
```

普通 MAPE 的分母在这些样本上非常不稳定。

当前报告应优先使用：

```text
MAE
WAPE
sMAPE
curve-level MAE quantiles
```

## 11.2 Curve-level MAE

每一条报价曲线首先计算 21 点平均价格绝对误差。

TEST：

```text
P50 = 14.164 $/MWh
P75 = 31.475 $/MWh
P90 = 70.457 $/MWh
P95 = 110.456 $/MWh
```

误差覆盖：

```text
MAE <= 5   : 26.78%
MAE <= 10  : 40.33%
MAE <= 20  : 61.32%
MAE <= 50  : 84.56%
```

因此当前结果可用于证明：

> 主体策略画像、市场状态和机组状态能够在不直接输入完整历史报价曲线的情况下，对多数测试样本生成具有一定一致性的报价曲线。

但当前还不应视为生产级高精度预测模型。

---

# 12. 当前数量预测结果

Full pipeline：

```text
q_anchor MAE  = 10.456 MW
q_anchor WAPE = 13.077%

q_span MAE    = 16.437 MW
q_span WAPE   = 9.526%

q_max MAE     = 12.637 MW
q_max WAPE    = 5.005%
```

分段结构：

```text
q_share MAE               = 0.023715
breakpoint fraction MAE   = 0.031384
breakpoint MW MAE         = 13.759 MW
```

平均归一化 breakpoint 位置误差约为：

```text
3.14% of quantity span
```

数量侧当前已经明显优于价格侧。

---

# 13. 分段价格误差

当前五个分段中点价格 MAE：

```text
Segment 1 = 24.264 $/MWh
Segment 2 = 27.812 $/MWh
Segment 3 = 32.511 $/MWh
Segment 4 = 37.746 $/MWh
Segment 5 = 42.079 $/MWh
```

误差随报价曲线向后段逐渐增加。

这说明当前最主要的剩余问题仍然是：

```text
高价段 / 尾部价格水平
```

而不是模板类别或数量尺度。

---

# 14. 误差来源分解

当前可以通过三个评价模式区分误差来源。

### 14.1 `representation_oracle`

直接使用真实 21 点曲线：

```text
Price WAPE = 0
```

当前 21 点直接表示不存在额外参数化损失。

### 14.2 `oracle_template_model`

给连续回归模型真实模板，只评价价格和数量预测：

```text
Price WAPE = 25.8306%
```

### 14.3 `full_pipeline`

模板也由模型预测：

```text
Price WAPE = 26.1463%
```

因此：

```text
模板误差增加约 0.32 个 WAPE 百分点
```

模板分类不是当前主要误差来源。

04b 单独在真实均匀价格网格上的 TEST：

```text
Price WAPE = 24.1782%
```

进入完整物理数量轴评价后增加到：

```text
25.8306%
```

说明数量轴误差也会进一步放大最终曲线误差。

当前主要误差来源顺序可概括为：

```text
价格预测本身
    >>
数量轴预测
    >
模板分类
```

---

# 15. Validation 与 Test 的时间泛化差异

当前价格模型：

```text
Validation WAPE = 15.2081%
Test WAPE       = 24.1782%
```

存在明显 temporal generalization gap。

已经比较：

```text
Ridge
RandomForest
ExtraTrees
Ridge + RandomForest residual
Ridge + ExtraTrees residual
```

Validation 最终仍选择纯 `ExtraTrees`。

因此当前阶段不再继续单纯通过更换 RF / ExtraTrees / Ridge 来提升结果。

若下一阶段继续提高精度，应重点检查：

1. D-1 截止时刻的市场信息是否足以解释未来报价绝对价格水平；
2. 是否缺少燃料成本、机组边际成本、约束状态、节点/区域价格预期等可获得变量；
3. `Z_base + Z_tr + M + U` 在 TEST 时间段是否发生明显分布漂移；
4. 是否需要按市场状态进行 regime-aware 建模。

---

# 16. 当前阶段结论

当前 `bidprediction` 已经完成从预测特征到完整报价曲线的端到端可行性验证：

```text
输入
Z_base + Z_tr + M + U
        ↓
层级模板预测
        ↓
预测模板
        ↓
ExtraTrees 预测 21 点价格
+
RandomForest 预测数量尺度
+
RandomForest 预测分段位置
        ↓
完整 21 点未来报价曲线
```

当前方案的主要优点：

- 不采用深度学习；
- 输入和输出含义清楚；
- 模型训练、推理和工程部署较轻量；
- 数量尺度和 breakpoint 具有明确物理含义；
- 可直接输出完整报价曲线；
- 无泄漏时间划分已经验证；
- 模板分类已经达到较高稳定性。

当前主要限制：

- 测试期价格预测仍存在明显时间分布漂移；
- 高价尾段误差较大；
- 当前精度适合作为方法可行性验证，不宜直接表述为生产级报价预测精度。

---

# 17. 当前建议保留的核心流程

建议后续主流程只保留：

```text
01_build_prediction_dataset.py
02_validate_prediction_dataset.py

03 / 03b 模板预测相关脚本

04a_build_curve_parameter_dataset.py
04a2_build_template_adjustment_dataset.py
04a2b_sanitize_theta_quantity_targets.py
04a3_freeze_modeling_dataset.py

04b_train_template_parameter_models_frozen_v10.py
04c_reconstruct_bid_curves_frozen_v10.py

05_visualize_bid_curve_prediction.py
```

以下内容属于此前参数化探索路线，可保留作实验记录，但不再作为当前最终模型：

```text
p_base weighted specialist
p_base target-transform specialist
旧 HGB theta regression
旧 p_base / alpha / beta 参数路线
PCA residual parameter regression
```

---

# 18. 当前输出目录

冻结预测数据：

```text
data/processed/bidprediction/2025/frozen_modeling_dataset/
```

最终模型：

```text
data/processed/bidprediction/2025/template_parameter_models/
```

模板预测模型：

```text
data/processed/bidprediction/2025/final_template_predictor/
```

最终曲线重构结果：

```text
data/processed/bidprediction/2025/curve_reconstruction/
```

主要评价文件：

```text
curve_reconstruction/reconstruction_metrics.csv
curve_reconstruction/summary.txt
```

可视化输出：

```text
data/processed/bidprediction/2025/curve_visualization/
```
