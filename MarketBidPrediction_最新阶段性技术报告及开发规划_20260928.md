# MarketBidPrediction 最新阶段性技术报告及开发规划

> 项目名称：电力市场主体报价预测（MarketBidPrediction）  
> 更新时间：2026-09-28  
> 文档用途：作为后续在其他聊天窗口、AI 工具、本地 IDE 或协作开发环境中继续工作的最新技术基线与开发规划。  
> 当前主线：**主体多时间尺度策略画像 → 阶梯结构模板库 → 模板分类 → 模板专属关键参数预测 → 曲线重构 → 未来报价预测。**

---

# 1. 项目最终目标

项目目标不是直接把历史报价曲线输入模型并预测下一条曲线，而是先把历史报价行为压缩为具有明确经济和行为含义的主体策略画像，再利用当前市场环境和主体状态判断本时段更可能采用哪一种报价策略结构，最后只预测该结构下少量关键参数，恢复完整分段报价曲线。

总体形式为：

$$
\boxed{
\text{历史报价行为}
\rightarrow
\text{主体策略画像}
}
$$

$$
\boxed{
\text{主体策略画像}
+
\text{市场环境}
+
\text{主体状态}
+
\text{时间信息}
\rightarrow
\text{策略模板}
\rightarrow
\text{模板专属关键参数}
\rightarrow
\text{完整报价曲线}
}
$$

最终模型学习的是：

> **什么样的策略画像，在什么样的市场环境和主体状态下，会形成什么样的报价结构，以及该结构下的关键报价点如何变化。**

模型不首先人为规定“火电应该怎么报、水电应该怎么报、储能应该怎么报”，而是首先从历史行为中学习共同的策略模式。不同资源类型的物理差异主要通过主体状态和约束特征体现。

---

# 2. 当前仓库与代码基准

GitHub：

`https://github.com/hzan7112/MarketBidPrediction`

当前已确认基准：

- `main` 分支确认 commit：`5079afffce11880cdcad2d2e170364b727ac45fb`
- `scripts/bidprofile`：现有主体画像构建代码；
- `scripts/bidtemplate`：现有报价曲线样本、归一化和模板聚类代码；
- `scripts/bidprediction`：此前模板分类、参数回归、完整曲线预测等探索代码。

现有代码不直接删除。后续按照新的职责边界逐步重构，旧预测实验建议归档到：

```text
scripts/bidprediction/archive/
```

---

# 3. 三个项目目录的最新定位

整个项目重新划分为三个相互独立但前后衔接的阶段。

## 3.1 `bidprofile`

定位：

$$
\boxed{
\text{多时间尺度主体策略画像构建与有效性验证}
}
$$

负责回答：

> **我们设计的主体策略画像，到底有没有足够的信息量来解释报价行为？**

画像不是一个单纯的长期平均向量，而是一套多时间尺度、多状态维度的策略描述。

## 3.2 `bidtemplate`

定位：

$$
\boxed{
\text{阶梯报价结构发现、模板库构建与模板专属参数化}
}
$$

负责回答：

1. 实际报价曲线主要存在哪些典型阶梯结构；
2. 每种结构最少需要哪些关键参数才能准确还原；
3. 主体策略画像和市场状态能否解释模板选择。

模板不再只是一个“聚类标签”，而是后续曲线重构的**结构骨架**。

## 3.3 `bidprediction`

定位：

$$
\boxed{
\text{使用已验证画像和模板体系进行真正的未来报价预测}
}
$$

负责：

$$
X_{i,t}
\rightarrow
\hat{k}_{i,t}
\rightarrow
\hat{\theta}_{\hat{k}_{i,t}}
\rightarrow
\hat{C}_{i,t}
$$

其中：

- $X_{i,t}$：预测时可获得的画像、市场、主体状态和时间信息；
- $\hat{k}_{i,t}$：预测模板；
- $\hat{\theta}_{\hat{k}_{i,t}}$：该模板下的关键参数；
- $\hat{C}_{i,t}$：最终完整报价曲线。

---

# 4. 第一阶段：主体策略画像

## 4.1 画像不是只有长期特征

最新定义中，主体策略画像应表示为多时间尺度结构：

$$
\boxed{
Z_{i,t}
=
\left[
Z_i^{LT},
Z_{i,t}^{ST},
Z_{i,m}^{Season},
Z_{i,s}^{IntraDay},
Z_{i,t}^{Transition}
\right]
}
$$

其中：

### 长期画像 $Z_i^{LT}$

描述主体长期稳定的报价风格，例如：

- 长期报价水平；
- 报价曲线弯折程度；
- 尾部抬价倾向；
- 有效报价段数；
- 各报价段电量集中程度；
- 报价激进程度；
- 容量保留倾向；
- 历史报价波动性；
- 高价段出现概率；
- 策略持续性；
- 对供需紧张程度、燃料成本、市场价格等外部因素的长期敏感性。

### 短期画像 $Z_{i,t}^{ST}$

描述主体近期相对长期行为的策略偏移，例如：

- 最近若干天报价水平相对长期基准的偏离；
- 近期高价段使用频率；
- 近期曲线弯折程度变化；
- 近期报价激进程度变化；
- 近期价格跟随程度变化；
- 近期策略是否处于异常或快速切换状态。

短期画像属于整个画像体系的一部分，但其是否能用于某个具体未来预测场景，必须满足信息可获得性约束。

### 季节画像 $Z_{i,m}^{Season}$

描述主体随月份、季节或负荷季节变化的报价规律，例如：

- 夏季、冬季的典型报价水平；
- 不同季节的高价段出现概率；
- 不同季节的典型容量保留比例；
- 不同季节的曲线弯折和尾段抬价特征；
- 不同季节下对供需紧张程度的敏感度。

该部分用于避免用一个全年均值掩盖明显的季节性行为。

### 日内画像 $Z_{i,s}^{IntraDay}$

描述主体在一天不同交易时段的策略规律，例如：

- 峰、平、谷时段典型报价水平；
- 高峰时段的加价幅度；
- 不同时段的报价段电量分布；
- 高峰时段末段抬价概率；
- 不同时段的模板使用倾向。

### 策略转换画像 $Z_{i,t}^{Transition}$

描述主体不同报价策略之间的切换规律，例如：

- 模板切换频率；
- 策略惯性；
- 从某模板切换到另一模板的历史概率；
- 市场供需变化后模板切换的响应概率；
- 策略持续时长；
- 高价策略进入和退出特征。

---

# 5. 当前已有画像基础

目前 `bidprofile` 已形成：

$$
9\ LT + 9\ ST + 8\ Break = 26
$$

个可解释策略变量。

现有 9 个 LT 变量包括：

- `flat_curve_rate`
- `curve_bend_ratio`
- `tail_uplift_ratio`
- `shape_variability`
- `effective_segment_count`
- `quantity_hhi`
- `bid_level`
- `adjustment_magnitude`
- `strategy_persistence`

这些特征仍然是重要基础，但后续需要按照最新画像框架重新审查：

1. 哪些变量属于真正长期稳定特征；
2. 哪些变量应该改造成季节条件特征；
3. 哪些变量应该构造日内条件版本；
4. 哪些短期特征属于真实预测时可用状态；
5. 哪些策略转换特征应当保留为画像的一部分。

因此后续不是简单删除原有 ST / Break，而是把现有 26 维画像重新归类到：

$$
\mathrm{LT}+\mathrm{ST}+\mathrm{Season}+\mathrm{IntraDay}+\mathrm{Transition}
$$

的统一画像体系中。

---

# 6. 画像的“可用性”与“解释性”必须区分

主体画像可以包含短期、季节、日内和转换特征，但在真正未来预测时，任何特征都必须满足：

$$
\boxed{
\text{在目标预测时刻之前能够获得}
}
$$

因此后续每个画像特征应明确标记：

- `historical_static`：可由完整历史年度预先计算并冻结；
- `calendar_conditioned`：由目标月份/时段直接调用历史规律；
- `market_conditioned`：由当前市场信息结合历史规律生成；
- `online_recent`：只有在真实预测时确实已有近期报价时才能使用。

对于“完整历史年训练后直接预测未来 8 月、且没有未来年份报价输入”的严格场景：

- 长期画像可用；
- 季节画像可用；
- 日内画像可用；
- 历史策略转换规律可用；
- 依赖未来年份已实现报价的短期滚动特征不可作为必要输入。

因此短期画像属于总体画像体系，但最终模型必须根据应用场景选择实际可用子集。

---

# 7. 第二阶段：重新定义报价模板库

## 7.1 实际数据的关键事实

当前实际报价曲线几乎都是阶梯状。

因此不应把“阶梯型”本身作为一个粗粒度模板类别。更合理的模板定义应直接描述：

$$
\boxed{
\text{阶梯有几级、阶梯在哪里、跳多少、每一级占多少电量}
}
$$

即模板应该表示“阶梯结构拓扑”，而不仅是一条 21 点平均 shape 曲线。

---

# 8. 从原始报价提取“有效阶梯”

对每条历史报价曲线，首先进行报价段清洗和有效阶梯提取。

需要处理：

- 相邻价格几乎相同的平台合并；
- 极小电量占比的微小报价段合并；
- 重复 MW 点清洗；
- 无实际策略意义的小幅价格抖动过滤。

最终得到有效阶梯数：

$$
K_{\mathrm{eff}}
$$

以及各阶梯的结构信息。

每条曲线可以表示为：

$$
\mathcal{S}=
\left[
K_{\mathrm{eff}},
b_1,\ldots,b_{K-1},
r_1,\ldots,r_K,
s_1,\ldots,s_{K-1}
\right]
$$

其中：

### 阶梯位置

$$
b_j=
\frac{q_{b_j}-q_{\min}}
{q_{\max}-q_{\min}}
$$

表示第 $j$ 个跳价点位于整个容量区间的什么位置。

### 各平台电量占比

$$
r_k=
\frac{\Delta q_k}
{q_{\max}-q_{\min}}
$$

并满足：

$$
r_k\ge0,
\qquad
\sum_k r_k=1
$$

### 各次跳价占比

$$
s_j=
\frac{\Delta p_j}
{p_{\max}-p_{\min}}
$$

用于描述总价格跨度主要分布在哪几个阶梯。

---

# 9. 模板聚类不再直接只依赖 21 维 shape

当前已有模板聚类基于归一化 21 点 shape：

$$
\mathrm{shape\_v00},\ldots,\mathrm{shape\_v20}
$$

最终形成 13 类模板。

该结果保留作为现有 baseline，但后续正式模板体系建议改为：

$$
\boxed{
\text{先提取 staircase signature，再进行聚类}
}
$$

推荐结构特征包括：

- `effective_segment_count`
- 各段电量占比；
- 各断点归一化位置；
- 各次跳价幅度占比；
- 最大跳价位置；
- 最大跳价占总价格跨度比例；
- 尾部价格增幅占比；
- 价格跳变集中度；
- 前半段/后半段价格增长占比。

---

# 10. 模板聚类算法

轻量算法即可。

优先比较：

- K-means；
- K-medoids；
- GMM。

其中：

- K-means 可以作为基准；
- K-medoids 更适合作为最终候选，因为模板中心可以直接对应真实历史样本，结构更容易解释；
- GMM 可用于检查是否存在明显重叠型结构群体。

模板数量不预先固定为 13。

新的 $K$ 应由以下三个条件共同决定：

$$
\boxed{
\text{聚类稳定性}
+
\text{结构可解释性}
+
\text{模板内参数化重构误差}
}
$$

不再仅依据 silhouette 或其他单一聚类指标选 $K$。

---

# 11. 初始阶梯结构族建议

在分析真实 staircase signature 分布之前，不应直接锁死模板数量。

初始可以从以下结构族出发理解数据：

| 结构族 | 结构描述 |
|---|---|
| F0 | 单平台 / 近似平价 |
| F1 | 前段集中跳升 |
| F2 | 中段主阶跃 |
| F3 | 末段陡升 |
| F4 | 多级均匀抬升 |
| F5 | 多级后置抬升 |
| F6 | 多级前置抬升 |

这些只是结构族的初始解释框架。

最终模板数量可能为 5、6、8 或其他数量，必须由实际数据决定。

---

# 12. Template Class 与 Structural Family 的区别

后续建议明确区分：

$$
\boxed{
\text{Template Class}
}
$$

与：

$$
\boxed{
\text{Structural Family}
}
$$

例如多个聚类模板可能都属于：

> “双平台单阶跃”

这一结构族，只是：

- 阶梯位置不同；
- 高低平台比例不同；
- 电量分配不同。

因此：

- Template Class：数据驱动的细粒度报价模式；
- Structural Family：共享相同数学重构形式的结构族。

这样可以避免 13 个模板维护 13 套完全不同的手工 decoder。

---

# 13. 每种模板单独设计关键参数

模板不是一条固定曲线，而是一种曲线结构。

对于第 $k$ 个模板，定义其专属参数：

$$
\theta_k
$$

不同模板可以有不同维数。

## 13.1 平台型

$$
\theta_{\mathrm{flat}}
=
\left[p,\ q_{\min},q_{\max}\right]
$$

## 13.2 单阶梯型

$$
\theta_{\mathrm{step}}
=
\left[
p_1,p_2,
q_{\min},q_{\max},
q_b
\right]
$$

其中：

- $p_1,p_2$：两个价格平台；
- $q_b$：阶梯切换位置。

## 13.3 双阶梯 / 三平台型

$$
\theta=
\left[
p_1,p_2,p_3,
q_{\min},q_{\max},
q_{b_1},q_{b_2}
\right]
$$

## 13.4 末段陡升型

$$
\theta_{\mathrm{tail}}
=
\left[
p_{\mathrm{base}},
p_{\mathrm{tail}},
q_{\min},q_{\max},
q_{\mathrm{knee}}
\right]
$$

---

# 14. 用参数定义直接满足结构约束

优先通过参数化保证结构约束。

价格单调：

$$
p_2=p_1+\Delta p_1
$$

$$
p_3=p_2+\Delta p_2
$$

并设置：

$$
\Delta p_j\ge0
$$

电量段占比：

$$
r_k\ge0,
\qquad
\sum_k r_k=1
$$

再由：

$$
q_k=
q_{\min}
+
(q_{\max}-q_{\min})
\sum_{j=1}^{k}r_j
$$

保证所有阶梯位置有序且位于容量范围内。

因此：

$$
\boxed{
\text{简单结构约束通过参数定义保证，复杂规则最后校验}
}
$$

---

# 15. 模板参数化必须先做 Oracle Reconstruction

在训练任何参数预测模型之前，必须先验证：

$$
\boxed{
\text{真实模板}
+
\text{真实关键参数}
\rightarrow
\text{能否准确恢复原始曲线}
}
$$

即：

$$
C
\overset{\mathrm{extract}}{\longrightarrow}
(k,\theta_k)
\overset{D_k}{\longrightarrow}
\tilde C
$$

评价至少包括：

- price MAE；
- RMSE；
- WAPE；
- sMAPE；
- breakpoint price error；
- segment midpoint price error；
- quantity breakpoint error；
- normalized shape error。

如果某模板的 oracle reconstruction 都不够准确，则：

$$
\boxed{
\text{先修改模板参数化，不进入预测}
}
$$

---

# 16. 第三阶段：模板分类

预测输入统一表示为：

$$
X_{i,t}
=
\left[
Z_{i,t},
M_t,
U_{i,t},
C_t
\right]
$$

其中：

- $Z_{i,t}$：主体多时间尺度策略画像；
- $M_t$：市场环境；
- $U_{i,t}$：主体自身物理状态；
- $C_t$：日历、月份、日内时段等当前时间信息。

模板分类：

$$
\hat{k}_{i,t}
=
f_{\mathrm{cls}}
(
X_{i,t}
)
$$

优先模型：

- Logistic Regression；
- Decision Tree；
- Random Forest。

必要时再比较小型 HGBR。

---

# 17. 模板分类的意义

分类器回答：

> 当前市场环境下，这个主体更可能采用哪一种阶梯报价结构？

因此模板分类本身也是画像有效性的验证。

如果主体画像与市场环境无法较好预测模板，则说明：

- 模板划分可能不稳定；
- 或者画像缺少决定策略切换的信息；
- 或者当前市场特征不足。

不能直接跳到参数回归掩盖问题。

---

# 18. 第四阶段：模板专属关键参数回归

已知真实模板 $k$ 时：

$$
\hat{\theta}_{i,t}^{(k)}
=
f_k
(
X_{i,t}
)
$$

这一阶段先使用**真实模板标签**训练和验证每个模板自己的参数模型。

目的：

$$
\boxed{
\text{先排除模板分类错误，只验证关键参数是否可预测}
}
$$

例如单阶梯型分别预测：

- $p_1$
- $p_2$
- $q_b$
- $q_{\min}$
- $q_{\max}$

模型仍优先使用：

- Linear Regression；
- Ridge；
- GAM；
- 分位数回归；
- Random Forest；
- 小型 HGBR。

不采用深度学习作为当前主路线。

---

# 19. 第五阶段：完整预测 Pipeline

只有在：

1. 模板库稳定；
2. oracle reconstruction 通过；
3. 模板分类通过；
4. 各模板参数预测通过；

之后，才组合完整 pipeline：

$$
\boxed{
X_{i,t}
\rightarrow
\hat{k}_{i,t}
\rightarrow
\hat{\theta}_{i,t}^{(\hat{k})}
\rightarrow
D_{\hat{k}}
\rightarrow
\hat{C}_{i,t}
}
$$

最终曲线：

$$
\hat{C}_{i,t}
=
D_{\hat{k}_{i,t}}
(
\hat{\theta}_{i,t}^{(\hat{k})}
)
$$

---

# 20. 最终报价生成后的规则校验

模板和参数恢复曲线后，再执行简单规则校验：

- 价格单调性；
- 电量单调性；
- 最大可用容量；
- 最小技术出力；
- 价格上下限；
- 报价段数要求；
- 市场规则中的其他硬约束。

优先只做：

$$
\boxed{
\text{必要的最小修正}
}
$$

而不是依靠后处理重新塑造整条曲线。

---

# 21. 对不同资源类型主体的统一处理

项目不以火电、水电、储能、新能源作为策略预测模型的第一层人工分类。

更准确的建模思想是：

$$
\boxed{
\text{什么样的策略画像}
+
\text{什么样的市场状态}
+
\text{什么样的物理状态}
\rightarrow
\text{什么样的报价行为}
}
$$

如果两个不同类型主体表现出相似策略行为，它们可以：

- 位于相近画像区域；
- 选择相同模板；
- 使用相同结构族 decoder。

不同资源类型的物理区别主要通过 $U_{i,t}$ 表达。

例如：

- 火电：可用容量、最小技术出力、启停成本、空载成本、最小运行时间等；
- 储能：SOC、充放电功率边界、能量约束等；
- 水电：水位、来水、库容、水头状态等；
- 新能源：预测出力、可用出力、限发状态等。

因此应表述为：

> **不以主体物理类型作为报价策略模型的首要人工分类依据，而是统一通过策略画像描述行为差异，通过主体状态特征体现不同资源的物理运行边界。**

---

# 22. 当前已有 `bidtemplate` 结果的定位

现有 2025 模板阶段已经完成：

- Raw = 11,498,525
- Written = 5,119,107
- Shape = 4,547,836（88.84%）
- Flat = 571,271（11.16%）
- 代表样本 = 300,000
- 当前模板数 = 13

现有模板基于：

$$
\mathrm{shape\_v00},\ldots,\mathrm{shape\_v20}
$$

归一化曲线聚类。

该结果后续保留为：

$$
\boxed{
\text{旧版 21 维 shape 聚类 baseline}
}
$$

但不直接认定 13 类就是最终模板库。

下一步需要重新进行：

$$
\boxed{
\text{staircase signature}
\rightarrow
\text{结构聚类}
}
$$

并与旧 13 类结果比较。

---

# 23. 当前已有画像—模板关系结果的定位

此前已经观察到：

- Participants = 652
- Profile-ready days = 174,948
- Templates = 13
- dominant template share 中位数 = 1.000

部分 LT 与模板之间存在较强相关，例如：

- `flat_curve_rate` 与 FLAT；
- `curve_bend_ratio` 与部分阶梯模板。

这些结果说明：

$$
\boxed{
\text{画像与报价结构之间确实存在关系}
}
$$

但下一阶段应在新的 staircase structural template 下重新验证。

---

# 24. 当前已有模板分类结果的定位

旧模板分类实验中：

- lag1-template baseline 很强；
- feature-only 模型也有一定分类能力；
- 但此前特征体系和最终应用约束尚未完全统一。

因此旧结果只作为参考。

新版本分类必须基于：

$$
\boxed{
\text{最终可部署的主体画像}
+
\text{市场}
+
\text{主体状态}
+
\text{时间特征}
}
$$

重新验证。

---

# 25. 旧 Template + Theta 路线为什么不直接沿用

以前的问题不是“模板思想本身错误”。

真正问题是：

> 所有模板使用了过于统一的连续参数定义，模板只作为类别条件，并没有真正让不同曲线结构采用不同的专属关键参数。

旧路线：

$$
\mathrm{Template}
+
\text{统一参数向量 }\theta
\rightarrow
\mathrm{Curve}
$$

新路线：

$$
\boxed{
\mathrm{Template}_k
+
\theta_k^{\text{template-specific}}
\rightarrow
\mathrm{Curve}
}
$$

即：

> **模板决定参数结构本身。**

这才是新的关键技术变化。

---

# 26. `bidprofile` 最新开发规划

建议后续逐步形成：

```text
scripts/bidprofile/
│
├─ 01_build_daily_strategy_core.py
├─ 02_build_long_term_strategy_profile.py
├─ 03_build_short_term_strategy_state.py
├─ 04_build_seasonal_strategy_profile.py
├─ 05_build_intraday_strategy_profile.py
├─ 06_build_strategy_transition_profile.py
├─ 07_build_unified_strategy_profile.py
└─ 08_validate_strategy_profile.py
```

这只是目标结构，不要求一次性重写现有全部代码。

当前应优先：

1. 整理现有 26 个画像变量；
2. 明确 LT / ST / Season / IntraDay / Transition 五个维度；
3. 找出目前尚缺失的季节和日内画像；
4. 明确每个特征在未来预测时的信息可用性。

---

# 27. `bidtemplate` 最新开发规划

建议核心 pipeline 调整为：

```text
scripts/bidtemplate/
│
├─ 01_build_curve_samples.py
├─ 02_extract_effective_staircase.py
├─ 03_build_staircase_signature.py
├─ 04_cluster_staircase_templates.py
├─ 05_validate_template_stability.py
├─ 06_define_template_parameterization.py
├─ 07_validate_oracle_reconstruction.py
├─ 08_profile_template_association.py
└─ 09_train_template_classifier.py
```

当前真正优先的是：

$$
\boxed{
02\rightarrow03\rightarrow04\rightarrow06\rightarrow07
}
$$

即：

> 先重新定义阶梯结构，再确定模板，再确定每种模板如何用少量关键参数重构。

---

# 28. `bidprediction` 最新开发规划

在 `bidprofile` 与 `bidtemplate` 通过后，再正式进入：

```text
scripts/bidprediction/
│
├─ archive/
│  └─ 旧预测探索代码
│
├─ 01_build_prediction_dataset.py
├─ 02_train_template_classifier.py
├─ 03_train_template_parameter_models.py
├─ 04_reconstruct_bid_curves.py
├─ 05_validate_bid_rules.py
└─ 06_evaluate_full_pipeline.py
```

---

# 29. 最新开发顺序

后续不要同时推进多个问题。

## Step 1：画像体系整理

$$
\mathrm{LT}+\mathrm{ST}+\mathrm{Season}+\mathrm{IntraDay}+\mathrm{Transition}
$$

明确每类画像的定义和可用性。

## Step 2：有效阶梯提取

从原始报价曲线得到：

- 有效段数；
- 阶梯位置；
- 电量占比；
- 跳价幅度。

## Step 3：Staircase Signature

构建低维结构特征。

## Step 4：重新聚类模板

比较 K-means / K-medoids / GMM。

模板数量由：

$$
\text{稳定性}
+
\text{解释性}
+
\text{重构误差}
$$

共同决定。

## Step 5：模板专属参数设计

对每个结构族定义：

$$
\theta_k
$$

## Step 6：Oracle Reconstruction

验证：

$$
k+\theta_k
\rightarrow
C
$$

能否准确还原。

这一关不过，不预测。

## Step 7：模板分类

验证：

$$
\mathrm{Profile}+\mathrm{Market}+\mathrm{Unit}+\mathrm{Calendar}
\rightarrow
\mathrm{Template}
$$

## Step 8：模板参数预测

真实 Template 条件下验证：

$$
X\rightarrow\theta_k
$$

## Step 9：完整 Pipeline

最后组合：

$$
X
\rightarrow
\hat{k}
\rightarrow
\hat{\theta}_{\hat{k}}
\rightarrow
\hat{C}
$$

---

# 30. 每一层的失败处理原则

### 画像无法解释模板

回到：

$$
\boxed{
\text{bidprofile}
}
$$

修改画像。

### 模板不稳定

回到：

$$
\boxed{
\text{bidtemplate 聚类}
}
$$

修改 staircase signature 或模板数。

### Oracle reconstruction 差

回到：

$$
\boxed{
\text{模板参数化}
}
$$

增加或修改关键参数。

### 已知真实模板但参数预测差

说明：

$$
\boxed{
\text{画像 / 市场 / 主体状态中缺少决定该参数的信息}
}
$$

先查特征，不直接换复杂模型。

### 单独各层都好但 full pipeline 差

再分析：

- 模板误分类传播；
- 参数模型误差；
- decoder 敏感性。

---

# 31. 当前不再作为主线的方向

以下路线不再优先：

- 直接预测 21 个曲线点；
- PCA latent 后直接回归；
- 所有模板共享同一个统一 theta；
- 单纯依靠 previous template；
- 把 lag1 完整报价曲线作为最终必要输入；
- 通过不断增加复杂模型解决表示问题；
- 在完整曲线效果差时直接更换 RF / HGBR / 深度模型。

---

# 32. 当前模型选择原则

继续坚持：

$$
\boxed{
\text{轻量 + 可解释}
}
$$

模板分类优先：

- Logistic Regression；
- Decision Tree；
- Random Forest。

模板参数回归优先：

- Linear Regression；
- Ridge；
- GAM；
- Quantile Regression；
- Random Forest；
- 小型 HGBR。

只有确认输入信息与目标定义都合理后，才考虑增加模型复杂度。

---

# 33. 当前完整技术路线

最终路线统一表述为：

$$
\boxed{
\begin{aligned}
&\text{历史报价行为提取}
\\
&\rightarrow
\text{长期 + 短期 + 季节 + 日内 + 策略转换主体画像}
\\
&\rightarrow
\text{历史阶梯报价结构提取}
\\
&\rightarrow
\text{Staircase Signature}
\\
&\rightarrow
\text{报价策略模板库}
\\
&\rightarrow
\text{逐模板设计关键曲线参数}
\\
&\rightarrow
\text{Oracle 重构验证}
\\
&\rightarrow
\text{画像 + 市场 + 主体状态预测当前模板}
\\
&\rightarrow
\text{模板专属轻量模型预测关键参数}
\\
&\rightarrow
\text{模板 Decoder 恢复完整报价曲线}
\\
&\rightarrow
\text{市场规则校验和必要最小修正}
\end{aligned}
}
$$

---

# 34. 模型解释示例

最终模型可以解释为：

> 某主体历史上表现为报价水平较高、末段抬价明显、容量集中度较高、策略持续性较强，并且在高负荷季节和高峰时段更容易采用后置抬价策略。当前系统供需趋紧、主体可用容量较高，因此模板分类器判断其更可能采用“末段陡升型”阶梯模板。随后模板专属参数模型预测基础价格平台、末段价格、尾部阶梯位置和最大报价电量，并通过该模板的确定性 decoder 恢复完整分段报价曲线。

这一解释链条为：

$$
\text{历史上怎么报}
+
\text{当前处于什么环境}
$$

$$
\Downarrow
$$

$$
\text{选择什么报价结构}
$$

$$
\Downarrow
$$

$$
\text{关键报价点是多少}
$$

$$
\Downarrow
$$

$$
\text{完整报价曲线}
$$

---

# 35. 当前阶段最优先任务

下一步不再继续 `bidprediction` 旧实验。

当前优先回到：

$$
\boxed{
\texttt{bidtemplate}
}
$$

第一项新工作是：

$$
\boxed{
\text{对 2025 历史报价曲线提取“有效阶梯结构”}
}
$$

需要首先统计：

1. 有效报价段数 $K_{\mathrm{eff}}$ 分布；
2. 最大跳价位置分布；
3. 最大跳价占总价格跨度比例；
4. 各平台电量占比分布；
5. 前段、中段、末段跳价比例；
6. 尾部价格增幅占比；
7. 不同阶梯数下的典型结构。

只有看清这些数据分布后，才决定最终：

- staircase signature；
- 模板数；
- structural family；
- 每个模板的参数定义。

---

# 36. 当前阶段一句话交接

后续开发的核心已经确定为：

> **首先用历史报价构建包含长期、短期、季节、日内和策略转换信息的主体策略画像；然后针对实际报价几乎均为阶梯状这一事实，从历史曲线中提取有效阶梯结构并重新建立模板库；不同模板采用各自的低维关键参数表示。预测时先根据主体画像、市场环境和主体状态选择报价模板，再预测该模板下的少量关键参数，最后通过模板 decoder 恢复完整分段报价曲线。模板分类、模板参数预测和曲线重构分别独立验收，任何一层失败都回到对应模块修改，而不是直接增加模型复杂度。**

