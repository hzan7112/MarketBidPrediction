# MarketBidPrediction 最新阶段性技术报告及开发规划

> 项目名称：电力市场主体报价预测（MarketBidPrediction）  
> 日期：2026-09-26  
> 文档用途：作为后续在其他聊天窗口、AI 工具、本地 IDE 或协作开发环境中继续工作的最新技术基线与开发规划。  
> 当前状态：**暂停继续优化完整报价曲线预测，重新梳理 `bidprofile`、`bidtemplate`、`bidprediction` 三个目录的职责边界，并把“用户画像是否真的具备报价解释能力”作为进入未来预测阶段之前的强制验收关卡。**

---

# 1. 项目最终目标

项目最终目标不是“利用近期报价惯性预测下一次报价”，而是：

$$
\boxed{
\text{利用完整历史年度报价构建主体画像}
\rightarrow
\text{冻结画像}
\rightarrow
\text{结合未来可获得外生特征预测未来报价}
}
$$

例如：

$$
2025\text{全年报价}
\rightarrow
Profile_{2025}
$$

随后在真实部署中预测 2026 年某个未来时点：

$$
Profile_{2025}
+
Market_{2026,t}
+
Unit_{2026,t}
+
Calendar_{2026,t}
\rightarrow
\hat{Bid}_{2026,t}
$$

未来预测时原则上**不能依赖未来年份已经发生的报价数据**，因此最终模型不得依赖：

- lag1 完整报价曲线；
- lag1 $p_{\min},p_{\max}$；
- 未来年份 7d / 30d 滚动报价统计；
- previous latent；
- previous template；
- 任何需要目标预测年度中的已实现报价才能构造的短期报价惯性特征。

这些量可以用于离线诊断，但不能成为最终部署模型的必要输入。

---

# 2. 当前仓库基准

GitHub：

`https://github.com/hzan7112/MarketBidPrediction`

当前确认基准：

- `main` 分支已确认 commit：
  `5079afffce11880cdcad2d2e170364b727ac45fb`
- `scripts/bidprofile`：原始主体画像构建代码基准。
- `scripts/bidtemplate`：原始报价模板构建与聚类代码基准。
- `scripts/bidprediction`：此前大量预测探索实验所在地。

后续代码结构需要按照本文重新整理，但原有代码建议先保留，避免丢失实验记录。

---

# 3. 当前项目为什么需要重新划分目录职责

此前三个模块逐渐发生混用：

- `bidprofile` 只负责构造画像，没有单独证明画像是否真的能解释报价；
- `bidtemplate` 一度承担“模板分类 + 参数回归 + 曲线重构”的强制技术路线；
- `bidprediction` 同时承担特征验证、曲线表示、模板选择、参数化、完整预测等过多任务。

结果是当完整曲线预测失败时，很难判断问题究竟来自：

1. 用户画像本身信息不足；
2. 报价曲线低维表示不合理；
3. 模板划分不合理；
4. 模型能力不足；
5. 未来预测特征不足。

因此现在重新定义三个目录：

$$
\boxed{
\text{bidprofile：画像有没有信息量}
}
$$

$$
\boxed{
\text{bidtemplate：报价行为有没有稳定、可解释的离散模式}
}
$$

$$
\boxed{
\text{bidprediction：已经验证有效的画像能否真正预测未来报价}
}
$$

---

# 4. 总体新技术路线

新的技术主线为：

$$
\boxed{
\text{阶段1：主体画像与报价表达能力验证}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{阶段2：报价模板的解释性验证}
}
$$

$$
\Downarrow
$$

$$
\boxed{
\text{阶段3：未来报价预测}
}
$$

其中：

- 阶段 1 和阶段 2 均属于**建模基础有效性验证**；
- 只有阶段 1 通过之后，阶段 3 才有意义；
- 阶段 2 不再是最终曲线预测的必经路径；
- 最终预测可以完全不经过 Template。

---

# 5. 第一阶段重新定位：`bidprofile`

## 5.1 核心任务

`bidprofile` 不再只是“计算若干主体画像特征”。

它必须回答：

$$
\boxed{
\text{我们设计的“用户画像”到底有没有足够的信息量来解释报价？}
}
$$

如果这一关都过不了：

$$
\boxed{\text{直接修改画像定义，不进入未来预测阶段}}
$$

因此 `bidprofile` 应包含三部分：

1. 年度静态主体画像构建；
2. 报价曲线低维连续表示；
3. 画像对这些报价低维变量的解释能力验证。

---

# 6. 第一阶段 A：年度静态主体画像

## 6.1 原有画像

当前主体策略画像为：

$$
9\ LT + 9\ ST + 8\ Break = 26
$$

原有 9 个 LT 变量主要包括：

- `flat_curve_rate`
- `curve_bend_ratio`
- `tail_uplift_ratio`
- `shape_variability`
- `effective_segment_count`
- `quantity_hhi`
- `bid_level`
- `adjustment_magnitude`
- `strategy_persistence`

此前 LT 主要按约 180 天滚动历史构造。

## 6.2 当前对 LT 的重新认识

需要区分两类特征。

### 结构型长期特征

例如：

- 平报价倾向；
- 曲线弯折程度；
- 尾部抬升；
- 有效报价段数；
- 数量集中程度；
- 策略持续性；
- 曲线形状波动程度。

这类特征描述：

> “这个主体通常怎么报。”

这类信息适合用较长时间甚至完整年度进行统计。

### 绝对水平特征

例如：

$$
lt\_bid\_level
$$

它描述：

> “这个主体通常报多少钱。”

如果一年内报价明显随季节或市场状态变化，单一 180 天均值可能过于粗糙。

因此后续年度画像需要重点重新评估：

- 哪些 LT 可以直接做全年静态统计；
- 哪些 ST / Break 本质上属于短期状态，不适合进入冻结年度画像；
- 是否需要引入季节条件画像或市场状态条件画像。

---

# 7. 第一阶段 B：先确定如何低维表示报价曲线

这一部分是新的关键任务。

此前很多预测实验直接在模板、PCA、latent 或参数化上做预测，但现在需要先独立回答：

$$
\boxed{
\text{一条报价曲线最少需要哪些低维、可解释变量才能较好还原？}
}
$$

这里暂时**不做预测**。

只验证：

$$
\text{原始报价曲线}
\rightarrow
\theta
\rightarrow
\text{重构报价曲线}
$$

## 7.1 推荐的总体分解

报价曲线优先拆为三部分：

$$
\boxed{
\text{价格尺度}
+
\text{数量尺度}
+
\text{归一化形状}
}
$$

### 价格尺度

候选：

$$
p_{\min},\quad p_{\max}
$$

或者：

$$
p_{\min},\quad p_{span}=p_{\max}-p_{\min}
$$

### 数量尺度

候选：

$$
q_{\min},\quad q_{\max}
$$

或者：

$$
q_{anchor},\quad q_{span}
$$

### 归一化形状

可基于归一化横轴：

$$
x=\frac{q-q_{\min}}{q_{\max}-q_{\min}}
$$

定义：

$$
y(x)=
\frac{p(x)-p_{\min}}
{p_{\max}-p_{\min}}
$$

然后尝试少量可解释控制点，例如：

$$
y(0.25),\ y(0.50),\ y(0.75)
$$

形成例如：

$$
\theta=
[
p_{\min},
p_{\max},
q_{\min},
q_{\max},
y_{25},
y_{50},
y_{75}
]
$$

这是 7 维连续表示。

也可以比较：

- 5 维；
- 7 维；
- 9 维；
- 其他少量控制点方案。

## 7.2 表示能力验收

先独立评价：

$$
\theta
\rightarrow
Curve
$$

必须至少包含：

- price MAE；
- RMSE；
- WAPE；
- sMAPE；
- breakpoint / segment midpoint price error；
- normalized shape error；
- quantity range error。

目标不是马上定死 7 维，而是：

$$
\boxed{
\text{找到最低维、足够准确、可解释的连续曲线表示}
}
$$

这一表示一旦确定，后续画像和预测统一围绕它展开。

---

# 8. 第一阶段 C：画像表达能力验证

低维表示确定之后，`bidprofile` 再做：

$$
\boxed{
Profile_{2025}
+
Market_t
+
Unit_t
+
Calendar_t
\rightarrow
\theta_t
}
$$

其中：

- `Profile_{2025}`：由 2025 全年报价构造的一次性冻结主体画像；
- `Market_t`：当前时点的市场供需状态；
- `Unit_t`：机组静态/准静态属性；
- `Calendar_t`：月份、日内时段、季节位置等；
- 不输入任何近期报价历史。

严格禁止：

- lag1 bid；
- previous curve；
- 7d / 30d recent bid；
- previous latent；
- previous template；
- 当前报价产生的任何泄漏特征。

---

# 9. 年度静态画像表达能力实验的性质

由于目前只有 2025 全年数据，无法真正完成：

$$
2025\ Profile
\rightarrow
2026\ Prediction
$$

所以当前可以先做一个“画像信息上限验证”。

方法：

$$
2025\text{全年报价}
\rightarrow
Profile_{2025}
$$

然后冻结画像，再用：

$$
Profile_{2025}
+
Market_{2025,t}
+
Unit_{2025,t}
+
Calendar_{2025,t}
\rightarrow
\theta_{2025,t}
$$

这个实验不能叫真实未来泛化测试。

因为 `Profile_{2025}` 使用了全年报价，其中包括当年后续时段的信息。

正确定位是：

$$
\boxed{
\text{年度静态主体画像的解释能力测试}
}
$$

它专门回答：

> 如果把一个主体整年的报价行为压缩成画像，这个画像是否还保留了足够多的报价信息？

---

# 10. 第一阶段的强制验收逻辑

例如首先预测：

$$
p_{\min},\quad p_{\max}
$$

如果在这种对画像非常有利的静态年度画像实验中仍只能达到：

$$
WAPE \approx 15\%-20\%
$$

甚至更差，则说明：

$$
\boxed{\text{当前画像定义本身不够}}
$$

此时不应该继续调 RF、GAM、HGBR 或其他模型。

应该直接返回画像设计。

如果：

$$
p_{\min},p_{\max}
$$

能够达到明显更低的误差，再继续验证：

$$
q_{\min},q_{\max}
$$

然后验证：

$$
shape_1,shape_2,\ldots
$$

最终验证画像是否足以解释完整低维报价参数：

$$
\theta
$$

---

# 11. 第二阶段重新定位：`bidtemplate`

`bidtemplate` 不再承担“必须依靠模板重构未来报价曲线”的责任。

其核心定位改为：

$$
\boxed{
\text{报价行为的解释性离散模式}
}
$$

主要回答两个问题。

---

# 12. 第二阶段 A：报价模板聚类

已有工作：

- 2025 Raw：11,498,525
- Written：5,119,107
- Shape：4,547,836（88.84%）
- Flat：571,271（11.16%）
- 最终 13 类归一化报价模板。

这部分继续保留。

其目的不是“模板必须进入最终预测”，而是回答：

> 报价曲线是否存在稳定、可解释的典型行为模式？

---

# 13. 第二阶段 B：模板分类可分性

需要验证：

$$
Curve
\rightarrow
Template
$$

是否能形成：

- 稳定类别；
- 有足够类间差异；
- 类内相似；
- 跨主体/跨月份仍具有解释意义。

如果模板本身过度依赖局部数据划分，则应重新聚类或减少模板数量。

---

# 14. 第二阶段 C：画像能否解释模板选择

进一步验证：

$$
\boxed{
Profile
+
Market
+
Unit
+
Calendar
\rightarrow
Template
}
$$

这里的目的不是最终预测，而是进一步验证：

> 用户画像是否能够解释主体在不同市场条件下倾向选择哪类报价行为模式。

因此 `bidtemplate` 可以提供画像有效性的另一个角度：

- 连续角度：画像能否解释 $\theta$；
- 离散角度：画像能否解释 Template。

---

# 15. 模板与最终曲线重构彻底解耦

以后不再默认：

$$
Template+\theta
\rightarrow
Curve
$$

是唯一最终路线。

最终预测完全可以是：

$$
Profile
+
Market
+
Unit
+
Calendar
\rightarrow
\theta
\rightarrow
Curve
$$

而 Template 只作为：

- 行为解释标签；
- 聚类分析结果；
- 策略类型展示；
- 画像有效性的辅助验证。

---

# 16. 第三阶段重新定位：`bidprediction`

`bidprediction` 只负责：

$$
\boxed{
\text{真正的未来报价预测}
}
$$

它不再负责证明画像是否有效，也不再负责确定曲线表示。

进入 `bidprediction` 之前必须满足：

1. 年度静态主体画像已经确定；
2. 低维连续曲线表示已经确定；
3. 画像对主要曲线参数的解释能力已通过；
4. Template 若保留，仅作为辅助解释，不是强制主线。

---

# 17. 最终未来预测形式

有 2026 数据之后：

$$
2025\text{全年报价}
\rightarrow
Profile_{2025}
$$

冻结。

然后：

$$
Profile_{2025}
+
Market_{2026,t}
+
Unit_{2026,t}
+
Calendar_{2026,t}
\rightarrow
\hat{\theta}_{2026,t}
$$

再：

$$
\hat{\theta}_{2026,t}
\rightarrow
\hat{Curve}_{2026,t}
$$

此时才做真正的：

- 跨年度验证；
- 未来泛化；
- 未见市场状态测试；
- 完整报价曲线误差评价。

---

# 18. 现有预测探索实验的处理

此前 `bidprediction` 中完成了大量 04～11 系列探索。

这些实验已经证明：

- Template 分类本身并非唯一瓶颈；
- PCA / latent 表示能力可以很好，但预测 latent 很差；
- 完整曲线直接预测存在明显纵向价格偏差；
- 固定时间切分对后期预测误差较大；
- 月度滚动训练能改善结果；
- 大部分误差集中于少量报价边界突变样本。

这些结果仍有研究价值，但不应继续作为最终技术主线。

建议：

```text
scripts/bidprediction/archive/
```

用于归档此前探索代码和结果。

---

# 19. 已有 11 系列实验及结论

## 19.1 11：简单上下界预测

$$
83\ features
\rightarrow
RF
\rightarrow
(p_{\min},p_{\max})
$$

固定时间切分 TEST：

- $p_{\min}$ WAPE = 26.31%
- $p_{\max}$ WAPE = 21.02%

当前结论：

$$
\boxed{\text{精度不足}}
$$

---

## 19.2 11b：失败原因诊断

主要结果：

`lt_bid_level`

- RF importance = 0.370
- Spearman($p_{\min}$) = 0.871
- Spearman($p_{\max}$) = 0.660

说明画像里已经有明显价格水平信息。

但：

- TEST 后期价格分布明显上移；
- Top 10% 样本贡献约一半总绝对误差。

---

## 19.3 11c：月度滚动验证

结果：

| Month | $p_{\min}$ WAPE | $p_{\max}$ WAPE |
|---|---:|---:|
| Jul | 20.23% | 16.60% |
| Aug | 18.19% | 13.94% |
| Sep | 12.25% | 9.71% |
| Oct | 14.16% | 12.65% |
| Nov | 15.20% | 11.41% |
| Dec | 19.55% | 16.21% |

说明：

> 固定一次训练到后期预测确实会低估模型能力，但滚动更新不符合最终“完整历史年画像 → 未来预测”的严格部署场景，因此该实验仅作为诊断。

---

## 19.4 11d：报价突变诊断

Top 10% 高误差样本贡献约：

$$
53\%\sim63\%
$$

总绝对误差。

例如 Dec $p_{\max}$：

- Q00-50：WAPE = 11.31%
- Q90-95：28.50%
- Q95-99：41.80%
- Q99-100：72.46%

说明：

$$
\boxed{
\text{现有模型主要失败于少量大幅报价边界切换样本}
}
$$

但由于最终预测不能依赖未来年份近期报价，这个发现只能用于理解难点，不能通过直接加入 recent bid 来解决。

---

## 19.5 11e

曾生成：

`11e_compare_83_vs_recent_price_state.py`

该实验加入：

- lag1 $p_{\min},p_{\max}$；
- 7d / 30d 报价统计；
- short-long price shift。

但这条路线当前正式停止。

原因：

$$
\boxed{
\text{它不符合最终部署时“未来年份没有近期报价输入”的约束}
}
$$

代码可保留为诊断实验，不进入正式 pipeline。

---

# 20. 推荐的新目录结构

建议逐步整理为：

```text
scripts/
│
├─ bidprofile/
│  │
│  ├─ 01_build_daily_strategy_core.py
│  ├─ 02_build_annual_strategy_profile.py
│  ├─ 03_validate_annual_strategy_profile.py
│  │
│  ├─ 04_build_curve_representation.py
│  ├─ 05_validate_curve_representation.py
│  │
│  ├─ 06_build_profile_explanation_dataset.py
│  ├─ 07_predict_price_bounds_from_profile.py
│  ├─ 08_predict_quantity_bounds_from_profile.py
│  ├─ 09_predict_shape_from_profile.py
│  └─ 10_evaluate_profile_explanatory_power.py
│
├─ bidtemplate/
│  │
│  ├─ 01_build_curve_samples.py
│  ├─ 02_cluster_curve_templates.py
│  ├─ 03_validate_curve_templates.py
│  ├─ 04_profile_template_association.py
│  └─ 05_predict_template_from_profile.py
│
└─ bidprediction/
   │
   ├─ archive/
   │  └─ 旧 04～11 预测探索实验
   │
   ├─ 01_build_future_prediction_dataset.py
   ├─ 02_train_curve_parameter_predictor.py
   ├─ 03_reconstruct_bid_curve.py
   └─ 04_evaluate_future_prediction.py
```

注意：

> 这是目标结构，不要求一次性重命名所有旧代码。优先开发新的 `bidprofile` 逻辑，旧代码先归档不删除。

---

# 21. 最新开发规划

## Phase A：冻结当前预测探索

立即停止：

- 完整曲线继续调参；
- Template + theta 路线继续优化；
- recent bid / lag1 bid 作为最终输入；
- Prediction Family 等分支；
- 在现有 83 特征上无目的更换模型。

---

# 22. Phase B：重构年度静态主体画像

第一步：

$$
\boxed{
2025\text{全年报价}
\rightarrow
\text{每个 participant 一个年度静态画像}
}
$$

先尽量复用现有 9 LT 指标，但重新定义统计范围。

需要逐项判断：

- 是否适合全年统计；
- 是否应该保留；
- 是否受季节变化严重影响；
- 是否需要条件化。

输出建议：

`annual_strategy_profile_2025.parquet`

每个 participant 一行。

---

# 23. Phase C：重新确定低维连续曲线表示

不要直接继承旧 theta。

先做纯表示实验：

$$
Curve
\rightarrow
\theta_K
\rightarrow
Curve
$$

比较：

- K=5
- K=7
- K=9

或其他少量合理候选。

必须坚持：

- 低维；
- 可解释；
- 可直接重构；
- 无模板依赖；
- oracle reconstruction error 足够低。

先选表示，再谈预测。

---

# 24. Phase D：年度画像解释价格上下界

冻结年度画像后：

$$
Profile_{2025}
+
Market_t
+
Unit_t
+
Calendar_t
\rightarrow
(p_{\min},p_{\max})
$$

模型先只用：

- Linear / Ridge；
- Decision Tree；
- Random Forest；
- 小型 HGBR 或 GAM 如确有必要。

当前优先 RF 作为基准。

禁止任何报价历史输入。

这一步只回答：

$$
\boxed{
\text{画像能不能解释价格尺度}
}
$$

---

# 25. Phase E：年度画像解释数量尺度

价格上下界通过后，再做：

$$
Profile
+
Market
+
Unit
+
Calendar
\rightarrow
(q_{\min},q_{\max})
$$

或最终选定的数量尺度参数。

---

# 26. Phase F：年度画像解释曲线形状

价格和数量尺度都通过后，再做：

$$
Profile
+
Market
+
Unit
+
Calendar
\rightarrow
shape
$$

如果形状无法解释：

- 先判断画像是否缺少结构信息；
- 再判断市场特征是否不足；
- 不要直接换复杂模型。

---

# 27. Phase G：画像总体表达能力验收

最终把所有低维参数拼起来：

$$
\hat{\theta}
\rightarrow
\hat{Curve}
$$

评价：

- price MAE；
- RMSE；
- WAPE；
- sMAPE；
- breakpoint error；
- segment midpoint error；
- quantity range error；
- normalized shape error。

此时仍然属于：

$$
\boxed{
\text{画像表达能力验证}
}
$$

不是未来预测。

---

# 28. Phase H：模板解释性验证

独立进行：

$$
Profile
+
Market
+
Unit
+
Calendar
\rightarrow
Template
$$

主要用于验证：

- 模板是否稳定；
- 画像是否能解释模板选择；
- 不同主体是否存在稳定行为倾向。

Template 不强制进入未来预测。

---

# 29. Phase I：真正未来预测

只有获得 2026 或其他未来年度数据后：

$$
2025\ Profile
\rightarrow
2026\ Prediction
$$

此时：

- 画像固定；
- 不读取 2026 已实现报价；
- 使用真实未来可获得外生特征；
- 做完整跨年度泛化评价。

---

# 30. 当前最优先的三个开发任务

按照顺序，只做以下三件事。

## Task 1

重新生成：

$$
\boxed{
2025\text{全年静态主体画像}
}
$$

不要先动预测模型。

## Task 2

重新验证：

$$
\boxed{
\text{最低维、可解释的连续报价曲线表示}
}
$$

先看表示误差，不预测。

## Task 3

在前两者确定后，做：

$$
\boxed{
Annual\ Profile
+
Market
+
Unit
+
Calendar
\rightarrow
(p_{\min},p_{\max})
}
$$

这是下一次真正意义上的模型实验。

---

# 31. 当前项目的强制开发原则

后续必须坚持：

### 原则 1：一次只解决一个问题

不要同时修改：

- 画像；
- 表示方式；
- 模型；
- 数据切分。

否则无法判断改进来自哪里。

### 原则 2：优先验证输入信息，而不是更换模型

如果 RF 都无法解释基本的价格上下界，优先怀疑：

- 画像缺信息；
- 外生特征缺信息；
- 表示目标不合理。

不要先换复杂模型。

### 原则 3：真实部署约束优先

任何需要未来年份近期报价的数据，都不能成为最终必要输入。

### 原则 4：模板不是必须预测路径

模板服务于解释，不再绑定曲线重构。

### 原则 5：完整曲线必须最后做

顺序固定：

$$
\boxed{
\text{价格尺度}
\rightarrow
\text{数量尺度}
\rightarrow
\text{形状}
\rightarrow
\text{完整曲线}
}
$$

---

# 32. 当前阶段最终结论

项目已经从：

$$
\text{直接预测完整报价曲线}
$$

重新调整为：

$$
\boxed{
\text{先验证主体画像是否真的包含足够报价信息}
}
$$

新的三个目录职责为：

$$
\boxed{
\texttt{bidprofile}
=
\text{画像构建 + 曲线低维表示 + 画像解释能力验证}
}
$$

$$
\boxed{
\texttt{bidtemplate}
=
\text{报价模板聚类 + 模板解释性验证}
}
$$

$$
\boxed{
\texttt{bidprediction}
=
\text{使用已经验证有效的画像进行真正未来报价预测}
}
$$

下一阶段不应继续修改 `bidprediction`。

应正式返回：

$$
\boxed{\texttt{scripts/bidprofile}}
$$

首先完成：

$$
2025\text{全年报价}
\rightarrow
\text{年度静态主体画像}
$$

以及：

$$
Curve
\rightarrow
\text{最低维连续可解释表示}
$$

然后再进行：

$$
Profile
+
Market
+
Unit
+
Calendar
\rightarrow
(p_{\min},p_{\max})
$$

专门回答整个项目当前最关键的问题：

$$
\boxed{
\text{我们设计的“用户画像”到底有没有足够的信息量来解释报价？}
}
$$

如果这一关失败：

$$
\boxed{\text{修改画像，不继续未来预测}}
$$

如果这一关通过：

$$
\boxed{\text{才进入真正的 bidprediction 阶段}}
$$

---

# 33. 一句话交接

后续开发者不要继续优化旧的完整曲线预测 pipeline。

当前唯一主线是：

> **在 `bidprofile` 中，用 2025 全年报价构造冻结年度主体画像，同时确定一个低维、可解释、能够准确还原报价曲线的连续表示；然后在不输入任何近期报价历史的条件下，验证该画像结合市场、机组和日历特征能否解释价格尺度、数量尺度和曲线形状。只有画像表达能力通过后，才进入真正的未来报价预测。**
