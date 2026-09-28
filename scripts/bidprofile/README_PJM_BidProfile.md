# MarketBidPrediction / bidprofile

## 1. 模块定位

`scripts/bidprofile` 用于从一个完整历史参考周期的报价行为中构建**主体多时间尺度策略画像**。其核心目标不是直接利用历史报价曲线预测下一条报价，而是先把历史行为压缩为低维、可解释、可复用的主体策略先验。

当前主体级策略画像定义为：

$$
\boxed{
Z_i=
\left[
Z_i^{LT},
Z_i^{STH},
Z_i^{SEA},
Z_i^{ID},
Z_i^{TR}
\right]
}
$$

其中：

| 画像组成 | 符号 | 当前维数 | 含义 |
|---|---:|---:|---|
| 长期稳定策略 | $Z_i^{LT}$ | 9 | 主体在完整历史周期内长期如何报价 |
| 历史短期策略倾向 | $Z_i^{STH}$ | 8 | 主体历史上出现短周期偏移、异常和形态迁移的典型程度 |
| 季节性策略摘要 | $Z_i^{SEA}$ | 6 | 主体不同月份相对全年习惯的变化幅度 |
| 日内策略摘要 | $Z_i^{ID}$ | 5 | 主体不同交易时段相对全年习惯的变化幅度及日内切换频率 |
| 策略转换特征 | $Z_i^{TR}$ | 5 | 历史报价模板偏好、惯性、切换频率和状态转移复杂度 |

当历史模板 assignment 已生成时，主体级默认画像维数为：

$$
\boxed{
9+8+6+5+5=33
}
$$

这 33 维是当前工程实现的基础候选集，不表示后续预测阶段必须永久固定为 33 维。后续仍应根据稳定性、缺失率、冗余性、模板区分能力和最终曲线预测增量进行筛选。

除主体级 $Z_i$ 外，还保留两个**历史条件画像表**：

$$
\boxed{
Z^{SEA}_{i,m}
}
$$

表示主体 $i$ 在历史月份 $m$ 的典型季节性偏移；

$$
\boxed{
Z^{ID}_{i,h}
}
$$

表示主体 $i$ 在历史交易时段 $h$ 的典型日内偏移。

未来预测某个时段 $t$ 时，仅根据目标时点已知的月份和交易时段从历史表中查找对应画像，不要求使用预测年度已经发生的近期报价：

$$
\boxed{
X_{i,t}
=
\left[
Z_i,\,
Z^{SEA}_{i,m(t)},\,
Z^{ID}_{i,h(t)},\,
M_t,\,
U_{i,t},\,
C_t
\right]
}
$$

其中 $M_t$ 为市场环境，$U_{i,t}$ 为主体物理状态，$C_t$ 为日历特征。

---

# 2. 当前代码流程

```text
历史 PJM 报价
    │
    ▼
01_build_daily_strategy_core.py
    │  时段级报价行为原子
    │  日级聚合
    │  participant × slot 历史统计
    ▼
02_build_long_term_strategy_profile.py
    │  9 个长期稳定策略特征
    ▼
03_build_historical_short_term_tendency.py
    │  历史年度内部 recent7d vs earlier60d
    │  汇总为 8 个主体级短期策略倾向
    ▼
04_build_seasonal_intraday_strategy_profile.py
    │  participant × month 季节条件画像
    │  participant × slot 日内条件画像
    │  6 个季节摘要 + 5 个日内摘要
    ▼
bidtemplate
    │  历史报价 → template_id
    ▼
05_build_strategy_transition_profile.py
    │  5 个模板/策略转换特征
    ▼
06_build_final_strategy_profile.py
    │  合并为最终主体级 Z_i
    ▼
07_validate_strategy_profile.py
    │  稳定性 / 覆盖率 / 冗余性
    ▼
08_visualize_strategy_profile.py
```

当前主脚本：

```text
scripts/bidprofile/
├─ 01_build_daily_strategy_core.py
├─ 02_build_long_term_strategy_profile.py
├─ 03_build_historical_short_term_tendency.py
├─ 04_build_seasonal_intraday_strategy_profile.py
├─ 05_build_strategy_transition_profile.py
├─ 06_build_final_strategy_profile.py
├─ 07_validate_strategy_profile.py
├─ 08_visualize_strategy_profile.py
├─ strategy_core.py
└─ README_PJM_BidProfile.md
```

---

# 3. 统一符号

设主体为 $i$，历史日期为 $d$，日内报价时段为 $t$。

某时段清洗后的报价点为：

$$
(q_{i,t,1},p_{i,t,1}),
\ldots,
(q_{i,t,m},p_{i,t,m})
$$

并满足：

$$
q_{i,t,1}<q_{i,t,2}<\cdots<q_{i,t,m}
$$

数量轴归一化为：

$$
x_j
=
\frac{
q_j-q_1
}{
q_m-q_1
},
\qquad
x_j\in[0,1]
$$

定义：

$$
\Delta x_j=x_{j+1}-x_j
$$

固定 shape 网格：

$$
\mathcal G
=
\{0,0.05,0.10,\ldots,1.00\}
$$

历史参考年度的有效日期集合记为：

$$
\mathcal D_i
$$

主体 $i$ 在月份 $m$ 的有效日期集合记为：

$$
\mathcal D_{i,m}
$$

主体 $i$ 在本地交易时段 $h$ 的历史报价集合记为：

$$
\mathcal T_{i,h}
$$

---

# 4. 01：时段级策略原子

脚本：

```text
01_build_daily_strategy_core.py
```

## 4.1 报价点清洗

当前技术验证采用：

| 规则 | 实现 |
|---|---|
| 有效报价点 | MW 与 BID 同时为有限值 |
| MW sentinel | 只有明确知道数据源 sentinel 时才通过 `--mw-sentinel` 删除 |
| BID = 0 | 保留 |
| BID < 0 | 保留 |
| MW 排序 | 升序 |
| 重复 MW | 同一 MW 下保留最大 BID |
| 最小有效点数 | 至少 2 个不同 MW 点 |
| 数量跨度 | $q_m-q_1>0$ |
| 缺失时段 | 保持缺失，不填 0、不插值 |

当前：

$$
m<2
$$

或：

$$
q_m-q_1\le 10^{-12}
$$

时，该时段不进入策略特征统计。

---

## 4.2 `bid_level`：整体报价水平

### Sloped 曲线

$$
\boxed{
L_{i,t}
=
\sum_{j=1}^{m-1}
\frac{
p_j+p_{j+1}
}{2}
\Delta x_j
}
$$

### Block 曲线

$$
\boxed{
L_{i,t}
=
\sum_{j=1}^{m-1}
p_j\Delta x_j
}
$$

该指标反映完整报价曲线在归一化容量轴上的整体价格水平。

字段：

```text
bid_level
```

---

## 4.3 `adjustment_bias` 与 `adjustment_magnitude`

先建立主体同一日内时槽的历史基准。

设当前时槽为 $h(t)$，最近最多 $W=30$ 条同槽历史有效报价水平为：

$$
\mathcal H_{i,t}^{slot}
=
\{
L_{i,\tau}:
\tau<t,\,
h(\tau)=h(t)
\}
$$

至少有 5 条历史记录时：

$$
B_{i,t}^{slot}
=
\operatorname{Median}
\left(
\mathcal H_{i,t}^{slot}
\right)
$$

则：

$$
\boxed{
A^{bias}_{i,t}
=
L_{i,t}
-
B_{i,t}^{slot}
}
$$

$$
\boxed{
A^{mag}_{i,t}
=
\left|
A^{bias}_{i,t}
\right|
}
$$

对应字段：

```text
adjustment_bias
adjustment_magnitude
```

---

## 4.4 `quantity_hhi`：容量区间集中度

$$
\boxed{
HHI^Q_{i,t}
=
\sum_{j=1}^{m-1}
(\Delta x_j)^2
}
$$

若容量主要集中在少数大区间，$HHI^Q$ 较高；若容量在多个区间分布较均匀，$HHI^Q$ 较低。

字段：

```text
quantity_hhi
```

---

## 4.5 `effective_segment_count`：有效价格段数

当前 01 中按价格实际变化次数定义：

$$
\boxed{
K^{eff}_{i,t}
=
1
+
\sum_{j=1}^{m-1}
\mathbf{1}
\left(
|p_{j+1}-p_j|>10^{-9}
\right)
}
$$

字段：

```text
effective_segment_count
```

说明：这里是 `bidprofile` 的历史行为原子，不等同于后续 `bidtemplate` 中按照新技术路线进行“近价段合并 + 微小容量段合并”后的最终模板有效阶梯数。

---

## 4.6 `flat_curve_flag`：平价报价标志

当前不再仅比较首尾价格，而是使用整条曲线价格极差：

$$
\boxed{
F_{i,t}
=
\mathbf{1}
\left[
\max_j p_j-\min_j p_j
\le 10^{-9}
\right]
}
$$

字段：

```text
flat_curve_flag
```

---

## 4.7 `tail_uplift_ratio`：尾部抬价程度

在固定网格 $\mathcal G$ 上得到报价函数 $P(x)$。

定义：

$$
P_0=P(0)
$$

$$
P_{0.8}=P(0.8)
$$

$$
P_1=P(1)
$$

则：

$$
\boxed{
T_{i,t}
=
\frac{
P_1-P_{0.8}
}{
|P_1-P_0|+\varepsilon
}
}
$$

其中：

$$
\varepsilon=10^{-12}
$$

字段：

```text
tail_uplift_ratio
```

---

## 4.8 `curve_bend_ratio`：曲线弯折程度

令：

$$
P_{0.2}=P(0.2)
$$

头部价格变化：

$$
\Delta P^{head}
=
P_{0.2}-P_0
$$

尾部价格变化：

$$
\Delta P^{tail}
=
P_1-P_{0.8}
$$

则：

$$
\boxed{
C_{i,t}
=
\frac{
\Delta P^{tail}
-
\Delta P^{head}
}{
|P_1-P_0|+\varepsilon
}
}
$$

字段：

```text
curve_bend_ratio
```

---

## 4.9 `shape_v00 ~ shape_v20`：归一化相对形态

非平价且首尾价格不相等时：

$$
\boxed{
V_{i,t}(u)
=
\frac{
P_{i,t}(u)-P_{i,t}(0)
}{
P_{i,t}(1)-P_{i,t}(0)
}
}
$$

其中：

$$
u\in\mathcal G
$$

得到：

```text
shape_v00
...
shape_v20
```

若整条曲线 flat，则 shape 不定义；若曲线非 flat 但首尾价格相等，当前端点归一化 shape 同样不定义。

---

# 5. 01：日级策略特征

对每个：

```text
participant_id × local_date
```

进行日级聚合。

## 5.1 日级基础特征

| 日级字段 | 公式 / 聚合方式 |
|---|---|
| `daily_bid_level` | $\operatorname{Median}_t(L_{i,t})$ |
| `daily_adjustment_bias` | $\operatorname{Median}_t(A^{bias}_{i,t})$ |
| `daily_adjustment_magnitude` | $\operatorname{Median}_t(A^{mag}_{i,t})$ |
| `daily_quantity_hhi` | $\operatorname{Median}_t(HHI^Q_{i,t})$ |
| `daily_effective_segment_count` | $\operatorname{Median}_t(K^{eff}_{i,t})$ |
| `daily_flat_curve_rate` | $\operatorname{Mean}_t(F_{i,t})$ |
| `daily_tail_uplift_ratio` | $\operatorname{Median}_t(T_{i,t})$ |
| `daily_curve_bend_ratio` | $\operatorname{Median}_t(C_{i,t})$ |
| `daily_shape_v00~20` | 每个 shape 坐标分别取日内中位数 |

此外保留：

```text
daily_valid_interval_count
daily_shape_defined_interval_count
```

作为数据质量字段。

---

## 5.2 `daily_curve_type_count`：日内不同报价曲线数量

当前 01 使用清洗后的完整：

```text
curve_mode + q sequence + p sequence
```

生成精确 `curve_key`。

一天内有效报价曲线 key 集合为：

$$
\mathcal K_{i,d}
$$

则：

$$
\boxed{
K^{curve}_{i,d}
=
|\mathcal K_{i,d}|
}
$$

字段：

```text
daily_curve_type_count
```

注意：该 key 仅用于 `bidprofile` 的日内行为原子，不等同于 `bidtemplate` 的报价策略模板 ID。

---

## 5.3 `daily_curve_switch_count`：日内曲线切换次数

按时间排序后的有效曲线 key 为：

$$
k_1,k_2,\ldots,k_N
$$

则：

$$
\boxed{
S_{i,d}
=
\sum_{t=2}^{N}
\mathbf{1}
(
k_t\ne k_{t-1}
)
}
$$

字段：

```text
daily_curve_switch_count
```

---

## 5.4 `daily_curve_switch_rate`：日内策略切换率

$$
\boxed{
R^{switch}_{i,d}
=
\frac{
S_{i,d}
}{
N-1
}
}
$$

仅当：

$$
N\ge2
$$

时定义。

字段：

```text
daily_curve_switch_rate
```

---

## 5.5 `daily_intraday_price_range`：日内价格跨度

对当天所有不同 `curve_key`，分别取：

- 最大报价价格 $p^{max}$；
- 最小报价价格 $p^{min}$；
- 整体报价水平 $L$。

定义：

$$
R^{max}_{i,d}
=
\max_c p^{max}_c
-
\min_c p^{max}_c
$$

$$
R^{min}_{i,d}
=
\max_c p^{min}_c
-
\min_c p^{min}_c
$$

$$
R^{level}_{i,d}
=
\max_c L_c
-
\min_c L_c
$$

最终：

$$
\boxed{
R^{price}_{i,d}
=
\frac{
R^{max}_{i,d}
+
R^{min}_{i,d}
+
R^{level}_{i,d}
}{
3
}
}
$$

字段：

```text
daily_intraday_price_range
```

若当天只有一种曲线：

$$
R^{price}_{i,d}=0
$$

---

# 6. 01：历史 `participant × slot` 原子

01 同时建立：

```text
intraday_slot_core_<year>.csv
```

对：

```text
participant_id × local_slot_seconds
```

累积历史均值。

对任一时段行为量 $f_{i,t}$：

$$
\boxed{
\bar f^{slot}_{i,h}
=
\frac{
1
}{
N_{i,h}
}
\sum_{t\in\mathcal T_{i,h}}
f_{i,t}
}
$$

当前保存：

| 字段 | 定义 |
|---|---|
| `slot_bid_level_mean` | $\operatorname{Mean}_{t\in\mathcal T_{i,h}} L_{i,t}$ |
| `slot_adjustment_magnitude_mean` | $\operatorname{Mean} A^{mag}_{i,t}$ |
| `slot_quantity_hhi_mean` | $\operatorname{Mean} HHI^Q_{i,t}$ |
| `slot_effective_segment_count_mean` | $\operatorname{Mean} K^{eff}_{i,t}$ |
| `slot_flat_curve_flag_mean` | $\operatorname{Mean} F_{i,t}$ |
| `slot_tail_uplift_ratio_mean` | $\operatorname{Mean} T_{i,t}$ |
| `slot_curve_bend_ratio_mean` | $\operatorname{Mean} C_{i,t}$ |

同时保存每个字段对应的：

```text
slot_<feature>_count
```

以及：

```text
slot_observation_count
```

---

# 7. 02：9 个长期稳定策略特征

长期画像基于完整历史参考周期 $\mathcal D_i$。

## 7.1 正式 LT 特征表

| 序号 | 字段 | 数学定义 | 策略含义 |
|---:|---|---|---|
| 1 | `lt_bid_level` | $\operatorname{Median}_{d\in\mathcal D_i}(daily\_bid\_level_{i,d})$ | 长期整体报价水平 |
| 2 | `lt_adjustment_magnitude` | $\operatorname{Median}_{d}(daily\_adjustment\_magnitude_{i,d})$ | 长期主动偏离自身历史习惯的幅度 |
| 3 | `lt_strategy_persistence` | 见 7.2 | 历史报价调整方向/幅度在连续日间的持续性 |
| 4 | `lt_quantity_hhi` | $\operatorname{Median}_{d}(daily\_quantity\_hhi_{i,d})$ | 长期容量区间集中程度 |
| 5 | `lt_effective_segment_count` | $\operatorname{Median}_{d}(daily\_effective\_segment\_count_{i,d})$ | 长期报价结构复杂度 |
| 6 | `lt_flat_curve_rate` | $\operatorname{Median}_{d}(daily\_flat\_curve\_rate_{i,d})$ | 长期采用平价/近似平价结构的倾向 |
| 7 | `lt_tail_uplift_ratio` | $\operatorname{Median}_{d}(daily\_tail\_uplift\_ratio_{i,d})$ | 长期尾部抬价倾向 |
| 8 | `lt_curve_bend_ratio` | $\operatorname{Median}_{d}(daily\_curve\_bend\_ratio_{i,d})$ | 长期前后段价格变化的不对称程度 |
| 9 | `lt_shape_variability` | 见 7.3 | 长期报价相对形态的稳定/波动程度 |

---

## 7.2 `lt_strategy_persistence`

令日级调整偏差为：

$$
a_{i,d}
=
daily\_adjustment\_bias_{i,d}
$$

只保留真实连续日：

$$
d_j-d_{j-1}=1
$$

构造：

$$
\mathbf{a}^{-}
=
\left[a_{i,d_1},\ldots,a_{i,d_{n-1}}\right]
$$

$$
\mathbf{a}^{+}
=
\left[a_{i,d_2},\ldots,a_{i,d_n}\right]
$$

则：

$$
\boxed{
lt\_strategy\_persistence_i
=
\operatorname{Corr}_{\mathrm{Pearson}}
(
\mathbf{a}^{-},
\mathbf{a}^{+}
)
}
$$

至少要求 5 对有效连续日。

---

## 7.3 `lt_shape_variability`

先建立主体长期 shape 原型：

$$
\boxed{
V_i^{LT}(u)
=
Median_{d\in\mathcal D_i}
V_{i,d}(u)
}
$$

对每个 shape 完整的历史日：

$$
D^{shape}_{i,d}
=
\sqrt{
\frac{1}{21}
\sum_{u\in\mathcal G}
\left[
V_{i,d}(u)
-
V_i^{LT}(u)
\right]^2
}
$$

最终：

$$
\boxed{
lt\_shape\_variability_i
=
Median_d
D^{shape}_{i,d}
}
$$

同时输出长期 shape 原型：

```text
lt_shape_v00
...
lt_shape_v20
```

这些 21 个坐标是辅助历史形态基准，不计入默认 9 LT 模型维数。

---

# 8. 03：历史短期策略状态原子

新技术路线中，03 的日级 ST **不是未来预测期在线输入**，而是用于从完整历史年度中提炼“主体历史上如何发生短周期策略变化”。

对于历史日期 $d$：

$$
\mathcal H^S(d)
=
[d-7,d-1]
$$

$$
\mathcal H^L(d)
=
[d-67,d-8]
$$

两窗口不重叠。

对日级行为量 $f$：

$$
f^{recent}_{i,d}
=
Median_{\tau\in\mathcal H^S(d)}
f_{i,\tau}
$$

$$
f^{base}_{i,d}
=
Median_{\tau\in\mathcal H^L(d)}
f_{i,\tau}
$$

$$
s^f_{i,d}
=
1.4826\,
\operatorname{MAD}_{\tau\in\mathcal H^L(d)}
(f_{i,\tau})
$$

若：

$$
s^f_{i,d}
>
s_f^{floor}
$$

则：

$$
\boxed{
z^f_{i,d}
=
\operatorname{clip}
\left(
\frac{
f^{recent}_{i,d}
-
f^{base}_{i,d}
}{
s^f_{i,d}
},
-10,
10
\right)
}
$$

最低数据要求：

$$
N_{recent}\ge3,
\qquad
N_{long}\ge20
$$

---

## 8.1 Historical Break

若：

$$
s^f_{i,d}
\le
s_f^{floor}
$$

且：

$$
\left|
f^{recent}_{i,d}
-
f^{base}_{i,d}
\right|
>
s_f^{floor}
$$

则：

$$
\boxed{
B^f_{i,d}=1
}
$$

否则：

$$
B^f_{i,d}=0
$$

当前 practical-zero floor：

| 特征 | $s_f^{floor}$ |
|---|---:|
| `bid_level` | $10^{-3}$ |
| `adjustment_bias` | $10^{-3}$ |
| `adjustment_magnitude` | $10^{-3}$ |
| `quantity_hhi` | $10^{-6}$ |
| `effective_segment_count` | $10^{-6}$ |
| `flat_curve_rate` | $10^{-6}$ |
| `tail_uplift_ratio` | $10^{-6}$ |
| `curve_bend_ratio` | $10^{-6}$ |

---

## 8.2 Historical shape shift

近期 shape 原型：

$$
V^{recent}_{i,d}
=
Median_{\tau\in[d-7,d-1]}
V_{i,\tau}
$$

更早历史 shape 原型：

$$
V^{base}_{i,d}
=
\operatorname{Median}_{\tau\in[d-67,d-8]}
V_{i,\tau}
$$

则：

$$
\boxed{
S^{shape}_{i,d}
=
\operatorname{RMSE}
\left(
V^{recent}_{i,d},
V^{base}_{i,d}
\right)
}
$$

最低要求：

$$
N_{recent}^{shape}\ge2,
\qquad
N_{long}^{shape}\ge10
$$

---

# 9. 03：8 个主体级历史短期策略倾向

日级 historical ST 只用于构造主体级汇总。

| 序号 | 字段 | 数学定义 | 含义 |
|---:|---|---|---|
| 1 | `short_bid_level_abs_z_median` | $\operatorname{Median}_d(|z^{bid}_{i,d}|)$ | 主体典型短期报价水平偏移强度 |
| 2 | `short_bid_level_abs_z_p90` | $Q_{0.90,d}(|z^{bid}_{i,d}|)$ | 主体历史较强报价偏移的上分位程度 |
| 3 | `short_bid_level_high_state_rate` | $\frac{1}{N}\sum_d\mathbf{1}(z^{bid}_{i,d}>1)$ | 历史上进入明显偏高报价状态的频率 |
| 4 | `short_adjustment_magnitude_abs_z_median` | $\operatorname{Median}_d(|z^{adjmag}_{i,d}|)$ | 历史调整幅度的典型短周期偏离 |
| 5 | `short_structure_abs_z_median` | 见下式 | 历史报价结构整体短周期变化强度 |
| 6 | `short_shape_shift_median` | $\operatorname{Median}_d(S^{shape}_{i,d})$ | 历史报价相对形态迁移的典型程度 |
| 7 | `short_break_rate` | $\frac{1}{N_d}\sum_d\mathbf{1}(\sum_fB^f_{i,d}>0)$ | 历史稳定策略被打破的频率 |
| 8 | `short_ready_day_share` | $\frac{1}{N_d}\sum_d Ready_{i,d}$ | 历史短期状态可可靠构建的日期比例 |

其中结构类集合：

$$
\mathcal F_{struct}
=
\{
quantity\_hhi,
segment\_count,
flat,
tail,
bend
\}
$$

先计算每天：

$$
A^{struct}_{i,d}
=
\frac{
1
}{
|\mathcal F_{valid}|
}
\sum_{f\in\mathcal F_{valid}}
|z^f_{i,d}|
$$

再：

$$
\boxed{
short\_structure\_abs\_z\_median_i
=
Median_d
A^{struct}_{i,d}
}
$$

历史日 readiness 定义为：8 个标量 ST 中至少 6 个已被连续 z 或 Break 明确表示。

---

# 10. 04：月份条件策略画像

对每个：

```text
participant_id × month
```

构造历史月份典型值。

最低要求：

$$
N_{i,m}^{active}\ge10
$$

对任一日级行为量 $f$：

$$
\boxed{
f^{month}_{i,m}
=
Median_{d\in\mathcal D_{i,m}}
f_{i,d}
}
$$

并以主体长期 LT 为基准：

$$
\boxed{
\Delta f^{season}_{i,m}
=
f^{month}_{i,m}
-
f_i^{LT}
}
$$

---

## 10.1 月份条件表全部字段

| 字段 | 数学定义 |
|---|---|
| `month_bid_level` | $Median_{d\in\mathcal D_{i,m}} daily\_bid\_level_{i,d}$ |
| `season_bid_level_delta` | $month\_bid\_level-lt\_bid\_level$ |
| `month_adjustment_magnitude` | $Median_d daily\_adjustment\_magnitude$ |
| `season_adjustment_magnitude_delta` | $month\_adjustment\_magnitude-lt\_adjustment\_magnitude$ |
| `month_quantity_hhi` | $Median_d daily\_quantity\_hhi$ |
| `season_quantity_hhi_delta` | $month\_quantity\_hhi-lt\_quantity\_hhi$ |
| `month_effective_segment_count` | $Median_d daily\_effective\_segment\_count$ |
| `season_effective_segment_count_delta` | $month\_effective\_segment\_count-lt\_effective\_segment\_count$ |
| `month_flat_curve_rate` | $Median_d daily\_flat\_curve\_rate$ |
| `season_flat_curve_rate_delta` | $month\_flat\_curve\_rate-lt\_flat\_curve\_rate$ |
| `month_tail_uplift_ratio` | $Median_d daily\_tail\_uplift\_ratio$ |
| `season_tail_uplift_ratio_delta` | $month\_tail\_uplift\_ratio-lt\_tail\_uplift\_ratio$ |
| `month_curve_bend_ratio` | $Median_d daily\_curve\_bend\_ratio$ |
| `season_curve_bend_ratio_delta` | $month\_curve\_bend\_ratio-lt\_curve\_bend\_ratio$ |
| `season_shape_shift` | $\operatorname{RMSE}(V^{month}_{i,m},V_i^{LT})$ |

月份 shape 原型：

$$
V^{month}_{i,m}(u)
=
Median_{d\in\mathcal D_{i,m}}
V_{i,d}(u)
$$

因此：

$$
\boxed{
season\_shape\_shift_{i,m}
=
\sqrt{
\frac1{21}
\sum_u
\left[
V^{month}_{i,m}(u)
-
V_i^{LT}(u)
\right]^2
}
}
$$

---

# 11. 04：6 个主体级季节性摘要

对每个主体，在 12 个月历史条件画像上进一步压缩。

定义某月份变量 $g_{i,m}$ 的跨月范围：

$$
Range_m(g_i)
=
\max_m g_{i,m}
-
\min_m g_{i,m}
$$

当前 6 个正式季节摘要：

| 序号 | 字段 | 数学定义 | 含义 |
|---:|---|---|---|
| 1 | `season_bid_level_range` | $\max_m month\_bid\_level-\min_m month\_bid\_level$ | 报价水平季节跨度 |
| 2 | `season_adjustment_magnitude_range` | $\max_m month\_adjustment\_magnitude-\min_m month\_adjustment\_magnitude$ | 调整幅度季节跨度 |
| 3 | `season_quantity_hhi_range` | $\max_m month\_quantity\_hhi-\min_m month\_quantity\_hhi$ | 容量结构季节跨度 |
| 4 | `season_effective_segment_count_range` | $\max_m month\_segment-\min_m month\_segment$ | 有效段数季节跨度 |
| 5 | `season_tail_uplift_ratio_range` | $\max_m month\_tail-\min_m month\_tail$ | 尾部抬价季节跨度 |
| 6 | `season_shape_shift_median` | $Median_m(season\_shape\_shift_{i,m})$ | 月份形态偏离长期原型的典型程度 |

辅助字段：

```text
season_months_available
season_nonmissing_count
season_ready_flag
```

其中：

$$
season\_ready\_flag
=
\mathbf{1}
(
season\_nonmissing\_count\ge4
)
$$

---

# 12. 04：时段条件策略画像

由 01 的 `participant × local_slot_seconds` 历史均值构建。

对于时段 $h$：

$$
\boxed{
\Delta f^{ID}_{i,h}
=
\bar f^{slot}_{i,h}
-
f_i^{LT}
}
$$

---

## 12.1 时段条件表全部字段

| 字段 | 数学定义 |
|---|---|
| `intraday_bid_level_delta` | $slot\_bid\_level\_mean-lt\_bid\_level$ |
| `intraday_adjustment_magnitude_delta` | $slot\_adjustment\_magnitude\_mean-lt\_adjustment\_magnitude$ |
| `intraday_quantity_hhi_delta` | $slot\_quantity\_hhi\_mean-lt\_quantity\_hhi$ |
| `intraday_effective_segment_count_delta` | $slot\_effective\_segment\_count\_mean-lt\_effective\_segment\_count$ |
| `intraday_flat_curve_rate_delta` | $slot\_flat\_curve\_flag\_mean-lt\_flat\_curve\_rate$ |
| `intraday_tail_uplift_ratio_delta` | $slot\_tail\_uplift\_ratio\_mean-lt\_tail\_uplift\_ratio$ |
| `intraday_curve_bend_ratio_delta` | $slot\_curve\_bend\_ratio\_mean-lt\_curve\_bend\_ratio$ |

这些变量是未来模板分类器最直接的历史日内条件特征。目标时段 $h(t)$ 已知时，直接查：

$$
Z^{ID}_{i,h(t)}
$$

---

# 13. 04：5 个主体级日内摘要

对每个主体的所有历史时段 $h$ 进一步汇总。

定义：

$$
Range_h(g_i)
=
\max_h g_{i,h}
-
\min_h g_{i,h}
$$

当前 5 个正式日内摘要：

| 序号 | 字段 | 数学定义 | 含义 |
|---:|---|---|---|
| 1 | `intraday_bid_level_range` | $\max_h slot\_bid\_level\_mean-\min_h slot\_bid\_level\_mean$ | 不同时段报价水平差异 |
| 2 | `intraday_bid_level_robust_scale` | $1.4826\,\operatorname{MAD}_h(slot\_bid\_level\_mean)$ | 报价水平日内离散程度 |
| 3 | `intraday_adjustment_magnitude_range` | $\max_h slot\_adjustment\_magnitude\_mean-\min_h(...)$ | 调整幅度日内差异 |
| 4 | `intraday_quantity_hhi_range` | $\max_h slot\_quantity\_hhi\_mean-\min_h(...)$ | 容量结构日内差异 |
| 5 | `intraday_switch_rate` | $\operatorname{Median}_d(daily\_curve\_switch\_rate_{i,d})$ | 主体长期典型日内策略切换频率 |

其中：

$$
\operatorname{MAD}(x)
=
\operatorname{Median}
\left(
\left|x-\operatorname{Median}(x)\right|
\right)
$$

辅助字段：

```text
intraday_slots_available
intraday_nonmissing_count
intraday_ready_flag
```

且：

$$
intraday\_ready\_flag
=
\mathbf{1}
(
intraday\_nonmissing\_count\ge4
)
$$

---

# 14. 05：日级模板状态原子

05 从 `bidtemplate` 历史 assignment 读取：

```text
participant_id
local_date
template_id
```

模板 ID 可以是当前的：

```text
T00, T01, ..., FLAT
```

也可以是后续新技术路线下的：

```text
F0, F1, ..., F6
```

05 不依赖模板名称本身。

设主体 $i$ 在日期 $d$ 的各模板出现次数为：

$$
n_{i,d,k}
$$

总模板观测数：

$$
N_{i,d}
=
\sum_k n_{i,d,k}
$$

---

## 14.1 `daily_active_template_count`

$$
\boxed{
K^{temp}_{i,d}
=
\left|
\{
k:n_{i,d,k}>0
\}
\right|
}
$$

字段：

```text
daily_active_template_count
```

---

## 14.2 `daily_dominant_template_id`

$$
\boxed{
k^*_{i,d}
=
\arg\max_k n_{i,d,k}
}
$$

字段：

```text
daily_dominant_template_id
```

---

## 14.3 `daily_dominant_template_share`

$$
\boxed{
D_{i,d}
=
\frac{
\max_k n_{i,d,k}
}{
N_{i,d}
}
}
$$

字段：

```text
daily_dominant_template_share
```

---

## 14.4 `daily_template_entropy`

令当日非零模板概率为：

$$
p_{i,d,k}
=
\frac{
n_{i,d,k}
}{
N_{i,d}
}
$$

非零模板种类数为 $K_{i,d}$。

若：

$$
K_{i,d}\le1
$$

则：

$$
H^{day}_{i,d}=0
$$

否则：

$$
\boxed{
H^{day}_{i,d}
=
-
\frac{
\sum_k
p_{i,d,k}\ln p_{i,d,k}
}{
\ln K_{i,d}
}
}
$$

字段：

```text
daily_template_entropy
```

范围：

$$
H^{day}_{i,d}\in[0,1]
$$

---

# 15. 05：5 个主体级策略转换特征

设完整历史期内模板 $k$ 出现次数为：

$$
n_{i,k}
$$

总模板观测数：

$$
N_i
=
\sum_k n_{i,k}
$$

模板使用概率：

$$
p_{i,k}
=
\frac{
n_{i,k}
}{
N_i
}
$$

---

## 15.1 正式 transition 特征表

| 序号 | 字段 | 数学定义 | 含义 |
|---:|---|---|---|
| 1 | `transition_dominant_template_share` | $\max_k p_{i,k}$ | 主体对最常用模板的长期依赖程度 |
| 2 | `transition_active_template_count` | $|\{k:n_{i,k}>0\}|$ | 历史使用过的模板数量 |
| 3 | `transition_template_usage_entropy` | $H_i^{usage}$ | 全历史模板使用分布复杂度 |
| 4 | `transition_daily_dominant_switch_rate` | $N_i^{switch}/N_i^{pair}$ | 相邻连续日主导模板发生切换的频率 |
| 5 | `transition_pair_entropy` | $H_i^{pair}$ | 历史模板状态转移对的多样性 |

---

## 15.2 `transition_template_usage_entropy`

若主体使用 $K_i$ 个非零模板：

$$
\boxed{
H_i^{usage}
=
-
\frac{
\sum_k p_{i,k}\ln p_{i,k}
}{
\ln K_i
}
}
$$

若：

$$
K_i\le1
$$

则：

$$
H_i^{usage}=0
$$

---

## 15.3 `transition_daily_dominant_switch_rate`

只使用日期严格相邻的主导模板：

$$
d_j-d_{j-1}=1
$$

若：

$$
k^*_{i,d_j}
\ne
k^*_{i,d_{j-1}}
$$

记为一次 switch。

设有效连续日对数量为：

$$
N_i^{pair}
$$

切换次数为：

$$
N_i^{switch}
$$

则：

$$
\boxed{
R_i^{TR}
=
\frac{
N_i^{switch}
}{
N_i^{pair}
}
}
$$

字段：

```text
transition_daily_dominant_switch_rate
```

辅助惯性：

$$
\boxed{
transition\_daily\_dominant\_inertia
=
1-R_i^{TR}
}
$$

---

## 15.4 `transition_pair_entropy`

定义相邻连续日模板状态对：

$$
(k^*_{i,d-1},k^*_{i,d})
$$

每种非零 transition pair 的出现次数记为：

$$
n_{i,a\rightarrow b}
$$

对应概率：

$$
p_{i,a\rightarrow b}
=
\frac{
n_{i,a\rightarrow b}
}{
N_i^{pair}
}
$$

若有效 transition pair 类别数为 $K_i^{pair}>1$：

$$
\boxed{
H_i^{pair}
=
-
\frac{
\sum_{a,b}
p_{i,a\rightarrow b}
\ln
p_{i,a\rightarrow b}
}{
\ln K_i^{pair}
}
}
$$

若只有一种转移对：

$$
H_i^{pair}=0
$$

字段：

```text
transition_pair_entropy
```

---

## 15.5 Transition 辅助字段

| 字段 | 定义 |
|---|---|
| `transition_dominant_template_id` | $\arg\max_k n_{i,k}$ |
| `transition_daily_dominant_inertia` | $1-transition\_daily\_dominant\_switch\_rate$ |
| `transition_daily_template_entropy_median` | $Median_d(H^{day}_{i,d})$ |
| `transition_pair_count` | $N_i^{pair}$ |
| `transition_nonmissing_count` | 5 个正式 transition 特征中非缺失个数 |
| `transition_ready_flag` | $\mathbf{1}(transition\_nonmissing\_count\ge4)$ |

---

# 16. 06：最终主体级 33 维基础画像

当 transition 可用时，最终主体级默认模型特征为：

## 16.1 长期稳定策略：9 维

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

## 16.2 历史短期策略倾向：8 维

```text
short_bid_level_abs_z_median
short_bid_level_abs_z_p90
short_bid_level_high_state_rate
short_adjustment_magnitude_abs_z_median
short_structure_abs_z_median
short_shape_shift_median
short_break_rate
short_ready_day_share
```

## 16.3 季节性摘要：6 维

```text
season_bid_level_range
season_adjustment_magnitude_range
season_quantity_hhi_range
season_effective_segment_count_range
season_tail_uplift_ratio_range
season_shape_shift_median
```

## 16.4 日内摘要：5 维

```text
intraday_bid_level_range
intraday_bid_level_robust_scale
intraday_adjustment_magnitude_range
intraday_quantity_hhi_range
intraday_switch_rate
```

## 16.5 策略转换：5 维

```text
transition_dominant_template_share
transition_active_template_count
transition_template_usage_entropy
transition_daily_dominant_switch_rate
transition_pair_entropy
```

因此：

$$
\boxed{
Z_i\in\mathbb{R}^{33}
}
$$

只是当前基础候选表示。

---

# 17. 预测时额外查表的 15 个历史条件特征

主体级 33 维画像之外，未来目标日期和时段还分别查找：

## 17.1 月份条件：8 维

```text
season_bid_level_delta
season_adjustment_magnitude_delta
season_quantity_hhi_delta
season_effective_segment_count_delta
season_flat_curve_rate_delta
season_tail_uplift_ratio_delta
season_curve_bend_ratio_delta
season_shape_shift
```

## 17.2 日内条件：7 维

```text
intraday_bid_level_delta
intraday_adjustment_magnitude_delta
intraday_quantity_hhi_delta
intraday_effective_segment_count_delta
intraday_flat_curve_rate_delta
intraday_tail_uplift_ratio_delta
intraday_curve_bend_ratio_delta
```

因此从 `bidprofile` 侧提供给后续预测模型的完整历史行为信息可写为：

$$
\boxed{
X^{profile}_{i,t}
=
[
Z_i^{33},
Z^{SEA}_{i,m(t)}{}^{8},
Z^{ID}_{i,h(t)}{}^{7}
]
}
$$

当前共：

$$
33+8+7=48
$$

个基础候选历史行为变量。

这 48 个同样不是永久固定特征集，后续仍需通过预测实验筛选。

---

# 18. 完整预测模型接口

最新技术路线下，模板分类器输入建议写成：

$$
\boxed{
\hat{T}_{i,t}
=
F_{\mathrm{cls}}
\left(
Z_i,\,
Z^{\mathrm{SEA}}_{i,m(t)},\,
Z^{\mathrm{ID}}_{i,h(t)},\,
M_t,\,
U_{i,t},\,
C_t
\right)
}
$$

其中：

- $Z_i$：主体完整历史策略画像；
- $Z^{\mathrm{SEA}}_{i,m(t)}$：目标月份对应的历史季节画像；
- $Z^{\mathrm{ID}}_{i,h(t)}$：目标时段对应的历史日内画像；
- $M_t$：当前市场环境；
- $U_{i,t}$：主体当前物理状态；
- $C_t$：日历和时段信息。

模板确定后，对模板 $k$：

$$
\boxed{
\hat{\theta}^{(k)}_{i,t}
=
F_{\mathrm{reg}}^{(k)}
\left(
Z_i,\,
Z^{\mathrm{SEA}}_{i,m(t)},\,
Z^{\mathrm{ID}}_{i,h(t)},\,
M_t,\,
U_{i,t},\,
C_t
\right)
}
$$

最终：

$$
\boxed{
\hat{\mathcal B}_{i,t}
=
\operatorname{Reconstruct}
\left(
\hat{T}_{i,t},
\hat{\theta}^{(\hat{T})}_{i,t}
\right)
}
$$

不要求：

```text
目标预测年度 recent bid lag
目标预测年度 previous template
目标预测年度 recent-7d ST
```

作为默认输入。

---

# 19. 输出目录

```text
data/processed/final_clean/
│
├─ daily/<year>/
│  ├─ daily_strategy_core_<year>.csv
│  └─ intraday_slot_core_<year>.csv
│
├─ long_term/<year>/
│  └─ long_term_strategy_profile_<year>.csv
│
├─ short_term_history/<year>/
│  ├─ historical_short_term_state_<year>.csv
│  └─ historical_short_term_tendency_<year>.csv
│
├─ context_profile/<year>/
│  ├─ participant_month_strategy_profile_<year>.csv
│  ├─ seasonal_strategy_summary_<year>.csv
│  ├─ participant_slot_strategy_profile_<year>.csv
│  └─ intraday_strategy_summary_<year>.csv
│
├─ transition_profile/<year>/
│  ├─ strategy_transition_daily_<year>.csv
│  └─ strategy_transition_profile_<year>.csv
│
└─ strategy_profile/<year>/
   ├─ participant_strategy_profile_<year>.csv
   ├─ participant_strategy_profile_percentiles_<year>.csv
   ├─ strategy_profile_feature_dictionary_<year>.csv
   └─ strategy_profile_manifest_<year>.json
```

---

# 20. 推荐运行顺序

先构建基础历史主体画像：

```powershell
python scripts\bidprofile\01_build_daily_strategy_core.py --year 2025

python scripts\bidprofile\02_build_long_term_strategy_profile.py --year 2025

python scripts\bidprofile\03_build_historical_short_term_tendency.py --year 2025

python scripts\bidprofile\04_build_seasonal_intraday_strategy_profile.py --year 2025
```

然后运行 `scripts/bidtemplate` 建立历史报价策略模板并得到每条历史报价的 `template_id`。

随后：

```powershell
python scripts\bidprofile\05_build_strategy_transition_profile.py --year 2025

python scripts\bidprofile\06_build_final_strategy_profile.py --year 2025 --require-transition

python scripts\bidprofile\07_validate_strategy_profile.py --year 2025

python scripts\bidprofile\08_visualize_strategy_profile.py --year 2025
```

如果模板库暂未完成，可先运行：

```powershell
python scripts\bidprofile\06_build_final_strategy_profile.py --year 2025
```

生成不含 $Z_i^{TR}$ 的基础主体画像。

---

# 21. 当前模块边界

`bidprofile` 负责：

$$
\boxed{
\text{历史报价}
\rightarrow
\text{多时间尺度主体策略画像}
}
$$

`bidtemplate` 负责：

$$
\boxed{
\text{历史报价曲线}
\rightarrow
\text{有效阶梯结构}
\rightarrow
\text{策略模板库}
}
$$

后续 `bidprediction` 负责：

$$
\boxed{
[
主体历史画像,
当前市场环境,
当前主体状态,
日历信息
]
\rightarrow
模板分类
\rightarrow
模板专属参数回归
\rightarrow
报价曲线重构
}
$$

因此，`bidprofile` 内不再通过 KMeans 给主体贴固定类型标签；聚类的核心用途转移到 `bidtemplate` 的报价结构模板发现。

