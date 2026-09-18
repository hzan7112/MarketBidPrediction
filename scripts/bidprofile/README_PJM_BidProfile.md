# PJM_BidProfile Final Clean Pipeline

> 文档基准：`scripts/` 目录中的当前实现（2026-09-18）  
> 默认数据年度：2025  
> 日级入口脚本：`01_build_daily_strategy_core_v3.py`

本项目把 PJM 发电侧能量市场报价曲线压缩为可解释的主体策略特征，并分别构建长期画像、短期状态、策略突变、辅助聚类发现、验证结果和中文图表。

完整的主体报价策略画像由三部分组成：

$$
\boxed{
\Pi_{i,d}^{strategy}
=
\left[
Z_i^{LT},
Z_{i,d}^{ST},
E_{i,d}^{break}
\right]
}
$$

| 组成部分 | 维数 | 数据粒度 | 回答的问题 |
|---|---:|---|---|
| 长期静态画像 $Z_i^{LT}$ | 9 | 主体级 | 这个主体平时怎么报价？ |
| 短期动态状态 $Z_{i,d}^{ST}$ | 9 | 主体 × 日期 | 最近相对自己的长期习惯发生了什么变化？ |
| 突变事件 $E_{i,d}^{break}$ | 8 | 主体 × 日期 | 哪些在长期基线窗口内稳定的策略维度近期被打破？ |

因此，供后续预测模型使用的完整策略画像输入共有：

$$
\boxed{9\ LT+9\ ST+8\ Break=26\text{ 个策略画像变量}}
$$

其中长期特征是“慢变量”，短期特征是“动态状态”，Strategy Break 是“突变事件”。如果只讨论不会随日期快速变化的主体静态画像，则仅指 9 个 LT；如果讨论主体在日期 $d$ 的完整报价策略表示，则必须使用上述 26 维输入。

每个长期特征另外保留三种表达：

$$
\boxed{\text{原始连续值}+\text{经验百分位}+\text{独立状态}}
$$

其中独立状态为“低 / 中 / 高”。这些百分位和状态用于解释与展示，不额外计入上述 26 个原始策略变量。

KMeans 只对 9 个长期特征做辅助自然组合发现，不属于 26 维画像输入，也不再定义主体画像本身或构造旧版 `3×4×2` 交叉类型。

---

## 1. 当前代码流程

```text
PJM 原始报价 CSV
        |
        v
01_build_daily_strategy_core_v3.py
        |  时段报价曲线 -> 日级核心行为
        v
02_build_long_term_strategy_profile.py
        |  日级行为 -> 9 个长期连续策略维度
        v
03_build_short_term_strategy_state_fast.py
        |  日级行为 -> 8 个短期标量状态
        |            + 1 个形态迁移状态
        |            + Strategy Break
        v
04_build_strategy_profile_v3_complete_case.py
        |  9 个维度分别生成：原值 + 百分位 + 低/中/高状态
        |  complete-case KMeans 仅做四个空间的自然聚类发现
        v
05_validate_final_strategy_profile_v3_fast.py
        |  LT 稳定性、特征冗余、ST 增量信息、状态覆盖
        |  候选 K、聚类质量、规模和逐特征分离度复核
        v
06_visualize_final_strategy_profile.py
        |  9 维长期特征独立画像图 + 被接受的自然聚类图
        v
完整 26 维动态策略画像输入、验证表与中文图表
```

当前脚本顺序：

```text
scripts/01_build_daily_strategy_core_v3.py
scripts/02_build_long_term_strategy_profile.py
scripts/03_build_short_term_strategy_state_fast.py
scripts/04_build_strategy_profile_v3_complete_case.py
scripts/05_validate_final_strategy_profile_v3_fast.py
scripts/06_visualize_final_strategy_profile.py
scripts/strategy_core.py
```

`strategy_core.py` 统一维护 9 个 LT 特征、三个 3 维子空间、8 个日级标量映射、21 点形态列及公共函数。

---

## 2. 01：构建日级策略核心

脚本：

```text
scripts/01_build_daily_strategy_core_v3.py
```

默认输入：

```text
data/raw/energy_market_offers/2025/*.csv
```

默认输出：

```text
data/processed/final_clean/daily/2025/daily_strategy_core_2025.csv
```

默认参数：

```text
--year 2025
--raw-root data/raw/energy_market_offers
--out-root data/processed/final_clean/daily
--chunksize 100000
--same-slot-window 30
--min-same-slot-history 5
```

### 2.1 报价曲线清洗

报价点先按 MW 排序；同一个 MW 对应多个价格时保留最高价格。设清洗后的点为：

$$
(q_1,p_1),(q_2,p_2),\ldots,(q_m,p_m)
$$

数量轴归一化为：

$$
x_j=\frac{q_j-q_1}{q_m-q_1}
$$

斜坡报价使用梯形积分计算容量加权价格水平，阶梯报价使用左端价格积分。单点曲线或零容量跨度曲线按单段曲线处理。

### 2.2 时段核心特征

| 特征 | 当前实现含义 |
|---|---|
| `bid_level` | 归一化数量轴上的容量加权报价水平 |
| `adjustment_bias` | 当前报价水平减去该主体相同时槽历史报价水平中位数 |
| `adjustment_magnitude` | `abs(adjustment_bias)` |
| `quantity_hhi` | 归一化容量区间份额平方和 |
| `effective_segment_count` | 价格实际发生变化的次数加 1 |
| `flat_curve_flag` | 首尾价差在数值容差内是否为 0 |
| `tail_uplift_ratio` | 最后 20% 容量区间的相对抬价程度 |
| `curve_bend_ratio` | 尾部涨幅与头部涨幅之差相对总价差的比例 |
| `shape_v00 ... shape_v20` | 归一化数量网格上的 21 点相对曲线形态 |

其中：

$$
HHI_q=\sum_j(\Delta x_j)^2
$$

$$
N_{segment}=1+\sum_j\mathbf{1}(|p_{j+1}-p_j|>\varepsilon)
$$

设 21 点曲线在 $x=0,0.2,0.8,1$ 的价格分别为 $p_0,p_{0.2},p_{0.8},p_1$，则：

$$
TailUplift=\frac{p_1-p_{0.8}}{|p_1-p_0|+\varepsilon}
$$

$$
CurveBend=\frac{(p_1-p_{0.8})-(p_{0.2}-p_0)}{|p_1-p_0|+\varepsilon}
$$

非平价曲线的相对形态定义为：

$$
S(u)=\frac{p(u)-p_0}{p_1-p_0},\qquad u\in\{0,0.05,\ldots,1\}
$$

### 2.3 同时槽历史与日级聚合

调整基线按 `(participant_id, local_slot)` 分组，只使用当前记录之前最多 30 个有效报价水平；历史不足 5 条时不计算调整残差。UTC 记录按时间稳定排序，因此不会使用未来报价。

日级表按“主体 × 本地日期”聚合：大多数连续变量取日内中位数，`flat_curve_flag` 取日内均值形成 `daily_flat_curve_rate`，21 个形态点逐点取中位数。

---

## 3. 02：构建 9 个长期连续策略维度

脚本：

```text
scripts/02_build_long_term_strategy_profile.py
```

默认输入与输出：

```text
data/processed/final_clean/daily/2025/daily_strategy_core_2025.csv
data/processed/final_clean/long_term/2025/long_term_strategy_profile_2025.csv
```

### 3.1 中文名称、代码字段和顺序

9 个中文维度与代码字段严格一一对应：

| 顺序 | 中文画像维度 | 正式字段 | 直接日级来源 | 核心问题 |
|---:|---|---|---|---|
| 1 | 报价水平 | `lt_bid_level` | `daily_bid_level` | 主体通常报在什么价格水平？ |
| 2 | 调整幅度 | `lt_adjustment_magnitude` | `daily_adjustment_magnitude` | 相对自身同一时槽历史，通常改动多大？ |
| 3 | 持续性 | `lt_strategy_persistence` | `daily_adjustment_bias` | 今天的调整方向与幅度是否会延续到次日？ |
| 4 | 容量集中度 | `lt_quantity_hhi` | `daily_quantity_hhi` | 容量是否集中在少数报价区间？ |
| 5 | 有效段数 | `lt_effective_segment_count` | `daily_effective_segment_count` | 曲线中有多少个价格真正不同的有效段？ |
| 6 | 平价偏好 | `lt_flat_curve_rate` | `daily_flat_curve_rate` | 主体多大程度上使用首尾同价曲线？ |
| 7 | 尾部抬价 | `lt_tail_uplift_ratio` | `daily_tail_uplift_ratio` | 总价差中有多少集中在最后 20% 容量？ |
| 8 | 曲线弯折 | `lt_curve_bend_ratio` | `daily_curve_bend_ratio` | 尾部涨幅相对头部涨幅有多强？ |
| 9 | 形态波动 | `lt_shape_variability` | `daily_shape_v00 ... daily_shape_v20` | 去掉绝对价格水平后，曲线形状跨日变化多大？ |

最终长期向量按上表顺序写为：

$$
\boxed{
Z_i^{LT}
=
\left[
L_i^{LT},
A_i^{mag},
A_i^{persist},
H_i^{LT},
K_i^{LT},
F_i^{LT},
T_i^{LT},
C_i^{LT},
D_i^{shape}
\right]
}
$$

即：

```text
Z_i^LT = [
    报价水平,
    调整幅度,
    持续性,
    容量集中度,
    有效段数,
    平价偏好,
    尾部抬价,
    曲线弯折,
    形态波动
]
```

### 3.2 统一符号与三层计算关系

为避免公式中的 `daily_xxx` 难以理解，先定义统一符号：

- $i$：市场主体；
- $t$：主体的一条时段报价记录；
- $d$：本地自然日；
- $\mathcal T_{i,d}$：主体 $i$ 在日期 $d$ 的有效时段集合；
- $\mathcal D_i$：主体 $i$ 的有效日期集合；
- $m$：清洗后报价断点数；
- $\varepsilon=10^{-12}$：分母数值保护量；
- $\varepsilon_p=10^{-9}$：判断价格是否变化的容差。

主体 $i$ 在时段 $t$ 的清洗后报价断点为：

$$
\mathcal B_{i,t}
=
\left\{(q_{i,t,j},p_{i,t,j})\right\}_{j=1}^{m}
$$

断点按 MW 从小到大排序；同一 MW 有多个价格时保留最高价格。数量跨度大于 0 时，将累计数量归一化：

$$
x_{i,t,j}
=
\frac{q_{i,t,j}-q_{i,t,1}}
{q_{i,t,m}-q_{i,t,1}}
\in[0,1]
$$

并记区间容量份额为：

$$
\Delta x_{i,t,j}=x_{i,t,j+1}-x_{i,t,j}
$$

由断点构造报价函数 $P_{i,t}(x)$。在区间 $x\in[x_{i,t,j},x_{i,t,j+1})$ 上：

$$
P_{i,t}(x)
=
\begin{cases}
p_{i,t,j},
& \text{block 报价},\\
\displaystyle
p_{i,t,j}
+
\frac{x-x_{i,t,j}}
{x_{i,t,j+1}-x_{i,t,j}}
\left(p_{i,t,j+1}-p_{i,t,j}\right),
& \text{sloped 报价}.
\end{cases}
$$

在 $x=1$ 时取最后一个报价点 $p_{i,t,m}$。

所有维度都遵循下面的追溯链：

```text
原始 (MW, price) 报价断点
        -> 时段原子特征
        -> 主体 × 日期的日级统计
        -> 主体跨日长期统计
        -> 原值 + 百分位 + 独立状态
```

日级层先压缩同一天的多个时段，长期层再跨日取统计量。因此长期画像让每个有效日期获得近似相同权重，不会因为某一天记录时段更多而自动获得更大权重。

下面逐维给出当前代码的完整公式。

### 3.3 维度 1：报价水平 `lt_bid_level`

#### 时段层

单时段整体报价水平记为 $L_{i,t}$。它是归一化容量轴上的报价面积：

$$
L_{i,t}=\int_0^1P_{i,t}(x)\,dx
$$

当前代码的离散计算为：

$$
L_{i,t}
=
\begin{cases}
\displaystyle
\sum_{j=1}^{m-1}
\frac{p_{i,t,j}+p_{i,t,j+1}}{2}\Delta x_{i,t,j},
& \text{sloped 报价},\\
\displaystyle
\sum_{j=1}^{m-1}
p_{i,t,j}\Delta x_{i,t,j},
& \text{block 报价}.
\end{cases}
$$

单点报价或数量跨度为 0 时，直接取该点价格。

#### 日级与长期层

日典型报价水平：

$$
\ell_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}L_{i,t}
$$

长期报价水平：

$$
\boxed{
L_i^{LT}
=
\operatorname{Median}_{d\in\mathcal D_i}\ell_{i,d}
}
$$

字段链：

```text
bid_level -> daily_bid_level -> lt_bid_level -> 报价水平
```

`lt_bid_level` 保留价格量纲。高值表示主体长期容量加权报价水平相对更高；低值表示相对更低。它描述报价本身，不是相对市场出清价的价差。

### 3.4 维度 2：调整幅度 `lt_adjustment_magnitude`

#### 同时槽自身历史基线

当前记录的调整不是相对上一时段，也不是相对市场均价，而是相对该主体“相同本地时槽”的自身历史。设 $\mathcal H_{i,t}^{slot}$ 为当前记录之前、同主体同本地时槽最多 30 条有效 $L$，则：

$$
B_{i,t}
=
\operatorname{Median}
\left(\mathcal H_{i,t}^{slot}\right)
$$

只有历史至少 5 条时才计算基线。时段有符号调整偏差为：

$$
R_{i,t}=L_{i,t}-B_{i,t}
$$

其中 $R_{i,t}>0$ 表示相对自身历史上调，$R_{i,t}<0$ 表示下调。

#### 日级与长期层

日典型调整幅度只看改动大小，不保留方向：

$$
m_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}
\left|R_{i,t}\right|
$$

长期调整幅度：

$$
\boxed{
A_i^{mag}
=
\operatorname{Median}_{d\in\mathcal D_i}m_{i,d}
}
$$

字段链：

```text
adjustment_magnitude
    -> daily_adjustment_magnitude
    -> lt_adjustment_magnitude
    -> 调整幅度
```

高值表示主体经常明显偏离自己的同时槽历史报价；低值表示长期报价更接近自身历史习惯。该维度不区分上调或下调，调整方向保留在 `daily_adjustment_bias` 和短期状态 `st_adjustment_bias_z` 中。

### 3.5 维度 3：持续性 `lt_strategy_persistence`

先对每个自然日计算有符号调整偏差中位数：

$$
a_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}R_{i,t}
$$

只保留日期恰好相差 1 天、且前后两日 $a_{i,d}$ 都有效的日对。设这些连续日对组成集合 $\mathcal C_i$，则：

$$
\boxed{
A_i^{persist}
=
\operatorname{Corr}
\left(
\{a_{i,d-1}\}_{d\in\mathcal C_i},
\{a_{i,d}\}_{d\in\mathcal C_i}
\right)
}
$$

这里使用 Pearson 相关系数。至少需要 5 对有效连续日；若前一日序列或后一日序列的标准差不大于 $10^{-12}$，结果也记为缺失。

字段链：

```text
adjustment_bias -> daily_adjustment_bias
                -> lt_strategy_persistence
                -> 持续性
```

解释：

- 接近 1：今天相对自身历史上调或下调的模式，次日往往延续；
- 接近 0：相邻日调整关系较弱；
- 小于 0：调整更可能在相邻日反向；
- 缺失：连续日对不足，或长期调整完全固定而无法计算相关系数。

因此“持续性低”既可能表示弱相关，也可能表示负相关；分析时应同时查看原始连续值，而不能只看低/中/高状态。

### 3.6 维度 4：容量集中度 `lt_quantity_hhi`

归一化后，各相邻断点之间的容量份额为 $\Delta x_{i,t,j}$，且总和为 1。单时段容量 HHI 为：

$$
H_{i,t}
=
\sum_{j=1}^{m-1}
\left(\Delta x_{i,t,j}\right)^2
$$

日级容量集中度：

$$
h_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}H_{i,t}
$$

长期容量集中度：

$$
\boxed{
H_i^{LT}
=
\operatorname{Median}_{d\in\mathcal D_i}h_{i,d}
}
$$

字段链：

```text
quantity_hhi -> daily_quantity_hhi -> lt_quantity_hhi -> 容量集中度
```

高值表示大部分容量集中于少数区间；低值表示容量更均匀地分散在多个区间。单点或零跨度曲线在当前代码中取 $H_{i,t}=1$。该指标衡量的是容量区间分配，不直接衡量价格高低。

### 3.7 维度 5：有效段数 `lt_effective_segment_count`

单时段有效段数不是原始断点数量，而是相邻清洗后报价点发生实际价格变化的次数加 1：

$$
K_{i,t}^{eff}
=
1+
\sum_{j=1}^{m-1}
\mathbf 1
\left(
|p_{i,t,j+1}-p_{i,t,j}|>\varepsilon_p
\right)
$$

其中 $\varepsilon_p=10^{-9}$。连续多个同价断点只算一个有效价格段。

日级有效段数：

$$
k_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}K_{i,t}^{eff}
$$

长期有效段数：

$$
\boxed{
K_i^{LT}
=
\operatorname{Median}_{d\in\mathcal D_i}k_{i,d}
}
$$

字段链：

```text
effective_segment_count
    -> daily_effective_segment_count
    -> lt_effective_segment_count
    -> 有效段数
```

高值表示报价曲线使用更多真正不同的价格段，结构更细；低值表示报价结构更简单。它与容量集中度含义不同：段数多不代表容量一定均匀，段数少也不代表容量一定集中。

### 3.8 维度 6：平价偏好 `lt_flat_curve_rate`

当前代码用报价曲线首尾价格是否相同判断单时段平价状态：

$$
F_{i,t}
=
\mathbf 1
\left(
|P_{i,t}(1)-P_{i,t}(0)|\le\varepsilon_p
\right)
$$

日平价曲线比例为日内平价标志的均值：

$$
f_{i,d}
=
\frac{1}{|\mathcal T_{i,d}|}
\sum_{t\in\mathcal T_{i,d}}F_{i,t}
$$

长期平价偏好为各日平价比例的中位数：

$$
\boxed{
F_i^{LT}
=
\operatorname{Median}_{d\in\mathcal D_i}f_{i,d}
}
$$

字段链：

```text
flat_curve_flag -> daily_flat_curve_rate -> lt_flat_curve_rate -> 平价偏好
```

高值表示主体在多数日期更偏好首尾同价曲线；低值表示更常使用首尾有价差的曲线。由于先计算每日比例、再跨日取中位数，`lt_flat_curve_rate` 是“典型日期的平价比例”，不是把全年所有时段直接合并后的单一比例。

### 3.9 维度 7：尾部抬价 `lt_tail_uplift_ratio`

对非平价曲线，在归一化容量位置 $x=0.8$ 和 $x=1$ 读取价格。单时段尾部抬价比例为：

$$
T_{i,t}
=
\frac{P_{i,t}(1)-P_{i,t}(0.8)}
{|P_{i,t}(1)-P_{i,t}(0)|+\varepsilon}
$$

平价曲线在当前代码中直接令 $T_{i,t}=0$。

日级与长期统计为：

$$
\tau_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}T_{i,t}
$$

$$
\boxed{
T_i^{LT}
=
\operatorname{Median}_{d\in\mathcal D_i}\tau_{i,d}
}
$$

字段链：

```text
tail_uplift_ratio
    -> daily_tail_uplift_ratio
    -> lt_tail_uplift_ratio
    -> 尾部抬价
```

正值越大，说明从 80% 容量到满容量的涨价在首尾总价差中越突出；接近 0 表示尾部几乎不额外抬价；负值表示尾部价格下降。该比值不是概率，也不被代码强制限制在 $[0,1]$。

### 3.10 维度 8：曲线弯折 `lt_curve_bend_ratio`

头部 20% 容量的价格变化为：

$$
\Delta P_{i,t}^{head}
=
P_{i,t}(0.2)-P_{i,t}(0)
$$

尾部 20% 容量的价格变化为：

$$
\Delta P_{i,t}^{tail}
=
P_{i,t}(1)-P_{i,t}(0.8)
$$

单时段曲线弯折比为：

$$
C_{i,t}
=
\frac{
\Delta P_{i,t}^{tail}-\Delta P_{i,t}^{head}
}
{|P_{i,t}(1)-P_{i,t}(0)|+\varepsilon}
$$

平价曲线在当前代码中直接令 $C_{i,t}=0$。日级与长期统计为：

$$
c_{i,d}
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}C_{i,t}
$$

$$
\boxed{
C_i^{LT}
=
\operatorname{Median}_{d\in\mathcal D_i}c_{i,d}
}
$$

字段链：

```text
curve_bend_ratio
    -> daily_curve_bend_ratio
    -> lt_curve_bend_ratio
    -> 曲线弯折
```

正值表示尾部涨幅强于头部，负值表示头部涨幅更强，接近 0 表示首尾两段变化较接近。它是首尾局部涨幅差的结构指标，不是严格意义上的二阶导数或几何曲率。

### 3.11 维度 9：形态波动 `lt_shape_variability`

该维度先去掉绝对价格水平和首尾总价差，只比较曲线的相对形状。

对非平价曲线，在固定网格：

$$
\mathcal U
=
\{0,0.05,0.10,\ldots,0.95,1\}
$$

上计算 21 点归一化形态：

$$
V_{i,t}(u)
=
\frac{P_{i,t}(u)-P_{i,t}(0)}
{P_{i,t}(1)-P_{i,t}(0)},
\qquad u\in\mathcal U
$$

因此有效形态总是满足 $V_{i,t}(0)=0$、$V_{i,t}(1)=1$。平价曲线由于分母为 0，当前代码不定义 21 点形态。

每日典型形态对日内有效时段逐点取中位数：

$$
V_{i,d}(u)
=
\operatorname{Median}_{t\in\mathcal T_{i,d}}V_{i,t}(u)
$$

02 只把 21 个坐标全部有效的日期纳入形态计算。主体长期形态原型为：

$$
V_i^{LT}(u)
=
\operatorname{Median}_{d\in\mathcal D_i^{shape}}V_{i,d}(u)
$$

日期 $d$ 相对长期原型的形态距离为：

$$
D_{i,d}^{shape}
=
\sqrt{
\frac{1}{21}
\sum_{u\in\mathcal U}
\left[
V_{i,d}(u)-V_i^{LT}(u)
\right]^2
}
$$

最终形态波动为这些日级距离的中位数：

$$
\boxed{
D_i^{shape}
=
\operatorname{Median}_{d\in\mathcal D_i^{shape}}
D_{i,d}^{shape}
}
$$

字段链：

```text
shape_v00 ... shape_v20
    -> daily_shape_v00 ... daily_shape_v20
    -> lt_shape_variability
    -> 形态波动
```

高值表示即使去掉绝对价格水平，主体的曲线相对形状在不同日期之间仍经常变化；低值表示相对形态稳定。它不等于报价水平波动，也不等于尾部抬价本身。

代码同时保存：

```text
shape_defined_days
lt_shape_v00 ... lt_shape_v20
```

前者是形态完整日期数，后者是长期 21 点形态原型；它们是辅助字段，不属于正式 9 个长期维度。

### 3.12 长期部分 $Z_i^{LT}$ 的完整定义表

| 序号 | 中文维度 | 正式字段 | 数学定义 | 高低值解释 |
|---:|---|---|---|---|
| 1 | 报价水平 | `lt_bid_level` | $L_i^{LT}=\operatorname{Med}_{d\in\mathcal D_i}\operatorname{Med}_{t\in\mathcal T_{i,d}}\int_0^1P_{i,t}(x)\,dx$ | 低：长期容量加权报价较低；高：长期报价较高 |
| 2 | 调整幅度 | `lt_adjustment_magnitude` | $A_i^{mag}=\operatorname{Med}_{d\in\mathcal D_i}\operatorname{Med}_{t\in\mathcal T_{i,d}}\left\lvert L_{i,t}-\operatorname{Med}_{h\in\mathcal H_{i,t}^{slot}}L_{i,h}\right\rvert$ | 低：接近自身同时槽历史；高：经常明显偏离自身历史 |
| 3 | 持续性 | `lt_strategy_persistence` | $A_i^{persist}=\operatorname{Corr}_{d\in\mathcal C_i}\left(a_{i,d-1},a_{i,d}\right)$，其中 $a_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}R_{i,t}$ | 低或负：弱延续或反向；高：相邻自然日调整模式更连续 |
| 4 | 容量集中度 | `lt_quantity_hhi` | $H_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t}\sum_j\left(\frac{q_{i,t,j+1}-q_{i,t,j}}{q_{i,t,m}-q_{i,t,1}}\right)^2$ | 低：容量分散；高：容量集中于少数区间 |
| 5 | 有效段数 | `lt_effective_segment_count` | $K_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t}\left[1+\sum_j\mathbf 1\left(\left\lvert p_{i,t,j+1}-p_{i,t,j}\right\rvert>\varepsilon_p\right)\right]$ | 低：价格结构简单；高：真正不同的价格段更多 |
| 6 | 平价偏好 | `lt_flat_curve_rate` | $F_i^{LT}=\operatorname{Med}_{d}\left[\frac{1}{\lvert\mathcal T_{i,d}\rvert}\sum_{t\in\mathcal T_{i,d}}\mathbf 1\left(\left\lvert P_{i,t}(1)-P_{i,t}(0)\right\rvert\le\varepsilon_p\right)\right]$ | 低：较少采用首尾同价曲线；高：更偏好平价曲线 |
| 7 | 尾部抬价 | `lt_tail_uplift_ratio` | $T_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t}\left\{\mathbf 1\left(\left\lvert P_{i,t}(1)-P_{i,t}(0)\right\rvert>\varepsilon_p\right)\frac{P_{i,t}(1)-P_{i,t}(0.8)}{\left\lvert P_{i,t}(1)-P_{i,t}(0)\right\rvert+\varepsilon}\right\}$ | 低或负：尾部抬价弱或下降；高：最后 20% 容量抬价突出；平价曲线取 0 |
| 8 | 曲线弯折 | `lt_curve_bend_ratio` | $C_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t}\left\{\mathbf 1\left(\left\lvert P_{i,t}(1)-P_{i,t}(0)\right\rvert>\varepsilon_p\right)\frac{[P_{i,t}(1)-P_{i,t}(0.8)]-[P_{i,t}(0.2)-P_{i,t}(0)]}{\left\lvert P_{i,t}(1)-P_{i,t}(0)\right\rvert+\varepsilon}\right\}$ | 负：头部涨幅相对更强；正且高：尾部涨幅相对更强；平价曲线取 0 |
| 9 | 形态波动 | `lt_shape_variability` | $D_i^{shape}=\operatorname{Med}_{d\in\mathcal D_i^{shape}}\sqrt{\frac{1}{21}\sum_{u\in\mathcal U}[V_{i,d}(u)-V_i^{LT}(u)]^2}$ | 低：跨日相对形态稳定；高：跨日形态变化大 |

因此：

$$
Z_i^{LT}
=
\left[
L_i^{LT},
A_i^{mag},
A_i^{persist},
H_i^{LT},
K_i^{LT},
F_i^{LT},
T_i^{LT},
C_i^{LT},
D_i^{shape}
\right]
\in\mathbb R^9
$$

这 9 个特征是主体级慢变量，描述“这个主体平时怎么报”。

### 3.13 长期画像完整度

`active_days` 是主体有效日期数。`lt_nonmissing_count` 是 9 个长期维度中的非缺失数量：

$$
N_i^{LT,obs}
=
\sum_{j=1}^{9}
\mathbf 1
\left(Z_{i,j}^{LT}\text{ 非缺失}\right)
$$

当：

$$
N_i^{LT,obs}\ge 7
$$

时设置：

```text
lt_ready_flag = 1
```

否则 `lt_ready_flag = 0`。这不表示主体“没有画像”，只表示 9 个维度中缺失较多。长期平价主体可能缺少 21 点形态，长期调整完全固定的主体也可能无法定义持续性。

`lt_ready_flag` 用于确定辅助 KMeans 的发现队列，不决定一个主体是否能保留已有的单维原值、百分位或独立状态。

---

## 4. 03：构建 9 个短期状态与 8 个 Strategy Break

脚本：

<code>scripts/03_build_short_term_strategy_state_fast.py</code>

默认输入与输出：

| 类型 | 路径 |
|---|---|
| 输入日级策略数据 | <code>data/processed/final_clean/daily/2025/daily_strategy_core_2025.csv</code> |
| 输出短期状态与 Break | <code>data/processed/final_clean/short_term/2025/short_term_strategy_state_2025.csv</code> |

短期部分是主体—日期级动态状态：

$$
\boxed{
Z_{i,d}^{ST}
=
\left[
z_L,\,
z_{A^{bias}},\,
z_{A^{mag}},\,
z_{HHI},\,
z_{K^{eff}},\,
z_{F^{flat}},\,
z_{T^{rel}},\,
z_{B^{rel}},\,
D_{shape}^{ST}
\right]
}
$$

它回答的是：

> “截至日期 $d$，这个主体最近相对自己的历史习惯发生了什么变化？”

### 4.1 近期窗口、长期基线与稳健标准化

对主体 $i$ 的目标日期 $d$：

$$
\mathcal R_{i,d}
=
\{d-7,\ldots,d-1\}
$$

是近期 7 个日历日窗口，标量特征至少需要 3 个有效值；

$$
\mathcal L_{i,d}
=
\{d-67,\ldots,d-8\}
$$

是更早 60 个日历日基线窗口，标量特征至少需要 20 个有效值。

对任一日级标量序列 $x_{i,s}$，定义：

$$
x_{i,d}^{recent}
=
\operatorname{Med}_{s\in\mathcal R_{i,d}}x_{i,s}
$$

$$
x_{i,d}^{long}
=
\operatorname{Med}_{s\in\mathcal L_{i,d}}x_{i,s}
$$

$$
MAD_{i,d}^{long}(x)
=
\operatorname{Med}_{s\in\mathcal L_{i,d}}
\left\lvert
x_{i,s}-x_{i,d}^{long}
\right\rvert
$$

长期稳健尺度为：

$$
S_{i,d}^{long}(x)
=
1.4826\,MAD_{i,d}^{long}(x)
$$

当该尺度大于数值容差时：

$$
\boxed{
z_{i,d}(x)
=
\frac{
x_{i,d}^{recent}-x_{i,d}^{long}
}{
1.4826\,MAD_{i,d}^{long}(x)
}
}
$$

因此正值表示近期高于自身历史习惯，负值表示近期低于自身历史习惯，绝对值越大表示变化相对长期波动越显著。

### 4.2 短期部分 $Z_{i,d}^{ST}$ 的完整定义表

下表中的 $\ell_{i,s}$、$a_{i,s}$、$m_{i,s}$、$h_{i,s}$、$k_{i,s}$、$f_{i,s}$、$\tau_{i,s}$、$c_{i,s}$ 分别是第 3 节已经从原始报价曲线定义的日级报价水平、调整方向、调整幅度、容量 HHI、有效段数、平价比例、尾部抬价和曲线弯折。

| 序号 | 短期状态 | 正式字段 | 完整数学定义 | 解释 |
|---:|---|---|---|---|
| 1 | 报价水平偏移 $z_L$ | <code>st_bid_level_z</code> | $\displaystyle z_L=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}\ell_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}\ell_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert\ell_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}\ell_{i,r}\right\rvert}$ | 最近报价水平相对历史基准的稳健标准化偏移 |
| 2 | 调整方向偏移 $z_{A^{bias}}$ | <code>st_adjustment_bias_z</code> | $\displaystyle z_{A^{bias}}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}a_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}a_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert a_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}a_{i,r}\right\rvert}$ | 最近更倾向相对自身历史上调还是下调 |
| 3 | 调整幅度偏移 $z_{A^{mag}}$ | <code>st_adjustment_magnitude_z</code> | $\displaystyle z_{A^{mag}}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}m_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}m_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert m_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}m_{i,r}\right\rvert}$ | 最近调整大小相对长期习惯是否扩大 |
| 4 | 容量集中度偏移 $z_{HHI}$ | <code>st_quantity_hhi_z</code> | $\displaystyle z_{HHI}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}h_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}h_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert h_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}h_{i,r}\right\rvert}$ | 最近容量配置是否变得更集中或更分散 |
| 5 | 有效段数偏移 $z_{K^{eff}}$ | <code>st_effective_segment_count_z</code> | $\displaystyle z_{K^{eff}}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}k_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}k_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert k_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}k_{i,r}\right\rvert}$ | 最近报价段数是否相对长期结构增加 |
| 6 | 平价偏好偏移 $z_{F^{flat}}$ | <code>st_flat_curve_rate_z</code> | $\displaystyle z_{F^{flat}}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}f_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}f_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert f_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}f_{i,r}\right\rvert}$ | 最近是否更偏好首尾同价曲线 |
| 7 | 尾部抬价偏移 $z_{T^{rel}}$ | <code>st_tail_uplift_ratio_z</code> | $\displaystyle z_{T^{rel}}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}\tau_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}\tau_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert\tau_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}\tau_{i,r}\right\rvert}$ | 最近尾部抬价相对历史是否增强 |
| 8 | 曲线弯折偏移 $z_{B^{rel}}$ | <code>st_curve_bend_ratio_z</code> | $\displaystyle z_{B^{rel}}=\frac{\operatorname{Med}_{s\in\mathcal R_{i,d}}c_{i,s}-\operatorname{Med}_{s\in\mathcal L_{i,d}}c_{i,s}}{1.4826\,\operatorname{Med}_{s\in\mathcal L_{i,d}}\left\lvert c_{i,s}-\operatorname{Med}_{r\in\mathcal L_{i,d}}c_{i,r}\right\rvert}$ | 最近曲线前后段涨幅结构是否改变 |
| 9 | 形态迁移 $D_{shape}^{ST}$ | <code>st_shape_shift</code> | $\displaystyle D_{shape}^{ST}=\sqrt{\frac{1}{21}\sum_{u\in\mathcal U}\left[\operatorname{Med}_{s\in\mathcal R_{i,d}}V_{i,s}(u)-\operatorname{Med}_{s\in\mathcal L_{i,d}}V_{i,s}(u)\right]^2}$ | 近期 21 点曲线形态原型相对更早长期窗口原型的 RMSE |

前 8 个是有正负方向的稳健 z 状态。第 9 个不是 z-score，而是非负的形态迁移距离；越大表示近期曲线相对形态改变越明显。形态计算要求近期至少 2 个完整形态日、长期窗口至少 10 个完整形态日。

### 4.3 Strategy Break 的统一规则

Strategy Break 只针对前 8 个标量状态。若长期窗口尺度为 0：

$$
1.4826\,MAD_{i,d}^{long}(x)\le10^{-12}
$$

且近期中位数与长期中位数不同：

$$
\left\lvert
x_{i,d}^{recent}-x_{i,d}^{long}
\right\rvert
>10^{-12}
$$

则普通 z-score 无法合理表示“从长期完全固定到近期突然改变”，因此定义：

$$
break_{i,d}(x)=1
$$

此时相应的 <code>st_*_z</code> 保持缺失，由 Break 标志承载突变信息。若长期尺度为 0 且近期也没有变化，则令对应 z 值为 0，Break 为 0。

### 4.4 Break 部分 $E_{i,d}^{break}$ 的完整定义表

| 序号 | Break 事件 | 正式字段 | 触发条件 |
|---:|---|---|---|
| 1 | 报价水平 Break | <code>break_bid_level</code> | $MAD^{long}(\ell)=0$ 且 $\operatorname{Med}_{recent}(\ell)\ne\operatorname{Med}_{long}(\ell)$ |
| 2 | 调整方向 Break | <code>break_adjustment_bias</code> | $MAD^{long}(a)=0$ 且 $\operatorname{Med}_{recent}(a)\ne\operatorname{Med}_{long}(a)$ |
| 3 | 调整幅度 Break | <code>break_adjustment_magnitude</code> | $MAD^{long}(m)=0$ 且 $\operatorname{Med}_{recent}(m)\ne\operatorname{Med}_{long}(m)$ |
| 4 | 容量集中度 Break | <code>break_quantity_hhi</code> | $MAD^{long}(h)=0$ 且 $\operatorname{Med}_{recent}(h)\ne\operatorname{Med}_{long}(h)$ |
| 5 | 有效段数 Break | <code>break_effective_segment_count</code> | $MAD^{long}(k)=0$ 且 $\operatorname{Med}_{recent}(k)\ne\operatorname{Med}_{long}(k)$ |
| 6 | 平价偏好 Break | <code>break_flat_curve_rate</code> | $MAD^{long}(f)=0$ 且 $\operatorname{Med}_{recent}(f)\ne\operatorname{Med}_{long}(f)$ |
| 7 | 尾部抬价 Break | <code>break_tail_uplift_ratio</code> | $MAD^{long}(\tau)=0$ 且 $\operatorname{Med}_{recent}(\tau)\ne\operatorname{Med}_{long}(\tau)$ |
| 8 | 曲线弯折 Break | <code>break_curve_bend_ratio</code> | $MAD^{long}(c)=0$ 且 $\operatorname{Med}_{recent}(c)\ne\operatorname{Med}_{long}(c)$ |

表中的 `MAD = 0` 和“中位数不相等”是语义简写。代码实际使用的数值条件分别是 $1.4826\,MAD\le10^{-12}$ 和 $\left\lvert Median_{recent}-Median_{long}\right\rvert>10^{-12}$。

因此：

$$
E_{i,d}^{break}
=
\left[
e_L,\,
e_{A^{bias}},\,
e_{A^{mag}},\,
e_{HHI},\,
e_{K^{eff}},\,
e_{F^{flat}},\,
e_{T^{rel}},\,
e_{B^{rel}}
\right]
\in\{0,1\}^8
$$

代码还提供三个派生汇总字段：

| 字段 | 定义 | 是否额外计入 26 维 |
|---|---|---|
| <code>strategy_break_count</code> | 8 个 Break 标志之和 | 否，属于重复汇总 |
| <code>strategy_break_any</code> | 只要任一 Break 为 1 就取 1 | 否，属于重复汇总 |
| <code>strategy_break_types</code> | 所有触发维度名称组成的文本 | 否，仅用于解释 |

当前代码没有单独的 <code>break_shape_shift</code>，因此 Break 部分是 8 维而不是 9 维。

### 4.5 最终 26 维策略画像输入

主体 $i$ 在日期 $d$ 的完整报价策略画像定义为：

$$
\boxed{
\Pi_{i,d}^{strategy}
=
\left[
Z_i^{LT},
Z_{i,d}^{ST},
E_{i,d}^{break}
\right]
}
$$

三个组成部分分别为：

| 组成 | 维数 | 正式变量 |
|---|---:|---|
| $Z_i^{LT}$ | 9 | <code>lt_bid_level</code><br><code>lt_adjustment_magnitude</code><br><code>lt_strategy_persistence</code><br><code>lt_quantity_hhi</code><br><code>lt_effective_segment_count</code><br><code>lt_flat_curve_rate</code><br><code>lt_tail_uplift_ratio</code><br><code>lt_curve_bend_ratio</code><br><code>lt_shape_variability</code> |
| $Z_{i,d}^{ST}$ | 9 | <code>st_bid_level_z</code><br><code>st_adjustment_bias_z</code><br><code>st_adjustment_magnitude_z</code><br><code>st_quantity_hhi_z</code><br><code>st_effective_segment_count_z</code><br><code>st_flat_curve_rate_z</code><br><code>st_tail_uplift_ratio_z</code><br><code>st_curve_bend_ratio_z</code><br><code>st_shape_shift</code> |
| $E_{i,d}^{break}$ | 8 | <code>break_bid_level</code><br><code>break_adjustment_bias</code><br><code>break_adjustment_magnitude</code><br><code>break_quantity_hhi</code><br><code>break_effective_segment_count</code><br><code>break_flat_curve_rate</code><br><code>break_tail_uplift_ratio</code><br><code>break_curve_bend_ratio</code> |

总维数为：

$$
\boxed{
\dim\left(\Pi_{i,d}^{strategy}\right)
=
9+9+8
=
26
}
$$

构造预测样本时，应以 <code>participant_id</code> 将主体级 9 LT 合并到每个主体日，再与该日的 9 ST 和 8 Break 拼接。长期百分位、低/中/高状态、KMeans Cluster、完整度标志及 Break 汇总字段都不重复计入这 26 个原始策略变量。

下一步进行报价策略模板或完整报价曲线预测时，策略侧真正输入的是：

$$
\Pi_{i,d}^{strategy}
=
\left[
9\ LT,\,
9\ ST,\,
8\ Break
\right]
$$

如果再接入主体物理状态、市场状态和日历信息，完整预测关系可写为：

$$
\widehat{\mathcal B}_{i,d+1}
=
F\left(
\Pi_{i,d}^{strategy},
X_{i,d}^{physical},
X_d^{market},
X_d^{calendar}
\right)
$$

其中 $\widehat{\mathcal B}_{i,d+1}$ 是下一目标时段或下一目标日的完整报价曲线/报价模板。

8 个标量状态中至少 6 个可以正常表示、可以确认零变化或被识别为 Break 时，<code>st_ready_flag = 1</code>。当前 fast 版本先将每个主体重建到逐日历日索引，再用 rolling 窗口计算上述统计量，最后只输出原始有效交易日；该加速不会改变窗口、MAD、Break 或就绪定义。

---

## 5. 04：构建长期 9 维解释层与辅助聚类

脚本：

```text
scripts/04_build_strategy_profile_v3_complete_case.py
```

### 5.1 长期画像解释层

04 处理的是完整 26 维策略画像中的长期 $Z_i^{LT}$ 部分。长期画像不是一个 Cluster，也不是多个 Cluster 的拼接标签，而是下面 9 个可分别解释的连续策略维度：

各中文名称与代码字段、原始报价公式和高低值含义见第 3.1—3.12 节；本节只说明 04 如何为这些连续维度增加百分位和独立状态。

```text
报价水平
调整幅度
持续性
容量集中度
有效段数
平价偏好
尾部抬价
曲线弯折
形态波动
```

对任一长期特征 `f`，主体表保留：

| 表达 | 输出列 | 说明 |
|---|---|---|
| 原始连续值 | `f` | 来自 02 的长期统计量，保留量纲和实际差异 |
| 经验百分位 | `f_percentile` | 当前年度所有非缺失主体中的平均秩百分位，大于 0 且不超过 100 |
| 独立状态 | `f_state` | 根据该维度自身的经验百分位独立划分为低、中、高 |

经验百分位按：

$$
P_{i,f}=100\times\operatorname{rank}_{avg}(LT_{i,f})/N_f
$$

计算，其中并列值使用平均秩，$N_f$ 是该维度非缺失主体数。当前代码先对主体表中每个维度分别排名；`lt_ready_flag` 只用于后面的辅助聚类训练。

状态规则严格对应当前代码：

$$
State_{i,f}=
\begin{cases}
\text{低}, & P_{i,f}\le 33.333333\\
\text{中}, & 33.333333<P_{i,f}<66.666667\\
\text{高}, & P_{i,f}\ge 66.666667
\end{cases}
$$

特征缺失时，其百分位和状态也保持缺失。

这里的“低 / 中 / 高”只描述该特征数值在总体中的相对位置，不代表“差 / 一般 / 好”。例如“形态波动 = 高”表示跨日形态变化更大，而不是策略质量更高。

一个主体可以直接解释为：

```text
报价水平 = 高
调整幅度 = 低
持续性 = 高
容量集中度 = 中
有效段数 = 高
平价偏好 = 低
尾部抬价 = 高
曲线弯折 = 中
形态波动 = 低
```

`independent_state_fingerprint` 会把已有的 9 个独立状态拼成一行便于检索的文本，但它只是显示用指纹，不是 Cluster、Archetype 或新的组合画像分类。

### 5.2 KMeans 的新定位

KMeans 只回答一个辅助问题：某个特征空间中是否存在分离度和稳定性都足够高的自然分组。

它分别检查四个空间：

| 空间 | 特征 |
|---|---|
| `full_9d` | 全部 9 个长期策略维度 |
| `price_adjustment_3d` | 报价水平、调整幅度、持续性 |
| `quantity_structure_3d` | 容量 HHI、有效段数、平价偏好 |
| `curve_shape_3d` | 尾部抬价、曲线弯折、形态波动 |

聚类发现采用 complete-case 原则：

1. 先以 `lt_ready_flag = 1` 确定发现队列；若该队列少于 50 个主体，脚本直接报错；
2. 对每个空间分别筛选该空间全部特征均非缺失的主体；
3. 缺少该空间任一特征的主体不会参与该空间的训练，也不会用中位数或其他方法填补；
4. 用 complete-case 训练样本的 1% 和 99% 分位数缩尾；
5. 使用 `RobustScaler(quantile_range=(25, 75))` 标准化；
6. 缩尾和标准化只用于聚类副本，不覆盖画像中的原始连续值。

因此四个空间可以有不同的有效样本数。`complete_discovery_n` 记录 LT-ready 队列中的空间完整样本数，`complete_all_n` 记录全部主体中的空间完整样本数。

### 5.3 自动搜索与双门槛

每个空间自动搜索：

$$
K=2,\ldots,6
$$

对每个可计算的 $K$：

- 参考解使用 `random_state=42`、`n_init=50`；
- 计算参考解的 Silhouette；
- 再使用 10 个随机种子拟合，每次 `n_init=20`；
- 计算这些结果相对参考解的 Adjusted Rand Index；
- 以 10 个 ARI 的中位数作为稳定性指标 `median_ari`。

只有同时满足：

$$
Silhouette\ge 0.40
$$

和：

$$
Median\ ARI\ge 0.80
$$

该 $K$ 才是合格候选。如果有多个合格候选，先选 Silhouette 最大者，再用 `median_ari` 打破并列。最终模型使用 `random_state=42`、`n_init=100` 重新拟合。

空间被接受后，只对该空间全部特征非缺失的主体预测 Cluster。缺少任一空间特征的主体仍保持未分类：其 `*_cluster` 为空，`*_cluster_accepted = 0`。对完整主体，`*_cluster_accepted = 1`。

如果某个空间没有任何 $K$ 同时通过两个门槛：

- 该空间不保留聚类结果；
- 对应的 `*_cluster` 全部为空；
- 对应的 `*_cluster_accepted` 全部为 0；
- 不会为了展示而强行指定类别数或命名策略类型。

通过门槛时，数值 Cluster 编号仍然只是辅助分析标识，不属于主体画像定义，跨年度也不应直接按编号比较。

### 5.4 04 的输出

默认目录：

```text
data/processed/final_clean/strategy_profile/2025/
```

固定输出：

```text
participant_strategy_profile_2025.csv
cluster_discovery_summary_2025.csv
full_9d_k_selection.csv
price_adjustment_3d_k_selection.csv
quantity_structure_3d_k_selection.csv
curve_shape_3d_k_selection.csv
```

`participant_strategy_profile_2025.csv` 是长期主体画像表，包含 9 个 LT 原值、9 个百分位、9 个独立状态、状态指纹及辅助聚类字段。完整 26 维动态策略输入还必须按 `participant_id` 合并 03 输出的 9 ST 和 8 Break。

每个 `*_k_selection.csv` 保存该空间各候选 $K$ 的 `silhouette` 和 `median_ari`。`cluster_discovery_summary_2025.csv` 保存四个空间的 `complete_discovery_n`、`complete_all_n`、是否通过门槛、最终 $K$、质量指标、最小 Cluster 主体数和最小 Cluster 占比。

只有空间被接受时，才额外生成：

```text
{space}_cluster_percentile_profile.csv
```

该文件保存各辅助 Cluster 的主体数，以及相关特征的中位经验百分位。这里的百分位在该空间的 complete-case 发现训练样本内部重新计算，不等同于主体主表中基于全部非缺失主体计算的 `*_percentile` 列。

---

## 6. 05：验证最终画像体系

脚本：

```text
scripts/05_validate_final_strategy_profile_v3_fast.py
```

该脚本验证的是“9 个独立维度 + 动态短期状态 + 辅助自然聚类发现”，不再验证旧版交叉组合画像。

### 6.1 长期特征跨期稳定性

对能直接映射回日级标量的 7 个 LT 特征，分别计算 1—6 月与 7—12 月主体中位数，再计算主体间 Spearman 相关系数。

当前不在该项中直接检验 `lt_strategy_persistence` 和 `lt_shape_variability`，因为代码中的 `LT_DAILY_MAP` 没有把它们视为单一日级标量。

输出：

```text
lt_split_half_stability.csv
```

### 6.2 长期特征冗余

对 9 个 LT 特征的全部两两组合计算 Spearman 相关和绝对相关。共检查：

$$
\binom{9}{2}=36
$$

对。`summary.txt` 会统计 $|Spearman|\ge0.90$ 的高冗余特征对数量。

输出：

```text
lt_feature_redundancy.csv
```

### 6.3 短期状态增量信息

对 8 个日级标量，比较两种对当日值的绝对误差：

- 长期基线：前 60 个日历日窗口，即 $d-67$ 到 $d-8$；
- 近期基线：前 7 个日历日窗口，即 $d-7$ 到 $d-1$。

改进比例定义为：

$$
Improvement=\frac{MAE_{long}-MAE_{recent}}{MAE_{long}}\times100\%
$$

该项当前只覆盖 `DAILY_SCALARS` 中的 8 个标量，不包含 `st_shape_shift`。

新版验证脚本与 03 fast 版本一样，先重建逐日历日索引，再使用 rolling 窗口批量计算近期和长期中位数，避免对每个主体日反复扫描历史。窗口边界和最少样本要求不变。

输出：

```text
st_incremental_signal.csv
```

### 6.4 独立状态覆盖

逐个长期特征统计低、中、高和缺失的主体数，并同时给出：

- `share_of_valid`：占该特征非缺失主体的比例；
- `share_of_all`：占全部主体的比例。

缺失状态的 `share_of_valid` 留空。由于并列秩和缺失值的存在，三种有效状态不保证机械地各占恰好三分之一。

输出：

```text
independent_state_coverage.csv
```

### 6.5 所有候选 K 的完整复核

对 `full_9d`、`price_adjustment_3d`、`quantity_structure_3d` 和 `curve_shape_3d` 分别复核 $K=2,\ldots,6$。每个空间只使用该空间的 complete cases，不做缺失值填补；缩尾和 `RobustScaler` 与 04 的聚类流程一致。

05 的候选复核基于最终画像表中该空间的全部 complete cases；04 的发现模型则先限定 `lt_ready_flag = 1` 再取空间 complete cases。两者样本口径应结合输出中的 `complete_n`、`complete_discovery_n` 和 `complete_all_n` 阅读。

每个候选 $K$ 计算：

| 指标 | 输出字段 | 解释 |
|---|---|---|
| Silhouette | `silhouette` | 越高表示类内更紧、类间更分离 |
| Calinski-Harabasz | `calinski_harabasz` | 越高通常表示分离结构更清晰 |
| Davies-Bouldin | `davies_bouldin` | 越低通常越好 |
| 多随机种子稳定性 | `median_ari` | 10 个随机种子相对参考解的 ARI 中位数 |
| 最小/最大 Cluster 规模 | `min_cluster_count`、`max_cluster_count` | 检查分组是否极不平衡 |
| 最小/最大 Cluster 占比 | `min_cluster_share`、`max_cluster_share` | Cluster 规模占完整样本的比例 |

候选复核的参考模型使用 `random_state=42`、`n_init=100`；稳定性复核使用 10 个随机种子，每次 `n_init=20`。

另外输出：

```text
small_cluster_lt_5pct_flag
tiny_cluster_lt_2pct_flag
```

小 Cluster 只会被标记，不会自动否决，因为它可能代表真实的少数策略。需要结合相邻 $K$、分离指标和业务解释判断是少数策略还是过度切分。

输出：

```text
cluster_candidate_k_review.csv
```

### 6.6 逐特征分离度

对每个空间、每个候选 $K$ 和每个空间特征执行 Kruskal-Wallis 检验，并输出：

```text
kruskal_H
kruskal_p
epsilon_squared
```

效应量按：

$$
\epsilon^2=\max\left(0,\frac{H-k+1}{N-k}\right)
$$

计算。它用于判断 Cluster 是否真的在各个组成特征上形成实质分离，而不仅仅依赖一个总体聚类指标。

输出：

```text
cluster_candidate_feature_separation.csv
```

### 6.7 04 最终选择的复核汇总

`selected_cluster_review.csv` 将 04 的接受/拒绝结果与 05 的候选指标合并。对被接受空间汇总最终 $K$ 的 Silhouette、Calinski-Harabasz、Davies-Bouldin、ARI、Cluster 规模以及特征 $\epsilon^2$；对未接受空间标记 `not_accepted`。

`summary.txt` 汇总长期稳定性、高冗余特征对、短期 MAE 改进和 04 最终选择。最小 Cluster 占比低于 5% 时会附加提示，但不会自动改写 04 的接受结果。

默认验证目录：

```text
results/final_clean_validation/2025/
```

完整输出清单：

```text
lt_split_half_stability.csv
lt_feature_redundancy.csv
st_incremental_signal.csv
independent_state_coverage.csv
cluster_candidate_k_review.csv
cluster_candidate_feature_separation.csv
selected_cluster_review.csv
summary.txt
```

---

## 7. 06：中文可视化

脚本：

```text
scripts/06_visualize_final_strategy_profile.py
```

默认输出目录：

```text
results/final_clean_visualization/2025/
```

固定生成四张中文图：

| 文件 | 内容 |
|---|---|
| `01_independent_feature_percentile_distributions.png` | 9 个独立长期维度的经验百分位分布 |
| `02_participant_9d_profile_heatmap.png` | 主体 × 9 维经验百分位热图 |
| `03_independent_feature_state_shares.png` | 每个特征低、中、高状态的有效主体占比 |
| `04_kmeans_natural_cluster_discovery.png` | 四个空间的自然聚类是否通过双门槛 |

对于每个被接受的空间，额外生成：

```text
05_{space}_accepted_cluster_heatmap.png
```

该图展示各辅助 Cluster 在相关特征上的中位经验百分位。未通过门槛的空间不会生成 Cluster 热图。

脚本优先使用 `Microsoft YaHei`、`SimHei`、`Noto Sans CJK SC`、`Source Han Sans CN` 或 `Arial Unicode MS`，并以 `bbox_inches="tight"` 保存图片，减少中文标签和图例被裁切的问题。

所有主图都围绕 9 个独立维度展开，不再绘制旧版组合画像数量图、组合画像 9 维热图或人为类型分布图。

---

## 8. 运行方式

先安装依赖：

```powershell
python -m pip install -r requirements.txt
```

当前仓库根目录没有 `run_all.ps1`，因此按以下顺序运行：

```powershell
python scripts\01_build_daily_strategy_core_v3.py --year 2025
python scripts\02_build_long_term_strategy_profile.py --year 2025
python scripts\03_build_short_term_strategy_state_fast.py --year 2025
python scripts\04_build_strategy_profile_v3_complete_case.py --year 2025
python scripts\05_validate_final_strategy_profile_v3_fast.py --year 2025
python scripts\06_visualize_final_strategy_profile.py --year 2025
```

04 和 05 在导入 NumPy / scikit-learn 前使用 `setdefault` 将 `OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 和 `LOKY_MAX_CPU_COUNT` 默认限制为 5，并屏蔽 Windows + MKL 下已知的非致命 KMeans/joblib 警告。如果运行环境已经显式设置这些变量，`setdefault` 不会覆盖现有值。

各脚本可配置路径：

| 脚本 | 路径参数 |
|---|---|
| `01_build_daily_strategy_core_v3.py` | `--raw-root`、`--out-root` |
| `02_build_long_term_strategy_profile.py` | `--daily-root`、`--out-root` |
| `03_build_short_term_strategy_state_fast.py` | `--daily-root`、`--out-root` |
| `04_build_strategy_profile_v3_complete_case.py` | `--lt-root`、`--out-root` |
| `05_validate_final_strategy_profile_v3_fast.py` | `--root`、`--results-root` |
| `06_visualize_final_strategy_profile.py` | `--profile-root`、`--results-root` |

默认依赖来自 `requirements.txt`：

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

---

## 9. 默认目录结构

```text
data/
├─ raw/
│  └─ energy_market_offers/2025/*.csv
└─ processed/
   └─ final_clean/
      ├─ daily/2025/
      │  └─ daily_strategy_core_2025.csv
      ├─ long_term/2025/
      │  └─ long_term_strategy_profile_2025.csv
      ├─ short_term/2025/
      │  └─ short_term_strategy_state_2025.csv
      └─ strategy_profile/2025/
         ├─ participant_strategy_profile_2025.csv
         ├─ cluster_discovery_summary_2025.csv
         ├─ full_9d_k_selection.csv
         ├─ price_adjustment_3d_k_selection.csv
         ├─ quantity_structure_3d_k_selection.csv
         ├─ curve_shape_3d_k_selection.csv
         └─ {space}_cluster_percentile_profile.csv  # 仅被接受空间

results/
├─ final_clean_validation/2025/
│  ├─ lt_split_half_stability.csv
│  ├─ lt_feature_redundancy.csv
│  ├─ st_incremental_signal.csv
│  ├─ independent_state_coverage.csv
│  ├─ cluster_candidate_k_review.csv
│  ├─ cluster_candidate_feature_separation.csv
│  ├─ selected_cluster_review.csv
│  └─ summary.txt
└─ final_clean_visualization/2025/
   ├─ 01_independent_feature_percentile_distributions.png
   ├─ 02_participant_9d_profile_heatmap.png
   ├─ 03_independent_feature_state_shares.png
   ├─ 04_kmeans_natural_cluster_discovery.png
   └─ 05_{space}_accepted_cluster_heatmap.png  # 仅被接受空间
```

---

## 10. 正确使用最终结果

完整 26 维动态策略画像不是单独保存在一张现成 CSV 中，而是由两张表按 `participant_id` 合并：

| 数据层 | 文件 | 取用字段 |
|---|---|---|
| 9 LT | `data/processed/final_clean/strategy_profile/2025/participant_strategy_profile_2025.csv` | 9 个原始 `lt_*` 连续值 |
| 9 ST + 8 Break | `data/processed/final_clean/short_term/2025/short_term_strategy_state_2025.csv` | 9 个 `st_*` 状态和 8 个 `break_*` 标志 |

合并后每一行代表主体 $i$ 在日期 $d$ 的完整策略状态：

$$
\Pi_{i,d}^{strategy}
=
\left[
9\ LT,
9\ ST,
8\ Break
\right]
\in\mathbb R^{26}
$$

推荐使用方式：

- 用 9 个 LT 原始连续值表示主体长期慢变量；
- 用 9 个 ST 表示主体截至当前日期的动态策略偏移；
- 用 8 个 Break 表示无法由普通 z-score 表达的突变事件；
- 用 9 个经验百分位进行跨特征尺度一致的比较、排序和画图；
- 用 9 个独立状态进行可解释的主体描述和规则筛选；
- 建模时不要把 LT 原值、LT 百分位和 LT 状态同时重复当作三套独立行为信息；
- `strategy_break_count`、`strategy_break_any` 和 `strategy_break_types` 是 8 个 Break 的派生汇总，不重复计入 26 维；
- 仅在某个空间通过 Silhouette 与 ARI 双门槛后，才把其 Cluster 作为辅助探索变量；
- 通过门槛后，也只解释该空间全部特征非缺失主体的 Cluster；
- 用 05 的相邻候选 $K$、Cluster 规模和逐特征 $\epsilon^2$ 复核最终选择。

不应再做以下解释：

- 不应把 KMeans Cluster 当成主体画像本身；
- 不应把四个空间都假定为必然存在自然分类；
- 不应在未通过双门槛时强行保留聚类标签；
- 不应对缺失空间特征的主体填补后强行分配 Cluster；
- 不应仅因最小 Cluster 低于 5% 就自动删除，它可能是真实少数策略；
- 不应恢复旧版 `3×4×2` 类型体系；
- 不应把 `independent_state_fingerprint` 当成新的组合类别；
- 不应直接比较不同年度的数值 Cluster 编号。

最终原则为：

$$
\boxed{
\text{完整策略画像}
=
\text{长期慢变量}
+
\text{短期动态状态}
+
\text{Strategy Break 突变事件}
}
$$

$$
\boxed{
\Pi_{i,d}^{strategy}
=
\left[
Z_i^{LT}(9),
Z_{i,d}^{ST}(9),
E_{i,d}^{break}(8)
\right]
\in\mathbb R^{26}
}
$$
