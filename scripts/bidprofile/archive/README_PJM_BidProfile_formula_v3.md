# PJM_BidProfile：电力市场主体报价策略画像

> 当前正式版本：历史报价画像 V1.0  
> 正式表示：**9 维长期策略画像 + 9 维短期策略状态 + Strategy Break 事件**  
> 当前验证数据：2025 年 PJM Energy Market Generation Offers  
> 设计原则：PJM 只作为公开验证样例。除 `01_standardize_pjm_bids_v2.py` 属于 PJM 数据适配器外，02 以后均按市场无关、能源类型无关方式设计，可迁移到火电、风电、光伏、水电和储能等主体。

---

## 1. 项目目标、流程与统一符号

本项目不直接采用“历史报价 → 黑箱模型 → 下一时报价”，而是先把历史报价分解为长期策略、近期策略偏移和策略突变，再与主体物理状态、市场状态共同用于完整报价曲线预测：

$$
\boxed{
Z_i^{LT}+Z_{i,t}^{ST}+E_{i,t}^{break}
+X_{i,t}^{physical}+X_t^{market}
\rightarrow \hat{\mathcal B}_{i,t}
}
$$

其中：

- $Z_i^{LT}$：主体长期策略画像，表示“主体平时怎么报”；
- $Z_{i,t}^{ST}$：主体短期策略状态，表示“最近相对长期习惯发生了什么变化”；
- $E_{i,t}^{break}$：策略突变事件，表示“长期完全稳定的策略维度最近是否被打破”；
- $X_{i,t}^{physical}$：容量、SOC、水位、启停、风光预测等物理状态；
- $X_t^{market}$：价格、负荷、供需紧张度、波动和不确定性等市场状态；
- $\hat{\mathcal B}_{i,t}$：预测的完整价格—数量报价曲线。

当前代码流程：

```text
PJM 原始报价
  ↓
01_standardize_pjm_bids_v2.py
  ↓ 统一 interval + segment 数据格式
02_build_interval_strategy_atoms_v2.py
  ↓ 单时段报价原子特征
03_build_temporal_strategy_behavior.py
  ↓ 主体自身历史偏离 + 相邻时段变化
04_build_long_term_strategy_profile_v3.py
  ↓ 日级摘要 + 长期候选画像
05_build_short_term_strategy_state_v2.py
  ↓ 因果短期候选状态
05c_validate_strategy_profile_stage.py
  ↓ 候选特征审计与人工门禁（稳定性、区分度、冗余性、短期增量信息）
05d_build_final_strategy_profile.py
  ↓ 特征收缩
正式 9 LT + 9 ST + Break
  ↓
05e_validate_final_strategy_profile.py
  ↓ 稳定性、冗余性、短期增量信息验证
05f_visualize_final_strategy_profile_v2.py
  ↓ 聚类倾向、聚类质量、策略类型和典型报价曲线验证
```

正式运行顺序：

```powershell
cd D:\pythonproject\PJM_BidProfile

python scripts\01_standardize_pjm_bids_v2.py --year 2025
python scripts\02_build_interval_strategy_atoms_v2.py --year 2025
python scripts\03_build_temporal_strategy_behavior.py --year 2025
python scripts\04_build_long_term_strategy_profile_v3.py --year 2025
python scripts\05_build_short_term_strategy_state_v2.py --year 2025
python scripts\05c_validate_strategy_profile_stage.py --year 2025
python scripts\05d_build_final_strategy_profile.py --year 2025
python scripts\05e_validate_final_strategy_profile.py --year 2025
python scripts\05f_visualize_final_strategy_profile_v2.py --year 2025
```

其中，05c 是 04/05 候选画像进入 05d 正式画像前的审计步骤。它输出稳定性、主体区分度、特征冗余性和因果短期增量报告，供人工确认特征收缩方案；当前 05d 的 `LT_CORE` 与 `ST_CORE` 仍在代码中固定定义，并不会自动读取 05c 的报告。因此这里将 05c 纳入**完整审计流程**，但它尚不是自动阻断 05d 的机器门禁。

为避免后文出现 `Median(daily_xxx)` 这种需要反复回查代码字段的写法，全文统一采用以下数学符号。

### 1.1 单时段报价曲线与基础符号

主体 $i$ 在时段 $t$ 的报价曲线：

$$
\mathcal B_{i,t}=\{(q_{i,t,k},p_{i,t,k})\}_{k=1}^{K_{i,t}}
$$

其中 $q_{i,t,k}$ 为累计报价量，$p_{i,t,k}$ 为价格。若数量跨度为正，定义归一化数量：

$$
x_{i,t,k}
=
\frac{q_{i,t,k}-q_{i,t,1}}
{q_{i,t,K}-q_{i,t,1}}
\in[0,1]
$$

由断点构造报价函数 $P_{i,t}(x)$。`block` 使用阶梯函数，`sloped` 使用分段线性函数。

后文使用：

$$
L_{i,t}
$$

表示单时段整体报价水平；

$$
H_{i,t}
$$

表示单时段容量配置 HHI；

$$
K^{eff}_{i,t}
$$

表示单时段有效报价段数；

$$
F_{i,t}\in\{0,1\}
$$

表示单时段是否为平价曲线；

$$
T_{i,t}
$$

表示尾部抬价比例；

$$
C_{i,t}
$$

表示曲线弯折程度；

$$
V_{i,t}\in\mathbb R^{21}
$$

表示去掉绝对价格水平后的 21 维归一化曲线形态。

### 1.2 日级策略符号

对主体 $i$ 的自然日 $d$，用 $\mathcal T_{i,d}$ 表示该日所有有效市场时段。定义：

$$
\ell_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}L_{i,t}
$$

为日典型报价水平；

$$
a_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}R_{i,t}
$$

为日有符号调整方向；

$$
m_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}|R_{i,t}|
$$

为日典型调整幅度；

$$
u_{i,d}
=
Q_{0.90,t\in\mathcal T_{i,d}}(|R_{i,t}|)
$$

为日较大调整幅度；

$$
h_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}H_{i,t}
$$

为日容量配置集中度；

$$
k_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}K^{eff}_{i,t}
$$

为日有效报价段数；

$$
f_{i,d}
=
\frac{1}{|\mathcal T_{i,d}|}
\sum_{t\in\mathcal T_{i,d}}F_{i,t}
$$

为日平价曲线比例；

$$
\tau_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}T_{i,t}
$$

为日尾部抬价水平；

$$
c_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}C_{i,t}
$$

为日曲线弯折水平；

$$
g_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}|\Delta L^{adj}_{i,t}|
$$

为日相邻时段价格变化幅度；

$$
s^{slot}_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}D^{slot}_{i,t}
$$

为日同时槽形态变化；

$$
s^{adj}_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}D^{adj}_{i,t}
$$

为日相邻时段形态变化；

$$
V_{i,d}
=
\operatorname{Med}_{t\in\mathcal T_{i,d}}V_{i,t}
$$

为日典型归一化报价曲线，逐维取中位数。

后文长期和短期特征都直接用这些数学符号定义，不再通过代码字段间接表示。

---

## 2. 01–03：统一报价、单时段原子特征与时序行为

### 2.1 `01_standardize_pjm_bids_v2.py`：统一报价格式

该脚本只负责把 PJM 的 `mw1~mw20 / bid1~bid20` 宽表转换为通用结构，不构造策略画像。

Interval 表：

| 变量名 | 中文含义 | 定义 |
|---|---|---|
| `interval_index` | 时段内部索引 | 单个源文件内的唯一时段编号，仅用于表连接 |
| `participant_id` | 市场主体编号 | PJM 中由 `unit_code` 映射；未来可换成其他市场主体 ID |
| `timestamp_utc` | UTC 时刻 | 用于确定真实物理时间顺序 |
| `timestamp_local` | 本地市场时刻 | 用于构造自然日和相同时槽 |
| `market_product` | 市场品种 | 当前为 `ENERGY` |
| `curve_mode` | 曲线模式 | `sloped` 为分段线性；`block` 为阶梯报价 |
| `has_offer` | 是否存在有效报价 | 至少一个有效 $(q,p)$ 时为 1 |
| `raw_valid_point_count` | 原始有效断点数 | $\sum_k I(q_{i,t,k},p_{i,t,k}\text{ 均有效})$ |
| `source_market/source_file/source_row_index` | 数据溯源字段 | 不参与画像计算 |

Segment 表：

| 变量名 | 中文含义 | 定义 |
|---|---|---|
| `interval_index` | 所属时段索引 | 与 interval 表连接 |
| `segment_id` | 报价段编号 | 按数量升序编号 $1,\dots,K_{i,t}$ |
| `quantity` | 累计报价量 | $q_{i,t,k}$ |
| `price` | 报价价格 | $p_{i,t,k}$ |

---

### 2.2 `02_build_interval_strategy_atoms_v2.py`：单时段策略原子特征

所有单时段特征都直接由 $\mathcal B_{i,t}$ 计算。

| 变量名 | 中文含义 | 数学定义 | 解释 |
|---|---|---|---|
| `bid_level` | 整体报价水平 | $\displaystyle L_{i,t}=\int_0^1P_{i,t}(x)\,dx$ | `block` 用阶梯面积，`sloped` 用梯形积分；单点/零跨度报价取该点价格 |
| `bid_floor` | 起始价格 | $\displaystyle P^{floor}_{i,t}=P_{i,t}(0)$ | 曲线最低数量端价格 |
| `bid_ceiling` | 末端价格 | $\displaystyle P^{ceil}_{i,t}=P_{i,t}(1)$ | 曲线最高数量端价格 |
| `price_range` | 价格跨度 | $\displaystyle R^{price}_{i,t}=P_{i,t}(1)-P_{i,t}(0)$ | 曲线整体价格变化幅度 |
| `quantity_span` | 报价量跨度 | $\displaystyle Q^{span}_{i,t}=q_{i,t,K}-q_{i,t,1}$ | 仅是报价曲线自身量程，不等价于真实可用容量 |
| `raw_segment_count` | 原始有效段数 | $\displaystyle K^{raw}_{i,t}=K_{i,t}$ | 清洗后的有效断点数量 |
| `effective_segment_count` | 有效价格层次数 | $\displaystyle K^{eff}_{i,t}=1+\sum_{k=2}^{K}I(\lvert p_{i,t,k}-p_{i,t,k-1}\rvert>\epsilon_p)$ | 合并连续同价断点后真正不同的价格层次数 |
| `quantity_hhi` | 容量配置集中度 | $\displaystyle s_{i,t,k}=\frac{q_{i,t,k+1}-q_{i,t,k}}{q_{i,t,K}-q_{i,t,1}},\quad H_{i,t}=\sum_k s_{i,t,k}^2$ | 越高表示报价容量越集中在少数价格区间 |
| `tail_uplift_ratio` | 尾部抬价比例 | $\displaystyle T_{i,t}=\frac{P_{i,t}(1)-P_{i,t}(0.8)}{\lvert P_{i,t}(1)-P_{i,t}(0)\rvert+\epsilon}$ | 总价格变化中有多少集中于最后 20% 容量 |
| `curve_bend_ratio` | 曲线弯折结构 | $\displaystyle C_{i,t}=\frac{[P_{i,t}(1)-P_{i,t}(0.8)]-[P_{i,t}(0.2)-P_{i,t}(0)]}{\lvert P_{i,t}(1)-P_{i,t}(0)\rvert+\epsilon}$ | 正值大表示尾部涨价强于头部 |
| `flat_curve_flag` | 平价曲线标志 | $\displaystyle F_{i,t}=I(\lvert P_{i,t}(1)-P_{i,t}(0)\rvert\le\epsilon_p)$ | 1 表示曲线价格基本不变 |
| `shape_defined_flag` | 形态是否可定义 | $\displaystyle I(K_{i,t}\ge2,\ Q^{span}_{i,t}>0,\ \lvert R^{price}_{i,t}\rvert>\epsilon_p)$ | 单点、零跨度、平价曲线不能定义归一化 shape |
| `shape_v00~shape_v20` | 归一化曲线形态 | $\displaystyle \widetilde P_{i,t}(x)=\frac{P_{i,t}(x)-P_{i,t}(0)}{P_{i,t}(1)-P_{i,t}(0)},\ x\in\{0,0.05,\dots,1\}$ | $V_{i,t}=[\widetilde P(0),\dots,\widetilde P(1)]$，去掉绝对价格水平，只保留曲线结构 |

---

### 2.3 `03_build_temporal_strategy_behavior.py`：主体自身历史偏离

定义本地市场时槽：

$$
slot(t)=3600\,hour(t)+60\,minute(t)+second(t)
$$

默认最多使用当前时刻之前、相同时槽的 30 个有效 `bid_level`，至少 5 个历史观测才认为基准可靠。定义：

$$
\mathcal H^{slot}_{i,t}
=
\{\tau<t:slot(\tau)=slot(t)\}
$$

自身历史报价基准：

$$
B_{i,t}
=
\operatorname{Med}_{\tau\in\mathcal H^{slot}_{i,t}}
L_{i,\tau}
$$

当前相对自身常态的有符号偏离：

$$
R_{i,t}
=
L_{i,t}-B_{i,t}
$$

同时槽与相邻时段行为：

| 变量名 | 中文含义 | 数学定义 |
|---|---|---|
| `same_slot_history_count` | 同时槽历史样本数 | $\displaystyle N^{slot}_{i,t}=\lvert\mathcal H^{slot}_{i,t}\rvert$，最多 30 |
| `self_bid_level_baseline` | 自身同时槽历史基准 | $\displaystyle B_{i,t}=\operatorname{Med}_{\tau\in\mathcal H^{slot}_{i,t}}L_{i,\tau}$ |
| `self_bid_level_residual` | 相对自身常态的报价偏离 | $\displaystyle R_{i,t}=L_{i,t}-B_{i,t}$ |
| `same_slot_level_change` | 相比上一次同时槽的价格变化 | $\displaystyle \Delta L^{slot}_{i,t}=L_{i,t}-L_{i,t^{-slot}}$ |
| `same_slot_shape_distance` | 相比上一次同时槽的形态变化 | $\displaystyle D^{slot}_{i,t}=\sqrt{\frac1{21}\sum_{m=1}^{21}(V_{i,t,m}-V_{i,t^{-slot},m})^2}$ |
| `baseline_ready_flag` | 历史基准可用标志 | $\displaystyle I(N^{slot}_{i,t}\ge5)$ |
| `prev_time_gap_seconds` | 与上一条记录的时间差 | $\displaystyle \Delta\tau_{i,t}=t-t^{-}$ |
| `nominal_interval_seconds` | 市场标准时间间隔 | 全体唯一 UTC 时刻差值的众数，记为 $\Delta\tau^{nom}$ |
| `adjacent_interval_flag` | 是否为真实相邻时段 | $\displaystyle I(\lvert\Delta\tau_{i,t}-\Delta\tau^{nom}\rvert\le1s)$ |
| `adjacent_bid_level_change` | 相邻时段价格变化 | $\displaystyle \Delta L^{adj}_{i,t}=L_{i,t}-L_{i,t-1}$ |
| `adjacent_quantity_hhi_change` | 相邻时段容量集中度变化 | $\displaystyle \Delta H^{adj}_{i,t}=H_{i,t}-H_{i,t-1}$ |
| `adjacent_effective_segment_count_change` | 相邻时段有效段数变化 | $\displaystyle \Delta K^{adj}_{i,t}=K^{eff}_{i,t}-K^{eff}_{i,t-1}$ |
| `adjacent_tail_uplift_ratio_change` | 相邻时段尾部策略变化 | $\displaystyle \Delta T^{adj}_{i,t}=T_{i,t}-T_{i,t-1}$ |
| `adjacent_curve_bend_ratio_change` | 相邻时段弯折结构变化 | $\displaystyle \Delta C^{adj}_{i,t}=C_{i,t}-C_{i,t-1}$ |
| `adjacent_shape_distance` | 相邻时段形态变化 | $\displaystyle D^{adj}_{i,t}=\sqrt{\frac1{21}\sum_{m=1}^{21}(V_{i,t,m}-V_{i,t-1,m})^2}$ |
| `curve_mode_switch_flag` | block/sloped 切换 | $\displaystyle I(mode_{i,t}\neq mode_{i,t-1})$ |
| `flat_curve_switch_flag` | 平价/非平价切换 | $\displaystyle I(F_{i,t}\neq F_{i,t-1})$ |

---

## 3. 04–05：从日级策略到长期候选画像和短期候选状态

### 3.1 `04_build_long_term_strategy_profile_v3.py`：日级特征

所有 interval 先按 `local_date` 合并，再对主体 $i$ 的自然日 $d$ 做日级聚合。这样一天只占一个权重，不受 5/15/30/60 分钟市场时间粒度影响。

代码字段与前文日级数学符号一一对应：

| 代码变量 | 数学符号 | 中文含义 | 公式 |
|---|---:|---|---|
| `daily_bid_level` | $\ell_{i,d}$ | 日典型报价水平 | $\displaystyle \ell_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}L_{i,t}$ |
| `daily_self_adjustment_bias` | $a_{i,d}$ | 日整体调整方向 | $\displaystyle a_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}R_{i,t}$ |
| `daily_self_adjustment_magnitude` | $m_{i,d}$ | 日典型调整幅度 | $\displaystyle m_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}\lvert R_{i,t}\rvert$ |
| `daily_self_adjustment_p90` | $u_{i,d}$ | 日较大调整幅度 | $\displaystyle u_{i,d}=Q_{0.90,t\in\mathcal T_{i,d}}(\lvert R_{i,t}\rvert)$ |
| `daily_adjacent_level_change_magnitude` | $g_{i,d}$ | 日相邻时段价格变化幅度 | $\displaystyle g_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}\lvert\Delta L^{adj}_{i,t}\rvert$ |
| `daily_quantity_hhi` | $h_{i,d}$ | 日容量配置集中度 | $\displaystyle h_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}H_{i,t}$ |
| `daily_effective_segment_count` | $k_{i,d}$ | 日有效报价段数 | $\displaystyle k_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}K^{eff}_{i,t}$ |
| `daily_flat_curve_rate` | $f_{i,d}$ | 日平价曲线比例 | $\displaystyle f_{i,d}=\frac1{\lvert\mathcal T_{i,d}\rvert}\sum_{t\in\mathcal T_{i,d}}F_{i,t}$ |
| `daily_tail_uplift_ratio` | $\tau_{i,d}$ | 日尾部抬价水平 | $\displaystyle \tau_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}T_{i,t}$ |
| `daily_curve_bend_ratio` | $c_{i,d}$ | 日曲线弯折水平 | $\displaystyle c_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}C_{i,t}$ |
| `daily_same_slot_shape_change` | $s^{slot}_{i,d}$ | 日同时槽形态变化 | $\displaystyle s^{slot}_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}D^{slot}_{i,t}$ |
| `daily_adjacent_shape_change` | $s^{adj}_{i,d}$ | 日相邻形态变化 | $\displaystyle s^{adj}_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}D^{adj}_{i,t}$ |
| `daily_adjacent_shape_change_p90` | $s^{adj,90}_{i,d}$ | 当日较大相邻形态变化 | $\displaystyle s^{adj,90}_{i,d}=Q_{0.90,t\in\mathcal T_{i,d}}(D^{adj}_{i,t})$ |
| `daily_curve_mode_switch_rate` | $r^{mode}_{i,d}$ | 当日 block/sloped 切换率 | $\displaystyle r^{mode}_{i,d}=\frac1{\lvert\mathcal T_{i,d}\rvert}\sum_t I(mode_{i,t}\neq mode_{i,t-1})$ |
| `daily_flat_curve_switch_rate` | $r^{flat}_{i,d}$ | 当日平价结构切换率 | $\displaystyle r^{flat}_{i,d}=\frac1{\lvert\mathcal T_{i,d}\rvert}\sum_t I(F_{i,t}\neq F_{i,t-1})$ |
| `daily_shape_v00~20` | $V_{i,d}$ | 日典型归一化曲线 | $\displaystyle V_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}V_{i,t}$，逐维中位数 |

日级 QC 变量：

$$
N^{obs}_{i,d}=|\mathcal T^{obs}_{i,d}|
$$

表示当日观测时段数；

$$
N^{valid}_{i,d}
=
\sum_t I(L_{i,t}\text{ 有效})
$$

表示有效报价时段数；

$$
N^{base}_{i,d}
=
\sum_t I(N^{slot}_{i,t}\ge5)
$$

表示历史基准可用时段数；

$$
N^{shape}_{i,d}
=
\sum_t I(V_{i,t}\text{ 可定义})
$$

表示形态可定义时段数。因此：

$$
r^{valid}_{i,d}
=
\frac{N^{valid}_{i,d}}{N^{obs}_{i,d}}
$$

$$
r^{base}_{i,d}
=
\frac{N^{base}_{i,d}}{N^{obs}_{i,d}}
$$

$$
r^{shape}_{i,d}
=
\frac{N^{shape}_{i,d}}{N^{obs}_{i,d}}
$$

分别对应 `daily_valid_bid_rate`、`daily_baseline_ready_rate` 和 `daily_shape_defined_rate`。

### 3.2 长期典型形态和长期候选变量

主体长期典型归一化曲线：

$$
V_i^{LT}
=
\operatorname{Med}_{d\in\mathcal D_i}V_{i,d}
$$

逐维取中位数。日级形态偏离：

$$
d^{shape}_{i,d}
=
\sqrt{
\frac1{21}
\sum_{m=1}^{21}
(V_{i,d,m}-V^{LT}_{i,m})^2
}
$$

对应 `daily_shape_deviation`。

04 输出的长期候选变量全部可直接写成：

| 代码变量 | 数学定义 | 中文含义 |
|---|---|---|
| `lt_bid_level` | $\displaystyle L_i^{LT}=\operatorname{Med}_{d\in\mathcal D_i}\ell_{i,d}$ | 长期典型报价水平 |
| `lt_bid_level_scale` | $\displaystyle S^L_i=1.4826\,MAD_{d\in\mathcal D_i}(\ell_{i,d})$ | 报价水平长期稳健尺度 |
| `lt_self_adjustment_bias` | $\displaystyle A_i^{bias}=\operatorname{Med}_d a_{i,d}$ | 长期有符号调整偏置 |
| `lt_self_adjustment_magnitude` | $\displaystyle A_i^{mag}=\operatorname{Med}_d m_{i,d}$ | 长期典型调整幅度 |
| `lt_self_adjustment_p90` | $\displaystyle A_i^{90}=\operatorname{Med}_d u_{i,d}$ | 长期较大调整幅度 |
| `lt_strategy_persistence` | $\displaystyle A_i^{persist}=Corr(a_{i,d},a_{i,d-1})$ | 连续自然日策略偏移的滞后 1 持续性 |
| `lt_quantity_hhi` | $\displaystyle H_i^{LT}=\operatorname{Med}_d h_{i,d}$ | 长期容量配置集中度 |
| `lt_quantity_hhi_scale` | $\displaystyle S^H_i=1.4826\,MAD_d(h_{i,d})$ | HHI 长期稳健尺度 |
| `lt_effective_segment_count` | $\displaystyle K_i^{LT}=\operatorname{Med}_d k_{i,d}$ | 长期典型有效段数 |
| `lt_flat_curve_rate` | $\displaystyle F_i^{LT}=\operatorname{Med}_d f_{i,d}$ | 长期平价曲线偏好 |
| `lt_tail_uplift_ratio` | $\displaystyle T_i^{LT}=\operatorname{Med}_d \tau_{i,d}$ | 长期尾部抬价倾向 |
| `lt_curve_bend_ratio` | $\displaystyle C_i^{LT}=\operatorname{Med}_d c_{i,d}$ | 长期曲线弯折结构 |
| `lt_shape_defined_rate` | $\displaystyle R_i^{shape}=\operatorname{Med}_d r^{shape}_{i,d}$ | 长期 shape 可定义率，属于 QC |
| `lt_shape_v00~20` | $\displaystyle V_i^{LT}=\operatorname{Med}_dV_{i,d}$ | 长期典型归一化曲线 |
| `lt_shape_day_deviation_median` | $\displaystyle D_i^{shape}=\operatorname{Med}_d d^{shape}_{i,d}$ | 长期常规形态波动 |
| `lt_shape_day_deviation_p90` | $\displaystyle D_i^{shape,90}=Q_{0.90,d}(d^{shape}_{i,d})$ | 长期较大形态波动 |
| `lt_adjacent_level_change_magnitude` | $\displaystyle G_i^{LT}=\operatorname{Med}_d g_{i,d}$ | 长期相邻时段价格变化幅度 |
| `lt_same_slot_shape_change` | $\displaystyle S_i^{slot}=\operatorname{Med}_d s^{slot}_{i,d}$ | 长期同时槽形态变化习惯 |
| `lt_adjacent_shape_change` | $\displaystyle S_i^{adj}=\operatorname{Med}_d s^{adj}_{i,d}$ | 长期相邻时段形态变化习惯 |
| `lt_adjacent_shape_change_p90` | $\displaystyle S_i^{adj,90}=\operatorname{Med}_d s^{adj,90}_{i,d}$ | 长期较大相邻形态变化 |
| `lt_curve_mode_switch_rate` | $\displaystyle R_i^{mode}=\operatorname{Med}_d r^{mode}_{i,d}$ | 长期曲线模式切换率 |
| `lt_flat_curve_switch_rate` | $\displaystyle R_i^{flat}=\operatorname{Med}_d r^{flat}_{i,d}$ | 长期平价结构切换率 |
| `lt_active_days` | $\displaystyle N_i^{day}=\lvert\mathcal D_i\rvert$ | 主体有效活动日数 |
| `lt_total_intervals` | $\displaystyle N_i^{obs}=\sum_d N^{obs}_{i,d}$ | 全年总观测时段数 |
| `lt_valid_bid_intervals` | $\displaystyle N_i^{valid}=\sum_d N^{valid}_{i,d}$ | 全年有效报价时段数 |
| `lt_valid_bid_rate` | $\displaystyle R_i^{valid}=\operatorname{Med}_d r^{valid}_{i,d}$ | 长期有效报价覆盖率 |
| `lt_baseline_ready_rate` | $\displaystyle R_i^{base}=\operatorname{Med}_d r^{base}_{i,d}$ | 长期历史基准覆盖率 |
| `lt_persistence_pair_count` | $\displaystyle N_i^{pair}=\lvert\{d:(d-1,d)\text{ 连续且 }a_{i,d-1},a_{i,d}\text{ 有效}\}\rvert$ | persistence 使用的连续日对数 |

---

### 3.3 `05_build_short_term_strategy_state_v2.py`：短期候选状态

对状态日期 $d$，只使用过去数据：

$$
\mathcal H_S(d)=\{d-7,\dots,d-1\}
$$

$$
\mathcal H_L(d)=\{d-67,\dots,d-8\}
$$

最近 7 日和更早 60 日完全不重叠。

对任意一个日级策略变量

$$
x_{i,d}
\in
\{
\ell_{i,d},a_{i,d},m_{i,d},g_{i,d},
h_{i,d},k_{i,d},f_{i,d},\tau_{i,d},c_{i,d},
s^{slot}_{i,d},s^{adj}_{i,d},r^{mode}_{i,d},r^{flat}_{i,d}
\}
$$

统一定义：

最近策略水平：

$$
x^{S}_{i,d}
=
\operatorname{Med}_{\delta\in\mathcal H_S(d)}x_{i,\delta}
$$

长期参考基准：

$$
x^{L}_{i,d}
=
\operatorname{Med}_{\delta\in\mathcal H_L(d)}x_{i,\delta}
$$

长期稳健尺度：

$$
\sigma^{rob}_{x,i,d}
=
1.4826\,
MAD_{\delta\in\mathcal H_L(d)}(x_{i,\delta})
$$

原始近期偏移：

$$
\Delta x_{i,d}
=
x^S_{i,d}-x^L_{i,d}
$$

若 $\sigma^{rob}_{x,i,d}>0$，短期稳健标准化状态：

$$
Z^x_{i,d}
=
\frac{\Delta x_{i,d}}
{\sigma^{rob}_{x,i,d}}
$$

代码中的 `*_recent`、`*_baseline`、`*_baseline_scale`、`st_*_raw_shift`、`st_*_z` 分别就是上式的 $x^S_{i,d}$、$x^L_{i,d}$、$\sigma^{rob}_{x,i,d}$、$\Delta x_{i,d}$、$Z^x_{i,d}$。

最低有效样本要求：

$$
N^{S}_{x,i,d}\ge3,\qquad
N^{L}_{x,i,d}\ge20
$$

其中 $N^{S}_{x,i,d}$ 和 $N^{L}_{x,i,d}$ 分别对应 `*_recent_days` 和 `*_long_days`。

若：

$$
\sigma^{rob}_{x,i,d}=0
$$

且：

$$
\Delta x_{i,d}=0
$$

则定义：

$$
Z^x_{i,d}=0
$$

若：

$$
\sigma^{rob}_{x,i,d}=0,\qquad
\Delta x_{i,d}\neq0
$$

则不构造伪极端 z-score，而定义：

$$
Break^x_{i,d}=1
$$

对应 `*_zero_scale_change_flag=1`，同时 `st_*_z` 保持空值。

形态短期状态单独定义。最近 7 日典型形态：

$$
V^S_{i,d}
=
\operatorname{Med}_{\delta\in\mathcal H_S(d)}V_{i,\delta}
$$

更早 60 日典型形态：

$$
V^L_{i,d}
=
\operatorname{Med}_{\delta\in\mathcal H_L(d)}V_{i,\delta}
$$

近期形态迁移：

$$
D^{ST,shape}_{i,d}
=
\sqrt{
\frac1{21}
\sum_{m=1}^{21}
(V^S_{i,d,m}-V^L_{i,d,m})^2
}
$$

分别对应 `st_shape_v00~20` 和 `st_shape_prototype_shift`。

### 3.4 `05c_validate_strategy_profile_stage.py`：候选画像审计

05c 不构造新的画像变量，而是同时读取 04 生成的日级摘要与长期候选画像、05 生成的短期候选状态，从四个方面检查候选特征是否适合进入正式画像：

1. 长期特征在上下半年之间是否保持稳定的主体排序；
2. 主体间差异是否明显大于单个主体自身的日间波动；
3. 长期候选特征之间是否存在严重冗余；
4. 最近窗口形成的短期状态是否比更早长期基准包含额外的因果预测信息。

主要输出目录为：

```text
results/profile_stage_validation/2025/
```

这些结果用于人工确认后续特征收缩方案。05c 当前不会输出供 05d 自动读取的“入选特征清单”，也没有单一的自动通过/失败阈值；因此它是完整审计流程中的人工门禁，而 05e 则负责对已经冻结的最终 `9 LT + 9 ST + Break` 再次验证。两者验证对象不同，不能相互替代。

---

## 4. 05d：最终正式 9 LT + 9 ST + Break

`05d_build_final_strategy_profile.py` 不重新定义新行为，只从 04/05 候选变量中保留最终核心画像。

### 4.1 最终 9 维长期策略画像

$$
\boxed{
Z_i^{LT}
=
[
L_i^{LT},
A_i^{mag},
A_i^{persist},
H_i^{LT},
K_i^{LT},
F_i^{LT},
T_i^{LT},
C_i^{LT},
D_i^{shape}
]
}
$$

代码字段与公式：

| 正式变量名 | 数学符号 | 具体公式 | 含义 |
|---|---:|---|---|
| `lt_bid_level` | $L_i^{LT}$ | $\displaystyle L_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t\in\mathcal T_{i,d}}\int_0^1P_{i,t}(x)\,dx$ | 长期典型报价水平 |
| `lt_self_adjustment_magnitude` | $A_i^{mag}$ | $\displaystyle A_i^{mag}=\operatorname{Med}_{d}\operatorname{Med}_{t\in\mathcal T_{i,d}}\lvert L_{i,t}-B_{i,t}\rvert$ | 长期主动调整幅度 |
| `lt_strategy_persistence` | $A_i^{persist}$ | $\displaystyle A_i^{persist}=Corr(a_{i,d},a_{i,d-1})$，其中 $\displaystyle a_{i,d}=\operatorname{Med}_{t\in\mathcal T_{i,d}}(L_{i,t}-B_{i,t})$ | 策略偏移的跨日持续性 |
| `lt_quantity_hhi` | $H_i^{LT}$ | $\displaystyle H_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t\in\mathcal T_{i,d}}\sum_k\left(\frac{q_{i,t,k+1}-q_{i,t,k}}{q_{i,t,K}-q_{i,t,1}}\right)^2$ | 长期容量配置集中度 |
| `lt_effective_segment_count` | $K_i^{LT}$ | $\displaystyle K_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t\in\mathcal T_{i,d}}\left[1+\sum_{k=2}^{K}I(\lvert p_{i,t,k}-p_{i,t,k-1}\rvert>\epsilon_p)\right]$ | 长期报价结构复杂度 |
| `lt_flat_curve_rate` | $F_i^{LT}$ | $\displaystyle F_i^{LT}=\operatorname{Med}_{d}\left[\frac1{\lvert\mathcal T_{i,d}\rvert}\sum_{t\in\mathcal T_{i,d}}I(\lvert P_{i,t}(1)-P_{i,t}(0)\rvert\le\epsilon_p)\right]$ | 长期平价报价偏好 |
| `lt_tail_uplift_ratio` | $T_i^{LT}$ | $\displaystyle T_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t\in\mathcal T_{i,d}}\frac{P_{i,t}(1)-P_{i,t}(0.8)}{\lvert P_{i,t}(1)-P_{i,t}(0)\rvert+\epsilon}$ | 长期尾部抬价倾向 |
| `lt_curve_bend_ratio` | $C_i^{LT}$ | $\displaystyle C_i^{LT}=\operatorname{Med}_{d}\operatorname{Med}_{t\in\mathcal T_{i,d}}\frac{[P_{i,t}(1)-P_{i,t}(0.8)]-[P_{i,t}(0.2)-P_{i,t}(0)]}{\lvert P_{i,t}(1)-P_{i,t}(0)\rvert+\epsilon}$ | 长期曲线前后段增长结构 |
| `lt_shape_day_deviation_median` | $D_i^{shape}$ | $\displaystyle D_i^{shape}=\operatorname{Med}_{d}\sqrt{\frac1{21}\sum_{m=1}^{21}(V_{i,d,m}-V^{LT}_{i,m})^2}$ | 长期曲线形态稳定性 |

因此这 9 个最终 LT 不再依赖任何未解释的 `daily_xxx` 字段；每个变量都可以从最初报价断点 $(q,p)$ 或由其产生的自身历史偏离直接追溯。

05d 完整度辅助量：

$$
N_i^{LT,obs}
=
\sum_{j=1}^{9}I(Z_{i,j}^{LT}\text{ 非缺失})
$$

对应 `lt_core_nonmissing_count`；

$$
R_i^{LT,cover}
=
\frac{N_i^{LT,obs}}{9}
$$

对应 `lt_core_coverage`；

$$
Ready_i^{LT}
=
I(N_i^{LT,obs}\ge7)
$$

对应 `lt_core_ready_flag`。

`Ready_i^{LT}=0` 不表示主体“没有策略画像”，只表示 9 个核心维度中缺失较多。长期平价/单点报价主体天然可能缺少 shape，长期完全固定主体也可能无法定义 persistence。

### 4.2 最终 9 维短期策略状态

正式短期向量：

$$
\boxed{
Z_{i,d}^{ST}
=
[
Z^L_{i,d},
Z^A_{i,d},
Z^M_{i,d},
Z^H_{i,d},
Z^K_{i,d},
Z^F_{i,d},
Z^T_{i,d},
Z^C_{i,d},
D^{ST,shape}_{i,d}
]
}
$$

其中前 8 个统一使用：

$$
Z^x_{i,d}
=
\frac{
\operatorname{Med}_{\delta\in[d-7,d-1]}x_{i,\delta}
-
\operatorname{Med}_{\delta\in[d-67,d-8]}x_{i,\delta}
}{
1.4826\,MAD_{\delta\in[d-67,d-8]}(x_{i,\delta})
}
$$

具体映射：

| 正式变量名 | 使用的日级变量 $x_{i,d}$ | 完整公式含义 |
|---|---|---|
| `st_bid_level_z` | $x_{i,d}=\ell_{i,d}$ | 最近 7 日日典型报价水平相对更早 60 日报价水平基准的稳健标准化偏移 |
| `st_adjustment_bias_z` | $x_{i,d}=a_{i,d}$ | 最近 7 日有符号调整方向相对长期基准的偏移 |
| `st_adjustment_magnitude_z` | $x_{i,d}=m_{i,d}$ | 最近 7 日调整幅度相对长期习惯的偏移 |
| `st_quantity_hhi_z` | $x_{i,d}=h_{i,d}$ | 最近容量配置集中度相对长期习惯的偏移 |
| `st_effective_segment_count_z` | $x_{i,d}=k_{i,d}$ | 最近报价段数相对长期习惯的偏移 |
| `st_flat_curve_rate_z` | $x_{i,d}=f_{i,d}$ | 最近平价结构比例相对长期习惯的偏移 |
| `st_tail_uplift_ratio_z` | $x_{i,d}=\tau_{i,d}$ | 最近尾部抬价比例相对长期习惯的偏移 |
| `st_curve_bend_ratio_z` | $x_{i,d}=c_{i,d}$ | 最近曲线弯折结构相对长期习惯的偏移 |
| `st_shape_prototype_shift` | $V^S_{i,d},V^L_{i,d}$ | $\displaystyle D^{ST,shape}_{i,d}=\sqrt{\frac1{21}\sum_{m=1}^{21}(V^S_{i,d,m}-V^L_{i,d,m})^2}$ |

### 4.3 Strategy Break

Break 只对 8 个标量策略维度：

$$
x\in
\{
\ell,a,m,h,k,f,\tau,c
\}
$$

构造。

若：

$$
1.4826\,MAD_{\delta\in[d-67,d-8]}(x_{i,\delta})=0
$$

且：

$$
\operatorname{Med}_{\delta\in[d-7,d-1]}x_{i,\delta}
\neq
\operatorname{Med}_{\delta\in[d-67,d-8]}x_{i,\delta}
$$

则：

$$
Break^x_{i,d}=1
$$

否则：

$$
Break^x_{i,d}=0
$$

总 Break 数：

$$
N^{break}_{i,d}
=
\sum_{x\in\{\ell,a,m,h,k,f,\tau,c\}}
Break^x_{i,d}
$$

对应 `strategy_break_count`。

任意 Break 标志：

$$
E^{break}_{i,d}
=
I(N^{break}_{i,d}>0)
$$

对应 `strategy_break_any_flag`。

`strategy_break_types` 仅记录哪些 $x$ 满足 $Break^x_{i,d}=1$。

当前 2025 年：

$$
42,148/479,115=8.80\%
$$

的主体日至少出现一次 Break。

### 4.4 最终删减逻辑

最终 9+9+Break 不是主观挑选，而是基于长期稳定性、冗余性和短期因果增量验证收缩：

- `lt_self_adjustment_bias`：长期中位数天然围绕 0，不适合做长期主体区分，但其短期标准化偏移 `st_adjustment_bias_z` 有意义；
- `lt_self_adjustment_p90`：与 $A_i^{mag}$ 的 Spearman 约 0.978，冗余；
- `lt_bid_level_scale`：与调整幅度高度相关，保留作标准化辅助量；
- `lt_shape_defined_rate`：更接近 shape 是否可计算的 QC，不作为策略本身；
- `curve_mode_switch_rate`：几乎不变化，删除；
- `flat_curve_switch_rate`：多数日期为 0，降为稀有结构事件；
- `st_adjacent_level_change_z`：过于局部，且大量日期无变化；
- `st_same_slot_shape_change_z`：长期稳定但短期增量信息弱；
- `st_adjacent_shape_change_z`：作为辅助量保留，上位核心形态状态改用 $D^{ST,shape}_{i,d}$。

---

## 5. 05e：最终画像验证

### 5.1 长期跨期稳定性

对每个可以直接由日级变量 $x_{i,d}$ 重算的 LT 核心特征，定义上半年和下半年：

$$
x_i^{H1}
=
\operatorname{Med}_{d\in Jan-Jun}x_{i,d}
$$

$$
x_i^{H2}
=
\operatorname{Med}_{d\in Jul-Dec}x_{i,d}
$$

跨期稳定性：

$$
\rho_x
=
Spearman(x_i^{H1},x_i^{H2})
$$

当前最终结果：

```text
Validated final LT traits: 8
Median H1-H2 Spearman: 0.9489
Positive H1-H2 traits: 8/8
```

`lt_strategy_persistence` 本身已经是相关系数型特征，不对应一个简单日级原子量，因此 split-half 直接验证其余 8 个 LT。

### 5.2 主体间差异与主体内波动

主体长期中位数：

$$
\mu_i^x
=
\operatorname{Med}_{d}x_{i,d}
$$

主体间稳健尺度：

$$
S_x^{between}
=
1.4826\,MAD_i(\mu_i^x)
$$

主体自身日间稳健尺度：

$$
S_{i,x}^{within}
=
1.4826\,MAD_d(x_{i,d})
$$

典型主体内尺度：

$$
S_x^{within}
=
\operatorname{Med}_iS_{i,x}^{within}
$$

若 $S_x^{within}>0$，计算：

$$
R_x^{sep}
=
\frac{S_x^{between}}{S_x^{within}}
$$

若：

$$
S_x^{within}=0,\qquad
S_x^{between}>0
$$

不再输出无意义的超大比例，而标记：

```text
stable_cross_participant_difference
```

表示“主体内部长期稳定，但不同主体之间明显不同”。

### 5.3 最终核心冗余

对最终长期核心特征任意两维 $Z_{i,a}^{LT}$ 与 $Z_{i,b}^{LT}$：

$$
\rho_{ab}
=
Spearman(Z_{i,a}^{LT},Z_{i,b}^{LT})
$$

当前：

```text
Pairs with |Spearman| >= 0.90: 0
```

说明删减后不存在明显高度冗余的核心 LT 特征。

### 5.4 短期状态增量信息

对任一标量日级行为 $x_{i,d}$，长期基准预测：

$$
\hat x^{long}_{i,d}
=
\operatorname{Med}_{\delta\in[d-67,d-8]}x_{i,\delta}
$$

近期预测：

$$
\hat x^{recent}_{i,d}
=
\operatorname{Med}_{\delta\in[d-7,d-1]}x_{i,\delta}
$$

分别计算：

$$
MAE_{long}
=
Mean\left(
|x_{i,d}-\hat x^{long}_{i,d}|
\right)
$$

$$
MAE_{recent}
=
Mean\left(
|x_{i,d}-\hat x^{recent}_{i,d}|
\right)
$$

短期增量改善率：

$$
Improvement_x
=
\frac{
MAE_{long}-MAE_{recent}
}{
MAE_{long}
}\times100\%
$$

当前：

```text
Validated final scalar ST states: 8
Median recent-window MAE improvement: 49.81%
Positive MAE improvement: 8/8
```

因此最终 8 个标量 ST 状态全部具有正的近期增量信息。


---

## 6. 当前阶段结果与正式文件

当前 2025 年结果：

```text
Participants: 1,366
Participant-day short-term states: 479,115

Long-term core features: 9
Short-term core states: 9
Core break dimensions: 8

Long-term core-ready participants: 652 (47.73%)
Short-term core-ready states: 199,364 (41.61%)
Participant-days with >=1 strategy break: 42,148 (8.80%)
```

最终验证：

```text
Long-term split-half:
  Validated final LT traits: 8
  Median H1-H2 Spearman: 0.9489
  Positive H1-H2 traits: 8/8

Final-core redundancy:
  Pairs with |Spearman| >= 0.90: 0

Causal short-term signal:
  Validated final scalar ST states: 8
  Median recent-window MAE improvement: 49.81%
  Positive MAE improvement: 8/8
```

当前可以支持：

$$
\boxed{
\text{主体历史报价中存在稳定的长期策略属性}
}
$$

$$
\boxed{
\text{收缩后的 9 维 LT 不存在明显高度冗余}
}
$$

$$
\boxed{
\text{近期策略状态相对长期静态基准具有显著增量信息}
}
$$

$$
\boxed{
\text{部分长期完全稳定的策略需要用 Break 而不是连续 z-score 描述}
}
$$

尚未完成：

$$
\text{市场相对价格定位}
+
\text{市场响应}
+
\text{风险倾向}
$$

这些必须接入市场价格、负荷、供需紧张度、波动、预测误差以及主体物理状态后再定义。

正式建模和展示优先使用：

```text
data/processed/final_strategy_profile/2025/
├─ final_long_term_profile_2025.csv
├─ final_short_term_state_2025.csv
├─ strategy_break_events_2025.csv
└─ final_feature_dictionary_2025.csv
```

其中：

- `final_long_term_profile_2025.csv`：正式 9 维 LT；
- `final_short_term_state_2025.csv`：正式 9 维 ST；
- `strategy_break_events_2025.csv`：8 类 Break；
- `final_feature_dictionary_2025.csv`：正式变量字典。

04/05 的完整候选变量继续保留用于审计和敏感性分析，但不默认作为正式模型输入。
