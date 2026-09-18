# MarketBidPrediction / bidtemplate

## 项目定位

`bidtemplate` 是 `MarketBidPrediction` 中用于构建**历史报价曲线趋势模板库**的子项目。其目标是把 PJM 主体在不同时间提交的实际报价曲线转换为少量、稳定、可解释的标准形状模板，并保留恢复真实报价曲线所需的结构参数与尺度参数。

该子项目与第一阶段 `scripts/bidprofile` 相互独立。`bidprofile` 描述“主体是什么类型、近期策略如何变化”，核心为 `9 LT + 9 ST + 8 Break`；`bidtemplate` 描述“主体实际报出了什么样的曲线”。两者只在后续关联验证中连接，不使用主体画像参与模板发现。

本子项目最终输出的核心对象为：

- 12 个非平价趋势模板：`T00 ~ T11`；
- 1 个平价模板族：`FLAT`；
- 每条历史报价对应的 `template_id`；
- 报价表达结构：`block / sloped`、有效段数、断点位置；
- 曲线尺度参数：`q_anchor_mw / q_span_mw / p_anchor / p_span`；
- 第一阶段画像与模板之间的关联验证结果。

模板用于刻画**归一化报价趋势**，不等同于原始报价的精确分段形式。`block/sloped`、段数和断点位置作为独立结构参数保留，从而避免把模板体系拆成大量“模板 × 模式 × 段数”的组合类别。

`bidtemplate` 到此完成模板发现和有效性验证。后续无未来信息泄漏的滚动画像、预测样本构建、模板分类器及结构参数预测属于 `bidprediction` 子项目。

---

## 方法流程

整体方法可以概括为：

```text
PJM 历史 Energy Market Offers
        │
        ▼
01  报价曲线清洗与统一归一化
        │
        ├── FLAT
        │
        └── SHAPE: 21维 shape_v00 ~ shape_v20
                │
                ▼
02  趋势模板发现
        │
        ├── T00 ~ T11
        └── FLAT 独立保留
                │
        ┌───────┴────────┐
        ▼                ▼
04 真实结构验证       03 模板可视化
block/sloped            占比与典型形状
        │
        ▼
05 结构参数标签
mode / segment_count / breakpoints / anchor-span
        │
        ▼
06 画像-模板关系数据集
        │
        ▼
07 画像-模板关联验证
        │
        ▼
交由 bidprediction 做真正预测
```

当前版本中，`03` 的模式感知可视化依赖 `04` 生成的真实 `block/sloped` 代表样本，因此推荐执行顺序为：

```text
01 → 02 → 04 → 03
          │
          ├→ 05
          └→ 06 → 07
```

### 01_build_curve_samples.py

输入默认位于：

```text
data/raw/energy_market_offers/<year>/*.csv
```

支持 PJM 10 点或 20 点 `MW/BID` 报价格式。每条报价执行以下处理：

1. 仅保留有限的 `(MW, BID)` 对；
2. 按 MW 从小到大排序；
3. 对重复 MW，保留该 MW 下的最大价格；
4. 将数量轴标准化为

```text
x = (q - q0) / (q1 - q0),   x ∈ [0,1]
```

5. 在统一网格

```text
x = 0, 0.05, ..., 1.00
```

上采样为 21 点；
6. `bid_slope_flag=sloped` 时使用分段线性插值，`block` 时使用阶梯保持；
7. 非平价曲线按端点进行价格归一化：

```text
shape(x) = [P(x) - P(0)] / [P(1) - P(0)]
```

得到 `shape_v00 ~ shape_v20`；
8. 真正平价曲线满足 `p_range <= 1e-9`，独立记为 `FLAT`，不参与 shape 聚类；
9. 同时保留：

```text
q_anchor_mw = q0
q_span_mw   = q1 - q0
p_anchor    = P(0)
p_span      = P(1) - P(0)
```

因此标准趋势模板 `T_k(x)` 可恢复到实际尺度：

```text
q(x) = q_anchor_mw + q_span_mw · x
P(x) = p_anchor + p_span · T_k(x)
```

2025 年共读取 `11,498,525` 条原始记录，其中：

| 类型 | 样本数 | 占有效样本 |
|---|---:|---:|
| Shape family | 4,547,836 | 88.84% |
| Flat family | 571,271 | 11.16% |
| 有效曲线合计 | 5,119,107 | 100% |
| Invalid | 6,379,418 | - |

对 1 月 invalid 进行审计后，全部来自 `single_clean_point` 或 `no_valid_points`，未发现由归一化公式导致的大规模有效曲线误删。

### 02_cluster_curve_templates.py

聚类只使用：

```text
template_family == "shape"
shape_cluster_eligible_flag == 1
```

即 4,547,836 条 21 维 shape 曲线。为兼顾效率与全量覆盖，采用：

```text
全量 shape
  ↓
代表性随机抽样 300,000 条
  ↓
候选 K 评估
  ↓
确定 KMeans 模板中心
  ↓
全量样本分配 template_id
  ↓
全量重构误差与覆盖率统计
```

候选 `K = 4, 6, 8, 10, 12, 16, 20`。主要结果为：

| K | Shape MAE | Shape RMSE | Silhouette | 最小类占比 |
|---:|---:|---:|---:|---:|
| 4 | 0.050301 | 0.070427 | 0.6262 | 9.480% |
| 6 | 0.047673 | 0.065800 | 0.6321 | 1.172% |
| 8 | 0.038105 | 0.054441 | 0.6159 | 1.108% |
| 10 | 0.034204 | 0.049828 | 0.6371 | 1.077% |
| **12** | **0.032294** | **0.044925** | **0.6460** | **1.108%** |
| 16 | 0.027278 | 0.038584 | 0.6456 | 0.638% |
| 20 | 0.024166 | 0.034469 | 0.6455 | 0.745% |

`K=12` 的 silhouette 和 elbow 指标最好，同时保持较合理的最小类规模。`K=16/20` 虽继续降低重构误差，但几何分离度未继续改善，并会增加后续模板预测的类别碎片化，因此当前固定为 `12 shape + FLAT`。

需要强调：KMeans centroid 是趋势中心，不要求对应一条真实历史报价。其作用是定义归一化趋势模板，而真实 `block/sloped` 表达方式由后续结构参数保留。

### 03 / 04：模板展示与真实结构核查

`04_validate_template_representatives.py` 扫描全量样本，统计每个模板内部 `block/sloped` 构成，并分别寻找最接近模板趋势的真实历史 `block` 和 `sloped` 样本。

`03_visualize_curve_templates.py` 在此基础上展示：

- 模板在全部有效曲线中的占比；
- 12 个 shape 模板内部占比；
- KMeans centroid；
- 真实 block 代表曲线；
- 真实 sloped 代表曲线。

这一设计避免把平滑 centroid 误解释为所有真实报价都采用连续直线形式。

### 05_build_curve_parameter_labels.py

趋势模板固定后，进一步从原始 PJM 报价中提取结构标签。最终每条有效历史报价可表示为：

```text
template_id
curve_mode
effective_segment_count
breakpoint_x_json
q_anchor_mw
q_span_mw
p_anchor
p_span
```

其中：

- `FLAT`：有效段数为 1；
- `BLOCK`：连续相同价格段先合并，再计算有效段数和归一化断点；
- `SLOPED`：保留原始 knot 结构，段数由清洗后的有效点决定；
- 断点采用变长 JSON 保存，暂不强行固定维度。

这一层实现了“趋势模板”和“具体报价实现形式”的解耦。

### 06 / 07：画像—模板关联验证

`06_build_profile_template_relation_dataset.py` 不使用 `bidprofile` 中的辅助聚类标签，仅使用正式的：

```text
9 LT + 9 ST + 8 Break
```

将模板 assignment 按 `participant_id + local_date` 聚合，构建：

- 主体级表：`9 LT + 全年模板使用比例`；
- 主体日级表：`9 LT + 9 ST + 8 Break + 当日模板使用比例/主导模板`。

`07_validate_profile_template_relation.py` 从三方面验证：

1. LT 与主体全年模板偏好的 Spearman 关系；
2. ST 与主体日模板使用比例的 Spearman 关系，以及 Break 发生前后的模板分布变化；
3. 使用 HistGradientBoostingClassifier 作为关系诊断器，比较 `LT_9 / ST_9 / Break_8 / STBreak_17 / Full_26` 对当日 dominant template 的区分能力，并同时进行时间留出和未见主体留出。

这里的分类器仅用于验证画像是否包含模板区分信息，不是 `bidprediction` 的最终预测模型。

---

## 主要输出

```text
data/processed/bidtemplate/2025/
│
├─ curve_samples/
│   └─ curve_samples_*.csv
│
├─ curve_samples_manifest_2025.csv
│
├─ template_library/
│   ├─ candidate_k_metrics.csv
│   ├─ curve_template_library.csv
│   ├─ curve_template_cluster_summary.csv
│   ├─ assignments/
│   ├─ validation/
│   │   ├─ template_mode_composition.csv
│   │   ├─ template_representative_samples.csv
│   │   └─ centroid_vs_mode_representatives.png
│   └─ figures/
│       ├─ template_share_all.png
│       ├─ template_share_shape_only.png
│       ├─ template_gallery.png
│       └─ top_templates_overlay.png
│
├─ parameter_labels/
│   └─ curve_parameter_labels_*.csv
│
├─ curve_parameter_labels_manifest_2025.csv
├─ curve_parameter_structure_summary_2025.csv
│
└─ profile_template_relation/
    ├─ profile_template_daily_2025.csv
    ├─ profile_template_participant_2025.csv
    ├─ profile_template_relation_manifest_2025.csv
    └─ validation/
        ├─ lt_template_spearman.csv
        ├─ st_template_spearman.csv
        ├─ break_template_distribution_shift.csv
        ├─ feature_template_mutual_information.csv
        ├─ relation_model_metrics.csv
        └─ relation_model_metrics.png
```

## 当前结论与项目边界

当前 `bidtemplate` 已形成一套完整、轻量、可解释的报价曲线表示：

```text
趋势：template_id = T00~T11 / FLAT
结构：curve_mode + segment_count + breakpoints
尺度：q_anchor/q_span + p_anchor/p_span
```

验证结果表明模板具有稳定覆盖结构，KMeans 趋势中心与真实 block/sloped 报价可以同时解释；第一阶段 `9 LT + 9 ST + 8 Break` 与模板选择之间也存在明显关联，因此两阶段可以在后续预测环节连接。

本子项目不继续训练最终模板预测器。真正的未来报价预测应在 `bidprediction` 中使用仅由预测时点之前历史数据构造的滚动 LT/ST/Break 和市场特征，避免未来信息进入特征。
