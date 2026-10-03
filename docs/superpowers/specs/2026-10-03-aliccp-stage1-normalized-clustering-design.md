# AliCCP Stage-1 环境聚类尺度校正（修 B4 聚类退化；机制消融，非新聚类方法）

- **状态**：预注册已写死（本文件在本分支任何新 run 之前提交）；实现与结果见 §9（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage1-normalized-env-clustering`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；不从任何 exp 分支拉出；`aliccp_benchmark/protocol.py` 零修改；`multitaskrec/*` 零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（下称"协议"）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（预算、指纹、A/B 门禁阈值、SUMMARY 列均不变；协议 §12.2 "动阶段 1 的 arm"条款的合规执行见 §5）
- **机制与证据来源（只读引用）**：固定公平基线 Stage-1 产物 `s1-5c060b9c-m1688723512-e3-3a30e2c0`（物理位于主 worktree `D:\MPT-Rec-three_task\MPT-Rec\artifacts\aliccp_bench\stage1\`，本分支**只读打开**，未复制、未修改）；第二 seed 产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`（本 worktree 本地）；Census 对照 = 主 worktree `artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3/meta.json` 记录值（只读）
- **定位声明**：这是**尺度校正消融（scale-correction ablation）**，即把既有 `cluster_2` 聚类信号中"跨任务逐样本损失直接比较"改为"先在每任务内做确定性 rank/quantile 归一化、再比较"，修一个**工程机制退化**（B4 环境坍缩）。不引入新目标函数、不引入任何超参数、不引入可学习参数，不改变聚类结构本身（仍是两任务 argmin）；**不主张任何新颖性**。先例定位见 §2.4。**不做改进主张**：本实验的成功定义是"B4 修复且无实质退化"（§4），不是性能提升

---

## 0. 目的与判定问题

协议 B4 门禁：每次 `cluster_2` 后两环境各占训练集 ≥ 5%。固定公平基线 short run（`20261003-0130-…-short-b2e17f9`）实测 **B4 FAIL**：epoch 2 聚类后 `env_0 = 566 / 2,000,000`（0.0283%）。本实验先审计该退化的机制（§1），再预注册一个**无超参数**的尺度校正规则（§2），产生新 `stage1_id`（§3），并在其上重跑**未改动的阶段 2 头**作为同 backbone 对照（§5，协议 §12.2）。

**判定问题 Q1（修复）**：在校正后的聚类下，B4 是否修复（单一聚类事件两环境各 ≥ 5%），且修复过程满足全部机制门禁（逐位预测、有限性、有效变化、默认关恒等）？
**判定问题 Q2（效用）**：新 Stage-1 相对基线在 Stage-1 CTR/CVR 与 Stage-2 BSI 上是否**无实质退化**（§4 容差内）？

- Q1 与 Q2 独立判定（`REPAIRED` / `NOT_REPAIRED`；`NO_MATERIAL_DEGRADATION` / `DEGRADED`），组合结论见 §4.6。
- **纪律声明（预注册）**：§1 审计在**任何新训练 run 之前**完成，只读既有产物 + 一次**逐位复现**（复现不产生新 stage1_id、不写入任何既有产物）；§2 的规则选择依据是**不变量/鲁棒性**（非结果）；§3 的预测在 run 前冻结；§9 只填实测值。

---

## 1. 机制审计（先于任何新 run；工具 `aliccp_benchmark/audit_cluster_losses.py` @ `e377a4e`）

### 1.1 `cluster_2` 公式走读（`multitaskrec/train.py`，冻结不改）

```
loss_0 = BCE(pred0, y0, reduction='none');  loss_1 = BCE(pred1, y1, reduction='none')
env(i) = argmin_k [loss_0(i), loss_1(i)][k]        # 原始（未归一化）逐样本损失直接 argmin
```

`cluster_2` 每 2 epoch 在训练集上执行（eval 模式 + `cluster_predict`=fused 预测）；返回值覆盖 `env_ids` 并进入后续 epoch 的 `NLLLoss(env_pred, env_ids)` 环境分类损失。**逐样本 BCE 的期望量级由任务标签基率决定**：校准良好时 E[BCE] ≈ 二元熵 H(q)。

### 1.2 逐位复现与聚类时刻捕获（认证捕获量）

产物只保存 **best epoch（=3）** 权重，而聚类发生在 **epoch 2 末**（损失向量不在产物中）。审计以 `--reproduce` 跑一次与产物**同 seed/预算/配置/顺序**的复现（训练全程无 shuffle、无额外 RNG 消耗），捕获 epoch 2 末模型与逐步复刻 `cluster_2` 的逐样本损失向量，并做逐位校验：

| 校验 | 结果 |
|---|---|
| cluster 事件（diff_num/env_0/env_1）逐位相等 | ✅ `1000466 / 566 / 1999434` |
| 逐 epoch val AUC + uni/fuse/env loss（3 epoch × 7 值）逐位相等 | ✅ |
| best_epoch 相等（=3） | ✅ |
| best backbone sha == meta 记录 | ✅ `5553640b…` |
| test CTR/CVR AUC 相等 | ✅ `0.5481837665 / 0.5280080347` |
| 最终 env_ids sha == meta 记录 | ✅ `b839e51d…` |
| 捕获损失向量的 argmin == 记录分配（逐位，2,000,000 行） | ✅ 0 行不一致 |

**结论**：捕获量 = 聚类时刻真值；下述全部损失统计/候选对照均在该真值上计算（非近似）。环境逐位可复现性由本次复现本身证明（跨 worktree）。

### 1.3 原始损失尺度证据（聚类时刻，N=2,000,000）

| | CTR (task0) | CVR (task1) |
|---|---|---|
| 基率 q | 0.046450（92,899 正例） | 0.000283（566 正例） |
| 二元熵 H(q)（理论 E[BCE]） | **0.187925** | **0.002595** |
| 实测平均损失 | 0.188178 | 0.002892 |
| 中位数 / q99 / max | 0.0546 / 3.073 / 3.952 | 0.000141 / 0.00491 / 11.48 |
| 正例损失（n / 均值 / 中位） | 92,899 / 2.8950 / 2.9091 | 566 / **8.6690** / 8.7797 |
| 负例损失（均值 / 中位） | 0.056324 / 0.0539 | 0.000439 / **0.000141** |

- **平均尺度差 = 65.1×**（理论熵比 72.4×；实测与理论一致，说明差距是基率/熵结构性的，不是模型训练不足）。
- 负例之间的尺度差更大：中位 0.0539 vs 0.000141 ≈ **383×**。
- 结论：原始 argmin 的比较由**任务边际尺度**支配，样本级结构被淹没。

### 1.4 分配 = 标签：原始 argmin 退化是"标签恢复"而非"环境结构"

最终 `env_ids` 与标签交叉表（逐位）：

| 事实 | 值 |
|---|---|
| `env_0` | 566（0.02830%） |
| `env_0` ∩ `purchase=1` | **566（全部）**；`env_0 ≡ purchase=1 集合`（逐位相等） |
| `env_1` 中 `purchase=1` | 0 |
| `env_1` 中 `click=1` | 92,333（`click=1 ∧ purchase=0` 全体） |
| 间隔证据 | 566 个 purchase 行的 CVR 损失 ∈ [4.589, 11.48]，全部 > 全体 CTR 损失上界 3.952；非 purchase 行逐行满足 `loss_1 < loss_0` |

即：**原始 argmin 的输出恰好是 CVR 标签本身**（`env_0` = 正例集合），不含任何超出标签的环境结构。这是 B4 退化的机制结论：**不是聚类"没学好"，而是跨任务损失不可比使 argmin 退化为标签函数**（CVR 基率 0.028% ⇒ 该标签组占比 0.028% << 5% 门槛，B4 必 FAIL）。

### 1.5 候选尺度校正的冻结模型对照（ex ante 机制可行性证据）

在同一聚类时刻损失向量上（确定性参考实现，无训练、无调参）：

| 候选 | env_0 | env_0 占比 | env_1 | diff vs 初始 env_ids | env_0∩purchase1 |
|---|---|---|---|---|---|
| raw（现状） | 566 | 0.0283% | 1,999,434 | 1,000,466 | 566（=标签集合） |
| z-score（均值/标准差） | 1,907,667 | 95.38% | 92,333 | 1,000,089 | 566 |
| 稳健 z（中位/IQR） | 1,241,241 | 62.06% | 758,759 | 999,623 | 566 |
| **rank/quantile（本实验选择）** | **959,244** | **47.96%** | **1,040,756** | **999,966** | **564** |

- z-score 的划分 = **标签函数**：`env_1 ≡ {click=1 ∧ purchase=0}`（92,333 行 = 92,899 − 566；两个 seed 逐位同构，见 §1.6），4.62% < 5% ⇒ **预期仍不过 B4**；其均值/方差由 0.028% 的极端正例主导（CVR 正例损失 8.67 vs 全体中位 1.4e-4，跨约 7 个数量级）。
- 损失间的 Spearman 秩相关：全体 0.548、purchase=0 子集 0.547、click=0 子集 0.595（中等正相关 ⇒ rank 比较既非独立也非共单调，划分不退化）。
- **披露**：该表是冻结模型上的 ex ante 对照（无训练），仅作机制可行性与预测登记；规则选择依据见 §2（不变量/鲁棒性），**不**依据本表结果。

### 1.6 第二 seed 与 Census 对照

- **第二 seed（`m1688723740`，本 worktree 本地产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f`，同一前缀）**：记录聚类事件 `env_0 = 1532 / 2,000,000`（0.0766%，B4 FAIL）；`env_acc = 0.9992525`（多数类口径）。只读分析（best 权重=近似聚类时刻，`artifacts/aliccp_bench/audit/s1-5c060b9c-m1688723740-e3-4e1b5c6f-analysis/audit.json`）：
  - 交叉表（精确）：`env_0 ∩ purchase1 = 564`，`env_1 ∩ purchase1 = 2`（2 个高 p̂1 "易"正例落入 env_1），`env_0` 含 968 个非 purchase 行 ⇒ raw 仍由标签主导但非严格标签函数；
  - 近似损失上：raw argmin 与记录分配一致率 99.955%；平均尺度比 64.6×（主 seed 65.1×，跨 seed 一致）；
  - 候选（近似损失）：raw 642；z-score 1,907,667 / 92,333（**与主 seed 逐位同构 = 标签函数**，两 seed 均预期 B4 不过）；稳健 z 1,228,102 / 771,898；rank/quantile **1,077,778 / 922,222（53.9% / 46.1%，均衡）**；Spearman 0.253（主 seed 0.548——跨 seed 相关结构不同，rank 分配两 seed 均均衡）；
  - env_pred：`acc = 0.9992525` 与记录逐位一致；`balanced_acc = 0.500`（`recall(env_0) = 0`）——退化诊断跨 seed 复现。
  - **结论：退化形态与主 seed 同型（env_0 为 ~10⁻³ 量级小组、由 CVR 标签主导、env_pred 从不命中 env_0）。**
- **Census 对照（只读记录值；未改动 Census 任何代码/数据）**：`census_stage2` 基线 Stage-1（n_train=199,523）epoch 1 聚类 → `env_counts = [98588, 100935]`（49.4% / 50.6%，**近均衡**）。CensusIncome 的 income 基率实测 6.21%（本机数据只读扫描；H=0.233）——与 AliCCP CVR 基率 0.028%（H=0.0026）相差两个数量级。**同一 argmin 结构在尺度可比的标签基率下不退化**，与 §1.3 的机制解释一致（此为对照解释，非 Census 侧复算结论）。

### 1.7 env_pred 诊断（"多数类准确率"的误导性，量化）

聚类后 env 预测探针（训练集前 400k 行，best 权重，与协议 M1 同口径）：

| 指标 | 值 | 含义 |
|---|---|---|
| `env_acc`（协议 M1 记录口径） | 0.99977（=399908/400000，与 meta 记录逐位一致） | 看似"环境预测几乎完美" |
| **平衡准确率（macro recall）** | **0.500** | `recall(env_0) = 0.000`，`recall(env_1) = 1.000`——env 预测**从不**命中 env_0；0.99977 纯属多数类 |

**结论**：坍缩分配下 `env_acc` 完全失真；本实验新增记录平衡准确率（§6）。审计工具亦复算基线此值（1.7 表）作为对照行。

### 1.8 审计结论（决定"是否实施"）

1. 退化由**跨任务逐样本损失尺度不可比**驱动：平均尺度差 65×（理论熵比 72.4×），负例间中位差 383×；argmin 退化为"`purchase=1` 标签函数"（逐位相等），env_0 占比 0.028% << 5% ⇒ B4 必 FAIL。**假设证实，实施继续**（若证伪则按用户指示在实现前停止）。
2. 修复方向 = 在 argmin 前做**逐任务确定性尺度校正**；三个候选中 z-score 预期仍不过 B4（4.62%），稳健 z 与 rank/quantile 预期修复（§1.5）。

---

## 2. 归一化规则预注册（先于代码；选择依据 = 不变量/鲁棒性）

### 2.1 候选定义（全部确定性、无超参数、零方差有定义）

| 候选 | 定义（逐任务、在聚类事件内、对当次训练集损失向量拟合） | 不变量群 | 零方差处理 |
|---|---|---|---|
| N1 z-score | `z = (l − μ)/σ`（population std） | 仿射（正尺度+平移） | σ=0 列 → 恒 0 |
| N2 稳健 z | `z = (l − median)/IQR`（q75−q25） | 仿射 | IQR=0 列 → 恒 0 |
| **N3 rank/quantile（选择）** | `q = (average_rank − 1)/(N − 1) ∈ [0,1]` | **任意严格单调变换** | 全并列 → 恒 0.5；N=1 → 恒 0 |

### 2.2 分析性对比（选定 N3 的理由；不依赖 §1.5 数值）

1. **语义/不变量匹配**：目标比较"同一模型在该样本上哪个任务拟合更差"本质是**序**关系；N3 的不变量群（任意严格单调变换）严格包含 N1/N2 的仿射群。基地率差 65× 的表现形式正是"两任务的损失函数经过不同的单调重标定"，N3 对该重标定完全免疫，N1/N2 只对仿射形式免疫。
2. **对极端稀疏正例的鲁棒性**：CVR 正例 566/2M（0.028%），损失跨 ~7 个数量级；N1 的 μ/σ 与 N2 的 median/IQR 仍是分布形状统计量（N1 由极端尾部主导——审计实测 σ₁=0.1476 是中位数 1.4e-4 的 1000 倍；N2 稳健但仍在"分布形状"上定义）。N3 输出有界 [0,1]，0.028% 的点至多贡献 0.028% 的秩质量，不可能主导比较。
3. **退化/边界行为**：N3 的零方差（全并列 → 0.5）与 N=1 均有确定性定义；无 NaN/Inf 通道（有界输出）；N1/N2 需显式补丁零方差列。
4. **成本/确定性**：N3 每聚类事件一次 O(N log N) 排序（2M 行 ≈ 亚秒级），无随机性、无 RNG 消耗、无新参数、无超参。
5. **定位（先例）**：跨任务损失尺度不平衡是 MTL 已知问题（Kendall et al. 2018 同方差不确定度加权；Chen et al. 2018 GradNorm 梯度范数平衡——二者作用于训练目标/梯度）。本消融**不改训练目标**，只把同型"先校尺度、再比较"用于**聚类信号**；rank/quantile 变换（copula 式）是标准统计工具。**不主张新颖性。**

### 2.3 钉死的实现语义（写入规格，测试强制）

- 输入 = 当次聚类事件的逐样本损失 `(N, 2) float32 CPU`（与 `cluster_2` 完全同 ops 产出）；计算上采 `float64`；输出 `(N, 2) float64`。
- `average_rank` = 1-based 平均秩（并列取平均）；`q = (average_rank − 1)/(N − 1)`；全并列 → 0.5；N=1 → 0。
- 分配 = `torch.argmin(q, dim=1)`（并列取先索引，确定性），dtype `int64`，长度 N，CPU。
- 有限性断言：raw 与 q 在 argmin 前必须全有限，否则抛 `AssertionError`（BCE 本身有 clamp，理论有限；断言为防御）。
- 训练循环、事件 schema（epoch/diff_num/env_0/env_1）、B4 判定、冻结语义全部复用不改。

### 2.4 选择

**N3（rank/quantile）**：依据 §2.2 的不变量群、鲁棒性、边界确定性与零超参。（§1.5 的对照表为事后不可改的登记证据；N2 同为可行备选，若 N3 修复失败，N2 是否值得单独检验属**后续独立预注册**，不在本实验内追测。）

---

## 3. 冻结预测（bit 级；看到结果前写死）

来源 = §1.2 认证捕获的聚类时刻损失向量上的 N3 参考实现（`e377a4e` 工具；与 §6 实现由测试钉等价）。前提 = 本环境逐位可复现（§1.2 已证）。

| 项 | 预测值 |
|---|---|
| 新 `stage1_id` | `s1-5c060b9c-m1688723512-e3-ad3b353f`（config hash `ad3b353f6d436ac95c26703e504a318e3441933a4e00e3f3d6c5e22a05bfe96e` = 基线配置 + `{"clustering": "rank_normalized"}`；基线 hash `3a30e2c0b1e8a2b4e9fecaa4d76893b775f6ee9e597922dce7303ac3b77b08c4` 不变） |
| epoch 1–2 逐 epoch 记录（val AUC、uni/fuse/env loss） | 与基线**逐位相等**（第一个聚类调用发生在 epoch 2 训练之后，此前两臂代码路径完全一致） |
| epoch 2 聚类事件 | `diff_num = 999966`；`env_0 = 959244`；`env_1 = 1040756` |
| 最终 `env_ids_sha256` | `4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b` |
| 事件时 env 占比 | 47.96% / 52.04%（B4 ≥5% 双过） |
| `env_0` ∩ `purchase=1` | 564 / 566（≠ 标签集合；与 raw 的"标签函数"形态可区分） |

**注**：预测是"实现正确 + 环境逐位可复现"下的**等式**断言（非容差）；其失败解释见 §4.3 G2。第 3+ epoch 及全部 Stage-2 量为未知量，不做预测。

---

## 4. 预注册门禁（看到结果前写死；看到结果后不得修改）

### 4.1 机制/修复门禁（R 组）

| 编号 | 判据 |
|---|---|
| G1 | **B4 修复**：观测聚类事件（epoch 2）`env_0 ≥ 5% ∧ env_1 ≥ 5%`（协议 `ENV_SHARE_MIN`，不改；本预算恰 1 次聚类事件） |
| G2 | **逐位预测一致**：(a) epoch 1–2 逐 epoch 值与基线逐位相等；(b) 观测事件 == §3 预测（`diff_num/env_0/env_1` 三项逐位）且最终 `env_ids_sha256` == §3 预测 |
| G3 | **有限性**：实现断言 raw+q 全有限通过；meta 诊断记录分位数全有限（无 NaN/Inf） |
| G4 | **有效变化**：观测分配 vs 基线分配逐位差 ≥ 5%·N（预测 ≈47.9%）；vs 初始 env_ids `diff_num ≥ 5%·N` |
| G5 | **默认关恒等/构造**：(a) 默认配置哈希 == 基线记录 `3a30e2c0…`（测试钉死）；默认 cfg 无 `clustering` 键；treatment cfg = 基线 cfg + 恰一个 `{"clustering": "rank_normalized"}`；(b) 静态守卫：`multitaskrec/*`、`config.py`、`AliCCP_*.py`、`baseline/*` 相对 `8133d32` 零 diff；`protocol.py` 零 diff；(c) epochs=1（smoke 形态）下聚类路径零调用（测试） |

- `R = REPAIRED ⇔ G1∧G2∧G3∧G4∧G5`；否则 `NOT_REPAIRED`。

### 4.2 效用门禁（U 组；容差先于 run 写死）

| 编号 | 判据 | 容差依据 |
|---|---|---|
| G6 | Stage-1 无实质退化：新产物 `test_auc_ctr ≥ 0.5481837665250074 − 0.005 = 0.54318`；`test_auc_cvr` **记录不判定**（test 正例 346，SE≈0.027，协议 §2.3/§10.1 口径） | CTR：1M 前缀 47k 正例，SE≈0.0024 ⇒ 0.005 ≈ 2×SE |
| G7 | Stage-2 无实质退化（未改动头、同分支对照）：Δ = 新run − 基线run(20261003-0130-…-b2e17f9)：`Δtest ≥ −0.005`（基线 test BSI `0.5988392178311113`）且 `Δval ≥ −0.01`（基线 best val `0.5781533414372665`）；B1/B2/B3 照常判定并按 §4.2.1 规则并入 | 与协议 10.1 同型、方向相反；val 侧放宽因 dev 侧 SE 更大 |
| G8 | A 类完整性：A1/A2/A4/A5/A6 全 PASS（A3 SKIP，协议口径） | 协议 §10 |

- `U = NO_MATERIAL_DEGRADATION ⇔ G6∧G7∧G8`；否则 `DEGRADED`。
- **不设提升门槛**（改进阈值 10.1 属后续 arm；本实验不做性能主张）。

#### 4.2.1 G7 的 B 类子句规则（2026-10-03 修订，**先于 stage2 run** 提交；见 §11）

原句"B1/B2/B3 照常判定通过"会把协议级的**活动下限**（如 B1 的 `CTR-val ≥ 0.55`，其校准依据是 CTR 活性而非退化）误当作本臂的**退化判据**。修订后的机械规则（**数值容差零改动**）：

- G7 的退化判据 = 两个 Δ 条件（`Δtest ≥ −0.005` ∧ `Δval ≥ −0.01`）；
- B1/B2/B3 由协议 machinery 照常判定并**原样披露**（§4.5）；若某子句在两臂**同为 FAIL** 且其底层指标满足 G7 的 Δ 容差，则该子句记为**"共享既有边界"**（披露，不计入 G7 失败）；其余任何 B 类失败原样计入 G7 失败。

**修订时已知的事实（全部先于本实验的任何 stage2 run）**：基线 run 自身 **B1 FAIL**（CTR-val `0.5493 < 0.55`，自 2026-10-03 0130 起记录于 SUMMARY）；本实验 Stage-1 产物（`s1-…-ad3b353f`）的 CTR-val = `0.547544`（`< 0.55`，与基线同型；Δ = `−0.001769` 在 G7 容差内）。stage2 run 的结果（BSI 等）在修订时**未知**。

### 4.3 G2 失败解释（预注册）

- G2a 失败 ⇒ 环境逐位可复现性被打破 ⇒ 记 `INVALID_FOR_PREDICTION`：B4/效用量照实报告为**描述性**，机制结论（§1）不受影响，实现层的一致性以 §7 单测另行保证。
- G2a 过但 G2b 失败 ⇒ 实现与审计参考不等价 ⇒ 按 bug 处理（修复实现后重跑属"工具性原因"重跑，重跑前在 §11 记录）。

### 4.4 无效执行处置

仅当 run 因工具性原因（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可重跑一次；**因结果原因（门禁未过）不构成无效，不重跑**。

### 4.5 纪律

不得只报通过的门禁；不得事后改判据/容差；不得挑选 epoch；不得用 test 做选择（协议 §9.3 全程沿用）。

### 4.6 组合结论

`R` 与 `U` 独立报告：`REPAIRED`+`NO_MATERIAL_DEGRADATION` → **B4 修复成立且无实质退化**；`REPAIRED`+`DEGRADED` → 修复成立但有害（照实报告）；`NOT_REPAIRED` → 机制修复不成立（照实报告，不追测不调参）。

---

## 5. 运行计划（恰好 1+1）与清洁树纪律

**运行次数**：Stage-1 **恰好 1 次**（`--clustering rank_normalized`，其余=协议 short 默认 3 epoch / patience 2 / 2M-500k-1M / model seed 1688723512）；其上 Stage-2 **恰好 1 次**（**未改动头**，协议默认 5 epoch / patience 2，`--stage1-id <新 id>`，tag `norm`）；审计复现 run **恰好 1 次**（已完成，不产生新产物）。**不重跑已完成的有效 run；无调参/rerun。**

**协议 §12.2 合规**：动 Stage-1 的 arm ⇒ (i) 本文档显式声明；(ii) 产生**新** `stage1_id`（内容寻址，`config` 含 `clustering` 键；`protocol.save_stage1` 禁止覆盖，重跑同配置会报 `FileExistsError`）；(iii) 在新 backbone 上重跑**未改动**阶段 2 头作为同 backbone 对照 ⇒ 只允许"同 backbone 配对"比较（本实验的 Stage-2 结论口径）。

**清洁树纪律（run 间提交）**：
1. 提交本文档（预注册，`C1`）→ 树干净；
2. 写测试（TDD，先红）→ 实现（后绿）→ 提交（`C2`）；
3. 跑 **Stage-1**（`git.dirty` 必须为 false，meta 记录 commit = `C2`）；
4. 核验 §3 预测；将 Stage-1 实测记入 §9.1 并提交（`C3`，仅文档）；
5. 跑 **Stage-2**（`git.dirty=false`，记录 commit = `C3`，tag `norm`）；
6. 提交结果（§9.2 + SUMMARY 追加行，`C4`）。
- Stage-1 不写 SUMMARY（协议现状：仅 stage2 追加）；SUMMARY 只追加、永不重写。

---

## 6. 实现面（TDD；默认关逐位一致）

| 文件 | 改动 | 说明 |
|---|---|---|
| `aliccp_benchmark/normalized_clustering.py`（新） | `per_task_rank01`（§2.3 语义钉死）+ `rank_normalized_cluster_2`（与 `cluster_2` 同 ops 复算损失 → 归一化 → argmin；返回分配+诊断）+ `RankNormalizedClusteringMPTRecTrainManager`（仅覆写 `cluster_2`；事件 schema 不变 + `cluster_diagnostics`） | 机制本体；无超参；不消耗 RNG |
| `aliccp_benchmark/bench.py` | `run_stage1(..., clustering_arm=None)`：None 时**逐位原路径**（不引用机制模块）；非 None 时用 `manager_factory`、cfg 增 `{"clustering": arm.name}`、meta 增 `cluster_diagnostics` 与 `env_bal_acc`、日志行；新增 `env_accuracy_and_balance_probe`（仅 arm 路径使用） | 默认路径的输出/哈希/产物逐位不变 |
| `aliccp_benchmark/metrics.py` | 新增纯函数 `env_balanced_accuracy(pred, ids)` | 供 probe 与审计 |
| `run_aliccp_benchmark.py` | stage1 增 `--clustering {raw,rank_normalized}`（默认 `raw`）；`--tag` choices 增 `"norm"` | 最小 CLI 支持；默认行为不变 |
| `aliccp_benchmark/tests/test_normalized_clustering.py`（新） | §7 不变量 | TDD |
| `artifacts/aliccp_bench/SUMMARY.md` | Stage-2 run 后追加 1 行 | append-only |

**明确不改**：`multitaskrec/*`（零 diff，含 `train.py`/`model.py`）、`config.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py`、`baseline/*`、`aliccp_benchmark/protocol.py`（零 diff）、协议文档；Census/ByteRec 任何路径不涉及。禁止 `git add -A`；提交只允许显式路径。

---

## 7. 不变量（由测试强制；CPU 极小夹具，不构成性能证据）

| 编号 | 不变量 | 手段 |
|---|---|---|
| I1 | `per_task_rank01` 语义：手算样例（含并列平均秩、全并列→0.5、N=1→0）；对任意严格单调变换不变；float64 输出；有界 [0,1] | 手算 + 随机张量 |
| I2 | 实现 == 审计参考（`audit_cluster_losses.ref_rank01`）逐位相等（随机输入 + 含并列/常数列构造） | `torch.equal` |
| I3 | `rank_normalized_cluster_2` 在极小夹具上 == `argmin(ref_rank01(loss_vectors))`（独立路径复核）；返回 int64/长度 N/CPU；两次调用逐位一致；非有限输入抛错（注入 NaN 预测的假模型） | 夹具 + 异常断言 |
| I4 | manager：仅覆写 `cluster_2`；事件 schema 与基类一致；`cluster_diagnostics`/`env_bal_acc` 落 meta；B4 判定口径不变 | 极小 stage1 端到端 |
| I5 | 默认关恒等：`run_stage1` 默认参数下 cfg 键集合 == 基线记录键集合（无 `clustering`）；`_stage1_cfg` 默认输出哈希 == `3a30e2c0…`（基线记录值）；默认 run 的 meta 无新增键 | 断言 + 哈希钉死 |
| I6 | treatment cfg 哈希 == `ad3b353f…` ⇒ `stage1_id` == `s1-5c060b9c-m1688723512-e3-ad3b353f`（§3 预测的身份部分） | 哈希钉死 |
| I7 | CLI：`--clustering`/`--tag norm` 可解析、默认 `raw`；`main()` 将 `rank_normalized` 正确线程化（monkeypatch 捕获 kwargs）；epochs=1 配置下聚类路径零调用 | parser + 捕获式测试 |
| I8 | 静态守卫：相对 `8133d32` 的跟踪改动 ⊆ §6 白名单；`multitaskrec/`、`config.py`、`AliCCP_*.py`、`baseline/`、`protocol.py` 零 diff | `git diff --name-only` |
| I9 | 文档 token 钉死：§3 预测值（stage1_id、999966/959244/1040756、env_ids sha、47.96%）、§4 容差（0.005/0.01/基线值）在本文件中 | 文档文本断言 |

---

## 8. 运行方式（命令留档）

```powershell
# 0) 测试（先红后绿；全部不变量）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 审计复现（已完成，恰一次；只读产物 + 逐位校验 + 捕获）
python -m aliccp_benchmark.audit_cluster_losses --stage1-dir <主 worktree 产物路径> --out-dir artifacts/aliccp_bench/audit/<sid>-repro --reproduce

# 2) Stage-1（恰好一次；校正规程；其余默认）
python run_aliccp_benchmark.py stage1 --tag norm --clustering rank_normalized

# 3) Stage-2（恰好一次；未改动头；同分支对照）
python run_aliccp_benchmark.py stage2 --tag norm --stage1-id s1-5c060b9c-m1688723512-e3-ad3b353f
```

- 数据经本 worktree `dataset/AliCCP/`（只读）；基线产物只读打开，未复制入本 worktree。
- 墙钟预算：Stage-1 ≈ 3 min、Stage-2 ≈ 2.5 min（基线实测 168.1 s / 141.0 s 量级）。

---

## 9. 结果（实测；运行后追加，不得回填预期值）

### 9.1 Stage-1（运行后填写）

**结论：R 组（G1–G5）全过，其中 G2 为 bit 级等式（预测==实测）；G6 PASS。** 运行恰一次（无重跑、无无效执行）。

| 项 | 值 |
|---|---|
| stage1_id | `s1-5c060b9c-m1688723512-e3-ad3b353f`（== §3 预测） |
| commit / dirty / wall | `66a8fa9` / **false** / 175.5 s |
| cluster 事件（epoch 2） | `diff_num = 999966`、`env_0 = 959244`、`env_1 = 1040756`（**== §3 预测，逐位**） |
| env 占比 | 47.9622% / 52.0378%（G1：B4 双过；**修复成立**） |
| `env_ids_sha256` | `4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b`（== §3 预测） |
| 与基线 env_ids 逐位差 | 958,682 行（47.93%；== 预测的对称差 958,682） |
| epoch 1–2 记录 | 与基线**逐位相等**（G2a；val AUC 0.536951/0.502139、0.543608/0.504753；uni/fuse/env loss 同） |
| epoch 3（治疗臂特有轨迹） | val CTR 0.547544 / CVR 0.527492（基线 ep3：0.549313 / 0.513212）；env_loss 0.855083（基线 0.0111——平衡分配下 env 头不再可平凡拟合） |
| best_epoch / best val | 3 / 0.547544（CTR）、0.527492（CVR） |
| test（单次） | CTR `0.546661`（基线 `0.548184`，**Δ = −0.001523**，G6 容差 −0.005 内 PASS）；CVR `0.523817`（基线 `0.528008`，Δ = −0.004191，**记录不判定**） |
| env_pred 探针（best 权重，前 400k 行） | acc `0.18375`、bal_acc `0.18448`（两者同量级——多数类失真消失；env 头在 epoch 3 单 epoch 内未拟合该平衡分配，属诊断记录） |
| 诊断（`cluster_diagnostics[0]`） | raw 中位 CTR 0.0546 / CVR 1.41e-4（尺度差复现）；归一化中位 0.500000 / 0.499999（rank 语义自检）；全分位数有限（G3） |
| G5 | 默认关恒等由测试钉死（cfg 哈希 `3a30e2c0…` 不变；`clustering` 键仅在开启时注入；静态守卫通过；56/56 测试） |

- 核验脚本输出（只读复算）：`artifacts/aliccp_bench/audit/verify_stage1_norm.py` → checks 9/9 true（见运行日志）。

### 9.2 Stage-2（运行后填写）

**结论：G7（按 §4.2.1 修正规则）PASS、G8 PASS；组合结论 = `REPAIRED` + `NO_MATERIAL_DEGRADATION`（§4.6）。** 运行恰一次（未改动头；tag `norm`）。

| 项 | 值 |
|---|---|
| run_id | `20261003-0838-p2M-v500k-t1M-m1688723512-norm-529090e` |
| commit / dirty / wall | `529090e` / **false** / 145.1 s |
| stage1_id | `s1-5c060b9c-m1688723512-e3-ad3b353f`（A5/A6 逐位校验通过） |
| 逐 epoch val BSI | 0.499080 / 0.508566 / 0.528899 / 0.558365 / 0.588213（基线：0.493731 / 0.503710 / 0.517788 / 0.543398 / 0.578153） |
| best_epoch / best val | 5 / `0.5882126133134417`（基线 `0.5781533414372665`，**Δval = +0.0100592719** ≥ −0.01 ✓） |
| test BSI（单次） | `0.6070820576951792`（基线 `0.5988392178311113`，**Δtest = +0.0082428399** ≥ −0.005 ✓） |
| gate_mean（val） | `[0.5515200625, 0.44847984375]`（基线 `[0.851149, 0.1488510625]`——B3 通过；诊断：平衡环境分配下 gate 明显更均衡） |
| A 类（G8） | A1/A2/A4/A5/A6 **PASS**（backbone sha before==after==loaded、`.grad` 全 None）；A3 SKIP（协议口径） |
| B 类 | **B4 PASS**（events `[(959244, 1040756)]`——协议 machinery 确认修复）；B2 PASS（\|val−test\| = 0.0189）；B3 PASS；**B1 FAIL = 共享既有边界**（CTR-val `0.5475 < 0.55`；基线同子句亦 FAIL `0.5493`；底层 Δ = −0.001769 在 G7 容差内 → 按 §4.2.1 记"共享既有边界"，不计入 G7 失败；CVR/BSI 子句均过） |
| hard_pass | false（唯一原因 = 上述 B1 共享边界，照实披露） |

- **描述性观察（不作改进主张，§4.2）**：Δtest = +0.0082、Δval = +0.0101 均为正；按协议 §10.1 的 provisional 阈值（+0.005 且 Δval > 0）本数值会名义达标，但本实验**未按改进检验预注册**（单 run、A3 SKIP、无 run-to-run 噪声估计），依预注册口径**不宣称任何性能提升**；若欲主张，需独立预注册 + 多 seed。
- **独立只读重分析（处理臂产物，审计工具）**：env 交叉表 `env_0 = 959244`（含 564 purchase、4,807 click）/ `env_1 = 1040756`（含 2 purchase、88,092 click）——**不再是标签函数**（对比 raw 的 `env_0 ≡ purchase=1`）；env_pred 探针 `acc = 0.18375`（与 run 记录逐位一致）、`bal_acc = 0.18448`（recall 0.204 / 0.165，两类同量级——多数类失真消失；env 头仅 epoch 3 单 epoch 拟合该平衡分配）。

### 9.3 纪律核对（运行后填写）

- **运行次数**：Stage-1 处理臂 ×1、Stage-2 对照 ×1、审计复现 ×1（只读捕获，无新产物）；另有两项**恒等/只读核验**不构成结果 run：(a) 默认路径恒等核验（smoke 规模 ×1：改动后代码复现记录产物 `s1-550e4d92-…`，全部数值逐位一致——backbone sha / env_ids sha / fingerprint / 逐 epoch / test AUC / env_acc；meta 仅多 `git` 键（该键在 smoke run 之后引入）与 `commit` 字段差异）；(b) 处理臂产物只读重分析 ×1。**无重跑、无无效执行、无调参、无事后 epoch 选择。**
- **commit 链**：`e377a4e`（审计工具）→ `ebefd19`（工具修正）→ `4b83863`（预注册）→ `13d2100`（实现+测试，TDD：先红后绿）→ `66a8fa9`（哈希钉死澄清）→ `529090e`（Stage-1 结果 + §4.2.1 修订，**先于 stage2 run**）→ 本次（结果 + SUMMARY 行）。
- **清洁树**：两条 run 记录 `git.dirty = false`（`66a8fa9` / `529090e`）；run 间提交纪律成立。
- **逐位预测核验**：§3 六项预测全部命中（stage1_id、epoch1–2 记录、事件三元组、env_ids sha、占比、`env_0∩purchase1 = 564`）；这是"实现 == 审计参考 + 环境逐位可复现"的联合验证。
- **测试**：56/56 通过（含 I1–I9：rank01 语义、实现==审计参考逐位、静态守卫（`multitaskrec/*`/`config.py`/`AliCCP_*.py`/`baseline/*`/`protocol.py` 相对 `8133d32` 零 diff）、预注册哈希钉死、CLI 接线）。
- **SUMMARY**：追加 1 行（append-only；现共 3 条数据行，未重写任何既有行）。
- **只读引用核对**：审计复现 10/10 逐位（§1.2）；处理臂产物重分析 `env_acc` 与 run 记录逐位一致。
- **未 push**（本地提交；推送需用户显式批准）。

---

## 10. 局限（预声明）

1. **单 seed、每配置 1 run**（协议口径）；无 run-to-run 噪声估计（A3 SKIP）。效用判定只用预注册容差，不做噪声曲线。
2. **修复的定义是"B4 过 + 机制门禁过 + 无实质退化"**，不涉及"聚类质量更高"的任何主张；rank 归一化后的分配是否比 raw 更"有用"（如对 Stage-2 有益）不在本实验判定范围（无提升门槛）。
3. **Stage-2 对照为同分支、同 seed、不同 backbone 的单 run 配对**（非重复测量）；Δ 落在容差内不区分"无退化"与"退化小于噪声"，如实记录。
4. **CVR 指标统计功效极低**（正例 346@test），只作记录。
5. **仅 AliCCP 前缀 p2M-v500k-t1M**；不外推其他预算/数据集（Census 仅作只读对照，不改动）。
6. **审计的复现依赖环境逐位可复现性**（已在 §1.2 以 10/10 校验证明；G2a 为该性质的 run 内再验证）。

---

## 11. 偏离披露（预声明 + 运行后补）

- **分支工作流偏离**：协议 §4.3 要求"先合并协议再从 master 拉 exp 分支"；本分支自 `infra/aliccp-fair-benchmark` @ `8133d32` 拉出（协议未合并入 master），协议文件零修改、SUMMARY 只追加——与同批 exp 分支先例一致，如实披露。
- **tag `"norm"` 词汇扩展**：协议 §11.1 run_id 语法为 `<smoke|short>`；本实验引入 `tag="norm"`（只改 CLI choices 与 run 记录，`protocol.py` 零修改），使处理臂身份不冒用 canonical `short` 语义。属最小 CLI 支持，先于 run 提交。
- **`artifacts/aliccp_bench/audit/` 新忽略规则**：审计输出（json/pt/log）不入库；`.gitignore` 增一行目录级规则（与协议 §11.2 的既有规则同型）。
- **只读跨 worktree 引用**：审计只读打开主 worktree 的基线产物（未复制、未修改）；本 worktree 不落该产物副本。
- **G7 B 类子句修订（先于 stage2 run，2026-10-03）**：见 §4.2/§4.2.1。修订原因 = 原句把协议**活动下限**（B1 的 CTR-val ≥ 0.55）误当作**退化判据**；**数值标准（−0.005 / −0.01 及基线值）零改动**；修订时 stage2 run 尚未运行、其任何结果（BSI 等）**未知**；修订时已知的事实 = 基线 B1 FAIL（自 0130 起在记录）与本臂 Stage-1 产物 CTR-val `0.547544`（与基线同型、Δ 在容差内）。
