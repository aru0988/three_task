# AliCCP Stage-2 Null Expert：源任务路由追加"零候选"（跨数据集复现）

- **状态**：预注册已写死（本文件在本臂任何 run 之前提交）；实现与结果见 §9（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage2-null-expert`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；协议尚未合并回 `master`，按用户指示直接从 infra 顶端拉实验分支——对协议 §4.3.2 "先等协议合并"流程的显式偏离，协议文件本身零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）。本分支**只引用**该协议与 `aliccp_benchmark/protocol.py`，不修改其任何文件与常量（SUMMARY 列、A/B 门禁阈值、预算、指纹定义均不变）
- **机制来源（prior art，非本实验发明）**：`exp/stage2-null-expert`（CensusIncome）及其说明 `docs/superpowers/specs/2026-09-29-stage2-null-expert-design.md`。该臂在 CensusIncome 上：机制成立（`null_top1_rate = 0.3725667088`，null 候选确被选中）但主判据未达标（ΔAUC-Test = +0.0012962965 < 预注册最低增量 +0.0020314693）——"机制活跃、效用未过有意义阈值"的弱正向结果
- **定位声明**：本实验是**跨数据集机制复现**（generality replication），验证同一机制在 AliCCP 第三任务（BSI）上是否成立/是否越过有意义阈值；**不主张新颖性**（"弃权/零候选专家"在 MoE 文献中属既有思想，MPTRec 自身的 env 路由即为候选混合的先例）。判断口径全部在见结果前写死
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；需变更先改本文件再改代码

---

## 1. 机制（逐字复现 CensusIncome 臂，不按 AliCCP 调整）

阶段 2 新任务头把 K 个源任务 specific 表征按 router 权重混合：

```
W = softmax(H_out · keys / T)             # keys = [env_emb_0..K-1]
new_spec_rep = Σ_k W_k · spec_rep_k
```

K=2 且 ΣW=1 ⇒ 恒为两个源任务表征的凸组合，无法表达"本样本不借用任何旧任务"。本臂在候选集合**末尾**追加一个**值恒为零**的候选（Null Expert），并给它一个**可学习的 router key**（`null_key`，`rep_dim` 维，零初始化）：

```
keys   = [env_emb_0..K-1, null_key]        # (B, K+1)
values = [spec_rep_0..K-1, 0]              # (B, rep_dim, K+1)，最后一列逐位精确为零
W      = softmax(H_out · keys / T)         # (B, K+1)
```

- **零 or 可学习**（对 CensusIncome 臂的核对结论）：**值**是恒零常量，**router key** 可学习——即"是否弃权"由可学习的方向决定，"弃权时借什么表征"恒为不借。本臂逐字复现该语义。
- `W_null → 1` ⇔ `new_spec_rep → 0` ⇔ `fused_rep` 退化为 gate 的 general 分支（`fused = gate_spec·0 + gate_gen·gen_rep`）。语义在 AliCCP 上完全成立：BSI 头可选择对某些样本不取 CTR/CVR 的 specific 表征。
- **默认臂逐位一致**：`use_null_expert=False` 时不创建 `null_key`，参数集合、算子与顺序与 `8133d32` 的 `multitaskrec/model.py` 完全相同（前向与反向逐位一致，由 §3 I1/I2 测试强制）。
- 两臂共享参数初始化逐位一致：`null_key` 在 `__init__` **末尾**创建且零初始化，不消耗全局 RNG（两臂头初始化 RNG 消耗序列相同）。
- `null_key` 不进入 `get_l2_reg()`（损失形式不变）。

---

## 2. 改动面（最小）

| 位置 | 改动 |
|---|---|
| `multitaskrec/model.py` → `NewTask` | 逐字移植 `exp/stage2-null-expert` 的 diff（35 行）：`use_null_expert=False` 参数、`routing_weights` / `routing_values` 提取、`forward` 改为 matmul 两 helper；默认路径逐位等价 |
| `aliccp_benchmark/metrics.py` | 新增 `NullRouteStats`（M7 判定量，逐位可复现口径，逐字移植）、`NullRouteDiagnostics`（C 类分布诊断）、`SourceGateStats`（源任务 gate 均值）、`null_supervision_stats`（分层/相关/预测离散度）、`null_arm_verdict` + 预注册常量 |
| `aliccp_benchmark/bench.py` | `run_stage2(..., null_expert=False)`；`stage2_run_id`（`-nullx` 后缀）；处理臂专属 val 探针 `newtask_null_probe`（no_grad，一次遍历）；`config.json` / `metrics.json` 记录 `null_expert`，处理臂追加 `mechanism` / `trainable_params` / `null_arm` |
| `run_aliccp_benchmark.py` | `stage2` 子命令新增 `--null-expert`（默认 false） |
| `aliccp_benchmark/tests/test_null_expert.py` | 新增（§3 不变量） |

**明确不改**：`aliccp_benchmark/protocol.py`（含 `SUMMARY_COLUMNS` 与 A/B 门禁阈值）、`config.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py`、`baseline/*`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`multitaskrec/model.py` 中 `NewTask` 以外的任何类、CensusIncome / ByteRec 任何代码路径。`SUMMARY.md` 只追加行。

---

## 3. 不变量（由 `test_null_expert.py` 强制，不接受人工目测）

参照 CensusIncome 臂 I1–I10，逐条移植；参考实现从 git `8133d32:multitaskrec/model.py` 动态加载（本分支基点 = 基线 run 实际使用的模型代码）。

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂前向/反向与基点实现**逐位一致** | `TestDefaultArmBitIdenticalToBase`：`torch.equal` 比对输出与每个参数的梯度 |
| I2 | 默认臂参数集合 = 基点参数集合（无 `null_key`） | 同上 + `test_default_arm_has_no_null_key`（`load_state_dict(strict=True)` 可过） |
| I3 | 开启后 router 权重为 `(B, K+1)` 且是概率 | 形状 / 归一性 / 非负测试 |
| I4 | 零候选值**精确为零**，前 K 列逐位不变 | `torch.equal(values[:, :, -1], zeros)` |
| I5 | `null_key` 是唯一新增参数、形状 `(rep_dim,)`、可学习、有非零梯度 | 参数集合差集 + 梯度测试 |
| I6 | 开启后的前向等于本文件写死的 K+1 公式 | 独立参考实现比对（`torch.equal`） |
| I7 | M7（`null_mean` / `null_top1_rate`）计算正确、流式累加逐位可复现、CPU 累加器 | `TestNullRouteStats`（含 CUDA 回归，`skipUnless`） |
| I8 | C 类诊断（分位数/熵/方差/分层/相关/预测离散度/源 gate 均值）在构造数据上数值正确 | `TestNullRouteDiagnostics` / `TestSourceGateStats` / `TestNullSupervisionStats` |
| I9 | 预注册常量与本文档一致；判定边界为闭区间 | `TestNullArmVerdict`（含 `test_doc_preregisters_same_numbers`） |
| I10 | 两臂共用同一 Stage-1 产物、冻结门禁 A1 仍 PASS、默认臂 metrics 既有键与语义不变（仅新增顶层 `null_expert: false` 布尔标记，与 CensusIncome 臂口径一致；不出现 `mechanism`/`null_*`/`null_arm`）、处理臂落盘完整 | `TestRunnerNullArm`（CPU 极小夹具端到端） |
| I11 | 模型侧唯一改动是 `model.py`；`aliccp_benchmark/protocol.py` 相对基点零改动 | 静态守卫（`git diff --name-only`） |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 4. 运行方式（恰好一次）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 处理臂（本实验唯一要跑的 run；固定 Stage-1 产物 = 基线 run 用的那份）
.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 --tag short `
  --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0 --null-expert
```

- 基线臂**不重跑**：对照 = 协议 SUMMARY 已有 run `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`（同 `stage1_id`、同预算、同种子、同顺序，协议 §12.1 允许）。
- 预算 / 种子 / 顺序全部走协议默认（train 2M / val 500k / test 1M 前缀；model seed 1688723512；env seed 20261003；batch 2000、shuffle=False；stage2 epochs=5、patience=2）——**不搜参、不换结构、不调温度、不改初始化**。
- 处理臂 `run_id` 形如 `YYYYMMDD-HHmm-p2M-v500k-t1M-m1688723512-short-<commit>-nullx`。
- **前置条件**：开跑前先提交本批改动（否则 run_id / metrics.json 的 commit 记的不是本实验代码，可审计性有缺口）。运行期间不得有其他未提交的已跟踪文件改动（`git.dirty` 必须为 false）。

---

## 5. 预注册接受标准（写死；看到结果后不得修改）

| 项 | 值 |
|---|---|
| 对照基线 | run `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`（commit `b2e17f9`，`stage1_id=s1-5c060b9c-m1688723512-e3-3a30e2c0`，本臂与其共享同一 checkpoint） |
| 基线值（逐位取自该 run `metrics.json`） | `AUC-Test-BSI = 0.5988392178311113`；`AUC-Val-BSI(best) = 0.5781533414372665`（best epoch 5） |
| **主判据（PI 指定，2026-10-03，严于协议 §10.1）** | `ΔAUC-Test-BSI ≥ +0.0055`（绝对增量；等价于 `AUC-Test-BSI ≥ 0.6043392178311113`）**且** `ΔAUC-Val-BSI > 0`（即 `> 0.5781533414372665`，方向一致） |
| **机制判据** | `null_top1_rate ∈ [0.05, 0.95]`（两端含等号；null 从不被选或总被选都视为机制未成立——与 CensusIncome 臂逐字相同） |
| 判定 | 主判据（两条）与机制判据**同时**满足才允许进入多 seed 扩展；否则**停止** |
| 机械判定 | `metrics.null_arm_verdict(test_auc, val_auc_best, null_top1_rate)`，完整落盘 `metrics.json["null_arm"]` |

**阈值口径的透明处理**：协议 §10.1 的主判据是 `Δ ≥ +0.005`；本臂按 PI 指示采用更严的 `+0.0055` 作为预注册主判据（在看到任何本臂结果之前记录于本文件）。判定函数同时记录协议级检查 `protocol_checks`（Δtest ≥ +0.005 且 Δval > 0），报告中**两个口径都如实披露**；若两者不一致，以本文件主判据为准并显式说明分歧，不得用宽松口径重判。

**止损规则（与协议 §10.2、CensusIncome 臂 §5 一致）**：

1. 未达阈值 → 如实记录该次结果（SUMMARY 追加行 + `runs/` 留档 + §9 补记），**不得**重跑挑好的、不得换阈值、不得跳到多 seed 或全量。
2. 机制判据不成立（null 权重被压到 0 或 1）→ 视为"该机制在 AliCCP 上不激活"，判为失败并停止，不做温度 / 初始化等补救性调参。
3. 阈值只允许在**看到任何本臂 run 结果之前**修改；每次修改单独 commit 并在本文件记录理由；已判定的 run 不得用新阈值重判。
4. 运行期间发现实现缺陷 → 允许修复后重跑，但必须同时保留失败 run 的产物与说明。

---

## 6. 门禁与继承披露（B1/B4）

- 本臂执行协议 §10 全部 A 类与 B 类判定（`--tag short` ⇒ `enforce_b=True`），`null_arm` 是**臂级**判定，不并入 `hard_pass`（A/B 门禁语义与 `protocol.py` 不变）。
- **继承的失败（非本臂引入，两臂按构造完全相同）**：基线 short run 已记录 `B1 FAIL`（`AUC-Val-CTR 0.5493 < 0.55`）与 `B4 FAIL`（cluster 后 env_0 仅 566/2,000,000）。这两项由**共享的 Stage-1 产物**（`meta.json` 的 `best_val_auc_ctr` 与 `cluster_events`）决定，本臂与其基线引用同一 `stage1_id` ⇒ 判定逐位相同。报告中如实披露为**继承基线缺陷**，不归因于本臂、不据此否定本臂，也不据此宣称任何东西。
- A 类门禁必须全 PASS（A3 SKIP 视为通过），否则本臂比较作废（协议 §12.1）。
- B2（val-test 差）/ B3（gate_mean 未坍缩）为臂相关门禁，照常判定与披露。

---

## 7. 诊断指标（处理臂专属；C 类：只落盘不判定）

全部在 **val** 上算一次（no_grad；test 不参与任何机制诊断，更不参与选择）：

| 键 | 含义 | 口径 |
|---|---|---|
| `null_mean` / `null_top1_rate` | M7 判定量与机制判据 | 逐位可复现（逐样本左结合累加，与 CensusIncome 臂相同实现） |
| `null_std` / `null_var` | null 质量的样本离散度（总体 std / var，ddof=0） | val 全量 |
| `null_q10/25/50/75/90` | null 质量分位数（linear 插值） | val 全量 |
| `route_entropy_mean` / `route_entropy_std` | 逐样本 router 分布 Shannon 熵（nats，K+1 类）的均值 / 标准差 | val 全量 |
| `route_var_mean` | 逐样本 router 分布（K+1 类）方差的样本均值 | val 全量 |
| `null_mean_bsi_pos` / `null_mean_bsi_neg` / `null_label_gap` | 按 BSI 标签分层的 null 质量均值与差 | val 全量；BSI 正例率 ~99%，分层近乎退化，如实披露 |
| `corr_null_pred` | Pearson(null 质量, BSI 预测) | val 全量 |
| `corr_null_abs_err` | Pearson(null 质量, \|y − pred\|) | val 全量 |
| `source_gate_mean` | backbone `gate_networks` 每源任务 specific 分支在 val 的均值（2 维） | 与 CensusIncome `GateStats` 同口径 |
| `pred_std` | 新任务头 BSI 预测在 val 的标准差（预测离散度） | val 全量 |
| `trainable_params` | 新任务头可训练参数**精确计数**（int） | 处理臂应 = 默认臂 + `rep_dim`（= 64），由 I5/I10 强制 |

**实现约束**：no_grad 流式累积，不保留中间张量（null 质量逐样本列表除外，val 500k 行内存可忽略）；不做 FLOPs（`fvcore` 在 CPU 上极慢，与协议其余部分一致地跳过）。M7 判定量保持逐位可复现；C 类诊断仅要求同机同输入确定性（同一数据顺序、同一权重 → 同一值）。

---

## 8. 报告与分支处置

- `SUMMARY.md` 只追加、不重写：处理臂即使失败也必须有行（协议 §9.3）。
- 结果（含判定、两臂对照、门禁、诊断数值、止损结论）在运行后**追加**到本文件 §9；§1–§8 预注册内容不得回填、不得修改。
- 分支处置（用户硬规则）：无论成功失败，**只保留 `exp/aliccp-stage2-null-expert` 本分支，不合并回 `master`**；推送需用户显式批准。
- 与本实验无关的既有探索（TC-Prompt / CGR / affinity gate / KL-Prompt / T4 等）一律不引用、不重建。

---

## 9. 结果（实测；运行后追加，不得回填预期值）

**判定：失败 → 按 §5 止损规则停止。** 主判据（AUC）两条均**通过**，机制判据（`null_top1_rate ∈ [0.05, 0.95]`）**不成立**（= 0.0000）；两条须**同时**满足，故臂级判定 `pass=false`。**不扩多 seed、不跑全量、不做温度/初始化等补救性调参**（§5 止损规则 2）。

### 9.1 两臂对照（同 `stage1_id`、同预算、同种子、同顺序）

| 项 | 基线臂 | 处理臂（本实验） |
|---|---|---|
| run_id | `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9` | `20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx` |
| commit / dirty | `b2e17f9` / false | `6224c0f` / false |
| stage1_id | `s1-5c060b9c-m1688723512-e3-3a30e2c0` | 同左（同一 checkpoint） |
| best val AUC-BSI | `0.5781533414372665`（epoch 5） | `0.5919398413138859`（epoch 5） |
| **AUC-Test-BSI** | **`0.5988392178311113`** | **`0.6126857532817985`** |
| Δ（处理 − 基线） | — | test **+0.0138465355**；val **+0.0137864999** |
| 主判据（≥ 0.6043392178311113 且 val 同向） | — | **通过**（协议 0.005 口径 `protocol_checks` 亦通过） |
| `null_mean` | 无此指标 | `0.3027766270` |
| `null_top1_rate` | 无此指标 | **`0.0000000000`（机制判据未成立）** |
| 逐 epoch val AUC（两臂） | 0.4937 / 0.5037 / 0.5178 / 0.5434 / 0.5782 | 0.5059 / 0.5211 / 0.5434 / 0.5673 / 0.5919（每个 epoch 均高于基线） |

两臂 backbone sha（`5553640b…`）加载/结束一致；`env_ids_sha256` 相同；`hard_pass=false`（B1/B4 继承失败，见 9.2）。

### 9.2 门禁

| 检查 | 结果 |
|---|---|
| A1 / A2 / A4 / A5 / A6 | **PASS**（backbone sha before==after，`.grad` 全 None） |
| A3 | SKIP（按需复跑，不阻塞首轮） |
| B1 | **FAIL（继承基线，非本臂引入）**：CTR 0.5493 < 0.55 来自共享 Stage-1 产物；BSI 分量 0.6127 ≥ 0.53 **通过** |
| B2 / B3 | PASS（\|val−test\| = 0.0207 ≤ 0.05；gate_mean = [0.4833, 0.5167] ∈ [0.05, 0.95]） |
| B4 | **FAIL（继承基线，非本臂引入）**：cluster events = [(566, 1999434)]，与基线逐位相同（同一 Stage-1 meta） |

### 9.3 机制诊断（处理臂专属，val 一次遍历；见指标字典 §7）

| 键 | 值 |
|---|---|
| `null_mean` / `null_top1_rate` | 0.3027766270 / 0.0000000000 |
| `null_std` / `null_var` | 0.0085302226 / 7.2764698406e-05 |
| `null_q10 / q25 / q50 / q75 / q90` | 0.2911510766 / 0.2953142598 / 0.3031031787 / 0.3099197820 / 0.3137927443 |
| `route_entropy_mean` / `route_entropy_std` | 1.0810294647 / 0.0068617515（ln 3 ≈ 1.0986） |
| `route_var_mean` | 0.0040387792 |
| `null_mean_bsi_pos` / `null_mean_bsi_neg` / `null_label_gap` | 0.3027552573 / 0.3055707665 / −0.0028155092 |
| `corr_null_pred` / `corr_null_abs_err` | 0.1465714848 / 0.0210091643 |
| `source_gate_mean`（backbone，冻结） | [0.1761596858, 0.2537776756] |
| `pred_std` | 0.0043802393 |
| `trainable_params` / `_default` | 8193 / 8129（差 = 64 = `rep_dim`，I5/I10 强制） |
| `null_key` L2 范数（落盘 `newtask.pt`） | 0.2878327668（**已从零初始化移动**：key 确实被学习） |

**机制解读（诊断一致，但因果归属未验证）**：

1. **机制机器可学习但未激活**：`null_key` 明显被训练（L2 = 0.288），梯度通路成立；但学习结果是路由把 `W_null` 稳定压在一个远窄于均匀值的带内（q10–q90 = 0.291–0.314，**全部 < 1/3**）。
2. **top1 = 0 是数学必然，不是探针缺陷**：两源权重和 = 1 − W_null ≈ 0.70 ⇒ `max(W_0, W_1) ≥ 0.35 > max(W_null) ≈ 0.314`，null 在任何 val 样本上都不可能成为最大权重。诊断自洽（`null_q90 = 0.3138 < 1/3` 与 `top1_rate = 0` 一致）。
3. **行为退化**：null 分支退化为对 specific 混合的**近似常数阻尼**（×≈0.70；std 仅 0.0085），而非逐样本弃权开关；同时头 gate 从基线学习解 [0.851, 0.149] 重新平衡到 [0.483, 0.517]。
4. **增益归属**：test +0.01385 与 val 同向且逐 epoch 一致高于基线，数值上越过预注册阈值；但机制判据不成立 ⇒ 按 §5 **不得**宣称"零候选机制在 AliCCP 上有效"。该增益与上述"阻尼 + gate 重平衡"的行为改变同时出现，属于**未归因的正向信号**（仅备查），不构成本方法有效性的证据。

### 9.4 与 CensusIncome 臂的对照（跨数据集复现问题的直接回答）

| 项 | CensusIncome（`exp/stage2-null-expert`） | AliCCP（本臂） |
|---|---|---|
| 机制判据（null_top1_rate） | 0.3726 **成立** | 0.0000 **不成立** |
| 效用判据（ΔAUC-Test） | +0.00130 **未达标** | +0.01385 **达标** |
| 臂级判定 | 失败（效用） | 失败（机制） |

两个数据集各只过一半：CensusIncome 上机制激活但无有意义增益；AliCCP 上有正向增益但机制未按预期激活。**在该设置下"零候选路由"不构成跨数据集一般有效的机制**；本臂不主张新颖性、不主张有效性。

### 9.5 纪律核对

- 阈值与判据在 run 前写死，**事后未修改**；`null_arm` 落盘 `metrics.json`，完整证据在 `artifacts/aliccp_bench/runs/20261003-0221-…-nullx/`。
- SUMMARY 追加行（append-only，未重写任何既有行）；失败结果保留。
- 只跑了**一次**处理臂（§4 纪律）；无重跑、无挑 seed、无调参。
- 墙钟 151.1 s（基线 141.0 s；多出 ≈ 一次 val 机制探针遍历）。
