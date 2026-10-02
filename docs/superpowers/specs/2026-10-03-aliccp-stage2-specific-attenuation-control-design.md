# AliCCP Stage-2 特定混合固定衰减对照：null 增益的机理归因控制（post-hoc）

- **状态**：预注册已写死（本文件在本臂任何 run 之前提交）；实现与结果见 §10（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage2-specific-attenuation-control`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；**不从** `exp/aliccp-stage2-null-expert` 拉出，不继承其任何代码改动；协议文件零修改。与 null 臂相同，属对协议 §4.3.2 "先等协议合并"流程的显式偏离——协议本身零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（SUMMARY 列、A/B 门禁阈值、预算、指纹定义均不变）
- **假设来源（先例）**：`exp/aliccp-stage2-null-expert` @ `aba8ee9`，其 run `20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx`。该臂实测：机制判据不成立（`null_top1_rate = 0.0000000000`），但 PI 效用阈值通过（Δtest = +0.013846535450687258，Δval = +0.013786499876619396）。诊断显示 null 权重被学习为一个近似常数（`null_mean = 0.3027766269669533`，std = 0.008530222646916882，q10–q90 = 0.2911510765552521–0.3137927442789078），即 specific 混合被近似 ×0.70 常数阻尼。
- **定位声明（重要，post-hoc）**：本臂的假设是在**看到前臂结果之后**提出的——属于 **post-hoc 假设生成（mechanistic follow-up / attribution control），不是确证性新方法实验**。本臂不主张任何新颖性；唯一目的是把前臂已观察到的增益做机理归因（"常数阻尼 vs 其它"二分解）。全部判定口径在见结果前写死。
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；需变更先改本文件再改代码。

---

## 1. 假设、系数推导与冻结（不调参）

### 1.1 前臂事实（逐位引用前臂 run `20261003-0221-…-6224c0f-nullx` 的 metrics.json；非四舍五入）

| 量 | 值 |
|---|---|
| `null_mean`（val 500k 聚合 null 权重） | `0.3027766269669533` |
| `null_std` | `0.008530222646916882` |
| `null_q10 / null_q90` | `0.2911510765552521` / `0.3137927442789078` |
| `null_top1_rate` | `0.0000000000`（机制判据不成立） |
| `pred_std`（val） | `0.004380239336642408` |
| `source_gate_mean`（backbone，val） | `[0.176159685842067, 0.2537776756367646]` |
| 头 gate 均值（val） | `[0.48325546875, 0.5167445625]`（基线为 `[0.851149, 0.1488510625]`） |

### 1.2 假设 H（post-hoc）

前臂 test +0.013846535450687258 / val +0.013786499876619396 的增益，可能主要由"specific 混合被近似常数 ×0.70 阻尼"这一行为改变解释；而非由"逐样本弃权路由"解释（该机制实际并未激活：`null_top1_rate = 0`）。

### 1.3 对照构造（最小改动）与系数冻结

- **不新增 router 类别**：W 仍为 K=2 源任务的 softmax；**不新增** null key、任何参数（零新增可训练参数）。
- 仅把 specific 混合 `new_spec_rep = Σ_k W_k · spec_rep_k`（经 `new_env_emb`）乘以固定常数 c；general 路径、头内 gate、tower、loss 形式全部不变。
- **系数确定性推导（唯一来源 = 前臂 metrics.json 的 `null_mean` 全精度字段）**：
  - `c := 1 − null_mean = 1 − 0.3027766269669533 = 0.6972233730330467`（float64 字面量）
  - 模型数学为 float32：**有效因子 = float32 铸造值 `0.6972233653068542`**（记录备查，判定不依赖）
- c 在训练与评测全程一致生效；**非可训练**、不进 state_dict、不参与 `get_l2_reg()`、不消耗 RNG。
- **禁止调参**：c 只有这一个取值。任何系数搜索/网格/按结果调整 = 违反预注册。若需换 c：新分支 + 新预注册，已判定 run 不得重判。
- **已知近似（如实披露）**：前臂阻尼是逐样本的 `(1 − W_null(x))`，本对照以其 val 聚合均值常数替代；前臂 `null` 权重 std = 0.0085、q10–q90 带宽 = 0.0226，常数近似忠实。前臂未记录训练集聚合值，训练期均值与 val 均值可能有微小差异——无法测量，列入局限（§9）。

---

## 2. 改动面（最小）

| 位置 | 改动 |
|---|---|
| `multitaskrec/model.py` → `NewTask` | 新增构造参数 `spec_attenuation=1.0`（校验 `0 < c ≤ 1`）+ 属性；`forward` 中 `new_spec_rep` 计算后**条件乘**（默认 1.0 时整句跳过 → 与基点 `8133d32` 逐位一致） |
| `aliccp_benchmark/metrics.py` | `SourceGateStats`（源任务 gate 均值，与 null 臂实现**同口径**）、诊断纯函数 `pred_dispersion` / `pearson_corr` / `spearman_corr` / `logit_clip` / `paired_delta_stats` / `cross_arm_stats`、预注册常量 + `attenuation_arm_verdict` |
| `aliccp_benchmark/bench.py` | `run_stage2(..., spec_attenuation=1.0)`；`stage2_run_id`（`-sattn` 后缀）；处理臂专属 val 探针 `newtask_attenuation_probe`（no_grad 一次遍历）；config/metrics 记录臂信息 |
| `aliccp_benchmark/compare.py`（新文件） | 三方对照分析（§6.2）：同一 Stage-1 产物 + 三个头 `newtask.pt`，同一 test/val 序重评测；完整性检查；离散度/相关/paired delta；落盘 JSON |
| `run_aliccp_benchmark.py` | stage2 新增 `--spec-attenuation`（默认 1.0）；新增 `compare` 子命令 |
| `aliccp_benchmark/tests/test_spec_attenuation.py`（新文件） | §3 不变量 |

**明确不改**：`aliccp_benchmark/protocol.py`（零改动）、`config.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py`、`baseline/*`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`multitaskrec/model.py` 中 `NewTask` 以外的任何类、CensusIncome / ByteRec 任何代码路径。`SUMMARY.md` 只追加行。**前臂代码（null expert）不移植进本分支**（三方对照中 null 头类从 git blob 动态载入，见 §6.2）。

---

## 3. 不变量（由 `test_spec_attenuation.py` 强制，不接受人工目测）

参照前臂 I1–I11 口径；参考实现从 git `8133d32:multitaskrec/model.py` 动态加载（本分支基点 = 基线 run 实际使用的模型代码）。

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂（`spec_attenuation=1.0`）前向/反向与基点实现**逐位一致** | `torch.equal` 比对输出与每个参数的梯度 |
| I2 | 默认臂参数集合/state_dict = 基点；衰减不是参数、不进 state_dict；构造不消耗 RNG（同种子两臂共享参数逐位一致） | 参数集合差集 + `load_state_dict(strict=True)` + 种子对照 |
| I3 | 开启臂前向等于本文件写死的 K-way 公式（`new_spec_rep' = c × Σ W_k spec_rep_k`，同温度 softmax） | 独立参考实现比对（`torch.equal`） |
| I4 | 开启臂可训练参数计数 = 默认臂（零新增）；全部原参数梯度照旧存在；`get_l2_reg()` 与默认臂同权重下相等 | 计数 + 梯度 + 回归对照 |
| I5 | 系数钉死：代码常量 `== 1.0 − 0.3027766269669533`，与文档字面量 `0.6972233730330467` 一致；float32 有效值钉死 `0.6972233653068542`；非法值（≤0 或 >1）构造抛错 | 常量相等断言 + `test_doc_preregisters_same_numbers` + 异常测试 |
| I6 | 诊断量数值正确：`SourceGateStats`（流式==单批、构造数据正确）、`pred_dispersion`/`pearson_corr`/`spearman_corr`（scipy 参考）/`logit_clip`/`paired_delta_stats` | 构造数据精确断言（float64）|
| I7 | 判定函数边界正确：主判据 `test_auc ≥ 0.6085317926465923`（闭）且 `val_auc_best > 0.5781533414372665`（开）；PI 口径单独记录；记录带 MOST/PARTIAL/MINOR/NEGATIVE 与文档一致 | 边界/证据字段测试 |
| I8 | 对照管线：`predict_newtask` 的 AUC 与 `bench.evaluate_newtask` 逐位一致；null 头类从 git blob（sha256 = `836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c`）重构；三头 strict 载入；极小夹具端到端完整性通过 | 等价测试 + sha 钉死 + `skipUnless` 真实产物测试 + CPU 夹具 |
| I9 | 默认臂 metrics 既有键与语义不变（仅新增顶层 `spec_attenuation: 1.0` 标记；不出现 `probe`/`attenuation_arm`/`trainable_params`）；处理臂落盘完整 | `TestRunnerAttenuationArm`（CPU 极小夹具端到端） |
| I10 | 静态守卫：模型侧唯一改动是 `model.py`；`aliccp_benchmark/protocol.py` 相对基点零改动；相对 `8133d32` 的全部跟踪文件改动 ⊆ §2 白名单 | `git diff --name-only` |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 4. 运行方式（恰好一次）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 处理臂（本实验唯一要跑的 run；固定 Stage-1 产物 = 基线/null 两臂使用的那一份）
.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 --tag short `
  --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0 --spec-attenuation 0.6972233730330467

# 2) 三方对照（纯分析：无训练、不改任何 run 产物；run 之后执行恰好一次）
.venv\Scripts\python.exe run_aliccp_benchmark.py compare `
  --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0 --arm-run <上一步 run_id>
```

- 基线臂**不重跑**：对照 = 协议 SUMMARY 已有 run `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`（同 `stage1_id`、同预算、同种子、同顺序，协议 §12.1 允许）。
- 预算 / 种子 / 顺序全部走协议默认（train 2M / val 500k / test 1M 前缀；model seed 1688723512；env seed 20261003；batch 2000、shuffle=False；stage2 epochs=5、patience=2）——**不搜参、不换结构、不调温度、不改初始化**。
- 处理臂 `run_id` 形如 `YYYYMMDD-HHmm-p2M-v500k-t1M-m1688723512-short-<commit>-sattn`。
- **前置条件**：开跑前先提交本批改动（否则 run_id / metrics.json 的 commit 记的不是本实验代码，可审计性有缺口）。运行期间不得有其他未提交的已跟踪文件改动（`git.dirty` 必须为 false）。
- `compare` 子命令默认三臂 = 上述基线 run、前臂 null run `20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx`、以及 `--arm-run`（必填）。

---

## 5. 预注册接受标准（写死；看到结果后不得修改）

| 项 | 值 |
|---|---|
| 对照基线 | run `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`（`stage1_id=s1-5c060b9c-m1688723512-e3-3a30e2c0`） |
| 基线值（逐位取自该 run `metrics.json`） | `AUC-Test-BSI = 0.5988392178311113`；`AUC-Val-BSI(best) = 0.5781533414372665` |
| 前臂（null）值（逐位取自其 `metrics.json`） | `AUC-Test-BSI = 0.6126857532817985`；`AUC-Val-BSI(best) = 0.5919398413138859`；增益 Δtest = `0.013846535450687258`、Δval = `0.013786499876619396` |
| **主判据（本臂核心；分类判定）** | **重现率 r := Δtest(control) / Δtest(null) ≥ 0.70**（"most" 的预注册定义）**且** Δval(control) > 0 —— 两条同时满足 → 判定 **ATTENUATION_SUPPORTED**（衰减解释增益主体）；否则 **ATTENUATION_FALSIFIED**（该解释不成立） |
| 主判据的判定字面量（AUC 形式，权威口径） | `AUC-Test-BSI(control) ≥ 0.6085317926465923`（= 0.5988392178311113 + 0.70 × 0.013846535450687258）**且** `AUC-Val-BSI(best, control) > 0.5781533414372665` |
| 比率形式与 AUC 形式的浮点关系（披露） | 两形式数学等价；float64 下在精确边界处可差 ≤ 1 ulp（实测 roundtrip −5.55e-17）。判定以 **AUC 字面量形式**为准；ratio 仅作记录 |
| **provisional 效用检验（PI +0.0055；单独记录，不参与分类）** | `AUC-Test-BSI(control) ≥ 0.6043392178311113` 且 val 同向（同前臂字面量口径） |
| 记录带（descriptive，不判定） | r ≥ 0.70 → MOST；0.30 ≤ r < 0.70 → PARTIAL；0.00 ≤ r < 0.30 → MINOR；r < 0 → NEGATIVE（MOST 与主判据同源；下界仅记录） |
| 机械判定 | `metrics.attenuation_arm_verdict(test_auc, val_auc_best)`，完整落盘 `metrics.json["attenuation_arm"]` |

**结果解读的预注册口径**：SUPPORTED ⇒ 证据支持"常数阻尼（+ 头部 gate 自由重平衡）是前臂增益的主要驱动"；FALSIFIED ⇒ 证伪"聚合衰减解释该增益"，即前臂增益并非（仅）由平均意义上的阻尼造成（可能是逐样本调制或优化动力学差异，本臂不进一步归因）。

**止损规则（与协议 §10.2、前臂 §5 一致）**：

1. 未达主判据 → 如实记录该次结果（SUMMARY 追加行 + `runs/` 留档 + §10 补记），**不得**重跑挑好的、不得换阈值、不得跳到多 seed 或全量。
2. **不得调 c**（§1.3）；不得加跑第二个系数、第二个 seed、第二个 epoch 预算。
3. 阈值只允许在**看到任何本臂 run 结果之前**修改；每次修改单独 commit 并在本文件记录理由；已判定的 run 不得用新阈值重判。
4. 运行期间发现实现缺陷 → 允许修复后重跑，但必须同时保留失败 run 的产物与说明。

---

## 6. 诊断（C 类：只记录；test 不参与任何机制判定）

### 6.1 in-run 探针（处理臂专属；val 一次遍历，no_grad）

| 键 | 含义 | 口径 |
|---|---|---|
| `pred_mean` / `pred_std` / `pred_var` / `pred_q10..q90` / `pred_min` / `pred_max` | 头 BSI 预测的离散度（val 全量） | float64 终算、`std/var` 用 `unbiased=False`；**与前臂 `pred_std` 同公式**（前臂记录值 `0.004380239336642408`） |
| `source_gate_mean` | backbone `gate_networks` 每源任务 specific 分支在 val 的均值（2 维） | 与 null 臂 `SourceGateStats` **逐字同口径**（float64 逐批累加）；前臂记录值 `[0.176159685842067, 0.2537776756367646]`，预期逐位一致 |

### 6.2 三方对照（`compare` 子命令；run 后恰好一次；纯分析、无训练、不改任何 run 产物）

- 载入**同一** Stage-1 产物（真冻结、no_grad）；三个头 `newtask.pt` 各自 weight-only strict 载入：基线 run / null run / 本臂 run。
- **null 头类重构**：从 git blob `6224c0f:multitaskrec/model.py`（sha256 = `836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c`）动态载入（与测试同一机制；不改动任何分支文件；前臂代码不移植）。
- 同一 **test** 前缀（1M、batch 2000、shuffle=False、同序）逐样本预测；**val** 同序复算 gate 均值与离散度。
- **完整性检查（先于解读；任何一项失败 → 对照判 INTEGRITY_FAIL，只作废诊断解读，不影响 run 内主判据）**：
  - 重算 AUC（`bench.evaluate_newtask` / `metrics.auc_score` 原口径）必须 = 三 run 各自记录值（|Δ| ≤ 1e-9；预期逐位相等）；
  - 重算头 gate_mean（`bench.newtask_gate_mean` 原函数）与各自记录值 |Δ| ≤ 1e-12；
  - `source_gate_mean` 与 null 臂记录值 |Δ| ≤ 1e-12，且与本臂 in-run 探针值 |Δ| ≤ 1e-12；
  - 本臂重算 val `pred_std` 必须 = in-run 探针值（逐位，容差 0）；null 臂重算 val `pred_std` 与记录值 |Δ| ≤ 1e-12；
  - 指纹校验（A2 口径）通过；三臂 y_true 逐位一致（sha256 记录）；backbone sha 与产物 meta 一致。
- **输出量**（test 同序；逐臂）：pred 离散度（std/var/分位数）；两两 Pearson(logit) 与 Spearman(pred) 相关（control-null / control-baseline / null-baseline）；paired delta（mean/std/分位数/frac_pos 等）；val 离散度表；gate 表（头 gate 均值三臂 + 源 gate 均值）。
- **方向性预期（descriptive，不参与判定）**：若 SUPPORTED，则预期 `corr(control, null) > corr(control, baseline)`（Pearson-logit 与 Spearman 两口径都报告）；若 FALSIFIED，则不预期此排序。两种情况都如实报告。
- 落盘：`artifacts/aliccp_bench/runs/<control-run>/three_way_compare.json`（+ 日志）。

---

## 7. 门禁与继承披露（B1/B4）

- 本臂执行协议 §10 全部 A 类与 B 类判定（`--tag short` ⇒ `enforce_b=True`），`attenuation_arm` 是**臂级**判定，不并入 `hard_pass`（A/B 门禁语义与 `protocol.py` 不变）。
- **继承的失败（非本臂引入，三臂按构造完全相同）**：基线 short run 已记录 `B1 FAIL`（`AUC-Val-CTR 0.5493 < 0.55`）与 `B4 FAIL`（cluster 后 env_0 仅 566/2,000,000）。这两项由**共享的 Stage-1 产物**决定，本臂与其基线引用同一 `stage1_id` ⇒ 判定逐位相同。报告中如实披露为**继承基线缺陷**，不归因于本臂、不据此否定本臂，也不据此宣称任何东西。
- A 类门禁必须全 PASS（A3 SKIP 视为通过），否则本臂比较作废（协议 §12.1）。
- B2（val-test 差）/ B3（gate_mean 未坍缩）为臂相关门禁，照常判定与披露。

---

## 8. 报告与分支处置

- `SUMMARY.md` 只追加、不重写：处理臂即使失败也必须有行（协议 §9.3）。
- 结果（含分类判定、两臂/三臂对照、门禁、诊断数值、完整性检查、止损结论）在运行后**追加**到本文件 §10；§1–§9 预注册内容不得回填、不得修改。
- 分支处置（用户硬规则）：无论成功失败，**只保留 `exp/aliccp-stage2-specific-attenuation-control` 本分支，不合并回 `master`**；推送需用户显式批准。
- 与本实验无关的既有探索（TC-Prompt / CGR / affinity gate / KL-Prompt / T4 / null expert 代码本身等）一律不引用、不重建。

---

## 9. 局限（预声明）

1. **单 seed、单 run**：r 与 Δ 均为点估计，无重复噪声估计；判定按预注册阈值执行，不引入事后置信区间叙事。
2. **post-hoc 假设**：H 来自前臂行为观察（§1.2），本臂检验的是"常数阻尼 + 头部 gate 自由重平衡"这一**复合**假设的可复现性，不是独立发现的新机制。
3. **常数近似**：c 为 val 聚合常数，非逐样本 `1 − W_null(x)`；训练期聚合值不可得（前臂未记录）。
4. **前臂增益本身**为单 seed、机制门禁未过（`null_top1_rate = 0`）的未归因正信号；本臂不改变前臂的判定，只回答"该增益是否可由聚合阻尼重现"。
5. **对照仅检验一条路径**：FALSIFIED 时只证伪"聚合阻尼解释"，不构成对其它解释的确证；SUPPORTED 时归因上限为"常数阻尼（含其诱发的 gate 重平衡）"，不区分二者内部贡献。

---

## 10. 结果（实测；运行后追加，不得回填预期值）

**判定：`ATTENUATION_SUPPORTED`（band = MOST）。** 主判据两条**同时满足**：reproduction ratio r = `0.917640706647481` ≥ 0.70 **且** Δval = `+0.013279276931939643` > 0。provisional PI 口径（+0.0055）亦通过（单独记录，不参与分类）。不扩多 seed、不跑全量（本臂无多 seed 阶段）。

### 10.1 三臂对照（同 `stage1_id`；同预算、同种子、同顺序；test 各只评一次）

| 项 | 基线臂 | 前臂（null） | 本臂（control，c = 0.6972233730330467） |
|---|---|---|---|
| run_id | `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9` | `20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx` | `20261003-0322-p2M-v500k-t1M-m1688723512-short-f88dca4-sattn` |
| commit / dirty | `b2e17f9` / false | `6224c0f` / false | `f88dca4` / false |
| stage1_id | `s1-5c060b9c-m1688723512-e3-3a30e2c0` | 同左 | 同左（同一 checkpoint） |
| best val AUC-BSI | `0.5781533414372665`（ep5） | `0.5919398413138859`（ep5） | `0.5914326183692061`（ep5） |
| **AUC-Test-BSI** | `0.5988392178311113` | `0.6126857532817985` | `0.6115453624066993` |
| Δ vs 基线（test / val） | — | `+0.013846535450687258` / `+0.013786499876619396` | `+0.012706144575588052` / `+0.013279276931939643` |
| reproduction ratio（test / val） | — | 1.0 / 1.0 | **`0.917640706647481`** / `0.9632087223574451` |
| 逐 epoch val AUC | 0.4937 / 0.5037 / 0.5178 / 0.5434 / 0.5782 | 0.5059 / 0.5211 / 0.5434 / 0.5673 / 0.5919 | 0.5070 / 0.5226 / 0.5455 / 0.5687 / 0.5914 |
| 头 gate 均值（val） | `[0.851149, 0.1488510625]` | `[0.48325546875, 0.5167445625]` | `[0.49435615625, 0.50564384375]` |
| trainable params | 8129 | 8193（= 8129 + 64） | **8129（零新增，= 默认臂）** |
| PI provisional 口径 | 未过 | 通过 | **通过**（`checks.pi_utility = true`，单独记录） |

主判据两形式一致性核对：r 口径与 AUC 字面量口径（`0.6115453624066993 ≥ 0.6085317926465923`）判定一致，无边界分歧。

### 10.2 门禁

| 检查 | 结果 |
|---|---|
| A1 / A2 / A4 / A5 / A6 | **PASS**（backbone sha before==after==loaded，`.grad` 全 None；指纹与标签对齐通过） |
| A3 | SKIP（按需复跑，不阻塞首轮） |
| B1 | **FAIL（继承基线，非本臂引入）**：CTR 0.5493 < 0.55 来自共享 Stage-1 产物；BSI 分量 0.6115 ≥ 0.53 **通过** |
| B2 | PASS（\|val−test\| = 0.0201 ≤ 0.05） |
| B3 | PASS（gate_mean = `[0.49435615625, 0.50564384375]` ∈ [0.05, 0.95]） |
| B4 | **FAIL（继承基线，非本臂引入）**：cluster events = [(566, 1999434)]，与基线逐位相同（同一 Stage-1 meta） |

`hard_pass = false`（B1/B4 继承失败，与前两臂按构造完全相同）。

### 10.3 in-run 探针（处理臂专属，val 一次遍历，no_grad）

| 键 | 值 |
|---|---|
| `pred_mean` / `pred_std` / `pred_var` | `0.9913739628497362` / `0.004451373214009348` / `1.981472349039991e-05` |
| `pred_min` / `pred_max` | `0.9107213616371155` / `0.9987686276435852` |
| `pred_q10 / q25 / q50 / q75 / q90` | `0.9858221411705017` / `0.9895219206809998` / `0.9923179149627686` / `0.9943958520889282` / `0.9958062171936035` |
| `source_gate_mean`（backbone，冻结） | `[0.176159685842067, 0.2537776756367646]`（与前臂记录值**逐位一致**） |

### 10.4 三方对照（`compare` 子命令；同一 test/val 序；纯分析恰跑一次）

**完整性检查：全部 PASS，所有 |Δ| = 0.0（逐位相等）。** baseline/null/control 三臂重算 AUC 与各自记录值 diff = 0.0（含前臂 `0.6126857532817985`）；头 gate 均值、`source_gate_mean`、前臂 `pred_std`（`0.004380239336642408`）均逐位复现；null 头源码 sha256 = `836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c`（钉死通过）；三臂 y_true sha256 一致（`2e251c4acd8ba848…`）；本臂重算 val `pred_std` = in-run 探针值（逐位）。诊断解读有效。

**相关（test，同序 1M）：**

| 对 | Pearson(logit) | Spearman(pred) | Pearson(pred) |
|---|---|---|---|
| control vs **null** | **`0.9993984942943822`** | **`0.9993505670902368`** | `0.9992507917159521` |
| control vs baseline | `0.8153554490985575` | `0.7962207228613848` | `0.8189340623864756` |
| null vs baseline | `0.8136671729702486` | `0.7954514969256509` | `0.8148595534144193` |

**paired delta（test，同序）：**

| 对 | mean | std | q10 / q50 / q90 | frac_pos |
|---|---|---|---|---|
| control − null | `-0.00010642456352710724` | `0.00018741332194162008` | −3.47e-4 / −7.50e-5 / 7.21e-5 | 0.2741 |
| control − baseline | `+0.00039633107310533525` | `0.0026123970576252455` | −2.77e-3 / +7.71e-4 / +3.10e-3 | 0.6388 |
| null − baseline | `+0.0005027556366324424` | `0.0025899341986537695` | −2.61e-3 / +8.79e-4 / +3.16e-3 | 0.6608 |

**离散度（pred_std）：** val — 基线 `0.0033728455401762676` / null `0.004380239336642408` / control `0.004451373214009348`；test — 基线 `0.0033986210314953876` / null `0.0044506792920786945` / control `0.004521192069464136`。

**方向性预期（descriptive）：成立** —— `corr(control, null) > corr(control, baseline)` 在 Pearson-logit 与 Spearman 两口径均为 true（`prereg_expectation` 两项 true）。

### 10.5 解读（预注册口径，§5；不越界）

**SUPPORTED ⇒ 证据支持"常数阻尼（含其诱发的头 gate 重平衡）是前臂增益的主要驱动"**，具体为：

1. 零新增参数、无 null router、无逐样本路由的最简对照（仅把 specific 混合 × 0.6972233730330467）重现了前臂 test 增益的 **91.8%**、val 增益的 **96.3%**；
2. 三方对照显示 control 与 null 的预测**近乎同一**（Pearson-logit 0.99940；paired mean −1.06e-4），而二者与基线的关系近乎相同（0.8154 / 0.8137）；
3. 前臂的头 gate 重平衡被本臂重现：baseline `[0.8511, 0.1489]` → null `[0.4833, 0.5167]` → control `[0.4944, 0.5056]`（control 完成二者间位移的 ≈97%）；
4. 预测位移方向与幅度一致：control−baseline 平均 +3.96e-4 ≈ null−baseline +5.03e-4 的 79%。

**如实披露**：

- **未归因残差**：r = 0.9176 ⇒ 约 8.2% 的 test 增益、3.7% 的 val 增益未被常数阻尼重现（可能为逐样本调制、`null_key` 优化耦合或单 seed 噪声；单 seed 无法区分，不作进一步主张）。
- 本对照自身亦过 provisional PI 效用阈值（`0.6115453624066993 ≥ 0.6043392178311113`）——仅作记录；**本臂是归因控制而非方法声明**，不主张任何方法有效性/新颖性，也不改变前臂既有判定（前臂机制判据 FAIL 维持）。
- B1/B4 继承失败照常披露（与基线逐位相同，非本臂引入）。
- 局限见 §9（单 seed；post-hoc 假设；常数 vs 逐样本聚合近似）。

### 10.6 纪律核对

- 处理臂**恰好一次** run（§4），无重跑、无挑 seed、无调参；系数与阈值事后**未修改**；`compare` 恰跑一次（纯分析，无训练、不改任何 run 产物）。
- 产物留档：`artifacts/aliccp_bench/runs/20261003-0322-…-sattn/`（config/metrics/gate_report.json、newtask.pt、three_way_compare.json；均为 gitignore 产物），SUMMARY 追加行为入库记录。
- SUMMARY 追加行（append-only，未重写任何既有行）；`git.dirty = false`（run 时）。
- 墙钟 146.5 s（基线 141.0 s、前臂 151.1 s）；峰值显存 52.0 MB。
