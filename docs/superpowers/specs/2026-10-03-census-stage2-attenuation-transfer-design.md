# CensusIncome Stage-2 跨数据集外部验证：AliCCP 固定 specific 混合衰减系数迁移

- **状态**：预注册已写死（本文件在本臂任何 run 之前提交）；实现与结果见 §10（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/census-stage2-attenuation-transfer`（自 `infra/fair-stage2-benchmark` @ `87afe03` 独立拉出；**不从**任何 AliCCP 或 Census `exp/*` 分支拉出、不继承其任何代码改动；协议文件零修改）
- **协议依赖**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（Census 公平评测协议）与 `census_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（SUMMARY 列、A/B 门禁阈值、预算、指纹定义均不变）
- **定位声明（重要）**：本实验是**对 AliCCP 事后（post-hoc）因果发现的跨数据集外部验证（external validity check）**。它**不是**新方法、**不主张**任何新颖性、**不是**新的系数搜索：系数逐位冻结自 AliCCP 已完成的对照 run，只有这一个取值。Census 侧不做任何数据集特定调参（包括：不使用、不计算、不比较任何"Census 原生"系数——例如 null 臂的 `null_mean` 派生值，见 §1.4）。
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；需变更先改本文件再改代码。

---

## 1. 假设来源、系数审计与冻结（不调参）

### 1.1 AliCCP 侧事实（系数来源；逐位引用 AliCCP 分支既有产物，非四舍五入）

来源 run：`exp/aliccp-stage2-null-expert` @ `aba8ee9` 的 run `20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx`。

| 量 | 值 |
|---|---|
| `null_mean`（val 聚合 null 权重；**系数唯一来源**） | `0.3027766269669533` |
| `null_top1_rate` | `0.0000000000`（null 从未被选 ⇒ 前臂行为 = 近似常数阻尼） |
| `null_std` | `0.008530222646916882`；`null_q10–q90 = 0.2911510765552521–0.3137927442789078` |

来源 run：`exp/aliccp-stage2-specific-attenuation-control` @ `3314420` 的 run `20261003-0322-p2M-v500k-t1M-m1688723512-short-f88dca4-sattn`（对照臂，即本系数的已被执行过的 AliCCP 版本）。

| 量 | 值 |
|---|---|
| 基线（AliCCP run `20261003-0130-…-b2e17f9`） | `AUC-Test-BSI = 0.5988392178311113`；`AUC-Val-BSI(best) = 0.5781533414372665` |
| null 臂（前节 run） | `AUC-Test-BSI = 0.6126857532817985`；`Δtest = +0.013846535450687258`；`Δval = +0.013786499876619396` |
| 对照臂（固定系数，本系数） | `AUC-Test-BSI = 0.6115453624066993`；`AUC-Val-BSI(best) = 0.5914326183692061`；`Δtest = +0.012706144575588052`；`Δval = +0.013279276931939643` |
| 重现率 `r = Δtest(control)/Δtest(null)` | `0.917640706647481`（预注册带 MOST，`ATTENUATION_SUPPORTED`） |
| 头 gate 均值（val；对照/基线） | `[0.49435615625, 0.50564384375]` / `[0.851149, 0.1488510625]` |
| 相关（test 同序，Pearson-logit） | `corr(control, null) = 0.9994`；`corr(control, baseline) = 0.8154` |

### 1.2 系数审计（audit trail，逐字核对）

- **定义**：`c := 1 − null_mean = 1 − 0.3027766269669533 = 0.6972233730330467`（float64 字面量）。
- **代码位置（AliCCP 分支 @ `3314420`）**：`aliccp_benchmark/metrics.py` 第 109–111 行：`NULL_RUN_NULL_MEAN = 0.3027766269669533`；`SPEC_ATTENUATION_COEF = 1.0 − NULL_RUN_NULL_MEAN`（`= 0.6972233730330467`）；`SPEC_ATTENUATION_COEF_F32 = 0.6972233653068542`。
- **模型数学为 float32**：有效因子 = float32 铸造值 `0.6972233653068542`（本分支已实测一致；记录备查，判定不依赖）。
- **施加位置（AliCCP）**：`multitaskrec/model.py` → `NewTask.forward`，`new_spec_rep = matmul(stack(spec_reps), W).squeeze()` 之后、`env_aware_rep = new_spec_rep * new_env_emb` 之前，固定常数乘（非可训练、无参数、不进 `state_dict`、不消耗 RNG）。

### 1.3 Census 阶段 2 架构审计（本分支基点 `87afe03`）

- `git diff 8133d32:multitaskrec/model.py 87afe03:multitaskrec/model.py` **为空**（AliCCP 基点与 Census 基点的模型代码逐字节相同）⇒ 衰减的插入点语义在两数据集上**按构造完全同构**。
- 阶段 2 管线（`run_census_benchmark.py::run_stage2`，协议文件零修改）：冻结 backbone（`MPTRec`，参数级 + eval 级 + `no_grad` 抽表征三件套）→ 从固定 Stage-1 产物加载 → 每 batch `get_infos()` 得 `(dnn_input, gen_rep, spec_reps, env_embs)` →
  `NewTask`：`H_out = projection_network(dnn_input)`；source router `W = softmax(H_out @ stack(env_embs) / T)`（T=150，K=2 源任务）；`new_spec_rep = Σ_k W_k · spec_rep_k`；`env_aware_rep = new_spec_rep * new_env_emb`；head gate `gate_out = softmax(gate_network(dnn_input))`（2 维：specific / general）；`fused_rep = gate_out_0 · env_aware_rep + gate_out_1 · gen_rep`；`tower_network(fused_rep)` → sigmoid（BCE 目标）。
- 训练：仅 `NewTask` 参数，Adam `lr=1e-3`，loss = `BCE + reg_dnn × tower L2`；`epochs=5`、`patience=2`；val 只用于 early stop 选点，test 全结束后评一次（协议 §4.6、§7.3）。
- **本实验唯一改动**：在 `new_spec_rep` 计算后**条件乘**固定常数 `c`（§2）。general 路径、head gate、router、tower、loss、优化器全部不变。

### 1.4 假设 H（外部验证问题）

AliCCP 的对照实验已显示：固定常数 `c = 0.6972233730330467`（把 specific 混合整体乘以该常数、并允许头 gate 自由重平衡）可在 AliCCP 上重现 null 臂增益的 91.8%。**本实验的问题（唯一问题）**：把这**同一个、零改动的**常数原样应用到 CensusIncome 阶段 2，不做任何数据集特定适配，它是否仍然表现为一个正向干预（达到 §5 主判据）？

**如实披露的差异（预注册，不事后解释）**：Census 侧的前臂（`exp/stage2-null-expert` @ `1eadd05` run `20260929-1802-…-1eadd05-nullx`）与 AliCCP 前臂行为不同——其 `null_top1_rate = 0.37256670876686515`（逐样本路由**已激活**，而 AliCCP 为 0），`null_mean = 0.34177784155227137`。因此：

1. Census null 臂的 +0.001296296526501206 增益**不能**先验归因于常数阻尼；
2. **明确禁止**用 Census 的 `null_mean` 派生 "Census 原生系数"（`1 − 0.34177784155227137 = 0.6582221584477286`）参与本实验的任何判定或运行——那属于数据集特定调参，正是本外部验证要排除的变量。该值仅在此披露其存在性。
3. 本实验的系数只有 `0.6972233730330467` 一个取值；任何系数搜索/网格/按结果调整 = 违反预注册。

**已知近似（如实披露，同 AliCCP）**：AliCCP 前臂阻尼是逐样本 `(1 − W_null(x))`，对照以聚合均值常数替代；常数近似在 AliCCP 上忠实（std = 0.0085、q10–q90 带宽 = 0.0226）。Census 前臂的逐样本 null 权重分布更宽（top1 = 0.37），常数近似在 Census 上**先验地更粗**——这本身就是本外部验证要观测的一部分。

---

## 2. 改动面（最小；全部相对基点 `87afe03`）

| 位置 | 改动 |
|---|---|
| `multitaskrec/model.py` → `NewTask` | 新增构造参数 `spec_attenuation=1.0`（校验 `0 < c ≤ 1`）+ 属性；`forward` 中 `new_spec_rep` 计算后**条件乘**（默认 1.0 时整句跳过 → 与基点逐位一致） |
| `census_benchmark/metrics.py` | 预注册常量（§1.2 系数 + §5 阈值/基线/nullx 字面量）、`transfer_arm_verdict`、诊断纯函数（`pred_dispersion` / `pearson_corr` / `spearman_corr` / `paired_delta_stats`）、探针函数 `attenuation_probe`（val 一次遍历、no_grad、流式 float64） |
| `run_census_benchmark.py` | `run_stage2(..., spec_attenuation=1.0)`；`stage2` 子命令新增 `--spec-attenuation`（默认 1.0）；启用时 `run_id` 追加 `-sattn` 后缀；`config.json` / `metrics.json` 记录臂信息与参数审计；新增 `compare` 子命令（三方对照入口） |
| `census_benchmark/compare.py`（新文件） | 三方对照分析（§6.2）：同一 Stage-1 产物 + 三个头 `newtask.pt`，同一 test/val 序重评测；完整性检查；离散度/相关/paired delta/表征范数；落盘 JSON |
| `census_benchmark/tests/test_spec_attenuation.py`（新文件） | §3 不变量 |
| `census_benchmark/tests/test_smoke.py` | 静态守卫收窄：模型侧唯一允许改动 = `model.py` 的 `spec_attenuation` 分支（默认关闭）；其余守卫不变 |

**明确不改**：`census_benchmark/protocol.py`（零改动）、`config.py`、`CensusIncome_MPTRec.py`、`CensusIncome_NewTask.py`、`baseline/*`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`multitaskrec/model.py` 中 `NewTask` 以外的任何类、AliCCP / ByteRec 任何代码路径。`SUMMARY.md` 只追加行。**前臂代码（Census null expert）不移植进本分支**（三方对照中 null 头类从 git blob 动态载入，见 §6.2）。

### 2.1 为什么默认臂能逐位等于基点

- `spec_attenuation=1.0` 时：无任何新属性进入 `state_dict`、无参数、无 buffer；`forward` 中条件乘整句跳过 ⇒ 算子、操作数、顺序与基点完全相同。
- 衰减常数不是参数、不消耗 RNG（同种子两臂共享参数初始化逐位一致）、不进入 `get_l2_reg()`（loss 形式不变）。
- 由 `test_spec_attenuation.py::TestDefaultArmBitIdenticalToBase` 从 git `87afe03:multitaskrec/model.py` 动态加载参考实现，逐位比对前向与每个参数的梯度强制。

---

## 3. 不变量（由 `test_spec_attenuation.py` 强制，不接受人工目测）

参考实现从 git `87afe03:multitaskrec/model.py` 动态加载（本分支基点 = 全部对照 run 实际使用的模型代码）。

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂（`spec_attenuation=1.0`）前向/反向与基点实现**逐位一致** | `torch.equal` 比对输出与每个参数的梯度 |
| I2 | 默认臂参数集合/`state_dict` = 基点；衰减不是参数、不进 `state_dict`；构造不消耗 RNG（同种子两臂共享参数逐位一致） | 参数集合差集 + `load_state_dict(strict=True)` + 种子对照 |
| I3 | 开启臂前向等于本文件写死的公式（`new_spec_rep' = c × Σ_k W_k spec_rep_k`，同温度 softmax） | 独立参考实现比对（`torch.equal`） |
| I4 | 开启臂可训练参数计数 = 默认臂（零新增）；全部原参数梯度照旧存在；`get_l2_reg()` 与默认臂同权重下相等 | 计数 + 梯度 + 回归对照 |
| I5 | 系数钉死：代码常量 `== 1.0 − 0.3027766269669533`，与文档字面量 `0.6972233730330467` 一致；float32 有效值钉死 `0.6972233653068542`；非法值（≤0 或 >1）构造抛错 | 常量相等断言 + `test_doc_preregisters_same_numbers` + 异常测试 |
| I6 | 诊断量数值正确：`pred_dispersion` / `pearson_corr` / `spearman_corr`（scipy 参考）/ `paired_delta_stats` | 构造数据精确断言（float64） |
| I7 | 判定函数边界正确：主判据 `test_auc ≥ 0.8521`（闭）且 `val_best > 0.8527881905614896`（开）且 `construction_ok` 且 `a_gates_ok`；`TRANSFER_SUPPORTED` / `TRANSFER_NOT_SUPPORTED` 与文档一致 | 边界/证据字段测试 |
| I8 | 对照管线：`predict_newtask` 的 AUC 与 `metrics.evaluate_newtask` 逐位一致；null 头类从 git blob（sha256 = `836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c`）重构；三头 strict 载入；极小夹具端到端完整性通过 | 等价测试 + sha 钉死 + `skipUnless` 真实产物测试 + CPU 夹具 |
| I9 | 默认臂 metrics 既有键与语义不变（仅新增顶层 `spec_attenuation: 1.0` 标记；不出现 `probe` / `transfer_arm` / `trainable_params`）；处理臂落盘完整（含参数审计与探针） | `TestRunnerAttenuationArm`（CPU 极小夹具端到端） |
| I10 | 静态守卫：模型侧唯一改动是 `model.py`；`census_benchmark/protocol.py` 相对基点 `87afe03` 零改动；相对 `87afe03` 的全部跟踪文件改动 ⊆ §2 白名单 | `git diff --name-only` |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

> 注（null blob 交叉确认）：`1eadd05:multitaskrec/model.py` 的 sha256 与 AliCCP 分支对照实验引用的 null 头 blob sha256 相同——两数据集基点 model.py 逐字节相同（§1.3），此为一致性交叉验证，非巧合性依赖。

---

## 4. 运行方式（恰好一次）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v

# 1) 处理臂（本实验唯一要跑的 run；固定 Stage-1 产物 = 基线/null 两臂使用的那一份）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 `
  --tag short --epochs 5 --spec-attenuation 0.6972233730330467 --gpu 0

# 2) 三方对照（纯分析：无训练、不改任何 run 产物；run 之后执行恰好一次）
.venv\Scripts\python.exe run_census_benchmark.py compare `
  --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 `
  --control-run <上一步 run_id>
```

- 基线臂**不重跑**：对照 = 协议既有 run `20260929-1735-s20260929-m1685480945-short-904f8d0`（同 `stage1_id`、同 split/model/env seed、同预算、同顺序）。
- null 臂**不重跑**：对照 = 既有 run `20260929-1802-s20260929-m1685480945-short-1eadd05-nullx`（同上，`--null-expert` 臂）。
- 种子 / 划分 / 顺序 / epoch 预算全部走协议默认（split seed 20260929、model seed 1685480945、env seed 20260929、train 全量 199523、test.gz 50/50、batch 256、`shuffle=False`、stage2 `epochs=5`、`patience=2`、`lr=1e-3`）——**不搜参、不换结构、不调温度、不改初始化**。
- 处理臂 `run_id` 形如 `YYYYMMDD-HHmm-s20260929-m1685480945-short-<commit>-sattn`。
- **前置条件**：开跑前先提交本批改动（否则 `run_id` / `metrics.json` 里的 `commit` 记的不是本实验代码，可审计性有缺口）。运行期间不得有其他未提交的已跟踪文件改动。

---

## 5. 预注册接受标准（写死；看到结果后不得修改）

| 项 | 值 |
|---|---|
| 对照基线 | run `20260929-1735-s20260929-m1685480945-short-904f8d0`（`stage1_id=s1-096f8f16-m1685480945-e2-cb2094b3`） |
| 基线值（逐位取自该 run `metrics.json`） | `AUC-Test-Education = 0.8500685307175756`；`AUC-Val-Education(best) = 0.8527881905614896` |
| 前臂（Census null）值（逐位取自其 `metrics.json`；descriptive 参照，非系数来源） | `AUC-Test-Education = 0.8513648272440768`；`AUC-Val-Education(best) = 0.853020431042571`；`Δtest = 0.001296296526501206`、`Δval = 0.00023224048108139161` |
| **主判据（分类判定）** | `AUC-Test-Education(arm) ≥ 0.8521` **且** `AUC-Val-Education(best, arm) > 0.8527881905614896`（严格大于）**且** 构造审计通过（`extra_trainable_params == 0` 且参数名集合与默认臂一致）**且** A1/A2/A4/A5 全 PASS。四条同时满足 → **TRANSFER_SUPPORTED**；否则 → **TRANSFER_NOT_SUPPORTED** |
| 阈值出处（不新设阈值） | `0.8521` 逐字沿用 `exp/stage2-null-expert` 预注册文档 §5 主判据（`docs/superpowers/specs/2026-09-29-stage2-null-expert-design.md` @ `1eadd05`；同一基线 run、同一 `stage1_id`）。本分支不修改、不重解释该阈值 |
| **descriptive 记录（不参与分类）** | `Δtest(arm)` / `Δval(arm)`（vs 基线）；`r = Δtest(arm) / Δtest(nullx) = Δtest(arm) / 0.001296296526501206`；方向符号；与 AliCCP 参照值的对照（`Δtest_ref = +0.012706144575588052`、`r_ref = 0.917640706647481`） |
| 机械判定 | `metrics.transfer_arm_verdict(test_auc, val_best, construction_ok, a_gates_ok)`，完整落盘 `metrics.json["transfer_arm"]` 与 `gate_report.json["transfer_arm"]` |

**结果解读的预注册口径**：

- SUPPORTED ⇒ 证据支持"AliCCP 事后发现的固定常数衰减作为一个**未调参的干预**在 CensusIncome 上可迁移（正向，达到预先约定的效用阈值）"。注意：这不等于"该常数解释了 Census null 臂的增益"——Census null 路由已激活（§1.4），两者是不同的命题。
- NOT_SUPPORTED ⇒ 该固定常数在 Census 上未达到预先约定的效用阈值（无论方向正负，均如实记录 Δ 与 r）——即"跨数据集、零适配的迁移在本设置下不成立"；不得据此再做任何系数调整或补跑。

**止损规则（与协议 §6.3、前臂 §5 一致）**：

1. 未达主判据 → 如实记录该次结果（SUMMARY 追加行 + `runs/` 留档 + §10 补记），**不得**重跑挑好的、不得换阈值、不得跳到多 seed 或全量。
2. **不得调 c**（§1.4）；不得加跑第二个系数、第二个 seed、第二个 epoch 预算；不得用 Census `null_mean` 派生系数。
3. 阈值只允许在**看到任何本臂 run 结果之前**修改；每次修改单独 commit 并在本文件记录理由；已判定的 run 不得用新阈值重判。
4. 运行期间发现实现缺陷 → 允许修复后重跑，但必须同时保留失败 run 的产物与说明。

---

## 6. 诊断（C 类：只记录；test 不参与任何机制判定）

### 6.1 in-run 探针（处理臂专属；val 一次遍历，no_grad）

| 键 | 含义 | 口径 |
|---|---|---|
| `pred_mean` / `pred_std` / `pred_var` / `pred_q10..q90` / `pred_min` / `pred_max` | 头预测的离散度（val 全量） | float64 终算、`std/var` 用 `unbiased=False` |
| `head_gate_mean` | `NewTask.gate_network` 输出（specific / general 两维）在 val 的均值 | float64 逐批累加 |
| `source_gate_mean` | source router `W`（K=2 源任务权重）在 val 的均值 | float64 逐批累加；与 AliCCP `SourceGateStats` 同口径（不含候选/键扩展） |
| `spec_mix_norm_pre` / `spec_mix_norm_post` | `new_spec_rep` 衰减前 / 后的平均 L2 范数 | 后 = 前 × c（浮点）；比值即衰减对 specific 混合的范数效应 |
| `env_aware_norm` / `fused_norm` / `gen_norm` | 融合路径各环节平均 L2 范数 | float64 逐批累加 |

### 6.2 三方对照（`compare` 子命令；run 后恰好一次；纯分析、无训练、不改任何 run 产物）

- 载入**同一** Stage-1 产物（真冻结、no_grad）；三个头 `newtask.pt` 各自 weight-only strict 载入：基线 run / null run / 本臂 run。
- **null 头类重构**：从 git blob `1eadd05:multitaskrec/model.py`（sha256 钉死 `836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c`）动态载入（与测试同一机制；不改动任何分支文件；前臂代码不移植）。
- 同一 **test / val** 全序（`Subset` 保序、batch 256、`shuffle=False`）逐样本预测。
- **完整性检查（先于解读；任何一项失败 → 对照判 INTEGRITY_FAIL，只作废诊断解读，不影响 run 内主判据）**：
  - 重算 test AUC（`metrics.evaluate_newtask` 原口径）必须 = 三 run 各自记录值（|Δ| ≤ 1e-9；预期逐位相等）；
  - 三臂 y_true 逐位一致（sha256 记录）；split 指纹校验（A2 口径）通过；backbone sha256 与 Stage-1 产物 meta 一致；
  - null 头重构 sha256 与 §6.2 钉死值一致；
  - 三臂头参数计数记录（基线 27064 / null 27192 / 本臂 27064；本臂与基线必须相等）。
- **输出量**（test 同序；逐臂）：pred 离散度（std/var/分位数）；两两 Pearson(logit) 与 Spearman(pred) 相关（control-baseline / control-null / null-baseline）；paired delta（mean/std/分位数/frac_pos（control−baseline、control−null、null−baseline））；val 离散度表；gate 表（head gate / source gate 三臂）；表征范数表（spec mixture / env_aware / fused / gen 三臂）。
- **方向性预期（descriptive，不参与判定）**：Census null 路由已激活（top1 = 0.3726），因此**预期 `corr(control, baseline) > corr(control, null)`**（与 AliCCP 的排序相反，AliCCP 为 `corr(control,null)=0.9994 > corr(control,baseline)=0.8154`）。两种情况都如实报告，不作事后反转解释。
- 落盘：`artifacts/census_stage2/runs/<control-run>/three_way_compare.json`（+ 日志）。

---

## 7. 门禁与继承披露（B3）

- 本臂执行协议全部 A 类与 B 类判定，`transfer_arm` 是**臂级**判定，不并入 `overall_pass`（A/B 门禁语义与 `protocol.py` 不变）。
- **继承的失败（非本臂引入，按构造与基线完全相同）**：基线 run 已记录 `B3 FAIL`（`gate_mean = [0.952401451979339, 0.8543901456350915]`，income 维 > 0.95 坍缩界）。该量由 `NewTask` **之外**的冻结 backbone `gate_networks` 在 val 上算出，三臂引用同一 `stage1_id` ⇒ 判定逐位相同。报告中如实披露为**继承基线姿态**，不归因于本臂、不据此否定本臂、也不据此宣称任何东西（"preserve baseline inherited B3 interpretation"）。
- B1 / B2 / B4 为臂相关门禁，照常判定与披露（预期 PASS；B2 若因臂行为变化而不通过则如实记录）。
- A 类门禁（A1/A2/A4/A5）必须全 PASS，否则主判据自动不成立（§5 已含）。

---

## 8. 报告与分支处置

- `SUMMARY.md` 只追加、不重写：处理臂即使失败也必须有行（协议 §9.3）。
- 结果（含分类判定、三方对照、门禁、诊断数值、完整性检查、止损结论、与 AliCCP 方向/比例对照）在运行后**追加**到本文件 §10；§1–§9 预注册内容不得回填、不得修改。
- 分支处置：无论成功失败，**只保留 `exp/census-stage2-attenuation-transfer` 本分支，不合并回 `master`、不触碰 AliCCP 分支**；推送需用户显式批准（本轮只本地 commit）。
- 与本实验无关的既有探索（null expert 代码本身、CGR、affinity gate、residual prompt gate、T4、LoRA adapter 等）一律不引用、不重建。

---

## 9. 局限（预声明）

1. **单 seed、单 run**：主判据与 Δ/r 均为点估计，无重复噪声估计；判定按预注册阈值执行，不引入事后置信区间叙事。
2. **外部验证的边界**：本实验只检验"该固定常数在 Census 短跑配置下是否达到预先约定的效用阈值"；不检验、不宣称该常数在其它预算/种子上的行为。
3. **常数近似**（同 AliCCP）：c 为 AliCCP val 聚合常数，非逐样本 `1 − W_null(x)`；Census 前臂 null 权重分布更宽（top1 = 0.37），常数近似在 Census 上先验更粗。
4. **来源与目标的机制差异**：AliCCP 来源臂 `null_top1_rate = 0`（纯阻尼）；Census null 臂路由已激活且其增益与 AliCCP 量级不同（+0.0013 vs +0.0138）。本实验回答的是"常数作为干预是否迁移"，**不是**"常数是否重现 Census null 增益"。
5. **对照仅一条路径**：NOT_SUPPORTED 时只说明"该常数在此设置下不达阈值"，不构成对其它解释的确证；SUPPORTED 时归因上限为"常数阻尼（含其诱发的 gate 重平衡）作为一个干预是正向的"，不区分内部贡献。

---

## 10. 结果（实测；运行后追加，不得回填预期值）

（运行后追加）
