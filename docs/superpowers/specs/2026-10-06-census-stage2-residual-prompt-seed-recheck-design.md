# CensusIncome 阶段 2 范数受控残差 Prompt —— 第二 canonical seed 独立配对复检（预注册）

- **状态**：**预注册已冻结**（第 3–6 节全部种子、命令、判据、分类与决策规则在看到任何本次运行结果之前写定；第 11 节结果待回填）。
- **日期**：2026-10-06
- **适用分支**：`exp/census-stage2-residual-prompt-seed-recheck`（自 `infra/fair-stage2-benchmark` @ `87afe03` **独立拉出**，未携带任何其它实验分支的代码或结果）。
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。本文件**只引用**它，不修改它；划分、三个种子、真冻结三件套、A/B 门禁、SUMMARY 台账一律沿用。
- **被测候选（只读引用，不在本分支修改）**：`exp/stage2-residual-prompt-gate` @ `e0ad6af`（预注册 `e87c945`、实现 `99b9510`、结果 `VALID_NEGATIVE`）。该分支的**结果行与结论不构成本次证据**；本次只迁移其机制代码并在**新种子**上重新独立运行。
- **定位**：独立复检（independent re-check），不是新颖性主张、不是新方法、不改机制、不调参。
- **数据集**：只跑 CensusIncome（评测协议 2.3）。
- **提交时序（预注册纪律的一部分）**：本文件先于实现与运行**单独提交并推送**→ 迁移 + 等价性守卫测试提交并推送 → 配对判定工具提交并推送 → 运行（实现冻结于上述提交）→ 结果以**新增小节**回填并推送。第 3–6 节任何数字不得在看到结果后改动。

---

## 1. 审计结论（全部为只读证据，本次会话内核实）

### 1.1 协议与候选

| 项 | 事实 | 出处（只读） |
|---|---|---|
| 公平协议 | Census 阶段 2 唯一事实来源；三个独立种子；内容寻址只读 Stage-1 产物；真冻结三件套；A/B 门禁 | `docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md` @ `87afe03` |
| 候选机制 | 上下文生成的、有界的（逐维 `tanh`）、范数受控的残差 prompt + 零初始化标量门控 `α`，注入 `gen_rep` 与两路 `spec_reps`（进入融合前） | `exp/stage2-residual-prompt-gate` @ `99b9510`：`census_benchmark/residual_prompt.py` |
| 候选原种子结果 | seed `1685480945`：基线 `0.8500685307175756`，处理臂 `0.8477376871797389`，Δ = **−0.0023308435**；G1–G5 全过、E1 未达标 ⇒ `VALID_NEGATIVE` | 同分支 `e0ad6af`（含 `artifacts/census_stage2/runs/20261001-0123-…-99b9510-rpg/`，主工作树内保存） |

原种子两臂的完整读数（本次 headroom 评估的参照值，只读）：

| 量 | 基线臂（run `20260929-1735-…-904f8d0`，commit `904f8d0`） | 处理臂（run `20261001-0123-…-99b9510-rpg`，commit `99b9510`） | 配对差 |
|---|---|---|---|
| AUC-Test-Education | `0.8500685307175756` | `0.8477376871797389` | **−0.0023308435** |
| best val AUC（Education） | `0.8527881905614896` | `0.8501679359179222` | **−0.0026202546**（验证方向也为负） |
| best epoch / 运行 epoch 数 | 5 / 5 | 5 / 5 | 两臂均**预算受限**（val 曲线仍在上升，无 early stop） |
| Stage-1 产物 | 两臂同一：`s1-096f8f16-m1685480945-e2-cb2094b3` | 同左 | —— |

### 1.2 canonical seed 列表与"下一个未测种子"的确定性选择

- **canonical 列表**：`[1685480945, 1685463909, 1685477428, 1685459668, 1685496394]`。
  出处（tracked 文件，15 处一致）：`CensusIncome_MPTRec.py:87`（注释中的训练循环）、`CensusIncome_NewTask.py:152`、`baseline/{csrec,mmoe,ple,sharedbottom,singletask,sparsesharing,stem}/*.py`、`mask/CensusIncome/CensusIncome_train_single.py`。
- **本候选（范数受控残差 prompt）已测种子**：**仅** `1685480945`（全仓库 `rpg` run 仅有 `20261001-0123-…-99b9510-rpg` 一条，已核实所有 worktree 的 `artifacts/census_stage2/runs/`）。
- **选择规则（先于结果、确定性）**：按 canonical 列表顺序取"本候选尚未测过的第一个种子" ⇒ **`1685463909`（列表第 2 位）**。不重排、不跳号、不看结果。
- 旁证（不作为选择依据，仅供交叉参考）：`1685463909` 是仓库多处训练脚本的第二 canonical seed（第一为 `1685480945`）。

### 1.3 兄弟线只读参照（**不同代码**，仅近似参考，不构成配对）

null-expert 线在 seed `1685463909` 有已完成的 baseline run（其分支修改过 `multitaskrec/model.py` / `census_benchmark/metrics.py` 并声称默认路径逐位等价——本分支不复用其代码，仅读其数字作为环境级参照）：

| 参照（只读，非配对） | 值 |
|---|---|
| run `20261005-0231-s20260929-m1685463909-short-9c6cc38`（baseline 臂） | test `0.845256873871603`，best val `0.8476002975447581`，best epoch 5/5 |
| 其 Stage-1 产物 | `s1-096f8f16-m1685463909-e2-8b53a3bf`（config_hash `8b53a3bf…`，与本分支代码预测值一致） |
| 其 B3 gate_mean | `[0.9291278140357668, 0.8962754976794589]` ⇒ **该种子下 B3 预期 PASS**（原种子 `1685480945` 的 B3 FAIL 是种子/产物特性） |

### 1.4 环境（本会话实测）

| 项 | 值 |
|---|---|
| 解释器 | `D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe`（Python 3.10.11；worktree 无独立 venv——**环境偏差，记录在案**） |
| 依赖 | torch `2.6.0+cu124`、numpy `2.2.6`、scikit-learn `1.7.2` |
| 设备 | `cuda:0`，NVIDIA GeForce RTX 3060 Laptop（6 GB） |
| 数据 | **只读 junction**：本 worktree `dataset/` → `D:\MPT-Rec-three_task\MPT-Rec\dataset`（`cmd mklink /J`；`dataset/` 被 `.gitignore` 拦截，git 状态不受影响）。实测加载：train.gz `199523` 行、test.gz `99762` 行，与兄弟产物 `meta.json` 的 `n_train/n_val/n_test` 一致。**不下载、不重建** |
| 依赖对象在场 | 先分支 `87afe03`（本分支祖先）、`e0ad6af/99b9510/e87c945`（origin 已推送，本地在场） |

---

## 2. 本次实验的问题（可证伪）

在**同一新种子 Stage-1 产物、同一划分、同一预算**下，配对运行 基线臂（论文 `NewTask`）与处理臂（范数受控残差 prompt，代码与 `99b9510` 逐字节一致）：

**原种子上观测到的 Δ = −0.00233（有效阴性）是否在新种子重现？** 本次以"配对差 `delta_test`"为主报告量，并按第 4 节双重判定（历史判据 + 新增分类）。

---

## 3. 冻结项（种子 / 预算 / 命令 / 身份）

### 3.1 三个独立种子与预算（不得改动）

| 项 | 值 |
|---|---|
| `--split-seed` | `20260929`（协议常量；跨 model seed 恒定，A2 要求指纹一致） |
| `--env-seed` | `20260929`（协议常量） |
| `--model-seed` | **`1685463909`**（§1.2 确定性选择） |
| `tag` | `short`（协议 smoke 配置） |
| Stage-1 | `epochs=2`、`patience=2` |
| Stage-2 | `epochs=5`、`patience=2` |
| 其余超参 | 论文值冻结：`batch_size=256`、`lr=1e-3`、`reg_embedding=0.006`、`reg_dnn=3e-5`、`uni_coe=0.9`、`env_coe=0.1`、`expert=(256,128)`、`tower=(64,32)`、`input_size=123`、`embedding_size=4` |
| 设备 | `cuda:0` |

### 3.2 Stage-1 产物：**只产一次**，两臂共用

```powershell
# cwd = 本 worktree 根
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_census_benchmark.py stage1 --model-seed 1685463909
```

**预期值与交叉核对（运行前已写定）**：

| 项 | 预期 | 处置 |
|---|---|---|
| `stage1_id` | `s1-096f8f16-m1685463909-e2-8b53a3bf`（config_hash `8b53a3bf6c0216577c849e6a31a76b1819eba9f16ed5dac3014f485960d10d5f`，本会话用协议常量本地预计算，与兄弟产物逐位一致） | **不等 ⇒ 立即停止**（配置漂移，协议完整性问题，不产出任何臂结果） |
| `split_fingerprint_sha256` | `096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c` | 不等 ⇒ 停止（A2 前置） |
| `n_train / n_val / n_test` | `199523 / 49881 / 49881` | 不等 ⇒ 停止 |
| `backbone_sha256` vs 兄弟产物 `da0458b6c2b781145aa6aaab552bdac29d73911aa90a1acfc28cd777dc39a4c2` | **诊断（非阻塞）**：相同 ⇒ 记录"跨分支复现"强证据；不同 ⇒ 记录环境/分支差异说明。**两臂共用本产物，配对有效性不受其影响** |

Stage-1 崩溃处置见 §7。

### 3.3 配对 Stage-2 两臂：**各恰好一次**

顺序固定为 **基线臂先行 → 处理臂**（处理臂的参照 `newtask.pt` 指向基线臂，与原种子实验同构）。

```powershell
# 臂 B（baseline，默认 variant）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_census_benchmark.py stage2 `
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685463909-e2-8b53a3bf --tag short --gpu 0

# 臂 T（residual-prompt；<baseline_run_id> 为臂 B 实际 run_id）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_census_benchmark.py stage2 `
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685463909-e2-8b53a3bf --tag short --gpu 0 `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/census_stage2/runs/<baseline_run_id>/newtask.pt
```

- 两臂**同一 HEAD**（= 迁移 + 工具提交，代码身份一致；SUMMARY 两行在结果提交时一并入库 ⇒ 两条 run_id 的 `commit` 字段相同）。
- 两臂之间**唯一差异** = `--variant`（及处理臂只读参照）；`--stage1-dir`、tag、全部超参、数据、划分、early-stop 规则完全相同。
- 处理臂 `run_id` 带 `-rpg` 后缀（沿用 `RUN_ID_SUFFIX`）。
- **不跑 A3 复跑**（本次"每臂恰好一次"优先；A3 属历史上仅 CONFIRMED 触发）。

### 3.4 迁移与等价性守卫（提交于运行之前）

| 文件 | 要求 | 钉子（git blob SHA-1，`git hash-object` 口径） |
|---|---|---|
| `census_benchmark/residual_prompt.py` | 与 `99b9510` **逐字节一致** | `107221b26382da7fd44990e167d9a61da2680d92` |
| `census_benchmark/tests/test_residual_prompt.py` | 与 `99b9510` 逐字节一致 | `847097b7cb48a56c8593dd76d63a71a21a4c9e4f` |
| `run_census_benchmark.py` | 与 `99b9510` 逐字节一致（只含最小接线） | `e15cad7565d4487f6a61055f195c662f64eb4f15` |

- 新增测试 `census_benchmark/tests/test_migration_equivalence.py` 固化上表三钉子；`99b9510` 对象在场时另行交叉核对 `git rev-parse 99b9510:<path>`（对象缺失则该项显式 skip 并说明，主钉子不降级）。
- 沿用既有静态守卫：`multitaskrec/model.py`、`config.py`、`census_benchmark/protocol.py`、`census_benchmark/metrics.py` 相对 `87afe03` 零改动（该守卫已存在于迁移测试内）。
- **新增模块** `census_benchmark/paired_verdict.py`（+ 测试 `census_benchmark/tests/test_paired_verdict.py`）：只实现第 4.2/4.3 节的新分类与配对差值计算；**不属于迁移、不修改任何迁移文件或协议文件**。TDD：先写测试（红）后实现（绿）。
- 迁移 TDD：先落测试（`git hash-object` 红、`import residual_prompt` 红）→ 再落实现与接线（绿）。
- 实验前必须跑**全量** `census_benchmark/tests` 套件：既有 23 项 + 迁移 27 项 = **50**，加新增等价性/工具测试项，**全绿才允许运行**。

---

## 4. 判据（看到结果之前写定）

### 4.1 历史判据（**immutable**，与原预注册 `e87c945` §3 逐字一致；本次**只读报告**，不覆盖）

| 判据 | 规则 |
|---|---|
| **E1** | `AUC-Test-Education >= 0.8521`（绝对阈值；历史基线常量 `0.8500685307175756` 仅作参照显示） |
| **G1** 构造恒等 | 共享参数与基线 `NewTask` 逐位相同；构造后全局 RNG 端点逐位一致；新增键恰为预注册 5 键 |
| **G2** 初值恒等 | `alpha_at_construction == 0.0`；真实首个训练 batch 上处理头与基线头前向 `torch.equal` |
| **G3** 门控活性 | 首 batch 反传 α 梯度 ≠ 0；末 epoch 首 batch α ≠ 0 且生成器梯度 > 0；`best_state` 载入后 α_final ≠ 0 |
| **G4** 范数受控界 | val 全样本三路 `ratio ≤ |α_final| + 1e-6` |
| **G5** 活性带 | val 逐路平均 `ratio ∈ [0.005, 0.5]`（含端点） |
| 结局分类 | 六类：`INVALID_IMPLEMENTATION`（G1/G2/G4 任一 FAIL）/ `MECHANISM_INACTIVE`（G3 FAIL）/ `MECHANISM_SILENT`（G5 低于下界）/ `MECHANISM_OVER_PERTURB`（G5 高于上界）/ `CONFIRMED`（全过且 E1 PASS）/ `VALID_NEGATIVE`（全过且 E1 FAIL） |

历史判据由迁移代码 `RP.arm_verdict` 原样计算并落盘于 `metrics.json:rp_arm` 与 `gate_report.json:residual_prompt`；结果文档**并列**报告"历史判据结论"与"新增分类"，互不覆盖。

### 4.2 新增分类（additive；本次任务指定；先于结果写定）

设 `delta_test = AUC_T − AUC_B`（本次配对两臂实测，同一 Stage-1 / 划分 / 预算）：

| 条件 | 分类 |
|---|---|
| `delta_test >= +0.001` | **POSITIVE_IMPROVEMENT** |
| `−0.02 < delta_test < +0.001` | **NO_CLEAR_IMPROVEMENT** |
| `delta_test <= −0.02` | **CLEAR_DEGRADATION** |

同时必录 `delta_val_best = best_val_T − best_val_B` 及其方向。

### 4.3 决策规则（先于结果写定）

- **EXPAND**（注册**新独立分支动作**：扩展到 3 个种子；**本分支不执行**）当且仅当：新增分类 = `POSITIVE_IMPROVEMENT` **且** 无实质矛盾。
- **实质矛盾**（任一即算）：
  - (a) 处理臂历史结局 ∈ {`INVALID_IMPLEMENTATION`, `MECHANISM_INACTIVE`, `MECHANISM_SILENT`, `MECHANISM_OVER_PERTURB`}（即 G1–G5 未全过）；
  - (b) `delta_val_best <= −0.001`（验证集以同级 materiality 反向，与 test 正号矛盾）；
  - (c) 任一臂 A1/A2/A4/A5 或 B1/B2/B4 FAIL（B3 例外：该量只读冻结 backbone、两臂应逐位相同；若两臂 B3 不一致则同样记 (c)）。
- 否则 **STOP**（本候选到此为止），并按 §4.4 记录 headroom 是否可信。**不因结果好坏做任何补救运行。**

### 4.4 NO_CLEAR / STOP 时的 headroom 评估维度（维度与参照值先冻结，结论在结果后按维度陈述）

| 维度 | 参照（原种子，只读） | 本次记录量 |
|---|---|---|
| (a) 机制有效性 | G1–G5 全过；α 0 → 0.314；g_eff 0.066–0.284；ratio_mean 0.1277；cos(δ,h) −0.37/−0.07/−0.22 | G1–G5 实测、α/g_eff/ratio/cos 逐项 |
| (b) 验证方向 | val 配对差 −0.002620（同向为负） | `delta_val_best` 符号与量级 |
| (c) 跨种子一致性 | 原种子 `delta_test = −0.0023308435` | 本次 `delta_test` 的符号与量级（并列比较，不平均、不合并判定） |
| (d) 预算/最优 epoch 趋势 | 两臂 best_epoch 均 = 5/5（预算受限） | 本次两臂 best_epoch / epoch 数；若仍 5/5 ⇒ 预算受限延续 |

### 4.5 阈值纪律

本文件先于运行单独提交并推送即凭据。判据、分类、决策规则、种子、预算、命令在看到本次结果后**一律不得修改**；工具代码中的分类常量（`+0.001` / `−0.02`）由测试钉死。

---

## 5. 必录清单（结果回填时逐项落盘/入文）

1. 两臂 `run_id`、`commit`、`variant`、完整 `config.json`；
2. 两臂 **AUC-Test-Education**、**best val AUC**、逐 epoch val AUC、`best_epoch`、运行 epoch 数；
3. **配对差** `delta_test`、`delta_val_best`；新增分类 + 决策（EXPAND/STOP）；
4. Stage-1 身份：`stage1_id`、`config_hash`、`split_fingerprint_sha256`、`env_ids_sha256`、`backbone_sha256`（与兄弟产物对照）；
5. 机制 G1–G5 逐项实测 + `arm_verdict` 历史结局；
6. prompt 门控/比值分布：`alpha_by_epoch`、`alpha_final`、`g_eff mean/std/min/max`、逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean`、`cos_mean`；
7. 预测离散度：处理头 val `pred_mean/std/min/max/q05/q50/q95` + 参照头（基线臂 `newtask.pt` 只读复算）同口径值与复算 val AUC 对账；
8. 冻结/划分证据：两臂 A1/A2/A4/A5/B1–B4 逐项 + `backbone_sha256_before/after`；
9. 失败/无效尝试：任何崩溃、无效 run、丢弃产物及其原因与处置（无则明确写"无"）；
10. 与原种子对照表（§1.1 值 vs 本次值）；
11. 参数预算：`new_params_total=4161`、`head_params=27063`、占比（工具核算，与预注册一致）；
12. `SUMMARY.md` 只追加两行（runner 自动）。

---

## 6. 运行规程（顺序固定，各步一次）

1. **环境准备**：junction 已建立并实测（§1.4）；不做任何依赖安装/数据下载。
2. **预注册提交并推送**（本文件）。
3. **迁移 + 等价性守卫 + 配对判定工具**：TDD 红→绿；提交并推送。
4. **全量测试**：`census_benchmark/tests` 全绿（§3.4）。
5. **Stage-1**（§3.2）→ 交叉核对 → 通过才继续。
6. **臂 B**（§3.3 命令 1）→ 记录；**臂 T**（§3.3 命令 2）→ 记录。
7. **配对判定**：`python -m census_benchmark.paired_verdict --baseline-run <臂B run 目录> --treatment-run <臂T run 目录>`（工具只读两 run 目录，输出分类/差值/决策）。
8. **结果回填**：本文件第 11 节 + `SUMMARY.md`（runner 已追加）；提交并推送。
9. **终检**：`git status` 干净、`origin/exp/census-stage2-residual-prompt-seed-recheck` tip = 本地 tip。

---

## 7. 失败处置与止损（先于结果写定）

1. **崩溃即记录**：任何失败尝试（含中间产物）按 §5.9 留痕；只有在**能证实**为基础设施原因（如 junction 失效、CUDA 初始化失败）时才允许**一次**重试，两次尝试全部留痕、判据不变；否则停止，不产出解释。
2. **禁止结果驱动的一切动作**：不换种子、不换 Stage-1 产物、不改预算/patience/阈值/命令、不因结果近阈值而复跑、不做 A3 复跑。
3. **不跑第三种子**（本次任务边界；EXPAND 时只**注册**新分支动作）；不跑 `full`；不跑 AliCCP/ByteRec/第三数据集。
4. 无论结果如何，`SUMMARY.md` 只追加；历史结局与新增分类并列报告，不得只报告通过的部分。

---

## 8. 与评测协议 / 候选分支的接口（改动清单）

| 文件 | 处置 |
|---|---|
| `multitaskrec/model.py`、`config.py`、`census_benchmark/protocol.py`、`census_benchmark/metrics.py` | **零改动**（静态守卫，逐字节 = `87afe03`） |
| `census_benchmark/residual_prompt.py`、`census_benchmark/tests/test_residual_prompt.py`、`run_census_benchmark.py` | 从 `99b9510` **逐字节迁移**（§3.4 钉子） |
| `census_benchmark/tests/test_migration_equivalence.py` | 新增（等价性守卫） |
| `census_benchmark/paired_verdict.py` + `census_benchmark/tests/test_paired_verdict.py` | 新增（本次分类与配对差值；additive） |
| `docs/superpowers/specs/2026-10-06-…-seed-recheck-design.md` | 本文件（预注册先行；结果只回填第 11 节） |
| `artifacts/census_stage2/{stage1,runs,splits}` | 只读/只追加；`SUMMARY.md` 追加两行 |

---

## 9. 明确的非目标

- 不改机制、不做超参搜索、不做 rank/宽度/α 初值/注入流的任何变体；
- 不主张方法新颖性、不写"我们提出"、不产出论文级性能结论；
- 不比较/引用先分支的**结果行**作为本次证据（只引用其机制代码与设计公式）；
- 不修改评测协议文件、不修改 `master`、不合并任何分支。

---

## 10. 已声明的风险与偏差

1. **环境偏差**：worktree 无独立 venv（使用主树 venv，§1.4）；数据经 junction 只读引用（`.gitignore` 已拦截，实测 git 状态干净）。
2. **fresh Stage-1 vs 兄弟产物**：兄弟产物由**不同代码**（null 线改过 `model.py`/`metrics.py`）产出；本次 fresh 产物的 `backbone_sha256` 若与兄弟不同，属已预见情形（§3.2 表），配对不受影响，只记诊断。
3. **B3 种子依赖**：`1685480945` 上 B3 FAIL（继承）、`1685463909` 兄弟读数预期 PASS；两臂 B3 必须逐位相同，否则记实质矛盾 (c)。
4. **预算受限**：原种子两臂 best_epoch 均 = 5/5；本种子若同样触顶，"预算受限"是 headroom 评估 (d) 的正当维度（结论只陈述、不借此改预算）。
5. **两个 canonical seed 的间距**：`1685480945` 与 `1685463909` 差约 1.7 万，种子空间内非相邻整数但都是仓库既有 canonical 值（列表序），选择规则不依赖其数值关系。

---

## 11. 结果（待运行后回填）

（本节在运行完成后以新增小节形式回填：运行标识与配置、两臂实测、配对差、新增分类与决策、G1–G5、逐源读数、预测离散度、冻结/协议证据、与原种子对照、headroom 评估、止损执行情况。第 1–10 节一字不改。）
