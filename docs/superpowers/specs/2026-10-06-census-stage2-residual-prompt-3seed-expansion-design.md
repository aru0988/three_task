# CensusIncome 阶段 2 范数受控残差 Prompt —— 第三 canonical seed 扩展与 3-seed 符号稳定性仲裁（预注册）

- **状态**：**预注册**（本文件先于任何实现与出数运行单独提交并推送）。第 3–6 节的任何数字不得在看到本次结果后修改；结果只以第 11 节新增小节回填。
- **日期**：2026-10-06
- **适用分支**：`exp/census-stage2-residual-prompt-3seed-expansion`（自 `infra/fair-stage2-benchmark` @ `87afe03` **独立拉出**，未携带任何其它实验分支的代码或结果）。
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。本文件**只引用**它，不修改它；划分、三个种子、真冻结三件套、A/B 门禁、SUMMARY 台账一律沿用。
- **被测候选（只读引用，不在本分支修改机制）**：范数受控残差 prompt（上下文生成、逐维 `tanh` 有界、按流范数归一、零初始化标量门控 `α`）。迁移来源 = `exp/census-stage2-residual-prompt-seed-recheck` @ `41feb29`（该分支自 `exp/stage2-residual-prompt-gate` @ `99b9510` 逐字节迁移实现）。两个先分支的**结果行与结论不构成本次证据**；本次只迁移其机制代码并按第 3.4 节钉死等价性，在**新种子**上重新独立运行。
- **定位**：独立第三种子扩展 + 3-seed 符号稳定性仲裁（arbitration）。不是新颖性主张、不改机制、不调参。
- **数据集**：只跑 CensusIncome（评测协议 2.3）。
- **提交时序（预注册纪律的一部分）**：本文件先于实现与运行**单独提交并推送** → 迁移 + 等价性守卫 + 3-seed 聚合工具提交并推送（TDD 红→绿）→ 运行（实现冻结于上述提交）→ 结果以**新增小节**回填并推送。

---

## 1. 审计结论（全部为只读证据，本次会话内核实）

### 1.1 协议与候选

| 项 | 事实 | 出处（只读） |
|---|---|---|
| 公平协议 | Census 阶段 2 唯一事实来源；三个独立种子（`--split-seed` / `--model-seed` / `--env-seed`）；内容寻址只读 Stage-1 产物；真冻结三件套；A/B 门禁；SUMMARY 只追加 | `docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md` @ `87afe03` |
| 候选机制 | 残差 `δ_s(x) = α·(‖h_s‖/√d)·tanh(MLP(x))`，注入 `gen_rep` 与两路 `spec_reps`（进入融合前）；`α` 零初始化、`ratio = ‖δ‖/‖h‖ = g_eff ≤ |α|` | `41feb29:census_benchmark/residual_prompt.py`（blob `107221b2…`，与 `99b9510` 逐字节相同） |
| 实现祖先 | 预注册 `e87c945`、实现 `99b9510`、结果 `VALID_NEGATIVE` | `exp/stage2-residual-prompt-gate` @ `e0ad6af` |

### 1.2 两个既有种子的只读证据（本会话自磁盘 run 产物与 tracked 文档复核）

| 量 | seed `1685480945`（原分支） | seed `1685463909`（recheck 分支） |
|---|---|---|
| baseline run_id | `20260929-1735-s20260929-m1685480945-short-904f8d0` | `20261006-0403-s20260929-m1685463909-short-f454205` |
| treatment run_id | `20261001-0123-s20260929-m1685480945-short-99b9510-rpg` | `20261006-0405-s20260929-m1685463909-short-f454205-rpg` |
| baseline test / best val | `0.8500685307175756` / `0.8527881905614896` | `0.845256873871603` / `0.8476002975447581` |
| treatment test / best val | `0.8477376871797389` / `0.8501679359179222` | `0.8482477806320898` / `0.8499928855458697` |
| `delta_test`（T − B） | **−0.0023308435** | **+0.0029909067604868556** |
| `delta_val_best` | −0.0026202546 | +0.0023925880011116396 |
| best_epoch / epoch 数 | 两臂均 5 / 5（**预算受限**） | 两臂均 5 / 5（**预算受限**） |
| Stage-1 产物 | `s1-096f8f16-m1685480945-e2-cb2094b3` | `s1-096f8f16-m1685463909-e2-8b53a3bf` |
| G1–G5 / E1 / 历史结局 | 全过 / 未达标 / `VALID_NEGATIVE` | 全过 / 未达标 / `VALID_NEGATIVE` |
| `α_final` / `ratio_mean` | 0.314076 / 0.12771816 | 0.34239962697029114 / 0.15696257 |
| B3 | FAIL（继承自冻结 backbone，非臂引入） | PASS |

**先证据文件钉死（sha256 of file bytes，回填时重算对账）**：

| 文件 | sha256 |
|---|---|
| s1-baseline `metrics.json` | `8c1f6d706c3a4f960b452280494008f3be8e6d5f921a14c759a66b53d0933387` |
| s1-baseline `gate_report.json` | `417f9216500c122c1ad770dc0993d24691917f0befd2fce53395490f2041872f` |
| s1-treatment `metrics.json` | `84bc4e1ba18f6b10aa5a44fcf7d57759eb202495b119420752709d7e1d675aaa` |
| s1-treatment `gate_report.json` | `5a7bd04887d2ff44abdc686a58babe273f6b6dd10267f4e13bdb9077b2757009` |
| s2-baseline `metrics.json` | `76c95eed8f248e4a12bcae4ca092fd2bcb261ea5bf9282595398238e4dc1ba3b` |
| s2-baseline `gate_report.json` | `b215a0c5f7dde2c1ca5c5c476a59d2e95164c3dce6ba0d7220ccaf2e41246d51` |
| s2-treatment `metrics.json` | `a2c5b79d31370cae2e6b2c9dc6c585041e28f9bbd564e70d33905eb644de15ad` |
| s2-treatment `gate_report.json` | `b8f6a2ca4f8e0db096b14cf65561cbd4a8210d9618fcd1e271c2ad172377c8e3` |

### 1.3 canonical seed 列表与第三个种子的确定性选择

- **canonical 列表**：`[1685480945, 1685463909, 1685477428, 1685459668, 1685496394]`。
  出处（tracked 文件，本次核实 15 处一致）：`CensusIncome_MPTRec.py:87`、`CensusIncome_NewTask.py:152`、`baseline/{csrec,mmoe,ple,sharedbottom,singletask,sparsesharing,stem}/*.py`、`mask/CensusIncome/CensusIncome_train_single.py`。
- **本候选（范数受控残差 prompt）已测种子**：`1685480945`（原分支）、`1685463909`（recheck 分支）。
- **选择规则（先于结果、确定性）**：按 canonical 列表顺序取"本候选尚未测过的第一个种子" ⇒ **`1685477428`（列表第 3 位）**。不重排、不跳号、不看结果。该种子与 recheck 分支 §11.9 注册的"下一步种子"一致。

### 1.4 环境与数据（本会话实测）

| 项 | 值 |
|---|---|
| 解释器 | `D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe`（Python 3.10.11；worktree 无独立 venv——**环境偏差，记录在案**） |
| 依赖 | torch `2.6.0+cu124`、numpy `2.2.6`、scikit-learn `1.7.2`、scipy（t 分位点计算） |
| 设备 | `cuda:0`，NVIDIA GeForce RTX 3060 Laptop（6 GB） |
| 数据 | **只读 junction**：本 worktree `dataset/` → `D:\MPT-Rec-three_task\MPT-Rec\dataset`（`New-Item -ItemType Junction`；`dataset/` 被 `.gitignore` 拦截，git 状态不受影响）。实测：`train.gz` 199523 数据行（199524 含表头）、`test.gz` 99762 数据行（99763 含表头）。**不下载、不重建** |
| 基线代码 | 本分支从 `87afe03` 拉出；全量 `census_benchmark/tests` 实测 **23 通过**（迁移前的基线绿） |
| 远端 | `origin` 尚无本分支——首次推送创建 |
| 依赖对象在场 | `87afe03`（祖先）、`e0ad6af / 99b9510 / e87c945`、`41feb29 / f454205 / ab7a624 / ba02b11`（本地对象库在场，迁移钉子可离线交叉核对） |

---

## 2. 本分支的问题（可证伪）

在**第三个 canonical seed `1685477428`** 的 fresh Stage-1 产物、同一划分、同一预算下，配对运行基线臂（论文 `NewTask`）与处理臂（范数受控残差 prompt，代码与 `41feb29` 逐字节一致），每臂恰好一次；再与两个只读既有种子合并成 3-seed 视图。

**主问题：处理臂相对同种子配对基线的效应符号，是否在三个固定种子上稳定？** 已知两种子的 `delta_test` 符号翻转（−0.00233 → +0.00299），本分支即为该翻转的仲裁：第三个种子给出 2/3 与 1/3 之间的裁决。

**次问题（记录，不单独判定）：** 3-seed 均值/样本标准差/符号计数/正提升计数/最差种子/配对 95% CI/机制通过计数/验证方向一致性（第 4.4 节）。

---

## 3. 冻结项（种子 / 预算 / 命令 / 身份 / 迁移）

### 3.1 三个独立种子与预算（不得改动）

| 项 | 值 |
|---|---|
| `--split-seed` | `20260929`（协议常量；跨 model seed 恒定，A2 要求指纹一致） |
| `--env-seed` | `20260929`（协议常量） |
| `--model-seed` | **`1685477428`**（§1.3 确定性选择；不得替换） |
| `tag` | `short`（协议 smoke 配置） |
| Stage-1 | `epochs=2`、`patience=2` |
| Stage-2 | `epochs=5`、`patience=2` |
| 其余超参 | 论文值冻结：`batch_size=256`、`lr=1e-3`、`reg_embedding=0.006`、`reg_dnn=3e-5`、`uni_coe=0.9`、`env_coe=0.1`、`expert=(256,128)`、`tower=(64,32)`、`input_size=123`、`embedding_size=4`；处理臂 `prompt_hidden=16`（机制常量，不搜索） |
| 设备 | `cuda:0` |
| 选点规则 | 两臂均为：逐 epoch 评 val（Education AUC），取**最优 val epoch** 的状态；test 只在训练全部结束后评**一次**；全程不用 test 选点 |

### 3.2 Stage-1 产物：**只产一次**，两臂共用

```powershell
# cwd = 本 worktree 根
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_census_benchmark.py stage1 --model-seed 1685477428
```

**预期值与交叉核对（运行前已写定；本会话用协议常量本地预计算，预计算管线已用 recheck 种子的已知值 `8b53a3bf…` 反向验证逐位命中）**：

| 项 | 预期 | 处置 |
|---|---|---|
| `stage1_id` | `s1-096f8f16-m1685477428-e2-15eabcb4`（config_hash `15eabcb4d63c18c471a6c689309bfbf4454f4a9d4a4c1391ce448ee3918bf40f`） | **不等 ⇒ 立即停止**（配置漂移，协议完整性问题，不产出任何臂结果） |
| `split_fingerprint_sha256` | `096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c` | 不等 ⇒ 停止（A2 前置） |
| `n_train / n_val / n_test` | `199523 / 49881 / 49881` | 不等 ⇒ 停止 |
| 初始 `env_ids` sha256（诊断） | `ec50367f76c2524bc57478d4c183984d50b8c381c66364ddc9a3a4f844eeba91`（env seed 决定，跨种子恒定；产物落盘的是 cluster 后的最终 `env_ids`，逐种子不同，不可预测） | 不等 ⇒ 记录并停查（env 路径被动过） |
| `backbone_sha256` | 不可预测（依赖训练动力学；新种子的独立值，无对照） | 落盘留痕 |

**运行前已建立的划分基准**：本 worktree `artifacts/census_stage2/splits/20260929/` 下的 `split_fingerprint.json` + `split_indices.npz` 复制自主树 smoke 基线（文件 sha256 `37a6ff553780e191a3d81a2691afc099ac23223f48b00a5e9ea34c4adac99411`、fingerprint `096f8f16…`，复制后逐字节复核）。因此运行期 A2 走**校验**路径（基准存在 → 必须一致），而非"首次写入"路径。

Stage-1 崩溃处置见 §7。

### 3.3 配对 Stage-2 两臂：**各恰好一次**

顺序固定为 **基线臂先行 → 处理臂**（处理臂的参照 `newtask.pt` 指向基线臂，与两个先种子实验同构）。

```powershell
# 臂 B（baseline，默认 variant）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_census_benchmark.py stage2 `
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685477428-e2-15eabcb4 --tag short --gpu 0

# 臂 T（residual-prompt；<baseline_run_id> 为臂 B 实际 run_id）
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_census_benchmark.py stage2 `
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685477428-e2-15eabcb4 --tag short --gpu 0 `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/census_stage2/runs/<baseline_run_id>/newtask.pt
```

- 两臂**同一 HEAD**（= 迁移 + 工具提交，代码身份一致）。
- 两臂之间**唯一差异** = `--variant`（及处理臂只读参照）；`--stage1-dir`、tag、全部超参、数据、划分、early-stop 规则完全相同。
- 处理臂 `run_id` 带 `-rpg` 后缀（沿用 `RUN_ID_SUFFIX` 约定）。
- **不跑 A3 复跑**（每臂恰好一次优先；A3 属历史上仅 CONFIRMED 触发）。

### 3.4 迁移与等价性守卫（提交于运行之前）

**迁移源 = `exp/census-stage2-residual-prompt-seed-recheck` @ `41feb29`**（全部逐字节，`git hash-object` 口径）：

| 文件 | 源 blob @ `41feb29` | 说明 |
|---|---|---|
| `census_benchmark/residual_prompt.py` | `107221b26382da7fd44990e167d9a61da2680d92` | 机制 + 运行期审计 + 诊断 + E1/G1–G5 判定（与 `99b9510` 同 blob） |
| `census_benchmark/paired_verdict.py` | `50469811c2e61e91d4508e0ddb3151a2c08dfeef` | 配对判定工具（层 B 分类 + 实质矛盾 + 决策） |
| `run_census_benchmark.py` | `e15cad7565d4487f6a61055f195c662f64eb4f15` | 最小接线（`--variant` / `--prompt-reference-newtask` / `-rpg` 后缀；与 `99b9510` 同 blob） |
| `census_benchmark/tests/test_residual_prompt.py` | `847097b7cb48a56c8593dd76d63a71a21a4c9e4f` | 机制单测（27 项） |
| `census_benchmark/tests/test_paired_verdict.py` | `773edf5c02ebff1b76e428fbc76493eda91db449` | 工具单测（12 项） |
| `census_benchmark/tests/test_migration_equivalence.py` | `2ced4274b7aef1d2d14284b1ba687b287f619490` | 先分支等价性守卫（钉子指向 `99b9510`，构成第二链条） |

**新增守卫（本分支，additive）**：`census_benchmark/tests/test_migration_equivalence_3seed.py`，对上述 6 个迁移文件提供三层证明：
1. **blob 主钉子**：`git hash-object <path>` == 上表 `41feb29` blob（始终运行）；
2. **diff 守卫**：`git diff --quiet 41feb29 -- <paths>` 为空（对象缺失时显式 skip）；
3. **AST 守卫**：工作树文件与 `git show 41feb29:<path>` 各自 `ast.parse` 后 `ast.dump` 相等（对象缺失时显式 skip；证明语义等价、不受行尾/格式影响）。

**新增工具（本分支，additive）**：`census_benchmark/three_seed_aggregate.py` + `census_benchmark/tests/test_three_seed_aggregate.py`——只实现第 4.4/4.5 节的 3-seed 聚合与终局稳定性判定；**不属于迁移**，不修改任何迁移文件或协议文件。TDD：先写测试（红）后实现（绿）。其判据常量（`+0.001` / `−0.02` / `−0.001` / 均值下限 / 两个 t 分位点）由测试钉死；分类函数复用 `paired_verdict.classify_delta`（不复制逻辑）。

**零改动清单（静态守卫，逐字节 = `87afe03`）**：`multitaskrec/model.py`、`config.py`、`census_benchmark/protocol.py`、`census_benchmark/metrics.py`。

**TDD 纪律**：先落新守卫测试（红：迁移文件缺失 / blob 不匹配）→ 再落迁移文件与接线（绿）。实验前必须跑**全量** `census_benchmark/tests` 套件：既有 23 + 迁移 27 + 迁移 12 + 先守卫 2 + 新增守卫 + 3-seed 工具测试，**全绿才允许运行**。

### 3.5 身份与顺序

- 迁移 + 工具提交并推送后，HEAD 冻结；两臂在**同一 HEAD** 上运行（SUMMARY 两行在结果提交时一并入库 ⇒ 两条 run_id 的 `commit` 字段相同）。
- `run_id` 的 `<commit7>` 即上述 HEAD；时间戳部分运行时产生，不预冻结。

---

## 4. 判据（看到结果之前写定）

### 4.1 层 A：历史判据（**immutable**，与预注册 `e87c945` §3 逐字一致；本次**只读报告**，不覆盖）

| 判据 | 规则 |
|---|---|
| **E1** | `AUC-Test-Education >= 0.8521`（绝对阈值；历史基线常量 `0.8500685307175756` 仅作参照显示） |
| **G1** 构造恒等 | 共享参数与基线 `NewTask` 逐位相同；构造后全局 RNG 端点逐位一致；新增键恰为预注册 5 键 |
| **G2** 初值恒等 | `alpha_at_construction == 0.0`；真实首个训练 batch 上处理头与基线头前向 `torch.equal` |
| **G3** 门控活性 | 首 batch 反传 α 梯度 ≠ 0；末 epoch 首 batch α ≠ 0 且生成器梯度 > 0；`best_state` 载入后 α_final ≠ 0 |
| **G4** 范数受控界 | val 全样本三路 `ratio ≤ |α_final| + 1e-6` |
| **G5** 活性带 | val 逐路平均 `ratio ∈ [0.005, 0.5]`（含端点） |
| 结局分类 | 六类：`INVALID_IMPLEMENTATION`（G1/G2/G4 任一 FAIL）/ `MECHANISM_INACTIVE`（G3 FAIL）/ `MECHANISM_SILENT`（G5 低于下界）/ `MECHANISM_OVER_PERTURB`（G5 高于上界）/ `CONFIRMED`（全过且 E1 PASS）/ `VALID_NEGATIVE`（全过且 E1 FAIL） |

层 A 由迁移代码 `RP.arm_verdict` 原样计算并落盘于 `metrics.json:rp_arm` 与 `gate_report.json:residual_prompt`；结果文档**并列**报告"层 A 历史结论"与"层 B 重分类"，互不覆盖。既有两个种子的层 A 结论（只读）见 §1.2。

### 4.2 层 B：用户重分类（每种子；相对**同种子**配对基线；先于结果写定）

设 `delta_test = AUC_T − AUC_B`（同一 Stage-1 / 划分 / 预算的配对两臂实测）：

| 条件 | 分类 |
|---|---|
| `delta_test >= +0.001` | **POSITIVE_IMPROVEMENT** |
| `−0.02 < delta_test < +0.001` | **NO_CLEAR_IMPROVEMENT**（须按 §4.3 做 headroom 评估） |
| `delta_test <= −0.02` | **CLEAR_DEGRADATION** |

同时必录 `delta_val_best = best_val_T − best_val_B` 及其方向。层 B 对三个种子逐一施加（两个既有种子按只读数值重述分类，不重跑）。

### 4.3 NO_CLEAR 的 headroom 评估维度（维度与参照先冻结，结论在结果后按维度陈述）

| 维度 | 参照（两个既有种子，只读） | 本次记录量 |
|---|---|---|
| (a) 机制有效性 | 两种子 G1–G5 全过；α_final 0.314 / 0.342；ratio_mean 0.1277 / 0.1570 | 新种子 G1–G5 实测、α/g_eff/ratio/cos 逐项 |
| (b) 验证方向 | s1: val −0.002620（同向负）；s2: val +0.002393（同向正） | 新种子 `delta_val_best` 符号与量级 |
| (c) 跨种子一致性 | 已测两种子符号翻转，量级 ~2–3e-3；baseline 本身跨种子差 ~4.8e-3 | 三种子 `delta_test` 并列表（不平均、不合并判定） |
| (d) 预算/最优 epoch 趋势 | 两种子两臂 best_epoch 均 = 5/5（预算受限） | 新种子两臂 best_epoch / epoch 数；若仍 5/5 ⇒ 预算受限延续 |

### 4.4 3-seed 聚合（必录；公式先冻结）

对三个固定种子 i ∈ {`1685480945`, `1685463909`, `1685477428`}：

- 逐种子：`delta_test,i`、`delta_val_best,i`、层 B 分类、层 A 结局、G1–G5 逐项、两臂 A/B 门禁逐项。
- 聚合量（先冻结公式）：
  - `mean = (Σ delta_test,i) / 3`；
  - `sd = 样本标准差（ddof=1）`；
  - `sign_positive_count = #{i : delta_test,i > 0}`；
  - `positive_improvement_count = #{i : delta_test,i >= +0.001}`；
  - `worst_seed = argmin_i delta_test,i`（连同其数值）；
  - `mechanism_pass_count = #{i : G1–G5 全过}`；
  - `validation_direction_consistency = #{i : sign(delta_val_best,i) == sign(delta_test,i)}`（含符号表）；
  - **配对 95% CI**：双侧 `mean ± t2·sd/√3`，`t2 = 4.302652729696142`（t 分布 df=2 的 0.975 分位点，scipy 实测值，先冻结）；**单侧 95% 下界** `LB95 = mean − t1·sd/√3`，`t1 = 2.919985580355516`（df=2 的 0.95 分位点）；`√3 = 1.7320508075688772`。

### 4.5 终局稳定性判定（**先于结果写定**）

**`STABLE_UTILITY` 当且仅当 S1–S8 全部成立**：

| 编号 | 条件 |
|---|---|
| S1 | `sign_positive_count = 3/3`（三个种子 `delta_test` 全为正号） |
| S2 | `positive_improvement_count >= 2/3`（至少两个种子 ≥ +0.001） |
| S3 | `min_i delta_test,i > −0.001`（无种子实质反向） |
| S4 | `mean >= +0.001` |
| S5 | 单侧 95% 配对 t 下界 `LB95 > 0` |
| S6 | 验证方向一致性 = 3/3（`delta_val_best` 与 `delta_test` 逐种子同号） |
| S7 | `mechanism_pass_count = 3/3`（三种子 G1–G5 全过；既有两种子已核实全过，新种子按层 A 实测） |
| S8 | 新种子两臂形式完整性：A1/A2/A4/A5 两臂无 FAIL；B1/B2/B4 两臂无 FAIL；B3 `gate_mean` 两臂**逐位相同**（B3 通过与否本身是冻结 backbone 的种子属性，如实记录，不要求 PASS） |

**如实声明（先于运行的诚实性声明）**：由于种子 `1685480945` 的 `delta_test = −0.0023308435` 是不可变历史，**S1（3/3 正号）、S3（min > −0.001）与 S5（单侧下界 > 0，负值同时拉低均值并抬高样本标准差）在冻结历史下均不可满足**。因此本规则在本分支**不可达**；除非（不可能）§1.2 的只读值被证伪，本分支的预期终点是 `UTILITY_SEED_UNSTABLE`。此声明本身先于运行写定，防止事后把规则描述成"本可通过"或"事后收紧"。

**终局决策（先于结果写定）**：

- **若 `STABLE_UTILITY`**（仅为规则完整性保留）→ 决策 `REGISTER_NEXT_VALIDATION`：注册新独立分支（暂名 `exp/census-stage2-residual-prompt-canonical-seeds45`），在其**自己的预注册**下按同一配对协议跑剩余两个 canonical 种子 `1685459668` / `1685496394`，形成 5-seed 视图。**本分支不执行**。
- **否则** → 终局 `verdict = UTILITY_SEED_UNSTABLE`，决策 `CLOSE_LINE`：本候选**机制有效**（G1–G5 在全部已测种子上通过）但**效用随种子不稳定 / 无持久提升**；如实记录 3-seed 聚合表与逐种子分类，仅作描述性说明"正号倾向（2/3 正）"或"负号倾向（≤1/3 正）"，**不**将其表述为支持稳定效用。**本分支不做任何补救运行**：不调参、不加预算、不加种子、不换划分、不做 A3。

### 4.6 阈值纪律

本文件先于运行单独提交并推送即凭据。判据、分类、决策规则、种子、预算、命令在看到本次结果后**一律不得修改**；工具代码中的分类与聚合常量由测试钉死。

---

## 5. 必录清单（结果回填时逐项落盘/入文）

1. 两臂 `run_id`、`commit`、`variant`、完整 `config.json`；
2. 两臂 **AUC-Test-Education**、**best val AUC（Education）**、逐 epoch val AUC、`best_epoch`、运行 epoch 数；
3. **配对差** `delta_test`、`delta_val_best`；层 B 分类；层 A 结局（并列）；
4. Stage-1 身份：`stage1_id`、`config_hash`、`split_fingerprint_sha256`、`env_ids_sha256`、`backbone_sha256`（与 §3.2 预测值逐项对账）；
5. 机制 G1–G5 逐项实测 + `arm_verdict` 结局；
6. prompt 门控/比值诊断：`alpha_by_epoch`（grad_probe）、`alpha_final`、`g_eff mean/std/min/max`、逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean`、`cos_mean`；
7. 预测离散度：处理头 val `pred_mean/std/min/max/q05/q50/q95` + 参照头（臂 B `newtask.pt` 只读复算）同口径值与复算 val AUC 对账；
8. 冻结/划分证据：两臂 A1/A2/A4/A5/B1–B4 逐项 + `backbone_sha256_before/after`；
9. **3-seed 聚合**（§4.4 全部量，含 CI 与两个 t 分位点）+ 终局判定（§4.5 S1–S8 逐项 + verdict/decision）；
10. 先证据对账：§1.2 八个文件的 sha256 重算一致；
11. 失败/无效尝试：任何崩溃、无效 run、丢弃产物及其原因与处置（无则明确写"无"）；
12. 参数预算（工具核算，与预注册一致）+ `SUMMARY.md` 只追加两行（runner 自动）。

---

## 6. 运行规程（顺序固定，各步一次）

1. **预注册提交并推送**（本文件）。
2. **迁移 + 等价性守卫 + 3-seed 聚合工具**：TDD 红→绿；提交并推送。
3. **全量测试**：`census_benchmark/tests` 全绿（§3.4）。
4. **Stage-1**（§3.2）→ 预测值与划分基准逐项核对 → 通过才继续。
5. **臂 B**（§3.3 命令 1）→ 记录；**臂 T**（§3.3 命令 2）→ 记录。
6. **配对判定**：`python -m census_benchmark.paired_verdict --baseline-run <臂B> --treatment-run <臂T>`（新种子两臂）。
7. **3-seed 聚合**：`python -m census_benchmark.three_seed_aggregate --pair 1685480945:<s1B>,<s1T> --pair 1685463909:<s2B>,<s2T> --pair 1685477428:<s3B>,<s3T> --out artifacts/census_stage2/paired/<date>-three-seed.json`（只读六个 run 目录；先证据路径见 §1.2，须重算 §1.2 的 8 个 sha256 并对账）。
8. **结果回填**：本文件第 11 节 + `SUMMARY.md`（runner 已追加两行）；提交并推送。
9. **终检**：`git status` 干净、`origin/exp/census-stage2-residual-prompt-3seed-expansion` tip = 本地 tip。

---

## 7. 失败处置与止损（先于结果写定）

1. **崩溃即记录**：任何失败尝试（含中间产物）按 §5.11 留痕；只有在**能证实**为基础设施原因（如 junction 失效、CUDA 初始化失败）时才允许**一次**重试，两次尝试全部留痕、判据不变；否则停止，不产出解释。
2. **禁止结果驱动的一切动作**：不换种子、不换 Stage-1 产物、不改预算/patience/阈值/命令、不因结果近阈值而复跑、不做 A3 复跑。
3. **本分支不跑**：其它种子、`full` tag、第三数据集、更长预算、任何机制变体；`STABLE_UTILITY` 情形下的下一步只**注册**、不执行。
4. 无论结果如何，`SUMMARY.md` 只追加；层 A 历史结局与层 B 分类并列报告，不得只报告通过的部分。

---

## 8. 与评测协议 / 候选分支的接口（改动清单）

| 文件 | 处置 |
|---|---|
| `multitaskrec/model.py`、`config.py`、`census_benchmark/protocol.py`、`census_benchmark/metrics.py` | **零改动**（静态守卫，逐字节 = `87afe03`） |
| `census_benchmark/residual_prompt.py`、`census_benchmark/paired_verdict.py`、`run_census_benchmark.py`、`census_benchmark/tests/test_residual_prompt.py`、`census_benchmark/tests/test_paired_verdict.py`、`census_benchmark/tests/test_migration_equivalence.py` | 从 `41feb29` **逐字节迁移**（§3.4 钉子） |
| `census_benchmark/tests/test_migration_equivalence_3seed.py` | 新增（本分支等价性守卫：blob/diff/AST 三层） |
| `census_benchmark/three_seed_aggregate.py` + `census_benchmark/tests/test_three_seed_aggregate.py` | 新增（3-seed 聚合与终局判定；additive） |
| `docs/superpowers/specs/2026-10-06-census-stage2-residual-prompt-3seed-expansion-design.md` | 本文件（预注册先行；结果只回填第 11 节） |
| `artifacts/census_stage2/{stage1,runs,splits}` | 只读/只追加；`SUMMARY.md` 追加两行 |

---

## 9. 明确的非目标

- 不改机制、不做超参搜索、不做 rank/宽度/α 初值/注入流的任何变体；
- 不主张方法新颖性、不写"我们提出"、不产出论文级性能结论、不绘制任何图形；
- 不比较/引用先分支的**结果行**作为本次证据——唯一例外是第 4.4 节 3-seed 聚合中对两个既有种子的**明确标注来源的只读引用**（其数值在回填时经 sha256 对账）；
- 不修改评测协议文件、不修改 `master`、不合并任何分支。

---

## 10. 已声明的风险与偏差

1. **环境偏差**：worktree 无独立 venv（使用主树 venv，§1.4）；数据经 junction 只读引用（`.gitignore` 已拦截，实测 git 状态干净）。
2. **终点已定的诚实声明**：§4.5 已声明 `STABLE_UTILITY` 在冻结历史下不可达；本分支的任务是**仲裁第三种子并如实收束**，不是寻找可通过的判定。
3. **B3 种子依赖**：s1 上 B3 FAIL（继承）、s2 上 PASS；新种子 B3 无论通过与否均如实记录，不归因于本臂，也不因此重跑挑结果。
4. **预算受限**：两种子两臂 best_epoch 均 = 5/5；新种子若同样触顶，"预算受限"是 §4.3 (d) 的正当维度（只陈述、不借此改预算）。
5. **fresh Stage-1 的 `backbone_sha256` 无对照**：系新种子的独立值（不同 model seed 必然不同），只落盘、不比较。
6. **配对 95% CI 的样本量声明**：n=3 的 t 区间仅作描述性区间；本分支不以 CI 做一般性统计推断之外的声明。

---

## 11. 结果（实测回填）

**判定（并列报告，互不覆盖）**：

| 判据体系 | 结论 |
|---|---|
| **终局稳定性判定（本次预注册 §4.5）** | **`UTILITY_SEED_UNSTABLE`**：S1 / S3 / S5 FAIL（符号 2/3、最差种子 −0.00233、单侧下界 −0.00441 < 0）；S2 / S4 / S6 / S7 / S8 PASS ⇒ 决策 **`CLOSE_LINE`**（机制有效但效用随种子不稳定 / 无持久提升） |
| 层 B 重分类（第三种子 `1685477428`） | **`POSITIVE_IMPROVEMENT`**：`delta_test = +0.004750278433788058`（≥ +0.001） |
| 层 A 历史判据（只读并列） | **`VALID_NEGATIVE`**：E1 = `0.847530 < 0.8521` 未达标；G1–G5 **全过** |

### 11.1 运行标识与配置（每臂恰好一次，无任何重跑）

| 项 | 臂 B（baseline） | 臂 T（residual-prompt） |
|---|---|---|
| run_id | `20261006-0506-s20260929-m1685477428-short-d4402d1` | `20261006-0507-s20260929-m1685477428-short-d4402d1-rpg` |
| commit（HEAD，两臂相同） | `d4402d1` | `d4402d1` |
| variant / tag / epochs / patience / lr / batch | baseline / short / 5 / 2 / 1e-3 / 256 | residual-prompt / short / 5 / 2 / 1e-3 / 256（+ `prompt_hidden=16`；参照 = 臂 B `newtask.pt`） |
| Stage-1 | 两臂同一：`s1-096f8f16-m1685477428-e2-15eabcb4` | 同左 |
| split / model / env seed | 20260929 / 1685477428 / 20260929 | 同左 |

命令即预注册 §3.2/§3.3 所写（Stage-1 一次 → 臂 B → 臂 T，含 `--prompt-reference-newtask` 指向臂 B）。

### 11.2 主报告量（配对，seed `1685477428`）

| 量 | 臂 B | 臂 T | 配对差（T − B） |
|---|---|---|---|
| **AUC-Test-Education** | `0.8427796966516823` | `0.8475299750854703` | **`+0.004750278433788058`** |
| best val AUC（Education，选点用） | `0.8436747451693688` | `0.8492923301160789` | `+0.0056175849467100525`（方向同号） |
| 逐 epoch val AUC | 0.833352 / 0.838978 / 0.841148 / 0.842422 / 0.843675 | 0.833710 / 0.838397 / 0.843514 / 0.847144 / 0.849292 | —— |
| best_epoch / 运行 epoch 数 | 5 / 5（**预算受限**） | 5 / 5（**预算受限**） | —— |

### 11.3 层 B 分类与工具输出（`artifacts/census_stage2/paired/20261006-paired-m1685477428.json`）

- `delta_test = +0.004750278433788058` ⇒ **`POSITIVE_IMPROVEMENT`**（阈值常量由迁移工具测试钉死）；
- `delta_val_best = +0.0056175849467100525`（> −0.001，(b) 不触发；方向同号）；
- 实质矛盾清单：**空**（(a) 历史结局 `VALID_NEGATIVE` ∉ 四类非全过集合；(c) 两臂 A1/A2/A4/A5/B1/B2/B4 全 PASS、两臂 B3 `gate_mean` 逐位相同）；
- 工具按迁移来的配对决策规则输出 `decision = EXPAND`（单种子配对层语义）；**本分支的终局决策以 §4.5 的 3-seed 判定为准（§11.9）**，两者并列报告、互不覆盖。

### 11.4 Stage-1 身份与交叉核对（运行前预测值全部命中）

| 项 | 预测（预注册 §3.2） | 实测 | 结果 |
|---|---|---|---|
| stage1_id | `s1-096f8f16-m1685477428-e2-15eabcb4` | 同 | **MATCH** |
| config_hash | `15eabcb4d63c18c471a6c689309bfbf4454f4a9d4a4c1391ce448ee3918bf40f` | 同 | **MATCH** |
| split_fingerprint_sha256 | `096f8f16db08…7d460c` | 同（A2 走校验路径，基准为运行前复制的 smoke 基线，文件 sha256 `37a6ff55…`） | **MATCH** |
| n_train / n_val / n_test | 199523 / 49881 / 49881 | 同；A4 disjoint + union_complete 复算 True | **MATCH** |
| 初始 env_ids sha256 | `ec50367f76c2524bc57478d4c183984d50b8c381c66364ddc9a3a4f844eeba91` | 同 | **MATCH** |
| 最终（cluster 后）env_ids | 不可预测（逐种子不同） | `1c7353491360ed98…f213f2`，counts `[109068, 90455]` | 落盘 |
| backbone_sha256 | 不可预测（诊断） | `291c86277a59cfd69213026b5fea21e7a086ee73a64546c17a0c182f0a2c0c37` | 落盘 |
| stage1 best_epoch / val AUC | —— | 2 / Income `0.9308640941564909` / Marital `0.9922620195083053`；env_acc `0.498925 → 0.336653`（M1 诊断） | —— |

### 11.5 机制门禁 G1–G5 与历史结局（`metrics.json:rp_arm`，迁移代码原样计算）

| 判据 | 实测 | 结果 |
|---|---|---|
| G1 构造恒等 | 共享参数逐位一致；构造后全局 RNG 端点逐位一致；新增键恰为预注册 5 键 | **PASS** |
| G2 初值恒等 | `alpha_at_construction = 0.0`；真实首个训练 batch（256 样本）处理头与基线头前向 `max_abs_diff = 0.0`（bit-identical） | **PASS** |
| G3 门控活性 | 首 batch α 梯度 = 0.008532 ≠ 0；α 逐 epoch 首 batch 探针 0.0 → 0.2020 → 0.2400 → 0.2727 → 0.2949；`best_state` 载入后 α_final = **0.31610456109046936** ≠ 0；生成器梯度首步 0（α=0 数学后果）→ 0.1371 / 0.1309 / 0.1438 / 0.1365 > 0 | **PASS** |
| G4 范数受控界 | 三路 `ratio_max` = 0.25897092 / 0.25897094 / 0.25897096 ≤ \|α_final\| + 1e-6 = 0.31610556（余量 0.0571） | **PASS** |
| G5 活性带 | 三路 `ratio_mean` = 0.12961406 ∈ [0.005, 0.5]（距下界 25.9×，距上界 3.9×） | **PASS** |
| **E1（历史，绝对阈值）** | `0.8475299750854703 < 0.8521` | **未达标** |
| **历史结局分类** | G1–G5 全过 + E1 未达标 | **`VALID_NEGATIVE`**（只读并列，不覆盖层 B/终局判定） |

### 11.6 prompt 门控/比值分布与逐源读数（val，`best_state` 载入后，n = 49881）

| 流 | h_norm_mean | delta_norm_mean | ratio_mean | ratio_max | cos(δ,h)_mean |
|---|---|---|---|---|---|
| gen | 49.419 | 6.432 | 0.1296140592 | 0.25897092 | −0.3004 |
| spec_0 | 6.678 | 0.903 | 0.1296140592 | 0.25897094 | −0.0393 |
| spec_1 | 10.859 | 1.369 | 0.1296140592 | 0.25897096 | −0.0190 |

- 逐样本有效门控 `g_eff`：mean 0.12961406 / std 0.03089554 / min 0.05987872 / max 0.25897094（真实分布，非单点）；
- S2 成立：三路相对扰动恒等（fp64 统计在 1e-10 内一致），绝对残差随各流范数缩放；`n_zero_rep = 0`；
- `cos(δ,h)` 明显偏离 ±1：残差方向不是对原表征的缩放；
- 本种子三路基范数（49.4 / 6.7 / 10.9）与另两种子的悬殊结构一致（gen ≫ spec）。

### 11.7 预测离散度（防坍缩）与参照对账

| 量（val，n = 49881） | 处理头 | 参照：臂 B 头（只读复算） |
|---|---|---|
| pred mean / std | 0.096950 / 0.128488 | 0.100518 / 0.129023 |
| pred min / max | 2.75e-10 / 0.621500 | 3.34e-08 / 0.587412 |
| pred q05 / q50 / q95 | 3.47e-07 / 0.050022 / 0.419557 | 1.76e-05 / 0.054690 / 0.412838 |
| 复算 val AUC | —— | `0.8436747451693688`（与臂 B 记录的 best_val **逐位一致**，对账通过） |

### 11.8 冻结/划分证据、参数预算与复现性交叉证据

- **两臂全门禁 PASS（含 B3）**：A1（backbone `291c8627…` 前后一致、`.grad` 全 None）、A2（划分指纹 `096f8f16…` 两臂一致）、A4（disjoint + union_complete）、A5（env_ids `1c735349…` = Stage-1 落盘值）、B1、B2（gap：臂 B 0.0008950 / 臂 T 0.0017624 ≤ 0.03）、B3（两臂 `gate_mean = [0.9301209185041758, 0.7759503682787815]` **逐位相同**；**本种子下 B3 = PASS**）、B4（env 份额 [0.5466, 0.4534]）。
- 参数预算：新增 4161（`prompt_gate` 1 + 生成器 4160），head 27063，占比 15.3752%；清单与预注册一致。
- 两臂 `commit = d4402d1`（同一 HEAD，代码身份一致）；A3 保持 `on_demand`（未触发复跑）。
- `SUMMARY.md` 只追加两行（runner 自动）：`20261006-0506-…` 与 `20261006-0507-…-rpg`，A/B 逐项全 PASS。

### 11.9 3-seed 聚合与终局判定（`artifacts/census_stage2/paired/20261006-three-seed.json`）

**逐种子表**（s1/s2 为只读先证据，8 个文件 sha256 回填前重算全部 MATCH，见 §11.10）：

| seed | Δ_test | Δ_val,best | 层 B | 层 A | 机制 G1–G5 | α_final | ratio_mean | best_epoch | B3 |
|---|---|---|---|---|---|---|---|---|---|
| `1685480945` | **−0.0023308435378367465** | −0.00262025464356741 | `NO_CLEAR_IMPROVEMENT` | `VALID_NEGATIVE` | 全过 | 0.314076 | 0.127718 | 5/5 | FAIL（继承） |
| `1685463909` | **+0.0029909067604868556** | +0.0023925880011116396 | `POSITIVE_IMPROVEMENT` | `VALID_NEGATIVE` | 全过 | 0.342400 | 0.156963 | 5/5 | PASS |
| `1685477428` | **+0.004750278433788058** | +0.0056175849467100525 | `POSITIVE_IMPROVEMENT` | `VALID_NEGATIVE` | 全过 | 0.316105 | 0.129614 | 5/5 | PASS |

**聚合量（预注册 §4.4 公式）**：

| 量 | 值 |
|---|---|
| mean | **+0.0018034472188127222** |
| sd（ddof=1） | 0.003686884486472501 |
| 配对 95% CI（双侧，t = 4.302652729696142） | **[−0.007355281572787369, +0.010962176010412814]**（含 0） |
| 单侧 95% 下界（t = 2.919985580355516） | **−0.004412103439338537**（< 0） |
| sign_positive_count | **2 / 3** |
| positive_improvement_count（Δ ≥ +0.001） | **2 / 3** |
| worst_seed | **`1685480945`（−0.0023308435378367465）** |
| mechanism_pass_count | **3 / 3** |
| validation_direction_consistency（Δ_val 与 Δ_test 同号） | **3 / 3**（−/−，+/+，+/+） |

**S1–S8 判定**：S1 FAIL（2/3 正号）、S2 PASS（2/3 ≥ +0.001）、S3 FAIL（最差 −0.00233 ≤ −0.001）、S4 PASS（均值 ≥ +0.001）、S5 FAIL（单侧下界 < 0）、S6 PASS（3/3 同号）、S7 PASS（3/3 机制全过）、S8 PASS（三种子两臂完整性约束全满足）⇒ **`verdict = UTILITY_SEED_UNSTABLE`，`decision = CLOSE_LINE`**。

**终局陈述**：范数受控残差 prompt 在三种子上**机制全部有效**（G1–G5 3/3 全过、α_final 0.314–0.342、ratio 带内、E1 全部未达标），但**效用随种子不稳定**：效应符号 −、+、+（2/3 正），第三种子给出三种子中最大的正效应（+0.00475）而第一种子为负（−0.00233），均值 +0.0018 但 95% 区间跨 0、最差种子实质为负 ⇒ **无持久提升**。baseline 本身跨种子波动（0.8428–0.8501，极差 ~0.0073）与效应量（~2–5e-3）同阶，n=3 下无法将该噪声水平与效应区分开。如实记录"正号倾向（2/3 正）"，**不**表述为支持稳定效用。

**`NO_CLEAR_IMPROVEMENT` 种子（`1685480945`）的 headroom 评估（预注册 §4.3 四维，如实陈述）**：
- (a) 机制有效性：G1–G5 全过，α 0.3141，ratio_mean 0.1277 在带内 ⇒ 机制在工作，"没跑起来"不成立；
- (b) 验证方向：Δ_val = −0.00262，与 Δ_test 同号（同向走低，非反向矛盾）；
- (c) 跨种子一致性：另两种子为 +0.00299 / +0.00475；该负读数是三种子中的最差/离群值，但 n=3 无法判定"离群 vs 常态"；
- (d) 预算趋势：两臂 best_epoch 均 = 5/5（预算受限，val 曲线仍上升），无法排除"更长训练反转"也无法据此改预算。
- 结论：该 NO_CLEAR 种子**没有可援引的正面 headroom 证据**；不计为"接近正提升"。

**注册的下一步**：无。按预注册 §4.5，`STABLE_UTILITY` 在冻结历史下不可达（运行前已如实声明），本分支终局为 `CLOSE_LINE`：**Residual Prompt 线就此关闭**（机制有效、效用种子不稳定）；不在本分支做调参 / 延长预算 / 追加种子 / 换划分 / A3 复跑。

### 11.10 失败/无效尝试、环境偏差与止损执行

- **运行失败：无**（Stage-1、臂 B、臂 T 全部一次成功；无崩溃、无丢弃产物）。
- **无效尝试 1 次（工具层，先于最终聚合、非 run）**：首次运行 3-seed 聚合工具时，历史基线 run（`904f8d0`，`variant` 字段引入之前所产）缺 `variant` 键 ⇒ 工具抛 `ValueError`。处置：在聚合工具中接受"基线臂 variant ∈ {None, 'baseline'}（None 视为历史缺省）"并新增回归测试（84 通过）；**不触碰任何判据、阈值、常量、种子或命令**，且该字段不参与任何 S1–S8 判定（S1/S3/S5 的 FAIL 与之无关）。修复提交晚于三臂运行，不改变 run 身份（两臂 `commit` 仍为 `d4402d1`）。
- **环境偏差**：① worktree 无独立 venv，使用主树 `D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe`（记录在案）；② 数据经只读 junction 引用主树 `dataset/`（`.gitignore` 拦截，git 状态干净）；③ `git push` 直连 github:443 两次被重置，改用本机代理 `git -c http.proxy=http://127.0.0.1:7897` 完成推送（预注册 `828d680` 与实现 `577c809`/`d4402d1` 均已推送成功）。
- **先证据对账**：§1.2 八个文件 sha256 回填前重算，全部 **MATCH**。
- **止损执行**：每臂恰好一次；无 A3 复跑；未改任何种子/预算/阈值/命令；未做任何结果驱动动作；`SUMMARY.md` 只追加两行（未重写）；三臂运行期间 HEAD 冻结于 `d4402d1`。
- **测试**：迁移后全量套件 **83 通过**（既有 23 + 迁移 27 + 迁移 12 + 先守卫 2 + 新增守卫 4 + 3-seed 工具 15），实验前全绿；修复后 **84 通过**。
