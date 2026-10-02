# AliCCP Stage-2 固定衰减增益的第二 seed 稳定性复现（replication / robustness，非新方法）

- **状态**：预注册已写死（本文件在本分支任何 run 之前提交）；实现与结果见 §10（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage2-attenuation-seed-replication`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；**不从**任何 exp 分支拉出、不继承其运行时状态；协议文件零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（SUMMARY 列、A/B 门禁阈值、预算、指纹定义均不变）
- **复现对象（先例记录，唯一来源）**：`exp/aliccp-stage2-specific-attenuation-control` @ `3314420`，处理臂 run `20261003-0322-p2M-v500k-t1M-m1688723512-short-f88dca4-sattn`（c = 0.6972233730330467，post-hoc 归因对照，ATTENUATION_SUPPORTED）
- **定位声明**：本实验是**复现 / 稳健性证据**，不是新方法，不主张任何新颖性；不改系数、不调参、不换结构、不改损失——与 seed-1 处理臂相比**唯一的变化 = 模型 seed**。结论口径：seed-1 观察到的固定衰减正增益在第二个模型 seed 下是否按预注册判据稳定复现。

---

## 0. 目的与判定问题

seed-1（`m1688723512`）在固定 Stage-1 产物上观察到：specific 混合乘以常数 c = 0.6972233730330467 的处理臂相对配对基线 Δtest = `+0.012706144575588052`、Δval = `+0.013279276931939643`（同 stage1_id、同顺序、单次 test 评测）。该结果单 seed、无重复噪声估计（协议 A3 复跑在全部 run 均为 SKIP）。

**判定问题 Q**：在第二个模型 seed（`1688723740`，选择规则见 §2）下，用同一系数、同一预算/超参/评测口径，处理臂相对同 seed 配对的基线是否仍满足 **Δtest ≥ +0.0055 且 Δval > 0**，且全部构造/冻结门禁通过？

---

## 1. 复现对象（seed-1 记录；逐位，已对照 main tree 实际 run 产物复核）

| 项 | seed-1 基线臂 | seed-1 处理臂（本实验的复现对象） |
|---|---|---|
| run_id | `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9` | `20261003-0322-p2M-v500k-t1M-m1688723512-short-f88dca4-sattn` |
| commit / dirty | `b2e17f9` / false | `f88dca4` / false |
| stage1_id | `s1-5c060b9c-m1688723512-e3-3a30e2c0` | 同左（同一 checkpoint） |
| model seed / env seed | `1688723512` / `20261003` | 同左 |
| **AUC-Test-BSI** | `0.5988392178311113` | `0.6115453624066993` |
| AUC-Val-BSI(best) | `0.5781533414372665` | `0.5914326183692061` |
| Δtest / Δval（相对基线） | — | **`+0.012706144575588052` / `+0.013279276931939643`** |
| 头 gate 均值（val） | `[0.851149, 0.1488510625]` | `[0.49435615625, 0.50564384375]` |
| trainable params | 8129 | 8129（零新增） |
| 臂级判定（seed-1 §5 口径） | — | ATTENUATION_SUPPORTED（r = 0.9176） |

复核方式（只读）：直接读取 main tree `artifacts/aliccp_bench/runs/<run_id>/metrics.json` 的 `test_auc_bsi` / `best_val_auc_bsi` / `model_seed` / `stage1_id` 字段，与上表逐位一致；Δ 为本文件重算的 float64 差值。

**披露（seed-1 证据链的性质，不因本实验改变）**：seed-1 的假设是 post-hoc 提出的（在观察到 null 臂结果之后），其 null 臂机制判据未过（`null_top1_rate = 0`），r ≥ 0.70 归类判据属 seed-1 专属。本实验只检验"该固定衰减增量在第二 seed 的稳定性"，不追溯、不重新归因其机理。

---

## 2. 种子审计与第二 seed 的选择（先于任何 seed-2 run 写死）

### 2.1 审计：仓库与协议中的 seed 常量与历史

| 事实 | 来源 |
|---|---|
| 协议唯一 model seed `MODEL_SEED = 1688723512`（= seed-1；只用于运行入口默认值与 run_id/stage1_id 记录） | `aliccp_benchmark/protocol.py:30`，spec §6.1（"仓库 AliCCP 全脚本默认、注释中的 5 seed 之一；首轮唯一 seed"） |
| 协议 env seed `ENV_SEED = 20261003`（只决定初始 `env_ids`；独立 `Generator`，不消耗全局 RNG） | `aliccp_benchmark/protocol.py:31`，spec §6.1 |
| canonical 5-seed 列表：`[1688723512, 1688723740, 1688738016, 1688749593, 1688762746]` | `AliCCP_MPTRec.py:90` 注释；`baseline/mmoe/AliCCP_MMOE.py:67`、`baseline/sparsesharing/AliCCP_train_mtl.py:66`、`baseline/csrec/AliCCP_train_single.py:171` 同列表 |
| 历史缺陷 F3（已由协议消除）：`AliCCP_MPTRec.py` 默认 `1688723512` vs `AliCCP_NewTask.py` 默认 `1688738016` | 协议 spec §3 F3 |
| 该 seed 的仓库既有使用：`analysis/AliCCP_Visualization.py:17`（`seed = 1688723740`）；`baseline/ple/AliCCP_PLE.py:71`、`baseline/ple/AliCCP_NewTask.py:81`（`for seed in [1688723740]`） | master 文件（零修改） |

**seed 覆盖安全性审计（回答"是否存在硬编码导致 override 不安全"）**：`run_aliccp_benchmark.py` 的 stage1/stage2 均有显式 `--model-seed`（默认 = `protocol.MODEL_SEED`）；`bench.run_stage1/run_stage2` 的全部随机性落点（`seed_model`、`make_stage1_id`、`make_run_id`、`config_hash`、`meta`）只读取传入参数，`protocol.py` 内除默认值外无任何分支读取 `MODEL_SEED`。**结论：seed 覆盖已是一等显式 CLI，无硬编码，无需新增 override；默认行为保持不变（由测试钉死默认值 == `protocol.MODEL_SEED`）。**

### 2.2 第二 seed 的选择规则（确定性；禁 cherry-pick）

**规则：取 canonical 5-seed 列表的"第 2 位"（= seed-1 `1688723512` 在列表中的后继元素）⇒ `1688723740`。**

- 规则只依赖列表顺序，不依赖任何 run 结果；在本文件提交时写死（早于任何 seed-2 run）。
- 未选备选（按同一规则的排除，非结果依赖）：`1688738016`（列表第 3 位；恰为历史 `AliCCP_NewTask.py` 默认）、`1688749593`、`1688762746`。
- `1688723740` 是"仓库既有 seed"（§2.1 辅助事实），满足"prefer an existing repository seed"。

---

## 3. 控制变量：哪些变、哪些不变

| 项 | seed-1 | seed-2（本实验） | 说明 |
|---|---|---|---|
| **model seed** | 1688723512 | **1688723740** | 唯一被操纵的随机性来源（模型初始化 + 训练 dropout + 阶段 2 头初始化） |
| env seed | 20261003 | 20261003（**不变**） | 初始 env_ids 生成策略受控；最终 env_ids 由 stage-1 训练轨迹（cluster_2）决定、随 seed 变——由 A5 逐位校验产物一致性，属预期行为 |
| 预算 / 前缀 | p2M-v500k-t1M | 同（不变） | 前缀指纹预期不变；A2 校验（预期 `fingerprint_sha256 = 5c060b9c…`，与 seed-1 相同） |
| 阶段 1 epochs / patience | 3 / 2 | 3 / 2（不变） | 协议 §8.3 |
| 阶段 2 epochs / patience | 5 / 2 | 5 / 2（不变） | 协议 §8.3 |
| 系数 c | 0.6972233730330467 | **同值钉死**（不调参） | float64 字面量，推导：`c := 1 − null_mean = 1 − 0.3027766269669533`（seed-1 null 臂 metrics.json 全精度字段）；float32 铸造值 `0.6972233653068542`（记录备查，判定不依赖） |
| batch / lr / 结构 / dropout / loss / 温度 | 全部协议 §8.1 默认 | 同（不变） | — |
| 评测口径 | val 选点 + test 单次评估（AUC-BSI） | 同（不变） | 协议 §7.5/§9.1 |

**头 seed / 顺序配对**：两臂 Stage-2 都以 `--model-seed 1688723740` 运行（进程重播种 → 建数据 → 初始化头），三个 DataLoader `shuffle=False`、`batch_size=2000`、前缀一致 ⇒ 两臂的头初始化与样本顺序**构造性相同**；配对性由"同一 stage1_id + 同一 model seed + 同序"保证。

---

## 4. 新 Stage-1 产物与运行计划（恰好一次）

- **新内容寻址产物**：`stage1_id = s1-<prefix8>-m1688723740-e3-<cfg8>`（model seed 进入 `config_hash` 与 `stage1_id`；`protocol.save_stage1` 禁止覆盖已有产物）。seed-1 产物（`s1-5c060b9c-…`）不被引用、不被修改。
- **配对运行**：在该新产物上跑 (a) 基线臂 `stage2`（默认行为）；(b) 处理臂 `stage2 --spec-attenuation 0.6972233730330467`。
- **运行次数纪律**：seed-2 stage1 **恰好 1 次**；每个臂 stage2 **恰好 1 次**；`replicate` 分析（纯分析，无训练、不改 run 产物）**恰好 1 次**。**已完成的有效 run 一律不重跑**（不挑 seed、不挑 epoch、不重跑挑好结果）。
- **无效执行处置（预注册）**：仅当 run 因**工具性原因**（进程崩溃/中断/环境故障，run 未完整落盘）无效时，保留全部现场（含失败日志与半成品目录，不删除），记录原因后可重跑一次；因**结果原因**（门禁失败、判定未过）不构成无效，不重跑，如实记录。§10 中无效执行与有效执行分开列明。

---

## 5. 预注册判据（看到结果前写死；看到结果后不得修改）

### 5.1 主判据（seed-2 稳定性）

对照 = **本分支新跑的 seed-2 基线臂**（同 `stage1_id`、同 model seed、同顺序；**不得**用 seed-1 的基线 run 做 seed-2 的对照）。

| 条件 | 判据 |
|---|---|
| C1 | Δtest := AUC-Test-BSI(arm, seed2) − AUC-Test-BSI(baseline, seed2) **≥ +0.0055**（闭区间） |
| C2 | Δval := AUC-Val-BSI(best, arm, seed2) − AUC-Val-BSI(best, baseline, seed2) **> 0**（严格） |
| C3 | 两 run 的**构造 / 冻结门禁**全过：A1（backbone sha before==after 且 `.grad` 全 None）、A2（指纹）、A4（标签对齐）、A5（env_ids sha）、A6（backbone sha == stage-1 记录）在**两臂**均 PASS；A3 SKIP 视为通过 |

- C1+C2+C3 全满足 → 分类 **`STABILITY_SUPPORTED`**；否则 **`STABILITY_NOT_SUPPORTED`**。
- 阈值来源：+0.0055 沿用 seed-1 记录中的 provisional PI 效用口径（协议 §10.1 "+0.005" 的保守取整版本，seed-1 处理臂即以 +0.0055 单独记录）；写成增量形式（而非 AUC 字面量形式）以避免基线字面量在 arm 运行时未知的问题，两形式在 seed-2 基线值确定后数学等价。
- B 类（B1–B4）照常判定并披露；**继承披露口径与 seed-1 相同**：B1/B4 由共享 Stage-1 产物决定（两臂逐位相同），若 FAIL 属继承缺陷、照常披露、不归因于处理臂、不据此否定本臂；B2/B3 为臂相关门禁照常记录。

### 5.2 pooled / 符号一致性（描述量 + 联合规则）

跨 seed 汇总量（seed-1 Δ 逐位取自 §1；seed-2 Δ 由 §5.1 计算）：

- `mean_delta_test` = (Δ1_test + Δ2_test) / 2；`mean_delta_val` = (Δ1_val + Δ2_val) / 2
- `sign_consistent_test` = (Δ1_test > 0) == (Δ2_test > 0)；`sign_consistent_val` 同理
- 联合判定（机械规则）：
  - **`STABLE`** ⇔ §5.1 主判据通过 ∧ `mean_delta_test ≥ +0.0055` ∧ `mean_delta_val > 0` ∧ 两指标符号一致
  - **`NOT_STABLE`** ⇔ §5.1 主判据未通过
  - **`MIXED`** ⇔ 其余（主判据过但 pooled 任一条件不成立）
- 算术注（预注册完备性说明）：seed-2 主判据过 ⇒ Δ2_test ≥ +0.0055、Δ2_val > 0，结合 Δ1_test = +0.012706…、Δ1_val = +0.013279… ⇒ mean_delta_test ≥ 0.009103… ≥ +0.0055、mean_delta_val > 0、符号必然一致 ⇒ 联合判定自动为 `STABLE`。`MIXED` 分支仅为规则完备性保留（实践中不可达）。

### 5.3 解释规则（因果口径，硬约束）

- **禁止**把跨 seed（= 不同 stage-1 backbone）的**原始 AUC** 作为任何因果/方法证据（不同 backbone 的 AUC 不可直接比较）。跨 seed 唯一允许的比较量 = **同 seed 内配对增量 Δ**（复现性统计）。
- seed-1 vs seed-2 的对比只以 Δ 与 pooled 量呈现；种子间基线 AUC 的差异不解释、不归因。
- 禁止：换 c、换 seed、换预算、换阈值、挑 seed/epoch、只报通过的 run（协议 §9.3）。阈值与规则的任何修改只允许在**看到本实验任何 run 结果之前**，每次修改单独 commit 并在本文件记录理由；已判定 run 不得用新阈值重判。

### 5.4 流程偏离披露（协议 §4.3.2 / §12.4）

- §4.3.2"先等协议合并再从 master 拉 exp 分支"：与 seed-1 处理臂相同，本分支属显式偏离（协议文件零修改、SUMMARY 只追加），沿用既有先例。
- §12.4 多 seed 扩展三条件：(i) 10.1 阈值——seed-1 处理臂已满足（Δtest +0.0127 ≥ +0.005、Δval > 0）；(ii) **A3 复跑证据——未到位（全部既有 run A3 = SKIP），如实披露**：本实验的 seed-1/seed-2 散布**不能**替代同 seed A3 复跑（seed 效应与 run 噪声不可分离，见 §9.1）；(iii) 扩展 seed 列表由用户显式指定——**满足**（用户指令本实验，seed = 1688723740 按 §2.2 规则确定）。本分支在披露 (ii) 缺口的前提下按用户指令执行。

---

## 6. 实现面（最小；TDD；默认关逐位一致）

| 文件 | 改动 | 来源 |
|---|---|---|
| `multitaskrec/model.py` → `NewTask` | 新增构造参数 `spec_attenuation=1.0`（校验 `0 < c ≤ 1`）；`forward` 中 `new_spec_rep = Σ_k W_k·spec_rep_k` 之后**条件乘**（默认 1.0 时整句跳过 → 与基点 `8133d32` 逐位一致） | **逐字移植**自 `exp/aliccp-stage2-specific-attenuation-control`（保证与 seed-1 处理臂代码一致） |
| `aliccp_benchmark/metrics.py` | 系数常量（含推导来源与 f32 铸造值）、seed-1 记录常量、`SourceGateStats`、`pred_dispersion`/`pearson_corr`/`average_rank`/`spearman_corr`/`logit_clip`/`paired_delta_stats`/`cross_arm_stats`、**新写** `stability_arm_verdict`（增量形式）/`pooled_seed_stability`/`joint_stability_classification` | 机制部分逐字移植；判定函数本分支新写（seed-1 的 `attenuation_arm_verdict` 与 r≥0.70 归类判据**不移植**——本实验无 null 臂、判据不同） |
| `aliccp_benchmark/bench.py` | `stage2_run_id`（`-sattn` 后缀）、`newtask_attenuation_probe`（val 一次遍历：预测离散度 + 源 gate 均值）、`run_stage2(..., spec_attenuation=1.0)` 接线与落盘（`probe`/`trainable_params`） | 逐字移植 + 一处适配：run 内**不做**臂级判定（基线字面量在 arm 运行时尚不存在；判定在 `replicate` 步由纯函数对记录值计算，规则见 §5） |
| `aliccp_benchmark/replicate.py`（新文件） | 双臂重评测对照：真冻结加载同一 Stage-1 产物 → 两 head strict 载入 → 同序 val/test 重评测 → 完整性检查（重算 AUC/gate/离散度与记录值比对）→ `arm_vs_baseline` 配对统计 → §5 判定 + pooled → 落盘 JSON | 由 seed-1 `compare.py` 裁剪（**去** null 臂与 git blob 头类重构） |
| `run_aliccp_benchmark.py` | stage2 新增 `--spec-attenuation`（默认 1.0）；新增 `replicate` 子命令 | 移植 + 适配 |
| `aliccp_benchmark/tests/test_seed_replication.py`（新文件） | §7 不变量 | 改编自 seed-1 测试文件 |
| `artifacts/aliccp_bench/SUMMARY.md` | 运行后追加行（append-only） | 协议 §11.2 |

**明确不改**：`aliccp_benchmark/protocol.py`（零改动）、`config.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py`、`baseline/*`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`multitaskrec/model.py` 中 `NewTask` 以外的任何类；CensusIncome / ByteRec 任何代码路径不涉及。**不移植** seed-1 分支的 `compare.py`/null 臂/"learnable attenuation" 代码；SUMMARY 只追加行、**永不重写**。

---

## 7. 不变量（由 `test_seed_replication.py` 强制，不接受人工目测）

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 默认臂（`spec_attenuation=1.0`）前向/反向与基点 `8133d32:multitaskrec/model.py` **逐位一致** | `torch.equal` 比对输出与每个参数的梯度 |
| I2 | 默认臂参数集合/state_dict = 基点；衰减不是参数、不进 state_dict；构造不消耗 RNG | 参数集合差集 + `load_state_dict(strict=True)` + 种子对照 |
| I3 | 开启臂前向等于写死公式（`new_spec_rep' = c × Σ W_k spec_rep_k`） | 独立参考实现比对（`torch.equal`） |
| I4 | 开启臂可训练参数计数 = 默认臂（零新增）；梯度照旧；`get_l2_reg()` 不变 | 计数 + 梯度 + 回归对照 |
| I5 | 系数钉死：`SPEC_ATTENUATION_COEF == 1.0 − 0.3027766269669533` == 文档字面量 `0.6972233730330467`；f32 铸造值 `0.6972233653068542`；非法值构造抛错 | 常量断言 + 文档一致性测试 + 异常测试 |
| I6 | 诊断量数值正确：`SourceGateStats`、`pred_dispersion`、`pearson_corr`/`spearman_corr`（scipy 参考）、`logit_clip`、`paired_delta_stats`、`cross_arm_stats` | 构造数据精确断言（float64） |
| I7 | 判定函数边界正确：`Δtest ≥ +0.0055` 闭区间（恰好 0.0055 通过、−1e-12 不通过）；`Δval > 0` 严格（恰好 0 不通过、+1e-12 通过）；`pooled_seed_stability` 均值/符号公式；`joint_stability_classification` 三分类真值表 | 边界/证据字段测试 |
| I8 | `replicate` 管线：`predict_newtask` 的 AUC 与 `bench.evaluate_newtask` 逐位一致；双臂 strict 载入；极小夹具端到端完整性通过；篡改记录值 → `integrity.pass=False` 且不抛异常；两 run 的 `model_seed` 必须 == `REPLICATION_MODEL_SEED` 且相等 | 等价测试 + CPU 夹具端到端 |
| I9 | 默认臂 run 的 metrics 既有键与语义不变（仅新增顶层 `spec_attenuation: 1.0` 标记；不出现 `probe`/`trainable_params`/臂判定）；处理臂 run 落盘完整（probe 键齐全、run_id `-sattn` 后缀、`trainable_params == trainable_params_default`） | `TestRunnerAttenuationArm`（CPU 极小夹具端到端） |
| I10 | 静态守卫：模型侧唯一改动是 `model.py`；`aliccp_benchmark/protocol.py` 相对基点零改动；相对 `8133d32` 的全部跟踪文件改动 ⊆ §6 白名单；CLI 默认 model seed == `protocol.MODEL_SEED`（默认行为不变）且显式覆盖可用 | `git diff --name-only` + parser 断言 |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 8. 运行方式（恰好一次；命令留档）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) seed-2 阶段 1（唯一一次；model seed 显式覆盖，env seed / 预算 / epochs / patience 走协议默认或显式同值）
python run_aliccp_benchmark.py stage1 --model-seed 1688723740 --env-seed 20261003

# 2) 配对阶段 2（在新 stage1_id 上；两臂同 model seed = 1688723740）
python run_aliccp_benchmark.py stage2 --model-seed 1688723740 --stage1-id <新 sid>
python run_aliccp_benchmark.py stage2 --model-seed 1688723740 --stage1-id <新 sid> --spec-attenuation 0.6972233730330467

# 3) 双臂复现分析（纯分析，恰一次）
python run_aliccp_benchmark.py replicate --stage1-id <新 sid> --baseline-run <基线 run_id> --arm-run <处理臂 run_id>
```

- **前置条件**：开跑前先提交本批改动（否则 run_id / metrics.json 的 commit 记的不是本实验代码）；运行期间不得有其他未提交的已跟踪文件改动（`git.dirty` 必须为 false）。
- 数据文件经目录联接（junction）复用 main tree 的 `dataset/AliCCP/`（只读；`dataset/` 被 gitignore，不入库）。
- **墙钟预算**：stage1 ≈ 3–4 min、stage2 ×2 ≈ 5 min、replicate ≈ 1–2 min（基于 seed-1 实测：stage1 168.1 s、stage2 141.0/146.5 s）。

---

## 9. 局限（预声明）

1. **总计 2 seeds、每 seed 1 run**：仍无同 seed run-to-run 噪声估计（A3 仍 SKIP）；seed-1 与 seed-2 的 Δ 差异混合了 seed 效应与 run 噪声，二者不可分离。本实验回答"是否复现"，不回答"效应量的方差分解"。
2. **复现对象本身为 post-hoc 归因控制**（seed-1 §9）：本实验不改变 seed-1 的判定与其局限；只检验其增量的跨 seed 稳定性。
3. **单数据集（AliCCP）**：结论不推及 CensusIncome / ByteRec（seed-1 的跨数据集迁移已独立记录为 TRANSFER_NOT_SUPPORTED，与本实验无关）。
4. **B1/B4 若 FAIL 为继承缺陷**（共享 Stage-1 语义，两臂逐位相同），本实验无法修复也不据此做任何主张。
5. **判定依赖单一汇总统计**（test 1M 前缀的 BSI AUC 差值）；BSI 负例 ≈1.6 万、SE ≈ 0.0025 @AUC0.7 ⇒ +0.0055 ≈ 2.2×SE——单点估计，不引入事后置信区间叙事。

---

## 10. 结果（实测；运行后追加，不得回填预期值）

（待运行）
