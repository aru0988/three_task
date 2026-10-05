# AliCCP Stage-1 归一化聚类的第三 canonical seed 判定（seed3 = 1688738016；配对效用分类 + 扩展门）

- **状态**：预注册已写死（本文件在本分支任何新 run 之前提交；P4 见 §3.1 的登记时点）；实现与结果见 §9（运行后追加，不回填）
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage1-normalized-clustering-seed3`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；不从任何 exp 分支拉出；`aliccp_benchmark/protocol.py` 零修改；`multitaskrec/*` 零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（下称"协议"）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（预算、指纹、A/B 门禁阈值、SUMMARY 列均不变）。
- **前置审计（同批提交，先于任何新 run）**：`docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-mechanism-audit.md`（下称"审计"）——对 seed1/seed2 既有证据的独立复算：**两 seed 机制修复稳定成立（B4 0.0283%/0.0766% FAIL → 47.96%/55.31% PASS；退化签名 bal=0.5、recall(env_0)=0 消失）；效用 seed1 正（Δtest +0.0082428399、Δval +0.0100592719）、seed2 近零且 val 反向（Δtest +0.0004588195、Δval −0.0014831645）**。
- **被判定对象（只读引用）**：seed1 修复 @ `exp/aliccp-stage1-normalized-env-clustering` `2058de8`（`REPAIRED` + `NO_MATERIAL_DEGRADATION`）；seed2 复现 @ `exp/aliccp-stage1-normalized-clustering-seed-replication` `7ab445c`（机制 10/10，效用方向未复现，`NOT_SUPPORTED`）。
- **定位声明**：对同一工程修复（scale-correction ablation，无超参、无学习参数、无新颖性主张）的**第三 canonical seed 判定**。成功/失败的分类规则由用户在指示中预先给定（§6.2），本文件只做机械执行与如实报告；**不主张性能提升**。

---

## 0. 目的与判定问题

- **Q1（机制）**：在 seed3 上，逐字节移植的 rank/quantile 归一化聚类是否再次复现修复形态（B4 双过、非标签恢复、有限诊断、两环境 recall>0、有效变化、默认关恒等、A 类完整、环境逐位可复现）？
- **Q2（效用分类）**：seed3 的**同 seed 配对** Δtest = test_bsi(norm) − test_bsi(raw) 落入用户三分法的哪一档（§6.2）？
- **Q3（扩展门）**：seed3 是否满足"扩展 seeds4/5"的冻结条件（§6.4）？若满足，本分支**不**启动 seeds4/5，只登记下一步独立动作；若不满足，按 §6.5 记录收束结论并对 NO_CLEAR 情形做 headroom 评估。

Q1/Q2/Q3 独立判定；组合规则见 §6。

---

## 1. 机制实现审计与逐字移植（pin @ `2058de8`）

### 1.1 移植范围与字节钉死

机制本体与其测试基础设施按 `2058de8`（seed1 修复 tip，已被 seed2 复现分支以同样方式钉死）**逐字节**移植（`git checkout 2058de8 -- <path>`，不手改）。sha256（LF 归一化 = `git show 2058de8:<path>` 字节）与 git blob：

| 文件 | sha256 @ `2058de8` | git blob | 角色 |
|---|---|---|---|
| `aliccp_benchmark/normalized_clustering.py` | `e87740a0b9cbd082b99e2538b6bd72b3103923055b76f78c877a37a8a36e7f95` | `1a68c0b48dad7b30b849a6cb9582fc17041cfbd1` | 机制本体（`per_task_rank01` + `rank_normalized_cluster_2` + manager 仅覆写 `cluster_2`） |
| `aliccp_benchmark/bench.py` | `ee28543255dd8d6b6c629adefd5ae94522bf753f0ba9b932f925b5e180841710` | `9c01947e1277e2e0e57c9215281ec8ad097411c6` | `clustering_arm=None` 默认路径逐位不变；arm 注入 + cfg/meta 标记 + balanced-accuracy 探针 |
| `aliccp_benchmark/metrics.py` | `f29eb951fb0ee2ba1b768d13c740eda5b6a2d0de11e370cab4c0b9878ca0bf8e` | `491b0847f3c9b35f622ba009b262e364763bd8ff` | `env_balanced_accuracy`（macro recall） |
| `run_aliccp_benchmark.py` | `534be3ca65191c2ab55898aa383e76c820691d3409e9072646856c11bbf8bb4e` | `74230871bf39d79e7203fb5ffcbdaff812ce7e76` | `--clustering {raw,rank_normalized}`（默认 raw）、`--tag norm` |
| `aliccp_benchmark/audit_cluster_losses.py` | `335383aa3da1ffb88fd4d50bcb7013dc5e95dce18bfb403e134db0fdc17e03f3` | `de65b2c6a817a139f6ac58fd77b7f95b94a879e9` | 只读审计工具（`ref_rank01` 被测试逐位对照引用；P4 捕获与重分析用） |

- **归一化规则与阈值零改动**：`average_rank → q = (rank − 1)/(N − 1)`、全并列 0.5、N=1 → 0、argmin 取先索引、有限性断言；容差 `0.005`/`0.01`/环境占比 `5%` 全部沿用（§4）。
- **`test_normalized_clustering.py`（@ `2058de8` sha256 = `652fed5a70c66c548cb886e6bc9b17fa8089234acd8a5f3e9377335ef9a76817`）**：随机制一并移植，**仅 2 处**分支适配（§8.2 由 AST 守卫逐项钉死为恰此 2 处）：(a) 模块级 `DOC_PATH` → 本文件；(b) `TestStaticGuards.test_tracked_changes_subset_of_whitelist` 白名单 += 本分支 5 个新文件。其余逐字保留：seed1 的默认协议身份钉死（`BASELINE_CFG_HASH` = `3a30e2c0b1e8a2b4e9fecaa4d76893b775f6ee9e597922dce7303ac3b77b08c4`、`TREATMENT_CFG_HASH` = `ad3b353f6d436ac95c26703e504a318e3441933a4e00e3f3d6c5e22a05bfe96e`、`PREDICTED_STAGE1_ID` = `s1-5c060b9c-m1688723512-e3-ad3b353f`）与文档 token 断言（`s1-5c060b9c-m1688723512-e3-ad3b353f`、`4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b`、`999966`/`959244`/`1040756`、基线值 `0.5481837665250074`/`0.5988392178311113`/`0.5781533414372665`、标签 `REPAIRED`/`NOT_REPAIRED`/`NO_MATERIAL_DEGRADATION`）对本文档继续成立（§2 引用这些钉死值）。
- **禁止**：`multitaskrec/*`、`config.py`、`AliCCP_*.py`、`baseline/*`、`protocol.py` 相对 `8133d32` 零 diff（静态守卫，§8）。

### 1.2 seed 与配置（固定，先于 run）

| 项 | 值 |
|---|---|
| MODEL_SEED | `1688738016`（canonical 列表 `[1688723512, 1688723740, 1688738016, 1688749593, 1688762746]` 位置 3；`AliCCP_NewTask.py` 仓库默认） |
| ENV_SEED | `20261003`（与全部既有 run 相同，协议常量） |
| 预算/前缀 | `p2M-v500k-t1M`（train 2,000,000 / val 500,000 / test 1,000,000；前缀指纹 `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`，与两既有 seed 相同） |
| Stage-1 | 3 epoch / patience 2 / batch 2000 / lr 1e-4 / uni_coe 0.9 / env_coe 0.1（协议默认，不变） |
| Stage-2 | 5 epoch / patience 2（协议默认，不变；未改动头） |

---

## 2. 配对比较定义（seed3 的"唯一有意差异 = 聚类规则"）

**四元结果 run（每项恰 1 次）**，全部在本分支、同一工作树、同一环境：

| # | 命令（留档） | 角色 |
|---|---|---|
| R1 | `python run_aliccp_benchmark.py stage1 --tag short --model-seed 1688738016` | raw 聚类 Stage-1（配对基线 backbone） |
| R2 | `python run_aliccp_benchmark.py stage1 --tag norm --clustering rank_normalized --model-seed 1688738016` | 归一化聚类 Stage-1（处理 backbone） |
| R3 | `python run_aliccp_benchmark.py stage2 --tag short --stage1-id <R1 sid> --model-seed 1688738016` | 未改动头，跑在 R1 backbone 上 |
| R4 | `python run_aliccp_benchmark.py stage2 --tag norm --stage1-id <R2 sid> --model-seed 1688738016` | 未改动头，跑在 R2 backbone 上 |

**同一性清单（R1↔R2 与 R3↔R4 之间必须完全一致；唯一有意差异 = Stage-1 聚类赋值规则）**：

- 数据集前缀与指纹：同 `p2M-v500k-t1M`、同前缀字节（A2 每次运行重算）；split 无种子（前缀确定性）；
- 随机性来源：同 model seed `1688738016`、同 env seed `20261003`（`make_env_ids` 独立 Generator，不消耗全局 RNG）；
- 架构/优化器/批/预算：同 `MPTRec(num_tasks=2)` 论文超参、Adam `1e-4`、batch 2000、2M/500k/1M、3 epoch/patience 2（Stage-1）、5 epoch/patience 2（Stage-2）；
- Stage-2 过程：同一 `bench.run_stage2` 代码路径、同 NewTask 头（`rep_dim=64`、tower `[32,32]`、`reg_dnn=7e-6`、BCELoss + l2）、真冻结三件套、val 选点/test 单评口径完全一致；两 run 的 Stage-2 代码相同（§9.4 记录各自 commit 与零代码 diff 证据）；
- Stage-1 config 的唯一差异：R2 的 config = R1 的 config + 恰一个键 `{"clustering": "rank_normalized"}`（测试钉死，§8）；内容寻址产生**不同** `stage1_id`（不同产物目录，禁止覆盖）；
- 两 Stage-1 进程在**首次聚类调用（epoch 2 末）之前**代码路径完全一致 → epoch 1–2 逐 epoch 记录（14 值）预测**逐位相等**（P3a）。

**配对判定量**：`Δtest = R4.test_bsi − R3.test_bsi`；`Δval = R4.best_val_bsi − R3.best_val_bsi`；Stage-1 侧 `Δtest_ctr = R2.test_ctr − R1.test_ctr`（U1）。跨 seed 比较**只在各 seed 自己的配对内**计算，跨 seed 不混合。

**记录项（每题必录，§9）**：环境占比（事件 env_0/env_1 计数与占比）、标签比例（env × click/purchase 交叉表）、归一化损失分布（`cluster_diagnostics` 的 raw/normalized 分位数）、聚类可预测性/退化诊断（env_acc / env_bal_acc / 各环境 recall；B4 门禁）、Stage-2 routing use（`gate_mean` 两维）、下游 BSI 的 test/val AUC 对（raw 与 norm 各一对 + Δ）、逐 epoch 轨迹、停止/截断（early stop epoch、best epoch）、门禁与失败、跨 seed 表。

---

## 3. 冻结预测（看到结果前写死；P4 的登记时点见 §3.1）

| 编号 | 预测 | 依据 |
|---|---|---|
| P1 | R1 `stage1_id` = **`s1-5c060b9c-m1688738016-e3-47619ce0`**（config hash `47619ce078ac75497b4da9a81f0db9de59d738f22759702788f0b4ae47640108` = 协议默认 cfg，`model_seed=1688738016`）；R2 `stage1_id` = **`s1-5c060b9c-m1688738016-e3-5f899cad`**（config hash `5f899cad0d89972af1ecd2280d9933384ddcad9db84d06069a33a0fd4843f9d3` = 协议默认 cfg + `{"clustering":"rank_normalized"}`） | 确定性哈希（实现前已复算，§8 测试钉死） |
| P2 | R3/R4 的 run 记录 `stage1_id` 分别 == P1 的两值；两者 A5/A6 逐位通过 | 内容寻址 + 冻结校验 |
| P3 | (a) R1 与 R2 的 epoch 1–2 逐 epoch 记录（7 值 × 2）**逐位相等**；(b) R1、R2 各恰 1 次聚类事件、记录 epoch = 2 | 首个 `cluster_2` 调用在 epoch 2 训练后、其 val 评估前；此前两臂代码路径一致（arm manager 仅覆写 `cluster_2`，不消耗 RNG、不改权重） |
| P3R | （方向性期望，非 bit 断言，不作门禁）R1 的 raw 聚类事件延续既有形态：`env_0` 占比 ≪ 5%（标签主导、B4 FAIL） | 机制审计（§前置审计 2.1/2.2）；seed3 为未知量，如实记录 |
| **P4** | （**在 R1 的认证捕获后、R2 运行前**登记，§3.1；已于 2026-10-05 冻结，派生 commit `69ae804`）R2 聚类事件 = **`diff_num = 999300`、`env_0 = 937876`、`env_1 = 1062124`**；`env_ids_sha256` = **`b60757be2e85ff8e621ac4093b30227099ed925fd5e126b08d7b8df0398ba978`**；环境占比 **46.8938% / 53.1062%**；`env_0∩purchase1 = 562`（另 `env_0∩click1 = 4219`、≠ purchase1 集合、非子集） | R1 `--reproduce` 认证捕获（10/10）的聚类时刻损失向量上，`per_task_rank01` 与 `ref_rank01` 两实现逐位一致的 argmin（谓词 `bitwise_agree=true`；登记文件 `artifacts/aliccp_bench/audit/s1-5c060b9c-m1688738016-e3-47619ce0-repro/p4_prediction.json`） |

### 3.1 P4 的登记程序（写死；无 bit 捕获则按 §4.3 降级）

1. R1 完成后，运行只读审计复现：`python -m aliccp_benchmark.audit_cluster_losses --stage1-dir artifacts/aliccp_bench/stage1/s1-5c060b9c-m1688738016-e3-47619ce0 --out-dir artifacts/aliccp_bench/audit/s1-5c060b9c-m1688738016-e3-47619ce0-repro --reproduce`；
2. 复现校验必须 10/10 逐位一致（cluster 事件、逐 epoch、best epoch、backbone sha、test AUC、env_ids sha、捕获 argmin 与父类分配一致、捕获分配 sha == meta）；此时捕获的 `cluster_losses.pt` 为**聚类时刻真值**；
3. 在捕获损失上分别用 `normalized_clustering.per_task_rank01` 与 `audit_cluster_losses.ref_rank01` 计算 `argmin` 分配（两者必须逐位相等），导出预测事件三元组、`env_ids_sha256`、占比、`env_0∩purchase1`（需要标签 → 只读扫描训练前缀）；
4. 将上列预测值填入本文件 §3 的 P4 行并**提交（C3）**；
5. 然后才运行 R2；R2 完成后逐位核对（M2c）。
6. 若复现未达 10/10：P4 记 `INVALID_FOR_PREDICTION`（不登记预测值），M6 FAIL、M2c 未判定，**照常运行 R2**（结果 run 计划不变），机制结论按 M 组如实报告。

---

## 4. 预注册门禁（看到结果前写死；看到结果后不得修改）

### 4.1 机制门禁（M 组）

| 编号 | 判据 |
|---|---|
| M0 | **前置审计**：`audit_prior_seeds` 对 seed1/seed2 四臂 `all_pins_pass = true`（已完成，见审计文档） |
| M1 | **B4 修复**：R2 恰 1 次聚类事件且两环境各 ≥ 5%·N（协议 `ENV_SHARE_MIN`，不改）；R4 的 `gate_report` B4 同口径 PASS |
| M2 | **身份与轨迹**：(a) R1/R2 `stage1_id` == P1；(b) R1 与 R2 epoch 1–2 记录（14 值）逐位相等；(c) R2 事件三元组 == P4 ∧ R2 `env_ids_sha256` == P4（P4 有效时；否则此项未判定并按 §4.3 记录） |
| M3 | **有限性**：R2 `cluster_diagnostics` 全部 raw/normalized 分位数有限（无 NaN/Inf）；运行未抛有限性断言 |
| M4 | **非标签恢复（norm）**：`env_0∩purchase1 ≤ 5%·|env_0|` ∧ `env_0 ≠ purchase1 集合`（逐位不等）∧ 非子集；raw 对照（R1）照实记录（不判定） |
| M4b | **退化签名消除（norm）**：`recall(env_0) > 0 ∧ recall(env_1) > 0`（聚类可预测性探针；raw 对照期望 bal≈0.5、recall(env_0)=0，两既有 seed 均如此） |
| M5 | **有效变化**：R2 事件 `diff_num ≥ 5%·N`；R2 vs R1 `env_ids` 逐位差 ≥ 5%·N |
| M6 | **环境逐位可复现**：R1 `--reproduce` 10/10（§3.1 第 2 步）；捕获 argmin 与父类分配 mismatch == 0 |
| M7 | **A 类完整性**：R3/R4 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP，协议口径） |
| M8 | **默认关恒等与静态守卫**：(a) 全测试套件绿（移植不变量 + 字节钉死 + AST 守卫 + seed3 身份/文档钉死）；(b) smoke 规模默认路径恒等核验（temp root、非结果 run）：meta 科学字段与既有记录产物 `s1-550e4d92-m1688723512-e1-d069eadf` 逐位一致（仅 commit/git/versions/wall/device/peak_vram 允许差异） |

- `MECHANISM_REPAIRED_SEED3 ⇔ M1∧M2∧M3∧M4∧M4b∧M5∧M6∧M7∧M8`；否则逐项如实报告。

### 4.2 效用（U 组；within-seed3 配对；容差先于 run 写死）

| 编号 | 判据 | 容差依据 |
|---|---|---|
| U1 | Stage-1 CTR 无实质退化：`Δtest_ctr ≥ −0.005`（CVR 记录不判定） | 同 seed1/seed2 口径（1M 前缀 47k 正例，SE≈0.0024） |
| U2 | **主分类量** `Δtest = R4.test_bsi − R3.test_bsi` 按 §6.2 三分法分类 | 用户预注册规则（§6.2） |
| U3（报告项，不判定） | `Δval`、逐 epoch val 轨迹、`gate_mean`、B1/B2/B3 原样披露；Δval 与 U1 参与 §6.3 的 material contradiction 规则 | — |

### 4.3 预测/复现失败处置

- M2c 未判定 ⇒ 该分量照实报告（机制结论以其余 M 判据为准）；**不重跑**。
- M6 失败（环境不可逐位复现）⇒ `INVALID_FOR_PREDICTION`；R2 照跑；机制结论降级为"模式级复现"并如实披露。
- 无效执行（崩溃/中断/产物不完整）仅因工具性原因可重跑一次并记录；**结果原因（门禁未过/分类不利）不构成无效，不重跑**。无调参、无 epoch 挑选、test 不参与任何选择。

### 4.4 纪律

不得只报通过的门禁；不得事后改判据/容差/分类阈值；不得挑选 epoch/seed；全部 run 的 `git.dirty` 必须为 false（§7）。

---

## 5. 计划外零容忍

本分支**不做**：seed2 之后的任何重跑；任何超参/结构/预算改动；其他数据集；Residual Prompt/Census/论文/图表/novelty 工作；`seeds4/5` 的实际运行（仅在 §6.4 条件下登记下一步，不执行）。**seeds4/5 绝不在本分支启动。**

---

## 6. 判定与收束规则（先于 run 写死）

### 6.1 机制判定

`MECHANISM_REPAIRED_SEED3`（§4.1）/ 否则逐项如实报告（含"机制在 seed3 未复现"的可能）。

### 6.2 效用主分类（用户预先给定；机械执行）

以 `Δtest = R4.test_bsi − R3.test_bsi` 分类：

| 条件 | 分类标签 |
|---|---|
| `Δtest ≥ +0.001` | `POSITIVE_IMPROVEMENT` |
| `−0.02 < Δtest < +0.001` | `NO_CLEAR_IMPROVEMENT` |
| `Δtest ≤ −0.02` | `CLEAR_DEGRADATION` |

### 6.3 material contradiction 规则（用于 §6.4；先于 run 写死）

在 `Δtest ≥ +0.001` 的前提下，称存在 **material contradiction** 当且仅当任一项成立：
(a) `Δval ≤ −0.01`（验证方向强负，seed1 修复容差量级）；或
(b) 任一 M 组门禁 FAIL；或
(c) U1 FAIL（Stage-1 CTR 退化超容差）；或
(d) R3 或 R4 的任一 A 类门禁 FAIL（A3 除外，协议口径）。

### 6.4 扩展 seeds4/5 的冻结条件与下一步登记

若 `Δtest ≥ +0.001` **且**无 material contradiction ⇒ 登记：**"seed3 条件通过"**；本分支**停止**，只记录**确切的下一步独立/续行动作**（供自动化执行）：在新分支（自本分支 tip 拉出）按本文件冻结协议对 canonical seed 4/5 = `1688749593`、`1688762746` 各跑 1+1+1+1（同 §2 配对定义、同门禁、同分类规则），并同样以"逐字节移植 + 同配对"方法推进；不合并、不改判据。

若条件不满足（`Δtest < +0.001` 或存在 material contradiction）⇒ 按 §6.5 收束。

### 6.5 收束结论（条件不满足时）

- 若 `MECHANISM_REPAIRED_SEED3`：记录 **"mechanism repair effective but utility unstable"**，并附 §6.6 headroom 评估；本线关闭（不再跑 seed）。
- 若机制未复现：如实记录"机制在 seed3 未复现"及失败分量；本线关闭。
- 两种情况下均不启动 seeds4/5、不调参、不重跑。

### 6.6 headroom 评估（仅 NO_CLEAR；输入与旗标先于 run 写死）

四组输入（全部来自 §9 记录，不引入新 run）：

| 编号 | 输入 | 判定 |
|---|---|---|
| H1 | 机制活性 | M 组全过 → favors；任一 FAIL → contradicts |
| H2 | 验证方向 | `Δval ≥ 0` → favors；`−0.01 < Δval < 0` → ambiguous；`Δval ≤ −0.01` → contradicts |
| H3 | seed1/2/3 一致性 | `Δtest_seed3 > 0` ∧（三 seed 中 Δtest>0 的个数 ≥ 2）∧（三 seed 平均 Δtest > 0）→ favors；否则 not |
| H4 | 预算轨迹（可得时） | 两臂 `best_epoch == 5`（预算受限、epoch 5 仍在上升）→ favors 长预算方向；否则照实记录 |

**结论标签**：`HEADROOM_PLAUSIBLE ⇔ H1 favors ∧ H2 ≠ contradicts ∧ H3 favors`；否则 `HEADROOM_NOT_EVIDENT`（H4 仅描述）。

---

## 7. 运行计划与清洁树纪律

**结果 run（恰 4 次，不重跑）**：R1 → R2 → R3 → R4（§2）。审计复现 ×1（§3.1，非结果）；smoke 恒等核验 ×1（M8b，非结果，temp root）；跑后只读核验（交叉表/探针/预测核对，非结果）。

**提交链（run 间提交；SUMMARY 追加行必须在下一 run 前提交——memory 纪律）**：

1. **C1**：本文件 + 审计文档 + `audit_prior_seeds.py` + `.gitignore`（audit/ 规则）→ 树干净；
2. **C2**：移植（§1.1 字节钉死）+ 移植测试（只 2 处适配）+ seed3 守卫测试 + `verify_seed3_results.py`（TDD：守卫先红后绿）→ 树干净；
3. 跑 **R1**（`git.dirty=false`，meta commit = C2）→ 审计复现（§3.1）→ 填 **P4** → **C3**（仅文档）→ 跑 **R2**（`dirty=false`，meta commit = C3）→ 核验 M2/M3/M5 + §9.1 → **C4**；
4. 跑 **R3**（`dirty=false`，commit = C4）→ **C5**（R3 记录：SUMMARY 行 + 文档；R3 的 SUMMARY 行追加后再提交，保证 R4 运行前树干净）→ 跑 **R4**（`dirty=false`，commit = C5）→ 核验 M1/M4/M4b/M7 + U 组 + §6 分类 → **C6**（§9.2/§9.3 + SUMMARY 最终行 + 验证 JSON）→ push。

- Stage-1 不写 SUMMARY（协议现状：仅 stage2 追加）；SUMMARY 只追加、永不重写。
- Stage-2 代码在 C4 与 C5 之间零 diff（差异仅为文档/SUMMARY）；§9.4 记录该证据。

---

## 8. 实现面（TDD；默认关逐位一致）与静态守卫

| 文件 | 改动 | 说明 |
|---|---|---|
| `aliccp_benchmark/normalized_clustering.py`、`bench.py`、`metrics.py`、`run_aliccp_benchmark.py`、`audit_cluster_losses.py` | 逐字节移植 @ `2058de8`（sha 见 §1.1） | 机制本体与基础设施 |
| `aliccp_benchmark/audit_prior_seeds.py`（C1 新增） | 只读审计（§前置审计） | seed1/seed2 证据复算 |
| `aliccp_benchmark/tests/test_normalized_clustering.py`（新） | 移植 @ `2058de8` + 恰 2 处适配（§1.1） | 原机制不变量测试全部保留 |
| `aliccp_benchmark/tests/test_normalized_clustering_seed3.py`（C2 新增） | 本分支守卫：字节钉死（§1.1 六文件 sha/blob）、AST 守卫（移植测试 == seed1 版 + 恰 2 处适配）、seed3 身份钉死（P1 两 id/哈希）、本文档 token 钉死 | §8.1–8.3 |
| `aliccp_benchmark/verify_seed3_results.py`（C2 新增） | 跑后只读验证 + §6.2/§6.3/§6.6 机械分类（读四产物 + 训练前缀标签扫描） | 独立验证与机械判定 |
| `.gitignore` | 增 `artifacts/aliccp_bench/audit/`（与两既有分支同型） | 审计输出不入库 |
| `artifacts/aliccp_bench/SUMMARY.md` | R3/R4 后各追加 1 行 | append-only |
| `docs/superpowers/specs/2026-10-05-*`（本文件 + 审计文档） | 预注册 + 审计 | — |

**明确不改**：`multitaskrec/*`（零 diff）、`config.py`、`AliCCP_*.py`、`baseline/*`、`aliccp_benchmark/protocol.py`（零 diff）、协议文档与两个既有分支的设计文档（只读引用）。禁止 `git add -A`；提交只允许显式路径。

### 8.1 字节钉死（守卫测试）

对 §1.1 的**五个机制文件**逐一断言 `sha256(工作树文件) == 2058de8 钉死值`（LF 归一化；含 git blob 对照）。任一文件与 seed1 修复不等价即红。**移植的测试文件**因含 §1.1 声明的 2 处适配、不可能等于 seed1 字节，改由 §8.2 的 AST 守卫 + "≠ seed1 原版 sha256"记录覆盖（守卫测试内非钉死断言）。

### 8.2 AST 守卫（移植测试文件 == seed1 版 + 恰 2 处适配）

逐类名/逐方法名集合相等 + 逐成员 `ast.dump(..., include_attributes=False)` 相等；模块级仅 `DOC_PATH` 豁免；差异集合必须恰等于 `{("TestStaticGuards","test_tracked_changes_subset_of_whitelist"), 模块级 DOC_PATH}`；适配后白名单 == seed1 白名单 ∪ §8 的 5 个新文件。种子 commit 不可得时响亮失败（不静默跳过）。

### 8.3 seed3 身份与文档钉死

`_stage1_cfg`（协议默认，`model_seed=1688738016`）哈希与 +`clustering` 哈希 == P1 两值 ⇒ 两 `stage1_id` == P1；本文档必须包含 token：P1 两 id/两哈希、`1688738016`、seed2 参考值（`0.5534062000766764`/`0.5420522697385088`/`0.5974422649550507`/`0.5809347091990792`/`+0.0004588195`/`−0.0014831645`）、seed1 参考值（`0.5481837665250074`/`0.5988392178311113`/`0.5781533414372665`/`+0.0082428399`/`+0.0100592719`）、事件计数（`566`/`1532`/`999966`/`959244`/`1040756`）、阈值（`+0.001`/`−0.02`/`0.005`/`0.01`）、分类标签（`POSITIVE_IMPROVEMENT`/`NO_CLEAR_IMPROVEMENT`/`CLEAR_DEGRADATION`）、机制标签（`MECHANISM_REPAIRED_SEED3`）、收束标签（`mechanism repair effective but utility unstable`）、headroom 标签（`HEADROOM_PLAUSIBLE`/`HEADROOM_NOT_EVIDENT`）。

### 8.4 继承不变量

I1–I9（rank01 语义、实现==审计参考逐位、manager 仅覆写 `cluster_2`、默认关恒等与 cfg 哈希、CLI 接线、静态守卫、seed1 文档 token）全部保留（移植文件内逐字不变）。

---

## 9. 结果（实测；运行后追加，不得回填预期值）

### 9.0 预运行核验（实测）

- **前置审计（C1）**：`audit_prior_seeds` 四臂 `all_pins_pass = true`（wall 79.3 s）；详见审计文档与 `artifacts/aliccp_bench/audit/prior-seeds/audit_prior_seeds.json`。
- **移植后审计复核（C2，`--seed1-capture`）**：seed1 认证捕获上 rank01 两实现逐位一致，复算分配 `959244/1040756`、`env_ids_sha256 = 4b983fc9…d52b` 与 seed1 归一化臂实测**逐位相等**（wall 78.7 s；`audit_prior_seeds_with_capture.json`）。
- **测试**：移植后全测试套件 **64/64 通过**（37 基线 + 19 移植 + 8 seed3 守卫；TDD 先红（8 tests, 2 failures + 4 errors）后绿）。
- **R1 的认证复现（§3.1 第 1–2 步；非结果 run）**：`all_pass=True (10/10)`（cluster 事件、逐 epoch、best epoch、backbone sha、test AUC、env_ids sha、捕获 argmin 与父类一致、捕获分配 sha == meta 全逐位）；捕获损失向量为聚类时刻真值。
- **P4 冻结（C3）**：见 §3；派生时两独立实现逐位一致（`bitwise_agree=true`）。
- **M8b smoke 恒等（C4；temp root，非结果 run）**：**PASS** —— `stage1_id` 内容寻址一致（`s1-550e4d92-m1688723512-e1-d069eadf`），除 `commit/git/versions/device/wall/peak_vram` 外全部科学字段与既有记录产物 meta **逐位一致**（0 diff）。**工具性修正披露**：首跑误用 CPU（记录产物为 `cuda:0`，跨设备浮点路径不同 → 差异 ~1e-4 量级、非逐位口径）；`verify_seed3_results._smoke_identity` 增加同设备断言后于 `cuda:0` 复跑通过（该修正为验证脚本工具修正，先于 R3/R4 任何结果，门禁/预测未变）。

### 9.1 Stage-1（R1/R2；实测，各运行恰一次）

**结论：M2（a/b/c）、M3、M5、P3R（方向性）全部命中；R2 事件与 P4 逐位相等（bit 级预测命中）。**

| 项 | R1（raw，`…-47619ce0`） | R2（norm，`…-5f899cad`） |
|---|---|---|
| stage1_id | `s1-5c060b9c-m1688738016-e3-47619ce0`（== P1） | `s1-5c060b9c-m1688738016-e3-5f899cad`（== P1） |
| commit / dirty / wall | `69ae804` / **false** / 222.2 s | `0e98345` / **false** / 254.2 s |
| 聚类事件（epoch 2，恰 1 次） | 1000467 / **567** / 1999433（env_0 占比 **0.02835%**，B4 FAIL） | 999300 / **937876** / 1062124（**46.8938%** / 53.1062%，B4 PASS；**== P4 逐位**） |
| epoch 1–2 记录（14 值） | — | **与 R1 逐位相等**（M2b；如 ep1 CTR `0.5300389834345405`、ep2 CTR `0.5263121746391954`、env_loss `1.0085222721099854`/`1.0006557703018188`） |
| epoch 3（治疗臂特有轨迹） | val CTR `0.5391685215033852` / CVR `0.5197565004111302`；env_loss `0.013464018702507019` | val CTR `0.5443114258164544` / CVR `0.5208893234133161`；env_loss `0.6862123012542725`（平衡分配下 env 头不再可平凡拟合） |
| best_epoch / best val | **1** / 0.5300389834345405（CTR）、0.5302680320389357（CVR） | **3** / 0.5443114258164544（CTR）、0.5208893234133161（CVR） |
| test（单次） | CTR `0.53167162252711`、CVR `0.5655139568464099` | CTR `0.5408833390410248`（**Δtest_ctr = +0.0092117165**，U1 大余量 PASS）、CVR `0.555491562797491`（记录不判定） |
| env_ids sha256 | `a175b93f1b3c43a3e7d0c6205931c2ba4bc26c218baeb50ef04f3f3117810ddb` | `b60757be2e85ff8e621ac4093b30227099ed925fd5e126b08d7b8df0398ba978`（**== P4**，M2c） |
| `cluster_diagnostics[0]`（聚类时刻） | 不适用（raw 臂记录原损失，未存分位数） | raw 中位 CTR `0.0621705353` / CVR `0.0001067370`（尺度差复现）；normalized 中位 `0.50000075` / `0.5`（rank 语义自检）；全分位数有限（M3 PASS） |
| 标签比例（env × 标签，独立复算） | env_0=567（含 purchase **566**、click 566；≈标签函数） | env_0=937876（含 purchase **562** = 0.0599%·\|env_0\| ≤ 5%；click 4219；≠ purchase1 集合、非子集 → **M4 PASS**） |
| env_pred 探针（best 权重，前 400k 行） | acc `0.42902`（best=epoch 1，聚类前权重，非退化签名口径） | acc `0.6167675`（== meta 记录）、bal `0.596992`、recall(env_0) `0.223029` > 0、recall(env_1) `0.970956` > 0 → **M4b PASS** |
| 有效变化（M5） | — | 事件 diff_num 999300 ≥ 5%·N；**R2 vs R1 env_ids 逐位差 937317 行（46.866% ≥ 5%）** |
| 认证复现（M6） | **10/10**（§9.0） | 不适用 |

- **P3R（方向性期望，非门禁）命中**：raw 臂事件延续既有形态（env_0 占比 0.028%、566/567 为 purchase 正例、B4 FAIL）。
- **披露（R1 的 best_epoch=1）**：seed3 raw 臂的 val 选点在 epoch 1（epoch 1–3 的 sum(AUC_val)：1.06031 / 1.05845 / 1.05893，最大值在 epoch 1）——与 seed1/seed2 raw 臂（best=3）不同；即 R3 将跑在"聚类发生之前"的 backbone 上，其 env 头对照为初始 env_ids 分布。属如实记录，配对判定不受影响（两臂同规则选点）。
- **M1** 的 R4 侧子句（gate_report B4 PASS）待 R4；**M7** 待 R3/R4。

### 9.1 Stage-1（R1/R2；运行后填写）

（含 P4 登记值、事件/占比/交叉表/探针/诊断分位数、逐位核对。）

### 9.2 Stage-2（R3/R4；实测，各运行恰一次，未改动头）

| 项 | R3（raw 配对） | R4（norm 配对） |
|---|---|---|
| run_id | `20261005-1341-p2M-v500k-t1M-m1688738016-short-a3a3e37` | `20261005-1345-p2M-v500k-t1M-m1688738016-norm-18a7008` |
| commit / dirty | `a3a3e37` / **false** | `18a7008` / **false** |
| stage1_id | `s1-5c060b9c-m1688738016-e3-47619ce0`（A5/A6 逐位通过） | `s1-5c060b9c-m1688738016-e3-5f899cad`（A5/A6 逐位通过） |
| 逐 epoch val BSI | 0.5063601783651219 / 0.521297223263171 / 0.5416431064878608 / 0.564903230805078 / 0.5839919245306395 | 0.5198187898322179 / 0.5382744235599318 / 0.5628996505080943 / 0.584755380823985 / 0.6015494613337896 |
| best_epoch / best val BSI | 5 / `0.5839919245306395` | 5 / `0.6015494613337896`（**Δval = +0.017557536803150087**） |
| test BSI（单次） | `0.6169265645506535` | `0.6404054911335456`（**Δtest = +0.02347892658289208**） |
| gate_mean（val，routing） | `[0.5505214375, 0.44947846875]` | `[0.544646375, 0.4553536875]` |
| A 类（M7） | A1/A2/A4/A5/A6 **PASS**（A3 SKIP） | A1/A2/A4/A5/A6 **PASS**（A3 SKIP） |
| B 类 | **B1 FAIL**（CTR-val 0.5300 < 0.55 活动下限；CVR/BSI 子句过）、B2 PASS（\|val−test\|=0.0329）、B3 PASS、**B4 FAIL**（`[(567, 1999433)]`，raw 退化，期望） | **B1 FAIL**（CTR-val 0.5443 < 0.55，同为活动下限；两臂 CTR-val Δ = +0.0143）、B2 PASS（0.0389）、B3 PASS、**B4 PASS**（`[(937876, 1062124)]`；协议 machinery 确认修复） |
| hard_pass | false（唯一原因 B1 活动下限；B4 FAIL 为 raw 臂特性） | false（唯一原因 B1 活动下限；**该 run 的 B4 首次在 seed3 通过**） |
| 停止/截断 | 无 early stop（5/5 epoch，best=5，曲线仍在上升） | 无 early stop（5/5 epoch，best=5，曲线仍在上升） |

- **B1 披露（两臂"共享既有边界"）**：seed3 两臂的 CTR-val 均低于协议活动下限 0.55（raw 0.5300 / norm 0.5443）——与 seed1 的 B1 共享边界同型（协议活动下限非本实验的退化判据；见 §4.2/§6.3：B 类不参与 material contradiction 判定，A 类才参与）。CVR-val / BSI 子句均过。
- **routing 观察（描述性）**：两臂 gate_mean 均在 ~0.55/0.45 附近（平衡），无坍缩（B3 PASS 两 run）。

### 9.3 分类与收束（实测；由 `verify_seed3_results.py` 机械计算，输出 `artifacts/aliccp_bench/audit/seed3_verification.json`）

**机制判定：`MECHANISM_REPAIRED_SEED3` = true（14/14 逐项 PASS）**：M1 B4 修复（937876/1062124 双过 + R4 的 B4 machinery PASS）、M2a 两身份 == P1、M2b epoch 1–2 与 R1 逐位相等（14 值）、M2c 事件与 env_ids sha **== P4 逐位（bit 级预测命中）**、M3 有限性、M4 非标签恢复（562/937876 = 0.0599% ≤ 5%）、M4b 两环境 recall 均 > 0（0.223029 / 0.970956）、M5 有效变化（事件 999300；vs raw 逐位差 937317 = 46.866%）、M6 认证复现 10/10、M7 A 类完整、M8a 测试 64/64、M8b smoke 恒等（cuda:0 逐位）。

**效用（within-seed3 配对）**：

| 量 | 值 | 判定 |
|---|---|---|
| `Δtest_ctr`（U1） | **+0.0092117165139148** ≥ −0.005 | **PASS**（Stage-1 CTR 无退化，且大幅为正——记录） |
| **`Δtest`（主分类量）** | **+0.02347892658289208** ≥ +0.001 | **`POSITIVE_IMPROVEMENT`**（§6.2） |
| `Δval` | **+0.017557536803150087** | 报告项；同时进入 §6.3 |

**material contradiction（§6.3，先于 run 写死）**：4 项旗标全部 false —— (a) Δval ≤ −0.01？否（+0.0176）；(b) M 组任一 FAIL？否（14/14）；(c) U1 FAIL？否；(d) A 类 FAIL？否（两 run A 类全 PASS）。**无 material contradiction。**

**扩展门（§6.4）：`Δtest ≥ +0.001` ∧ 无 material contradiction ⇒ `SEED3_CONDITION_PASSED`。** 按预注册：**本分支不启动 seeds4/5**，只登记下一步独立动作（见下）；本分支到此为终点（除记录/推送外不再有 run）。

**跨 seed 表（三次单 run 配对并列；不混合统计）**：

| seed | 聚类事件（diff_num/env_0/env_1） | 占比 | Δtest BSI | Δval BSI | Δtest CTR（Stage-1） | 分类（同一规则） |
|---|---|---|---|---|---|---|
| 1688723512（@ `2058de8`） | 999966 / 959244 / 1040756 | 47.96%/52.04% | +0.0082428399 | +0.0100592719 | −0.0015226 | ≥+0.001 → 正 |
| 1688723740（@ `7ab445c`） | 998878 / 1106286 / 893714 | 55.31%/44.69% | +0.0004588195 | −0.0014831645 | +0.0001965 | NO_CLEAR 档 |
| **1688738016（本次）** | **999300 / 937876 / 1062124** | **46.89%/53.11%** | **+0.0234789266** | **+0.0175575368** | **+0.0092117165** | **`POSITIVE_IMPROVEMENT`** |

- 三 seed 的机制修复（B4 由 ~10⁻³ 量级修复为 ~47–55%）**全部复现**；效用方向 **3/3 为正**（+0.0082 / +0.0005 / +0.0235）。seed3 的量级为三 seed 中最大。
- **headroom 旗标（§6.6，机械记录；本分类为 POSITIVE_IMPROVEMENT，不属 NO_CLEAR 情形，旗标仅备查）**：H1 favors（机制全过）、H2 favors（Δval ≥ 0）、H3 favors（Δtest_seed3 > 0 ∧ 3/3 为正 ∧ 均值 > 0）、H4 favors_longer_budget（两臂 best_epoch = 5 = 预算上限、epoch 5 仍在上升）→ 综合 `HEADROOM_PLAUSIBLE`。

**登记的下一步独立/续行动作（供自动化执行；不在本分支执行）**：

> 自本分支 tip（C6）拉新分支 `exp/aliccp-stage1-normalized-clustering-seeds45`；对 canonical seed4/5 = **`1688749593`**、**`1688762746`** 各执行与本文档完全相同的 1+1+1+1 配对（R1 raw stage1 → 审计复现 + P4 冻结 → R2 norm stage1 → R3/R4 stage2；同 tag/同前缀/同门禁 M1–M8、U1–U3、§6.2 分类、§6.3 material contradiction、§6.4 扩展门、§6.6 headroom 规则）；机制文件继续保持 `2058de8` 字节钉死（守卫测试复刻本分支模式）；协议/阈值/预测登记纪律不变；不合并 master、不修改协议文件；完成两 seed 后按 §6 规则汇总（若 seed4/5 中任一 ≥ +0.001 且无 contradiction，则为"多 seed 支持"证据；若否，按 §6.5 收束）。

### 9.4 纪律核对

- **运行次数**：结果 run = **R1/R2/R3/R4 各恰 1 次**（日志 `20261005-132524-stage1-short`、`…-stage1-norm`、`…-1341-stage2-short`、`…-stage2-norm`）；**无重跑、无无效执行、无调参、无 epoch 挑选、test 不参与任何选择**。非结果核验：seed1/seed2 审计 ×2（C1 base + C2 capture 附加）、R1 认证复现 ×1、smoke 恒等 ×1（另有 1 次无效 CPU 试跑，工具修正后 cuda:0 复跑，见 §9.0）、P4 冻结 ×1、终验 ×1。
- **清洁树**：四条 run 记录 `git.dirty = false`（`69ae804` / `0e98345` / `a3a3e37` / `18a7008`）；run 间提交纪律成立（SUMMARY 行在下一 run 前提交）。
- **commit 链**：C1 `1c07c6e`（审计 + 预注册）→ C2 `69ae804`（移植 + 测试 + 验证脚本）→ **R1** → 复现/P4 → C3 `0e98345`（P4 冻结 + §9.0）→ **R2** → C4 `a3a3e37`（§9.1 + 验证脚本设备修正）→ **R3** → C5 `18a7008`（R3 SUMMARY 行）→ **R4** → C6（本提交：§9.2/§9.3/§9.4 + R4 SUMMARY 行 + 验证 JSON 引用）。
- **R3/R4 零代码 diff**（机械证据）：`git diff a3a3e37 18a7008 -- aliccp_benchmark run_aliccp_benchmark.py multitaskrec config.py baseline` 为空；两 commit 间全部差异 = SUMMARY.md 的 1 行追加。
- **SUMMARY**：本分支追加 **2 行**（R3 short / R4 norm；append-only，未重写任何既有行；系谱 = `8133d32` 的 2 行 + 本 2 行）。
- **测试**：终验内 M8a 复跑 **64/64 OK**；守卫（字节钉死/AST/seed3 身份/文档 token）全绿。
- **验证产物**：`artifacts/aliccp_bench/audit/seed3_verification.json`（机械分类）、`…/p4_prediction.json`（P4）、`…-47619ce0-repro/audit.json`（认证复现 10/10）、`…/prior-seeds/audit_prior_seeds_with_capture.json`（前置审计）。
- **push**：按任务要求，本提交（C6）后立即 push 至 `origin/exp/aliccp-stage1-normalized-clustering-seed3`；远端 tip = C6。

---

## 10. 局限（预声明）

1. 每 seed 每配置 1 run 配对；无 run-to-run 噪声估计（A3 SKIP）；Δ 的绝对值不作显著性主张。
2. seed3 事件 bit 预测（P4）依赖环境逐位可复现（seed1 已证 10/10；本分支 M6 再验）；失败按 §4.3 降级。
3. CVR 指标统计功效极低（test 正例 346），只作记录。
4. 仅 AliCCP 前缀 `p2M-v500k-t1M`；不外推其他预算/数据集。
5. 跨 seed 表为三次单 run 配对的并列，不构成多重比较校正后的推断。

---

## 11. 偏离披露（预声明 + 运行后补）

- **分支工作流偏离**：协议 §4.3 要求"先合并协议再从 master 拉 exp 分支"；本分支自 `infra/aliccp-fair-benchmark` @ `8133d32` 拉出（协议未合并入 master），协议文件零修改、SUMMARY 只追加——与同批 exp 分支先例一致，如实披露。
- **分支本地 SUMMARY 系谱**：本分支 @ `8133d32` 的 `SUMMARY.md` 为 2 行（smoke + raw seed1 short）；本实验只追加本分支自己的 2 行（R3/R4），不复制其他分支行。
- **两阶段 run 的提交差**：R3（commit = C4）与 R4（commit = C5）的差异仅为文档/SUMMARY（零代码 diff，§9.4 记录）；系 SUMMARY 追加纪律（run 间提交）所致。
- **`artifacts/aliccp_bench/audit/` 忽略规则**：与两既有分支同型。
- **P4 的登记时点**：P4 在 R1 捕获后、R2 运行前登记提交（C3）——晚于本预注册文件、早于被预测的 run；机制依赖（环境逐位可复现）与捕获程序已在本文件写死。
- **预运行澄清（2026-10-05，先于任何 run；预测与门禁不变）**：(a) §1.1 补上 `BASELINE_CFG_HASH`/`TREATMENT_CFG_HASH` 的字面值（原仅引用常量名，token 钉死需要字面值）；(b) §8.1 澄清字节钉死的对象为五个机制文件（移植测试文件因 2 处声明适配由 AST 守卫 + 非钉死 sha 断言覆盖）。两项均为守卫测试 red 阶段发现的登记完备性问题，澄清时无任何 seed3 run 存在。
