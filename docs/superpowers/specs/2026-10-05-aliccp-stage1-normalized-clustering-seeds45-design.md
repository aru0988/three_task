# AliCCP Stage-1 归一化聚类的 canonical seed4/5 扩展判定（seed4 = 1688749593；seed5 = 1688762746；含五 seed 终局汇总）

- **状态**：预注册已写死（本文件在本分支任何新 run 之前提交并 push；P4 见 §3.2 的逐 seed 登记时点）；实现与结果见 §9（运行后追加，不回填）
- **日期**：2026-10-05
- **适用分支**：`exp/aliccp-stage1-normalized-clustering-seeds45`（自 seed3 tip `f8ff4cf` 独立拉出；不从其他 exp 分支拉出；`aliccp_benchmark/protocol.py` 零修改；`multitaskrec/*` 零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（下称"协议"）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（预算、指纹、A/B 门禁阈值、SUMMARY 列均不变）。
- **直接前置（只读引用，不可变证据）**：
  - seed1 修复 @ `exp/aliccp-stage1-normalized-env-clustering` `2058de8`（`REPAIRED` + `NO_MATERIAL_DEGRADATION`）；
  - seed2 复现 @ `exp/aliccp-stage1-normalized-clustering-seed-replication` `7ab445c`（机制 10/10，效用 `NOT_SUPPORTED`）；
  - seed3 判定 @ `exp/aliccp-stage1-normalized-clustering-seed3` `f8ff4cf`（`MECHANISM_REPAIRED_SEED3` 14/14；`SEED3_CONDITION_PASSED`；分类 `POSITIVE_IMPROVEMENT`，Δtest `+0.02347892658289208`、Δval `+0.017557536803150087`）；
  - seed3 预注册 §6.4 登记的下一步动作 = 本分支（seed4/5 各跑 1+1+1+1，同配对、同门禁、同分类）。
- **定位声明**：对同一工程修复（scale-correction ablation，无超参、无学习参数、无新颖性主张）的**第四、第五 canonical seed 判定**与**五 seed 终局汇总**。成功/失败均为结果；本文件只做机械执行与如实报告；**不主张性能提升**。
- **纪律**：不得只报通过的门禁；不得事后改判据/容差/分类阈值；不得挑选 epoch/seed；运行不得重跑（§4.3）；全部结果 run 的 `git.dirty` 必须为 false（§6）；提交只允许显式路径（禁止 `git add -A`）。

---

## 0. 目的与判定问题

- **Q1（seed4 机制）**：seed4 上逐字节移植的 rank/quantile 归一化聚类是否再次复现修复形态（B4 双过、非标签恢复、有限诊断、两环境 recall>0、有效变化、默认关恒等、A 类完整、环境逐位可复现）？
- **Q2（seed4 效用）**：seed4 的同 seed 配对 `Δtest = test_bsi(norm) − test_bsi(raw)` 落入三分法（§5.2）哪一档？
- **Q3（seed5 机制）**：同 Q1，seed5。
- **Q4（seed5 效用）**：同 Q2，seed5。
- **Q5（终局汇总）**：五 canonical seed 的配对 Δtest/Δval 汇总统计（均值、样本标准差、正提升计数、符号正计数、最差 seed、配对 95% 置信区间）与耐久性判定：`UTILITY_DURABLE` 或 `MECHANISM_REPAIR_ONLY_UNSTABLE_UTILITY`（§5.6）。任一 NO_CLEAR 结果附 headroom 评估（§5.7）。

Q1–Q5 独立判定；Q5 为终局标签，历史标签（§5.5）保持不变、并列呈现。

---

## 1. 预运行审计（本文件提交前已完成；证据为准入前提）

### 1.1 分支与基线

- 本工作树 `D:\MPT-Rec-three_task\MPT-Rec-exp-aliccp-norm-cluster-seeds45`，分支 `exp/aliccp-stage1-normalized-clustering-seeds45`，起点 = seed3 tip `f8ff4cf`（clean；`git status --porcelain` 空）。
- 远端 `origin/exp/aliccp-stage1-normalized-clustering-seeds45` 在首次 push 前不存在（本分支创建之）。
- 不合并、不触碰 master 与主工作树；全部运行在本工作树内。

### 1.2 环境审计（两解释器实测；决定见 §8.1 偏离披露）

| 项 | C:\...\Programs\Python\Python310\python.exe（任务字面指定的解释器） | D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe（**选定**） |
|---|---|---|
| python | 3.10.11 | 3.10.11 |
| torch / cuda | **2.12.0+cu126**（cuda_avail=True） | **2.6.0+cu124**（cuda_avail=True） |
| 导入本项目栈 | **失败**：`ModuleNotFoundError: No module named 'fvcore'`（`aliccp_benchmark.bench → multitaskrec.train` 导入期依赖） | 成功 |
| 全测试套件 | **无法执行**：`unittest discover` 采集中 `Windows fatal exception: access violation`（pandas 导入路径），退出码 `-1073741819`（0xC0000005） | `Ran 64 tests in 11.011s` → **OK**（exit 0） |
| 记录环境匹配 | **不匹配**：全部既有 canonical run 的 meta 记录均为 `python 3.10.11 / torch 2.6.0+cu124 / cuda 12.4`（协议 smoke、seed1–3） | **逐项匹配**（同上） |
| smoke 恒等核验（与记录产物 `s1-550e4d92-m1688723512-e1-d069eadf` 逐位比较，cuda:0） | 无法执行（同上） | **IDENTITY_PASS**（`stage1_id` 一致、科学字段差异 = ∅；排除字段仅 commit/git/versions/device/wall/peak_vram） |

**决定**：本分支全部运行、验证与审计使用 **`.venv`（torch 2.6.0+cu124）**。理由：Python310 无法执行本项目栈（缺 `fvcore` 且 unittest 采集崩溃），且其 torch 版本与全部既有 canonical run 的记录环境不同——P4 的 bit 级谓词、M2b/M8b 的逐位等式与跨 seed 可比性均定义在记录环境上，换环境属方法学破坏而非工具便利。偏离如实披露（§8.1）；Python310 的失败输出与 `.venv` 的通过输出为本次审计实测证据。

### 1.3 数据与访问

- 数据集位于 `D:\MPT-Rec-three_task\MPT-Rec-exp-aliccp-norm-cluster-seed3\dataset`（真实目录，未被修改）。本工作树经 **junction**（`dataset` → 上述目录）只读访问；`dataset/` 在 `.gitignore` 内（不进 git）。
- 文件大小复核 = 协议审计值：`ctr_cvr.train` 2,473,647,855 / `ctr_cvr.dev` 274,769,757 / `ctr_cvr.test` 2,711,167,840 字节。未下载、未再生、未修改任何数据文件。

### 1.4 M0 前置审计（本次独立复算）

`python -m aliccp_benchmark.audit_prior_seeds --artifacts-root <主工作树>/artifacts/aliccp_bench --artifacts-root <atten-seed2 工作树>/artifacts/aliccp_bench --out artifacts/aliccp_bench/audit/prior-seeds-45/audit_prior_seeds.json`

**结果：四臂（seed1/seed2 × raw/norm）`all_pins_pass = true`（wall 73.0 s）**；复算值：seed1 Δtest `+0.00824283986406793` / Δval `+0.010059271876175169`（B4 修复 confirmed）；seed2 Δtest `+0.00045881952817949934` / Δval `−0.0014831644646635667`（B4 修复 confirmed）。输出在 `artifacts/aliccp_bench/audit/prior-seeds-45/`（gitignore）。

### 1.5 seed3 记录只读复核

- `artifacts/aliccp_bench/audit/seed3_verification.json`（seed3 工作树，只读）：`delta_test_bsi = 0.02347892658289208`、`delta_val_bsi = 0.017557536803150087`、`delta_test_ctr = 0.009211716513914836`、`classification = POSITIVE_IMPROVEMENT`、`M8a/M8b = True` —— 与 seed3 文档 §9.2/§9.3 字面值逐项一致；seed3 两 stage1 产物 meta 的 `versions = python 3.10.11 / torch 2.6.0+cu124 / cuda 12.4`（= §1.2 的记录环境）。
- 本工作树 `f8ff4cf` 的机制文件与其钉死值：五个机制文件逐一 `sha256(工作树, LF 归一化) == 2058de8 钉死值 == git blob`（Python 实测 15/15 通过）。

### 1.6 结论

环境、数据、前置证据与字节钉死全部就绪；本分支可进入 C1（本文件）→ C2（实现）→ 运行阶段。若 §1.4 未达 `all_pins_pass=true` 则不得开始运行（本次已通过）。

---

## 2. 配对比较定义（每 seed 的 1+1+1+1；与 seed3 §2 逐项相同）

**四元结果 run（每 seed 每项恰 1 次）**，全部在本分支、同一工作树、同一环境（`.venv`）。`<S>` = seed4 `1688749593` 或 seed5 `1688762746`；`<R1sid>` = §3 P1 的 raw id；`<R2sid>` = P1 的 norm id。

| # | 命令（留档；`python` = §1.2 选定解释器） | 角色 |
|---|---|---|
| R1 | `python run_aliccp_benchmark.py stage1 --tag short --model-seed <S>` | raw 聚类 Stage-1（配对基线 backbone） |
| R2 | `python run_aliccp_benchmark.py stage1 --tag norm --clustering rank_normalized --model-seed <S>` | 归一化聚类 Stage-1（处理 backbone） |
| R3 | `python run_aliccp_benchmark.py stage2 --tag short --stage1-id <R1sid> --model-seed <S>` | 未改动头，跑在 R1 backbone 上 |
| R4 | `python run_aliccp_benchmark.py stage2 --tag norm --stage1-id <R2sid> --model-seed <S>` | 未改动头，跑在 R2 backbone 上 |

**同一性清单（R1↔R2 与 R3↔R4 之间必须完全一致；唯一有意差异 = Stage-1 聚类赋值规则）**：数据集前缀与指纹（同 `p2M-v500k-t1M`、同前缀字节，A2 每次运行重算）；随机性来源（同 model seed `<S>`、同 env seed `20261003`）；架构/优化器/批/预算（同 `MPTRec(num_tasks=2)` 论文超参、Adam `1e-4`、batch 2000、2M/500k/1M、3 epoch/patience 2（Stage-1）、5 epoch/patience 2（Stage-2））；Stage-2 过程（同一 `bench.run_stage2` 代码路径、同 NewTask 头 `rep_dim=64`/tower `[32,32]`/`reg_dnn=7e-6`/BCELoss + l2、真冻结三件套、val 选点/test 单评口径）；Stage-1 config 的唯一差异 = R2 的 config 恰多一个键 `{"clustering": "rank_normalized"}`（测试钉死）；内容寻址产生**不同** `stage1_id`（禁止覆盖）；两 Stage-1 进程在首次聚类调用（epoch 2 末）之前代码路径完全一致 → epoch 1–2 记录（14 值）逐位相等（P3a）。

**配对判定量**：`Δtest = R4.test_bsi − R3.test_bsi`；`Δval = R4.best_val_bsi − R3.best_val_bsi`；Stage-1 侧 `Δtest_ctr = R2.test_ctr − R1.test_ctr`（U1）。跨 seed 比较只在各 seed 自己的配对内计算；跨 seed 汇总只按 §5.6 的冻结统计量。

**记录项（每题必录，§9）**：环境占比（事件 env_0/env_1 计数与占比）、标签比例（env × click/purchase 交叉表）、归一化损失分布（`cluster_diagnostics` 的 raw/normalized 分位数）、聚类可预测性/退化诊断（env_acc / env_bal_acc / 各环境 recall；B4 门禁）、Stage-2 routing use（`gate_mean` 两维）、下游 BSI 的 test/val AUC 对（raw 与 norm 各一对 + Δ）、逐 epoch 轨迹、停止/截断（early stop epoch、best epoch）、墙钟/设备/峰值显存、门禁与失败、无效尝试（若有）、跨 seed 汇总表。

---

## 3. 冻结预测（看到结果前写死）

### 3.1 P1：四个新 `stage1_id`（确定性哈希；提交前已复算）

| seed | 臂 | `config_hash` | 预测 `stage1_id` |
|---|---|---|---|
| `1688749593`（seed4） | raw | `bb2b68de2dac257c20d7b50544f83e672a12eb2e04754ec476a66f90788a8cb3` | **`s1-5c060b9c-m1688749593-e3-bb2b68de`** |
| `1688749593`（seed4） | norm | `cf810ec493f52555dc7e69bcb971537c52b9f6b14dca6234759447ed1e9ac3fe` | **`s1-5c060b9c-m1688749593-e3-cf810ec4`** |
| `1688762746`（seed5） | raw | `6343490f5e16d99e31213d5bb4a91459a173b782616eda2184d9c4c3d94c75d2` | **`s1-5c060b9c-m1688762746-e3-6343490f`** |
| `1688762746`（seed5） | norm | `a8238feaa483aada38f0280971f6c5b6fd5fbfdb01ab67eda68dac99a501f0ae` | **`s1-5c060b9c-m1688762746-e3-a8238fea`** |

- 依据：`bench._stage1_cfg`（协议默认 cfg + `model_seed`）的确定性哈希；norm = 同 cfg + 恰一个键 `{"clustering":"rank_normalized"}`。
- **方法自证**：同一计算在本文件提交前复现了 seed1/2/3 的全部 6 个已知 `config_hash` 与 6 个已知 `stage1_id`（12/12 逐字相等，含 seed3 的 `47619ce0…`/`5f899cad…` 与 seed2 的 `4e1b5c6f…`/`820697f6…`）。
- 测试钉死：seeds45 守卫测试对四个哈希与四个 id 逐字断言（§7）。

### 3.2 P2–P4（每 seed 同型）

| 编号 | 预测 | 依据 |
|---|---|---|
| P2 | R3/R4 的 run 记录 `stage1_id` 分别 == P1 的两值；两者 A5/A6 逐位通过 | 内容寻址 + 冻结校验 |
| P3 | (a) R1 与 R2 的 epoch 1–2 逐 epoch 记录（7 值 × 2）**逐位相等**；(b) R1、R2 各恰 1 次聚类事件、记录 epoch = 2 | 首个 `cluster_2` 调用在 epoch 2 训练后、其 val 评估前；此前两臂代码路径一致（arm manager 仅覆写 `cluster_2`，不消耗 RNG、不改权重） |
| P3R | （方向性期望，非 bit 断言，不作门禁）R1 的 raw 聚类事件延续既有形态：`env_0` 占比 ≪ 5%（标签主导、B4 FAIL） | 机制审计；seed4/5 为未知量，如实记录 |
| **P4** | （**逐 seed 在 R1 的认证捕获后、R2 运行前登记**）R2 聚类事件三元组（`diff_num`/`env_0`/`env_1`）、`env_ids_sha256`、环境占比、`env_0∩purchase1` | R1 `--reproduce` 认证捕获（10/10）的聚类时刻损失向量上，`per_task_rank01` 与 `ref_rank01` 两实现逐位一致的 argmin |

**P4 登记程序（逐 seed 写死；与 seed3 §3.1 相同；无 bit 捕获则按 §4.3 降级）**：

1. R1 完成后运行只读审计复现：`python -m aliccp_benchmark.audit_cluster_losses --stage1-dir artifacts/aliccp_bench/stage1/<R1sid> --out-dir artifacts/aliccp_bench/audit/<R1sid>-repro --reproduce`；
2. 复现校验必须 10/10 逐位一致（cluster 事件、逐 epoch、best epoch、backbone sha、test AUC、env_ids sha、捕获 argmin 与父类分配一致、捕获分配 sha == meta）；此时捕获的 `cluster_losses.pt` 为聚类时刻真值；
3. 在捕获损失上分别用 `normalized_clustering.per_task_rank01` 与 `audit_cluster_losses.ref_rank01` 计算 argmin 分配（两者必须逐位相等），导出预测事件三元组、`env_ids_sha256`、占比、`env_0∩purchase1`（需要标签 → 只读扫描训练前缀）：`python -m aliccp_benchmark.verify_seeds45_results freeze-p4 --model-seed <S> --capture <…>/cluster_losses.pt --out artifacts/aliccp_bench/audit/<R1sid>-repro/p4_prediction.json`；
4. 将预测值填入本文件 §9 的对应 P4 行并**提交（seed4 → C3a；seed5 → C3b）**；
5. 然后才运行 R2；R2 完成后逐位核对（M2c）。
6. 复现未达 10/10：该 seed 的 P4 记 `INVALID_FOR_PREDICTION`（不登记预测值），M6 FAIL、M2c 未判定，**照常运行 R2**（结果 run 计划不变），机制结论按 M 组如实报告。

---

## 4. 预注册门禁（看到结果前写死；看到结果后不得修改）

### 4.1 机制门禁（M 组；逐 seed 判定，与 seed3 §4.1 同口径）

| 编号 | 判据 |
|---|---|
| M0 | **前置审计**：§1.4 的 `audit_prior_seeds` 四臂 `all_pins_pass = true`（已完成）+ §1.5 的 seed3 记录复核一致（已完成） |
| M1 | **B4 修复**：该 seed 的 R2 恰 1 次聚类事件且两环境各 ≥ 5%·N（协议 `ENV_SHARE_MIN`，不改）；该 seed 的 R4 `gate_report` B4 同口径 PASS |
| M2 | **身份与轨迹**：(a) R1/R2 `stage1_id` == P1（该 seed 两值）；(b) R1 与 R2 epoch 1–2 记录（14 值）逐位相等；(c) R2 事件三元组 == P4 ∧ R2 `env_ids_sha256` == P4（P4 有效时；否则未判定并按 §4.3 记录） |
| M3 | **有限性**：R2 `cluster_diagnostics` 全部 raw/normalized 分位数有限（无 NaN/Inf）；运行未抛有限性断言 |
| M4 | **非标签恢复（norm）**：`env_0∩purchase1 ≤ 5%·|env_0|` ∧ `env_0 ≠ purchase1 集合`（逐位不等）∧ 非子集；raw 对照（R1）照实记录（不判定） |
| M4b | **退化签名消除（norm）**：`recall(env_0) > 0 ∧ recall(env_1) > 0`（聚类可预测性探针；raw 对照期望 bal≈0.5、recall(env_0)=0） |
| M5 | **有效变化**：R2 事件 `diff_num ≥ 5%·N`；R2 vs R1 `env_ids` 逐位差 ≥ 5%·N |
| M6 | **环境逐位可复现**：该 seed 的 R1 `--reproduce` 10/10（§3.2 第 2 步）；捕获 argmin 与父类分配 mismatch == 0 |
| M7 | **A 类完整性**：该 seed 的 R3/R4 的 A1/A2/A4/A5/A6 全 PASS（A3 SKIP，协议口径） |
| M8 | **默认关恒等与静态守卫**：(a) 全测试套件绿（移植不变量 + 字节钉死 + AST 守卫 + seed3 守卫 + seeds45 守卫）；(b) smoke 规模默认路径恒等核验（temp root、非结果 run）：meta 科学字段与既有记录产物 `s1-550e4d92-m1688723512-e1-d069eadf` 逐位一致（仅 commit/git/versions/wall/device/peak_vram 允许差异；须在同设备 cuda:0 运行——跨设备浮点路径不同，CPU 比较非逐位口径） |

- `MECHANISM_REPAIRED_SEED4 ⇔ 该 seed 的 M1∧M2∧M3∧M4∧M4b∧M5∧M6∧M7∧M8`；seed5 同理；否则逐项如实报告。

### 4.2 效用（U 组；within-seed 配对；容差先于 run 写死）

| 编号 | 判据 | 容差依据 |
|---|---|---|
| U1 | Stage-1 CTR 无实质退化：`Δtest_ctr ≥ −0.005`（CVR 记录不判定） | 同 seed1/2/3 口径（1M 前缀 47k 正例，SE≈0.0024） |
| U2 | **主分类量** `Δtest = R4.test_bsi − R3.test_bsi` 按 §5.2 三分法分类（同时进入五 seed 汇总的 `reclassification` 字段） | 用户预注册规则（§5.2） |
| U3（报告项，不判定） | `Δval`、逐 epoch val 轨迹、`gate_mean`、B1/B2/B3 原样披露；Δval 与 U1 参与 §5.3 的 material contradiction 规则 | — |

### 4.3 预测/复现失败处置（与 seed3 §4.3 相同）

- M2c 未判定 ⇒ 该分量照实报告（机制结论以其余 M 判据为准）；**不重跑**。
- M6 失败（环境不可逐位复现）⇒ `INVALID_FOR_PREDICTION`；R2 照跑；机制结论降级为"模式级复现"并如实披露。
- 无效执行（崩溃/中断/产物不完整）仅因工具性原因可重跑一次并记录；**结果原因（门禁未过/分类不利）不构成无效，不重跑**。无调参、无 epoch 挑选、test 不参与任何选择。

### 4.4 纪律

不得只报通过的门禁；不得事后改判据/容差/分类阈值；不得挑选 epoch/seed；全部 run 的 `git.dirty` 必须为 false（§6）。**不得静默替换 seed 或预算**；seed4/5 为 canonical 列表第 4/5 位，唯一合法值。

---

## 5. 判定与收束规则（先于 run 写死）

### 5.1 机制判定

逐 seed：`MECHANISM_REPAIRED_SEED4` / `MECHANISM_REPAIRED_SEED5`（§4.1）；否则逐项如实报告（含"机制在该 seed 未复现"的可能）。

### 5.2 效用三分法（用户预注册；机械执行；= 五 seed 汇总的 `reclassification` 字段）

以 `Δtest = R4.test_bsi − R3.test_bsi` 分类：

| 条件 | 分类标签 |
|---|---|
| `Δtest ≥ +0.001` | `POSITIVE_IMPROVEMENT` |
| `−0.02 < Δtest < +0.001` | `NO_CLEAR_IMPROVEMENT` |
| `Δtest ≤ −0.02` | `CLEAR_DEGRADATION` |

- 该标签为**再分类字段**：对 seed1–5 统一机械计算；seed1/2/3 的**历史标签不覆盖、不重判**（§5.5 并列呈现）。

### 5.3 material contradiction 规则（先于 run 写死；逐 seed）

在 `Δtest ≥ +0.001` 的前提下，称存在 **material contradiction** 当且仅当任一项成立：
(a) `Δval ≤ −0.01`；或 (b) 该 seed 任一 M 组门禁 FAIL；或 (c) 该 seed U1 FAIL；或 (d) 该 seed R3 或 R4 的任一 A 类门禁 FAIL（A3 除外）。

### 5.4 续行规则与逐 seed 结论标签

- **seed4**：`Δtest ≥ +0.001` 且无 material contradiction ⇒ 登记 **`SEED4_CONDITION_PASSED`**，下一步 = seed5（同协议，本分支内继续）；若机制未复现 ⇒ **`MECHANISM_NOT_REPAIRED_ON_SEED4`**；若机制复现但 `Δtest < +0.001`（或存在 material contradiction）⇒ **`mechanism repair effective but utility unstable`** + §5.7 headroom。**无论 seed4 结果如何仍执行 seed5**（seed5 是 canonical 列表的一部分，其执行不取决于 seed4 结果——本分支对 seed4/5 的必要运行在预注册时即已固定）。
- **seed5**：完成后进入 §5.6 五 seed 汇总，seed5 的逐 seed 结论标签 = **`SEED5_CONDITION_PASSED`** / **`MECHANISM_NOT_REPAIRED_ON_SEED5`** / **`mechanism repair effective but utility unstable`**（同规则）；**本分支到此为终点**（不再启动任何后续 seed；不启动 Census residual prompt、不启动第三数据集、不合并 master）。

### 5.5 历史标签冻结表（不可变、不覆盖）

| seed | 分支 / tip | 历史标签（原样引用） | 备注 |
|---|---|---|---|
| 1688723512 | `exp/aliccp-stage1-normalized-env-clustering` @ `2058de8` | `REPAIRED` + `NO_MATERIAL_DEGRADATION` | G 组判定（无三分法标签） |
| 1688723740 | `exp/aliccp-stage1-normalized-clustering-seed-replication` @ `7ab445c` | `NOT_SUPPORTED`（`MECHANISM_REPLICATED ∧ ¬UTILITY_DIRECTION_REPLICATED`） | 机制 10/10 复现 |
| 1688738016 | `exp/aliccp-stage1-normalized-clustering-seed3` @ `f8ff4cf` | `MECHANISM_REPAIRED_SEED3` + `SEED3_CONDITION_PASSED`；分类 `POSITIVE_IMPROVEMENT` | §6.2 分类 = 再分类字段一致 |

### 5.6 五 seed 终局汇总（先于 run 写死；机械计算）

**输入（canonical 顺序 1–5）**：per-seed `Δtest`/`Δval` —— seed1/seed2 取自 `audit_prior_seeds.ARMS` 的精确记录浮点（= §1.4 复算值）；seed3 取自 `f8ff4cf` 提交记录（字面值 `+0.02347892658289208` / `+0.017557536803150087`，以 `--seed3-verification` JSON 交叉核对）；seed4/seed5 取自各自 verification JSON。

**统计量（对 Δtest 与 Δval 各算一遍）**：

| 量 | 定义 |
|---|---|
| mean | Σd / 5 |
| sample std | `sqrt(Σ(d−mean)² / 4)`（ddof=1） |
| positive-improvement count | `#{d ≥ +0.001}`（÷5） |
| sign-positive count | `#{d > 0}`（÷5） |
| worst seed | `argmin d`（名称 + 值） |
| paired 95% CI | `mean ± t·s/√5`，冻结常数 **`t = 2.7764451051977987`**（t 分布 df=4 的 0.975 分位） |

- "paired" 语义 = 每 seed 内部的 R4−R3 配对差，再对 5 个差做单样本 t 区间；不做跨 seed 混合的其它推断（无多重比较校正主张）。
- **耐久性终局标签（冻结规则）**：`UTILITY_DURABLE ⇔ CI_low(Δtest) > 0 ∧ positive-improvement count ≥ 3`；否则 `MECHANISM_REPAIR_ONLY_UNSTABLE_UTILITY`。机制一致性（B4 修复的 seed 数）为描述性报告项，不改变上述标签。

### 5.7 headroom 评估（每 seed 机械记录；NO_CLEAR 时为其结论分量）

| 编号 | 输入 | 判定 |
|---|---|---|
| H1 | 机制活性 | 该 seed M 组全过 → favors；任一 FAIL → contradicts |
| H2 | 验证方向 | `Δval ≥ 0` → favors；`−0.01 < Δval < 0` → ambiguous；`Δval ≤ −0.01` → contradicts |
| H3 | 跨 seed 一致性 | `Δtest_当前 > 0` ∧（seed 1..k 中 `Δtest>0` 的个数 ≥ `⌈k/2⌉`，k = 当前 seed 序号）∧（seed 1..k 平均 `Δtest > 0`）→ favors；否则 not |
| H4 | 预算轨迹 | 该 seed 两臂 `best_epoch == 5`（预算受限）→ favors_longer_budget；否则照实记录 |

- H3 的可用集合 = seed 1..k（canonical 顺序；不引入未来 seed）；k=3 时即 seed3 预注册的"≥2/3"规则。**结论标签**：`HEADROOM_PLAUSIBLE ⇔ H1 favors ∧ H2 ≠ contradicts ∧ H3 favors`；否则 `HEADROOM_NOT_EVIDENT`（H4 仅描述）。

---

## 6. 运行计划与清洁树纪律

**结果 run（每 seed 恰 4 次，共 8 次，不重跑）**：R1 →（审计复现 ×1 + P4 冻结）→ R2 → R3 → R4。非结果核验：审计复现 ×2、smoke 恒等 ×2（M8b，temp root）、P4 冻结 ×2、跑后只读核验（交叉表/探针/预测核对/汇总）。

**提交链（run 间提交；SUMMARY 追加行必须在下一 run 前提交——memory 纪律；全部 push 至 origin）**：

1. **C1**：本文件（+ §9.0）→ 树干净；
2. **C2**：`verify_seeds45_results.py` + seeds45 守卫测试（TDD 先红后绿）+ 白名单级联适配（§7.3，test-guarded + 披露）→ 树干净；
3. **每 seed 循环**（x ∈ {a: seed4, b: seed5}）：跑 **R1**（`git.dirty=false`）→ 审计复现（§3.2）→ 填 P4 → **C3x**（仅文档）→ 跑 **R2**（`dirty=false`）→ 核验 M2/M3/M5 + §9 记录 → **C4x** → 跑 **R3**（`dirty=false`）→ **C5x**（R3 的 SUMMARY 行）→ 跑 **R4**（`dirty=false`）→ 核验 M1/M4/M4b/M7 + U 组 + §5 分类 → **C6x**（§9 记录 + R4 的 SUMMARY 行 + 验证 JSON 引用 + P4 token 入守卫测试）；
4. **C7**：五 seed 汇总（§5.6；`aggregate` 输出 JSON）+ 终局耐久性标签 + §9.6 纪律核对 → push。

- Stage-1 不写 SUMMARY（协议现状：仅 stage2 追加）；SUMMARY 只追加、永不重写。
- 每 seed 的 R3/R4 之间零代码 diff（差异仅为文档/SUMMARY）；§9 记录该证据。

---

## 7. 实现面（TDD；默认关逐位一致）与静态守卫

| 文件 | 改动 | 说明 |
|---|---|---|
| `docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seeds45-design.md`（C1 新增） | 本预注册文件 | — |
| `aliccp_benchmark/verify_seeds45_results.py`（C2 新增） | 跑后只读验证 + M/U 门禁 + §5.2/§5.3/§5.6/§5.7 机械分类与汇总（按 `--model-seed` 参数化；复用 seed3 验证脚本的未改函数与方法学） | 独立验证与机械判定 |
| `aliccp_benchmark/tests/test_normalized_clustering_seeds45.py`（C2 新增） | 本分支守卫：本文件 token 钉死（P1 四 id/四哈希、阈值、标签、统计常数）、P4 登记时点后追加 P4 token、白名单级联的逐字节核对、seed3 证据文件不可变核对、SUMMARY 前缀（append-only）核对、验证器的机械函数测试（TDD 目标） | §7.1–§7.3 |
| `aliccp_benchmark/tests/test_normalized_clustering.py`（白名单 += 本分支 3 个新文件） | 仅 `test_tracked_changes_subset_of_whitelist` 内的白名单集合（= 声明的 2 处适配之一，AST 守卫口径不变） | §7.3 级联适配（披露） |
| `aliccp_benchmark/tests/test_normalized_clustering_seed3.py`（`NEW_WHITELIST_ENTRIES` += 同 3 个新文件） | 仅该集合字面值与消息（seed3 守卫的其余全部钉死不动） | §7.3 级联适配（披露；由新守卫测试逐字节核对"仅此一处差异"） |
| `artifacts/aliccp_bench/SUMMARY.md` | 每 seed R3/R4 后各追加 1 行（共 4 行） | append-only |

**机制文件零改动（逐字节钉死 @ `2058de8`）**：`aliccp_benchmark/normalized_clustering.py`、`bench.py`、`metrics.py`、`run_aliccp_benchmark.py`、`audit_cluster_losses.py`（sha256/blob 钉死同 seed3 §1.1；seeds45 守卫测试逐字断言，§1.5 已预验证 15/15）。

**明确不改**：`multitaskrec/*`（零 diff）、`config.py`、`AliCCP_*.py`、`baseline/*`、`aliccp_benchmark/protocol.py`（零 diff）、`aliccp_benchmark/audit_prior_seeds.py`、`aliccp_benchmark/verify_seed3_results.py`（不可变；seeds45 守卫测试钉其 LF-sha256）、seed3 两份文档（不可变；同钉）。

### 7.1 字节钉死（守卫测试）

对五个机制文件逐一断言 `sha256(工作树文件, LF 归一化) == 2058de8 钉死值` 且 `git hash-object == blob 钉死值`（含 `git rev-parse 2058de8:<path>` 对照）。任一文件与 seed1 修复不等价即红。

### 7.2 不可变证据与 append-only 核对（守卫测试）

- `verify_seed3_results.py`、seed3 两份文档：LF-sha256 == `f8ff4cf` 值（逐字不变）。
- `SUMMARY.md`：以 `f8ff4cf` 的 6 行（表头 + 分隔 + 4 数据行）为**前缀**（append-only；新行只允许追加在尾部）。
- `test_normalized_clustering_seed3.py`：与 `f8ff4cf` 版本逐字节比较，差异必须**恰为** `NEW_WHITELIST_ENTRIES` 集合字面值（5 项 → 8 项）与配套消息字符串的替换（机械替换核对）。
- `test_normalized_clustering.py`：与 `f8ff4cf` 版本逐字节比较，差异必须**恰为**白名单集合字面值（14 项 → 17 项）的替换；且 AST 守卫（对 `2058de8` seed1 版：恰 2 处适配，DOC_PATH + 该白名单方法）继续成立。

### 7.3 白名单级联适配（披露）

`test_tracked_changes_subset_of_whitelist` 用 `git diff --name-only 8133d32` 核对"全部 tracked 改动 ⊆ 白名单"。本分支新增 3 个 tracked 文件（§7 表中前 3 项）必须进入白名单，否则 M8a 必红；而白名单恰是 seed1 移植声明的 2 处适配之一、且 seed3 守卫断言 `白名单 == seed1 ∪ seed3 的 5 个文件`。因此**必要级联**：两个白名单副本同步 += 本分支 3 文件（仅集合字面值），其余一切不动；由 §7.2 的逐字节核对守卫。这是"严格必要的 seed4/5 身份/文档适配"，不改变机制或方法论。

---

## 8. 偏离披露（预声明）

1. **环境偏离（任务字面 vs 实测）**：任务字面指定 Python310 解释器；实测其无法执行本项目栈（缺 `fvcore`；unittest 采集访问违例崩溃），且 torch 版本（2.12.0+cu126）不同于全部既有 canonical run 的记录环境（2.6.0+cu124）。本分支改用 `.venv`（§1.2，逐项匹配记录环境、smoke 恒等 IDENTITY_PASS、全测试套件 64/64 OK）。该选择是为保持 bit 级方法学与跨 seed 可比性的**必要**偏离；全部证据与失败输出留档于 §1.2。
2. **白名单级联适配**：§7.3（必要、test-guarded、逐字节核对）。
3. **分支工作流偏离**：协议 §4.3 要求"先合并协议再从 master 拉 exp 分支"；本分支自 seed3 exp tip `f8ff4cf` 拉出（协议未合并入 master），协议文件零修改、SUMMARY 只追加——与 seed3（及同批 exp 分支）先例一致。
4. **数据访问方式**：本工作树经 junction 只读引用 seed3 工作树的数据集目录；源数据集零修改、零复制、零下载。
5. **P4 的登记时点（逐 seed）**：在 R1 捕获后、R2 运行前登记提交（C3a/C3b）——晚于本预注册、早于被预测的 run；程序依赖与捕获程序已在本文件写死。
6. **SUMMARY 系谱**：本分支起点 `f8ff4cf` 的 `SUMMARY.md` 为 6 行（smoke + seed1 short + seed3 short + seed3 norm）；本分支只追加自己的 4 行（每 seed R3/R4），不复制其他分支行。
7. **seed3 守卫文件的一处字面值修改**：§7.3（seed3 tip 上的原版保持不变、可查；本分支对该文件的唯一差异由新守卫测试逐字节锁定）。
8. **预运行澄清（2026-10-05，先于任何 run；预测与门禁不变）**：§5.4 补上逐 seed 结论标签字面值（`SEED4_CONDITION_PASSED` / `MECHANISM_NOT_REPAIRED_ON_SEED4` / `SEED5_CONDITION_PASSED` / `MECHANISM_NOT_REPAIRED_ON_SEED5` / `mechanism repair effective but utility unstable`）——原 §5.4 仅以中文描述，守卫测试 token 钉死需要字面值；澄清时无任何 seed4/5 run 存在。

---

## 9. 结果（实测；运行后追加，不得回填预期值）

### 9.0 预运行核验（实测）

- **环境（§1.2）**：Python310 实测失败（`ModuleNotFoundError: fvcore`；unittest 采集 access violation，exit `-1073741819`）；`.venv` 实测通过（全测试套件 `Ran 64 tests in 11.011s → OK`；smoke 恒等 `IDENTITY_PASS`、`field_diffs={}`）。
- **M0 前置审计（§1.4）**：`all_pins_pass = true`（wall 73.0 s）；seed1 Δtest `+0.00824283986406793` / Δval `+0.010059271876175169`；seed2 Δtest `+0.00045881952817949934` / Δval `−0.0014831644646635667`；两 seed B4 修复 confirmed。
- **seed3 记录复核（§1.5）**：`delta_test 0.02347892658289208` / `delta_val 0.017557536803150087` / `delta_test_ctr 0.009211716513914836` == 文档字面值；M8a/M8b True。
- **字节钉死预验证（§1.5）**：五机制文件 15/15（sha256 × 10 + blob × 5）通过。
- **P1 方法自证（§3.1）**：seed1/2/3 共 12/12 已知哈希与 id 逐字复现。
- **数据（§1.3）**：junction 只读；三文件字节数 == 协议审计值。

### 9.1 seed4 Stage-1（R1/R2；运行后填写）

**P4 登记（预先；C3a——R1 捕获后、R2 运行前）**：

- 捕获认证（M6 证据）：R1 `--reproduce` **10/10 all_pass**（cluster 事件、逐 epoch、best epoch、backbone sha、test AUC、env_ids sha、捕获 argmin == 父类分配、捕获分配 sha == meta 全逐位）。
- **P4 预测**（`per_task_rank01` 与 `ref_rank01` 双实现逐位一致，`bitwise_agree = true`）：事件 **`diff_num = 1000669`、`env_0 = 1021493`、`env_1 = 978507`**（占比 **51.07465% / 48.92535%**）；`env_ids_sha256` = **`523e129ca132bdab2ccfaeb22593e309dd30025ca792d3d6e945593dd8eeb2ae`**；`env_0∩purchase1 = 564`（另 `env_0∩click1 = 4209`、≠ purchase1 集合、非子集）。
- 登记文件：`artifacts/aliccp_bench/audit/s1-5c060b9c-m1688749593-e3-bb2b68de-repro/p4_prediction.json`（登记时 commit `d768ebc`）。
- R1 复现捕获（聚类时刻真值）诊断（只读工具记录）：raw 损失尺度 task0/task1 平均比 ≈ **35.4×**（尺度不可比复现）；候选对照 rank01 分配 = **1021493 / 978507**（即 P4）；raw 分配 = purchase 标签集合（env_0 == purchase1 集合 True）。

**R1 记录**：`stage1_id = s1-5c060b9c-m1688749593-e3-bb2b68de`（== P1）；commit `d768ebc` / dirty **false** / wall 258.8 s；恰 1 次聚类事件（epoch 2）：`1000466 / 566 / 1999434`（env_0 占比 **0.02830%**，B4 FAIL，P3R 方向性命中）；best_epoch = 3；test CTR `0.5475919346018614`、CVR `0.574534619903342`；env_acc `0.99977`。

**结论（R2 完成后）：M2（a/b/c）、M3、M5、P3R 全部命中；R2 事件与 P4 逐位相等（bit 级预测命中）。**

| 项 | R1（raw，`…-bb2b68de`） | R2（norm，`…-cf810ec4`） |
|---|---|---|
| stage1_id | `s1-5c060b9c-m1688749593-e3-bb2b68de`（== P1；M2a） | `s1-5c060b9c-m1688749593-e3-cf810ec4`（== P1；M2a） |
| commit / dirty / wall | `d768ebc` / **false** / 258.8 s | `c37024f` / **false** / 193.9 s |
| 聚类事件（epoch 2，恰 1 次） | 1000466 / **566** / 1999434（env_0 占比 **0.02830%**，B4 FAIL；env_0 == purchase1 集合） | 1000669 / **1021493** / 978507（**51.07465%** / 48.92535%，B4 PASS；**== P4 逐位**，M2c 命中） |
| epoch 1–2 记录（14 值） | — | **与 R1 逐位相等**（M2b；如 ep1 CTR `0.5362242695964587`、ep2 CTR `0.5380123424302347`、ep1/2 env_loss `1.1463263034820557`/`3.474088191986084`） |
| epoch 3（治疗臂特有轨迹） | val CTR `0.5474969702565669` / CVR `0.5385450186314242`；env_loss `0.07357814162969589` | val CTR `0.5462695471418261` / CVR `0.5261442095465247`；env_loss `1.631626009941101`（平衡分配下 env 头不再可平凡拟合） |
| best_epoch / best val | **3** / CTR `0.5474969702565669`、CVR `0.5385450186314242` | **3** / CTR `0.5462695471418261`、CVR `0.5261442095465247` |
| test（单次） | CTR `0.5475919346018614`、CVR `0.574534619903342` | CTR `0.5465741235184509`（**Δtest_ctr = −0.0010178110834104803**，U1 容差内 PASS）、CVR `0.5750194104732491` |
| env_ids sha256 | `b839e51df9cb46a6b46fa5ea5e757e006c8288e00a2716fa0d4a89967a4f2a15`（== seed1 raw 形态） | `523e129ca132bdab2ccfaeb22593e309dd30025ca792d3d6e945593dd8eeb2ae`（**== P4**，M2c） |
| `cluster_diagnostics[0]`（聚类时刻） | 不适用（raw 臂记录原损失，未存分位数） | raw 中位 CTR `0.048517823219299316` / CVR `5.378545830936332e-09`（尺度差复现）；normalized 中位 `0.5` / `0.5`（rank 语义自检）；全分位数有限（M3 PASS） |
| 标签比例（env × 标签，独立复算） | env_0=566（== purchase1 集合 True） | env_0=1021493（含 purchase **564** = 0.0552%·\|env_0\| ≤ 5%；click 4209；≠ purchase1 集合、非子集 → **M4 PASS**） |
| env_pred 探针（best 权重，前 400k 行） | acc `0.99977`（退化签名口径） | acc `0.4971375`、bal `0.5000354152434238` → **M4b 待 R4 verify 机械判定**（recall 双正 > 0；本 seed 探针显著弱于 seed1–3，如实记录） |
| 有效变化（M5） | — | 事件 diff_num 1000669 ≥ 5%·N；**R2 vs R1 env_ids 逐位差 1020931 行（51.0466% ≥ 5%）** |
| 认证复现（M6） | **10/10**（§9.0/P4 块） | 不适用 |

- **P3R（方向性期望，非门禁）命中**：raw 臂事件延续既有形态（env_0 占比 0.028%、566/566 为 purchase 正例、B4 FAIL）。
- **披露（R2 的 env 探针）**：seed4 归一化臂的 env_pred 探针 bal `0.5000354152434238`、acc `0.4971375` —— 接近多数类失真口径（seed1–3 的 norm 臂 bal 为 0.184/0.647/0.597）；M4b 以预注册判据（recall 双正）在 R4 verify 机械判定，如实记录。
- **M1** 的 R4 侧子句（gate_report B4 PASS）待 R4；**M7** 待 R3/R4。

### 9.2 seed4 Stage-2 与分类（R3/R4；运行后填写）

### 9.3 seed5 Stage-1（R1/R2；运行后填写）

### 9.4 seed5 Stage-2 与分类（R3/R4；运行后填写）

### 9.5 五 seed 汇总（终局；运行后填写）

### 9.6 纪律核对（运行后填写）

---

## 10. 局限（预声明）

1. 每 seed 每配置 1 run 配对；无 run-to-run 噪声估计（A3 SKIP）；Δ 的绝对值不作单 seed 显著性主张。
2. P4 事件 bit 预测依赖环境逐位可复现（seed1 已证 10/10；本分支每 seed M6 再验）；失败按 §4.3 降级。
3. CVR 指标统计功效极低（test 正例 346），只作记录。
4. 仅 AliCCP 前缀 `p2M-v500k-t1M`；不外推其他预算/数据集。
5. 五 seed 汇总为五个单 run 配对的并列统计（n=5，t 区间；无多重比较校正主张）；跨 seed 数值比较的精度受单 run 噪声限制。
6. 环境固定为 torch 2.6.0+cu124（§1.2/§8.1）；本分支结论不外推到其他数值环境。
