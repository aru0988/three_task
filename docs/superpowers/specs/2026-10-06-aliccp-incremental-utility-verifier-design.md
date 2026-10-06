# AliCCP 两路径增量效用路由（baseline vs 范数受控残差 Prompt）：预注册与执行计划（v2，按统筹审核修正）

- **状态**：预注册（修正版）。本文件在**任何实现、任何 GPU 运行之前**单独提交（commit C1）。第 0–9 节的判据、公式、阈值、特征集与运行规程在结果产生后不得改动；结果只在第 11 节以新增小节回填。
- **日期**：2026-10-06
- **适用分支**：`exp/aliccp-incremental-utility-verifier`（自 `exp/aliccp-stage2-residual-prompt-five-seed` @ `28aa1feae1fea0762136beebdfe6eb8387edb170` **独立拉出**，不合并 `master`，不触碰其它 worktree）。
- **修正说明（v1→v2→v3，按统筹审核）**：① 研究对象修正为 **baseline 与 norm-controlled 残差 Prompt 两条预测路径之间的增量效用路由**（非 seen-vs-OOF 样本加权）；② 效用标签 `u = BCE_baseline − BCE_prompt`，且**产生 u 的两臂均不得训练过被评分样本**（训练内部三分割/严格 out-of-fold 构造）；③ verifier **推理特征零标签、零标签依赖统计**；④ **主指标 = 实际路由后的目标 test AUC delta**（verifier AUROC 仅为机制诊断；oracle 仅诊断、不用于任何阈值）；⑤ 全部对照与基线**同数据可见性、同训练预算**——**历史全量训练基线不得作为公平性能对照**；历史 checkpoints 仅诊断用，且仅在无泄漏确认下复用；⑥ 方向 2/3/4 自纯公平基线 `8133d32` 建立，不继承 Residual Prompt；方向 1/5 自 F 建立、携带 RP 并标注为对应对照与最小 diff；⑦ **v3 增补（统筹第二轮）**：Stage-1 全 train 预训练定性为 **transductive unlabeled target exposure**（训练目标标签未用于 Stage-1，附源码证据；**禁止**表述为"严格完全未见样本"）；主收益必须**同时**相对 always-baseline、always-prompt 与 **calibration-only 固定混合**成立（仅优于较弱一侧不构成 router 价值证据）；有益标签固定 **`1[u>0]`**（不用中位数替代）；随机路由率**固定取 calibration**、test 端不调节。
- **被审计 / 只读引用**：
  1. AliCCP 公平评测协议：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（前缀预算、种子、Stage-1 内容寻址产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账纪律——沿用）。
  2. 五 seed canonical 线：`docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-five-seed-design.md`（canonical seed 列表、配对纪律、二级分类边界、披露纪律——沿用；其 run 一律不重跑）。
  3. 冻结公平基准 F 的既有产物（**仅作诊断参照与共享冻结 backbone 来源**，见 §2.4/§2.5）。
- **定位声明**：本文件**无新颖性主张、无改进主张**。可报告内容仅限：本仓库冻结协议 + canonical 短预算下、两路径增量效用路由的上述判定。历史结论原样保留、不改判、不重跑。
- **提交时序**：C1 本文件（单独提交）→ C2 实现 + 测试（TDD；本方向运行的 `commit` 字段即该提交）→ smoke（1 次，工具性）→ 筛查 run（canonical seed #1）→ 独立复核 → C3 结果回填（§11）+ SUMMARY 追加 → 条件满足才扩 seeds #2–5 → push 仅在统筹/用户明确批准后。

---

## 0. 判定问题与决策规则

- **Q1（主判定，效用；三比较同时成立才算 router 有价值）**：记
  `Δ_base = AUC_routed − AUC_always_baseline`、`Δ_prompt = AUC_routed − AUC_always_prompt`、
  `Δ_mix = AUC_routed − AUC_fixed_mix`（三者同 seed、同 A-臂、同 `stage1_id`）。
  - **分类（按 `Δ_base`，边界闭开，精确）**：`POSITIVE_IMPROVEMENT`：`Δ_base ≥ +0.001`；`NO_CLEAR_IMPROVEMENT`：`−0.02 < Δ_base < +0.001`（严格双开）；`CLEAR_DEGRADATION`：`Δ_base ≤ −0.02`。
  - **router 价值充分条件（预先写死）**：`Δ_base ≥ +0.001` ∧ `Δ_prompt > 0` ∧ `Δ_mix > 0`。**仅优于较弱一侧（baseline 或 prompt）不构成 router 价值证据**；`Δ_mix ≤ 0` 时结论必须为"固定混合已解释全部收益，router 无增量价值"。
- **Q2（机制诊断，非判定）**：verifier 的路由判别质量（对 `sign(u)` 的 **AUROC**，在 router-calibration 切分 C 上、out-of-sample）；oracle 路由（用真实 u 路由，test 上）仅作上限诊断。**二者均不参与任何阈值/分类/扩 seed 决策，不得作为部署指标或性能主张。**
- **禁止**（贯穿全文件）：
  1. verifier **推理特征**含任何标签或标签依赖统计（允许：原始特征、冻结表示、两臂 logits、其差异、范数）；
  2. test 与官方 validation（`ctr_cvr.dev`）参与 verifier 学习、校准、阈值选择或任何统计量估计；val 仅承担协议既有的两臂 early-stop/选点职责（与基线同规则）；
  3. 以历史 run（含历史全量训练基线/RP 臂）作为公平性能对照或臂来源（其训练见过被评分样本）；
  4. 外部 LLM/JEV 做样本推理（实现仅 numpy/sklearn/scipy/torch，本地）；
  5. 看到结果后改公式/阈值/特征/分割/网格口径；oracle/random 结果不得改判臂决策。

## 1. 审计（2026-10-06，只读；先于本文件提交）

| # | 项 | 值 / 结论 |
|---|---|---|
| A1 | 基准 F | `28aa1feae1fea0762136beebdfe6eb8387edb170`（2026-10-05 08:14:34 +0800，已推送，与 origin 同步）；纯公平基线 `8133d32`（fair-benchmark 基础设施，无任何模型机制）为其祖先（`git merge-base --is-ancestor` 已验证） |
| A2 | 前缀身份 | `PREFIX_TAG=p2M-v500k-t1M`；指纹 `5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8`；前缀 sha256：train `129b0a9e…` / val `ea65f591…` / test `91942851…`；标签计数与重复统计固化于 `splits/p2M-v500k-t1M/prefix_fingerprint.json` |
| A3 | Stage-1 产物（5/5 在位） | seed→`stage1_id`：1688723512→`s1-5c060b9c-m1688723512-e3-3a30e2c0`；1688723740→`s1-5c060b9c-m1688723740-e3-4e1b5c6f`；1688738016→`s1-5c060b9c-m1688738016-e3-47619ce0`；1688749593→`s1-5c060b9c-m1688749593-e3-bb2b68de`；1688762746→`s1-5c060b9c-m1688762746-e3-6343490f` |
| A4 | canonical 历史 run（5 基线 + 5 RP 臂，在位） | **仅诊断参照**（§2.5），不作对照、不作臂来源 |
| A5 | 运行环境 | Python 3.10.11 / torch 2.6.0+cu124 / CUDA 12.4；venv = 主树 `D:\MPT-Rec-three_task\MPT-Rec\.venv`；`numpy/scipy/sklearn/pandas` 在位；GPU = RTX 3060 Laptop 6GB |
| A6 | 计时实测（冻结协议记录） | stage2 5-epoch 全量 2M = 141.0s；每 1M 行/epoch ≈ 10s；峰值显存 ≤160 MB |
| A7 | 台账与守卫 | `SUMMARY.md` 只追加；静态守卫 `git diff --name-only 8133d32 ⊆ WHITELIST`（两处副本逐字节同步，先例 `28aa1fe`） |
| A8 | 结论 | F 既有产物齐备；本方向**零 stage1 重训**；两臂与全部对照在本分支内**新训**（§3），历史 run 不进任何比较 |

## 2. 对象定义

### 2.1 两条候选预测路径
- **路径 B（baseline）**：协议 NewTask 头（tower `[32,32]`、`rep_dim=64`、BCE + `get_l2_reg`、Adam 1e-4、batch 2000），训练于冻结的 canonical Stage-1 backbone 之上。
- **路径 P（prompt）**：范数受控残差 Prompt 头（`RP.build_newtask("residual-prompt", …)`；**与 F 逐字节相同**，sha 钉死；本分支不改机制文件、不调用 `arm_verdict`/pin 机制）。

### 2.2 增量效用（逐样本，两个方向都报）
- `u_i = BCE_B(i) − BCE_P(i)`（正 = 该样本上 prompt 路径更优 = 路由到 P 的增量效用）。
- 记 `Δp_i = p_P(i) − p_B(i)`、`u_i` 的 oracle 用真实标签计算（仅诊断）。
- 逐样本 `p_B(i)`、`p_P(i)` 一律来自**未训练过样本 i 的臂**（§3 分割保证）。

### 2.3 路由（部署形态）
- verifier 给每样本打分 `ŝ_i`（无标签特征，§4）→ 二值路由：`ŝ_i > thr` 选 P，否则选 B；`thr` 仅在 router-calibration 切分 C 上校准（§4.3）。
- 路由臂的 test 预测 = 每样本被选中臂的 `p`。

### 2.4 共享冻结 backbone 与泄漏边界（披露；禁止超额表述）
- 所有臂共享同一 canonical Stage-1 backbone（内容寻址产物，`requires_grad_(False)` + `eval()`，零更新，A1/A6 校验）。
- **训练目标标签未用于 Stage-1**（源码证据：`multitaskrec/train.py:395` 阶段 1 训练循环解包 `(y_0, y_1, _, features)`，第三标签（BSI，即本方向的目标标签）在阶段 1 被丢弃；阶段 1 仅 CTR/CVR 两路损失与 env 损失）。
- 但 Stage-1 预训练见过**全前缀的输入特征**（含 A/B/C 及与 test 分布同源的行）⇒ 本方向属 **transductive unlabeled target exposure**（目标域**输入**暴露、目标**标签**未暴露）。**本文件及一切报告禁止表述为"严格完全未见样本"**；唯一允许表述 = "**头级 out-of-sample**（两臂的头从未训练过被评分样本）+ **backbone 级 transductive unlabeled target exposure**（如实披露；两臂共有，在 u 的差分中同向抵消，残余风险不声称归零）"。
- 两臂的**头**训练数据 = 切分 A（§3），对 B/C/test 从未训练 ⇒ u 在 B/C 上、路由在 test 上均为头级 out-of-sample。

### 2.5 历史 checkpoints 的处置
- **可复用（零改动）**：5 份 canonical Stage-1 backbone/env_ids/meta（§2.4 条件下）。
- **仅诊断参照（不复用为臂/对照）**：5 个历史基线 run、5 个历史 RP 臂 run、20epoch/norm-cluster 等各线 run——其头训练见过全前缀（含 test 分布同源行），其 u 为 in-sample，**构成本质泄漏**，仅可用于诊断表（如“历史全量头 vs 本分支 A-训练头”的描述性对照，明确标注不可比）。
- 任何后续复用历史头输出（如作诊断特征）前必须先做无泄漏确认并单独登记；本文件不启用。

## 3. 训练内部三分割（A/B/C；确定性、行序、无 split 种子）

| 切分 | 行区间（train 前缀 2,000,000 内） | 用途 | 训练可见性 |
|---|---|---|---|
| **A** | `[0, 1,400,000)` | 训练两臂（B、P） | 两臂仅在 A 上训练 |
| **B** | `[1,400,000, 1,700,000)` | router-fit：由 A-臂评出 `u_i`，拟合 verifier | 两臂与 verifier 均未在 B 上训练过（verifier 在 B 上拟合，故 C 对 verifier out-of-sample） |
| **C** | `[1,700,000, 2,000,000)` | router-calibration：阈值网格选择 + verifier AUROC 诊断 | 两臂未训练；verifier 未拟合 |

- 载入实现：一次性构建全 2M `AliCCPDataset`（A2/A4 对全前缀校验），A/B/C 为同一 dataset 的**索引区间子集**（连续、互斥、并集恰 2M；索引集 sha 落盘）。
- 离散性披露：连续切分使 B/C 位于前缀后段，前缀内分布漂移（协议 §2.4 实测）如实披露；不改为交错划分（保持最小、可审计）。
- **两臂预算完全对等**（数据可见性 = A；epochs/patience/优化器/loss/选点规则逐项相同，§7 表）。

## 4. 路由特征（零标签）、verifier 与校准

### 4.1 推理特征（8 维，钉死；无标签、无标签依赖统计）
`[p_B, p_P, p_P−p_B, ‖dnn_input‖, ‖gen_rep‖, ‖spec_0‖, ‖spec_1‖, ‖env_emb‖]`
- 全部为两臂 logits 与冻结 backbone 表示（`get_infos`，no_grad）的逐样本函数；构造函数签名不接受标签参数（静态守卫）。
- 标准化统计（均值/标准差）**仅取 B**。

### 4.2 verifier（固定，无调参）
- Ridge 回归（闭式解，`alpha = 1.0`），target = `u_i`（B 上）；标准化后拟合，含截距。

### 4.3 阈值校准（仅 C）
- 网格 = C 上 `ŝ` 的 21 个分位点 `{0.05, …, 0.95, 步长 0.05}` ∪ `{−∞, +∞}`；在 C 上以“路由后 C-AUC 最大”选 `thr`；**并列时取被路由到 P 的样本占比更小者**（更保守）。被路由占比 `π̂`（C 上）作为随机对照的路由率。

### 4.4 诊断（不参与判定）
- 有益标签**固定 `1[u_i > 0]`**（v3；**禁止**用中位数二分或其他分位替代"有益"判定）。
- verifier out-of-sample（C 上）**AUROC vs `1[u>0]`**、Spearman(ŝ, u)；oracle 路由 **按 `u>0`** 选择路径（test 上，仅上限诊断，不用于任何阈值/决策）。

## 5. 评测配置（全部同数据可见性、同训练预算）

| 配置 | 预测 | 训练 |
|---|---|---|
| `always-baseline`（主对照） | `p_B(test)` | 臂 B（A 上 5 ep） |
| `always-prompt` | `p_P(test)` | 臂 P（A 上 5 ep） |
| `routed`（本方向臂） | `thr, ŝ` 二值路由 | 两臂 + verifier（B 拟合） + thr（C 校准） |
| `fixed-mix`（固定混合对照，**仅 C 定标**） | `α̂·p_P + (1−α̂)·p_B` | 两臂；`α̂` 仅由 C 选定（网格 `{0, 0.05, …, 1}` 取 C-AUC 最大，并列取更小 α），**test 端不调节** |
| `random-router`（控制） | Bernoulli(`π̂`) 路由 | 两臂；**路由率 `π̂` 固定取 calibration（C）**，test 端不调节；RNG = `default_rng(model_seed ^ ROUTER_RANDOM_SALT)`（钉死常量） |
| `oracle-router`（诊断，禁用为部署指标） | 按 `1[u(test)>0]` 路由 | 无 |

- **主指标（三比较；充分条件见 §0）**：`Δ_base = AUC_routed − AUC_always_baseline`、`Δ_prompt = AUC_routed − AUC_always_prompt`、`Δ_mix = AUC_routed − AUC_fixed_mix`（同 seed、同 A-臂、同 stage1_id）。
- 并报：`Δtest(prompt − baseline)`、`Δval` 各配置、两臂逐 epoch val 轨迹。
- test 仅为最终单次评估（两臂各一次前向整集，保存 `p_B/p_P` 后离线组配五配置；**路由决策不做任何 test 端再训练/再校准**）。
- **不做**与历史 `+0.0055`/U1/U2 的比较（对照口径不同）；历史数字仅诊断旁注。

## 6. 门禁（全部硬判据；先于结果写死）

| 编号 | 判据 | 实现 |
|---|---|---|
| A 类 | A1（冻结前后 backbone sha 一致 + 无残留梯度）/ A2（前缀指纹重算一致）/ A4（三切分样本数与标签计数逐项相等）/ A5（env_ids sha 与产物一致）/ A6（加载 backbone sha == meta 记录）；A3 SKIP | 复用 `metrics.evaluate_a_gates`（逻辑零改动） |
| R1 无标签特征 | 特征矩阵在 y 置换下**逐位不变**；构造函数签名无标签形参 | 运行期断言 + 单测 |
| R2 分割完整 | A/B/C 连续、互斥、并集 = 2M；两臂训练索引集 == A（sha 落盘） | 运行期断言 + 单测 |
| R3 学习隔离 | verifier 拟合只读 B；阈值只读 C（扰动测试：改 C 的 u 不改系数；改 B 的 u 不改 `thr`） | 单测 + run 记录分割 sha |
| R4 预算对等 | 两臂：同 A、5 ep、patience 2、Adam 1e-4、batch 2000、val 选点同规则 | 运行期记录 + 复核 |
| R5 覆盖 | B/C/test 的评分覆盖每索引恰一次，无重复计分 | 运行期断言 |

- **臂决策**：Q1 分类（§0）。**扩 seed 条件（六条全真）**：A 类 + R1–R5 全 PASS ∧ `Δtest_primary ≥ +0.001`（`POSITIVE_IMPROVEMENT`）∧ verifier 与两臂运行记录 `git.dirty=false`。
- **止损**：`CLEAR_DEGRADATION` ⇒ 记录并停止扩展；`NO_CLEAR` ⇒ 记录、不扩展（仅 POSITIVE 触发扩展）。verifier AUROC 为诊断，不作为判定或止损键。

## 7. 运行规程与各臂预算（前台、逐条等待；每步前全量单测绿）

```powershell
# cwd = D:\MPT-Rec-three_task\MPT-Rec-exp-aliccp-incremental-utility-verifier
$PY = D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe

# (0) 单测（CPU）全绿
& $PY -m unittest discover -s aliccp_benchmark/tests

# (1) smoke（工具性，1 次；自产 smoke stage1，小预算 20k/5k/10k；A/B/C = 14k/3k/3k）
& $PY run_aliccp_benchmark.py stage1 --tag smoke --train-budget 20000 --val-budget 5000 --test-budget 10000
& $PY run_incremental_utility_router.py --tag smoke --model-seed 1688723512 --stage1-id <smoke_sid>

# (2) 筛查（canonical seed #1）：两臂（A 上 5 ep）→ B/C 评分 → verifier/校准 → test 五配置
& $PY run_incremental_utility_router.py --tag short --model-seed 1688723512 `
    --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0

# (3) 独立复核（不 import 实现模块；重推 u/AUC/分类/门禁/分割/隔离）
& $PY verify_incremental_utility_router.py --run-id <iuv_run_id>

# (4) 仅当扩 seed 六条全真：seeds #2–5（各用其 stage1_id；不加 --controls）
& $PY run_incremental_utility_router.py --tag short --model-seed <S> --stage1-id <sid_S>
```

| 臂/项 | 预算（per seed） | 说明 |
|---|---|---|
| 臂 B | 5 ep × **1.4M** 行（A）；val 选点 patience 2 | 与臂 P 逐项同规则 |
| 臂 P | 5 ep × **1.4M** 行（A）；同规则 | RP 头（F 冻结实现） |
| B/C 评分 | 每臂 1 次前向（0.6M 行） | no_grad |
| test 评分 | 每臂 1 次前向（1.0M 行） | 单次评估 |
| verifier+校准 | CPU；Ridge 闭式 + 21 阈值网格 | 无 GPU |
| random/oracle | 离线后处理 | 无训练 |

- **台账纪律**：每个 run 之间先提交 `SUMMARY.md`/结果文档再跑下一个（`git.dirty=false` 必须成立；先例：五 seed 线 §10.0 事故与处置）。每次 runner 执行追加 **1 行**（后缀 `-iuv`）；列语义（本线约定，明细全部在 run JSON）：`auc_test_bsi := AUC_routed(test)`、`auc_val_bsi_best := routed 配置在 C 上的 AUC`、B 类门禁沿用臂 B 产物、`A*` 为本次执行 A 类判定。控制/诊断配置（fixed-mix/random/oracle）不单独占行，其数值在 `routing_report.json`。
- **一次纪律**：每 (seed, 配置) 恰一次有效执行；仅工具性无效可留痕重跑一次。
- 只读复用：canonical Stage-1 产物 + 指纹 + （诊断用）历史 run 目录，逐文件 sha256 校验后复制入本 worktree；`dataset` 以 junction 指向主树。smoke 结果不得用于任何性能陈述。

## 8. 最小测试（TDD，全部 CPU）

1. `test_split_partition`：A/B/C 连续、互斥、尺寸 1.4M/300k/300k、并集 2M、确定性。
2. `test_features_label_free`：特征构造签名无标签形参；y 置换 ⇒ 特征矩阵逐位不变（机械反泄漏）。
3. `test_utility_target`：`u = BCE_B − BCE_P` 逐位正确（tiny 合成 batch 手算对照）。
4. `test_router_isolation`：扰动 C 的 u ⇒ 系数逐位不变；扰动 B 的 u ⇒ `thr` 逐位不变。
5. `test_calibration_grid`：网格/并列保守口径/`π̂` 逐位可复算。
6. `test_routing_and_aucs`：routed/random/oracle 路由与 AUC 手算对照；random RNG 可复现。
7. `test_reproducibility`：同 seed 两次运行 ⇒ 系数、`thr`、路由决策逐位相同。
8. `test_e2e_tiny`：合成 tiny AliCCP 格式文件走通“stage1 产物 → 两臂 → 评分 → verifier → 校准 → 五配置 → 产物/SUMMARY”全链（CPU）。
9. 静态守卫：两处 WHITELIST 同步 + `git diff --name-only 8133d32` ⊆ WHITELIST + 冻结文件（`multitaskrec/*`、`config.py`、`protocol.py`、`metrics.py`、`bench.py`、`run_aliccp_benchmark.py`、`residual_prompt.py`、`five_seed.py`、既有测试）sha pin。

**新增文件（白名单；两处同步）**：`aliccp_benchmark/incremental_utility_router.py`、`aliccp_benchmark/tests/test_incremental_utility_router.py`、`run_incremental_utility_router.py`、`verify_incremental_utility_router.py`、本文件（`docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md`）。**RP 标注**：`residual_prompt.py` 逐字节沿用 F（= 对应对照路径的机制实现），最小 diff = 仅新增上述文件。

## 9. 五个独立研究分支（本轮建立）

| # | 方向 | 分支 | 基准 SHA | 说明 |
|---|---|---|---|---|
| 1 | 两路径增量效用路由（incremental utility verifier） | `exp/aliccp-incremental-utility-verifier` | **28aa1fe（F）** | 携带 RP，标注为对应对照路径（§2.1/§8），最小 diff = 新增文件 |
| 2 | generalization-driven environment weighting | `exp/aliccp-generalization-env-weighting` | **8133d32（纯公平基线）** | 不继承 RP；预注册在启动时另写 |
| 3 | parameter-efficient ensemble | `exp/aliccp-parameter-efficient-ensemble` | **8133d32（纯公平基线）** | 同上 |
| 4 | latent cross-task prediction diagnostic | `exp/aliccp-latent-cross-task-diagnostic` | **8133d32（纯公平基线）** | 同上 |
| 5 | training-state residual controller | `exp/aliccp-training-state-residual-controller` | **28aa1fe（F）** | 携带 RP，启动时标注对应对照与最小 diff；预注册另写 |

- 消融/预算子分支自对应方向 **C1（docs-only on 基准 SHA）** 建立，零代码漂移；一律不合并 `master`。
- 各分支流程与纪律：预注册单提交 → TDD → smoke → 筛查 → 独立复核 → 回填；每 run 间提交台账保 `dirty=false`；不重跑既有有效 run；不删除任何内容。

## 10. 成本估算（实测 10s/1M 行/epoch 外推；单 seed）

| 项 | 估算 |
|---|---|
| 臂 B + 臂 P（各 5 ep × 1.4M A） | ≈ 2 × 70s + val/test 评估 ≈ 3 min |
| B/C/test 评分（两臂前向 2.2M 行） | ≈ 1 min |
| verifier + 校准 + 五配置离线组配 | CPU < 1 min |
| **筛查 seed（含复核）** | **≈ 5–7 min GPU** |
| 扩展 seeds #2–5（各 1 run，无 controls） | ≈ 4 × 5 min ≈ 20 min |
| smoke + 单测 | ≤ 3 min |
| **全流程合计** | **≈ 30 min GPU（上限）**，串行、前台 |

- 子分支预留（不在本轮运行）：A 比例消融（如 60/20/20）≈ 同量级；`-iuv-crossfit`（严格 K 折交叉拟合，替代三分割）≈ OOF 重训 2 臂 × K 折 ≈ 13–17 min/seed。长跑一律前台（后台任务随会话退出被杀）。运行窗口：2026-10-06 国庆假期按用户规则允许；其它日期按既有高峰规则。

## 11. 结果（运行后回填，不回写第 0–10 节）

（待运行）

---

**非新颖性声明**：构件均为既有方法族（out-of-fold/交叉拟合估计、两模型路由/选择、学习型验证器、线性/岭回归打分）。本文件不声称独立于先行工作；可报告内容仅限本仓库冻结协议下的上述判定。
