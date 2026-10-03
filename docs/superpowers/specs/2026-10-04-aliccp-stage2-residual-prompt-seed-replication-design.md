# AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）的 canonical seed-2 复现（seed 1688723740；机制稳定性 + within-seed 效用方向）

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run 之前**单独提交**；commit C1）。结果只在第 10 节以新增小节回填；第 1–9 节的判据与数字不得在看到结果后改动。
- **日期**：2026-10-04
- **适用分支**：`exp/aliccp-stage2-residual-prompt-seed-replication`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**，不从 seed1 实现分支拉出；`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本与 `baseline/*` 零修改；不合并 `master`、不触碰其它 worktree）。
- **被复现对象（只读引用）**：
  1. seed1 实现分支 `exp/aliccp-stage2-residual-prompt-gate`：预注册 `221580a`、实现 `79ddefa`、结果 `3e2f083`（已 push）。其唯一有效 run `20261004-0214-p2M-v500k-t1M-m1688723512-short-79ddefa-rpg`：机制门禁 M0+G1–G8 全 PASS、A 类 PASS；`VALID_NEGATIVE`（U1 窄幅未过线：`Δtest = +0.005394543882337954 < +0.0055`，差 1.05e-4；U2 PASS：`Δval = +0.00448174078674779`）；`α_final = -0.060542766004800797`；三路 `ratio_mean ≈ 0.04480377`；三路 `cos_mean = +0.3582864391781954 / +0.4823664616168069 / +0.46924114503091274`；`pred_std = 0.003304574009664498`（参照 `0.0033728455401762676`）。
  2. seed1 设计文档 `docs/superpowers/specs/2026-10-03-aliccp-stage2-residual-prompt-gate-design.md`（下称"seed1 设计"，blob `34091fb3972e799b504fa43adac4f5416feade34`）——机制公式、门禁 M0+G1–G8、臂级三标签、报告清单**全部沿用**；本文件只做 seed/artifact/run-reference 级替换（§1.4）。
- **上游协议（只引用、不修改）**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（下称"AliCCP 评测协议"）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账全部沿用。
- **定位声明**：**独立 seed 复现**一项**工程消融**：seed1 上该机制"机制有效、效用窄幅未过线"（`VALID_NEGATIVE`，+0.0055 差 1.05e-4）。本实验在 canonical seed 列表位置 2（`1688723740`）上检验 (i) 机制是否跨 seed 复现、(ii) within-seed 效用方向与量级是否复现。**无新颖性主张、无改进主张、不放宽任何门限**；可报告内容仅限本仓库冻结协议下的双 seed 一次性判定（§6 复现状态 + pooled 报告）。

**提交时序（预注册纪律）**：本文件先于实现与运行**单独提交**（C1）→ 实现 + 测试（TDD；本次运行的 `commit` 字段即该提交 C2）→ **恰好一次** arm run（前台）→ 结果只在第 10 节与 `SUMMARY.md` 以**新增小节 / 新行**回填（C3）。基线臂不重跑（§7.3）。

---

## 0. 判定问题

- **Q1（机制复现）**：在 seed 2（`1688723740`）上，逐字移植（§1.4 适配清单）的范数受控残差 prompt 是否复现 seed1 形态：构造恒等（G1）、零初值恒等（G2）、门控/生成器梯度活（G3）、范数受控界（G4）、活性带（G5）、逐样本门控方差（G6）、无预测坍缩（G7）、参数预算（G8）、参照身份（M0）、A 类完整性全过？
- **Q2（within-seed 效用复现）**：在与参考基线（同 `stage1_id`、同 seed、同预算）的精确差值下，`Δtest ≥ +0.0055` 且 `Δval > 0`（**与 seed1 完全相同的 primary 门限，不放宽**）？方向（符号）是否与 seed1 一致？
- Q1/Q2 独立判定；组合为 §6 的复现状态（三项定义先于 run 写死）。

---

## 1. 机制审计与逐字移植（pin @ `79ddefa`）

### 1.1 被复现对象的钉死（内容寻址；2026-10-04 审计）

| 项 | 值 |
|---|---|
| 分支 / 实现提交 | `exp/aliccp-stage2-residual-prompt-gate` @ `79ddefa`（预注册 `221580a`；结果 `3e2f083`） |
| 机制实现 | `79ddefa:aliccp_benchmark/residual_prompt.py`（456 行；git blob `5dc7158ca999c9e7e6217a999c37869231411549`；LF-normalized sha256 `1c6131c7538718485ea7d7a65bd6e1d6d6a9c8dd21ba7645d66e62437020e996`） |
| 实现测试 | `79ddefa:aliccp_benchmark/tests/test_residual_prompt.py`（719 行；git blob `6484f5b8c9393ce22651d0f9161d81a0ef383f17`；LF-normalized sha256 `6a4a9c6ad6911606e69dce81c3116e36461f824da3f78ac188393b65296b6956`） |
| 运行接线 | `79ddefa:run_aliccp_benchmark.py`（blob `229c25a1c86719fa6fb6b057d6cff3f4120fa053`）与 `79ddefa:aliccp_benchmark/bench.py`（blob `bf3647686838157bf5fd7645615faad9f2d7185d`） |
| Census 祖本（沿用 seed1 的钉死，不改） | `99b9510:census_benchmark/residual_prompt.py`（blob `107221b26382da7fd44990e167d9a61da2680d92`；LF sha256 `8139e067a57fd3e144846ca5d39015a1d51657d55da67b9da9e4e06c007658df`）；其测试 blob `847097b7cb48a56c8593dd76d63a71a21a4c9e4f` |

### 1.2 机制要点（一句话版，逐字沿用 seed1 设计 §1.2）

逐样本方向 $P(x)=\tanh(W_2\,\mathrm{ReLU}(W_1x+b_1)+b_2)$（$x$=`dnn_input`，逐维有界）；有效门控 $g_{\text{eff}}(x)=|\alpha|\cdot\lVert P\rVert/\sqrt d$；残差 $\delta_s=\alpha\cdot(\lVert h_s\rVert/\sqrt d)\cdot P$ 注入**融合前的三路表征**（`gen_rep` + 两个 `spec_rep`），随后**原样委托** `super().forward`。结构性事实：S1 范数受控（$\le|\alpha|$）、S2 逐样本跨流一致、S3 α=0 ⇒ 与基线逐位一致。α 为标量 `nn.Parameter(torch.zeros(()))`；生成器构造在 `isolated_cpu_rng()` 内（不消耗共享随机流）。

### 1.3 维度 / 输入映射（与 seed1 相同的 AliCCP 事实，不变）

`dnn_input` `[B,80]`；`gen_rep` `[B,64]`；`spec_reps` 2 × `[B,64]`；`env_embs` 2 × `[64]`（机制不读取，只经 `super().forward` 透传）；注入流集合 = `("gen","spec_0","spec_1")` 三路；生成器 `Linear(80,16) → ReLU → Linear(16,64)`（hidden=16 不变，不搜索）；新增参数恰 5 键合计 **2385**（head **8129** 的 29.34%；总 10514）。优化器/损失沿用宿主协议 Stage-2 逐字：`Adam(newtask.parameters(), lr=1e-4)` + `BCELoss(pred,y) + newtask.get_l2_reg()`，batch 2000。

### 1.4 移植适配清单（**恰好以下 4 组**；其余以"钉死文本 + 恰好替换"等式与字节 sha 钉死）

| 编号 | 适配 | 内容 | 守卫 |
|---|---|---|---|
| A1′ | 机制文件 run-reference 常量与 docstring | `aliccp_benchmark/residual_prompt.py` 相对钉死 blob **恰好**两处文本改动：(a) 模块 docstring（指向本文件 + seed2 上下文 + `79ddefa` 系谱）；(b) 3 个 run-reference 常量行（值 + 行内注释）：`BASELINE_AUC_TEST 0.5988392178311113 → 0.5974422649550507`、`BASELINE_AUC_VAL 0.5781533414372665 → 0.5809347091990792`、`REFERENCE_PRED_STD 0.0033728455401762676 → 0.005217193225189258`。**其余逐字节一致** | 守卫测试：`working == pinned` 经"docstring 替换 + 3 行常量替换"的**重建等式**（每处替换前先断言旧文本在钉死文本中恰出现 1 次）；另 pin 工作树文件 LF sha256 |
| A2′ | 接线文件 | `run_aliccp_benchmark.py` 与 `aliccp_benchmark/bench.py` **逐字节**移植（相对 `8133d32` 的 diff 与 `79ddefa` 对该二文件的 diff 相同；无任何 seed 相关改动） | 守卫测试：工作树 LF sha256 == 钉死值（`2966ec39…` / `b063a366…`）；且 `git diff 8133d32` 的该二文件 diff 文本 == `git diff 8133d32 79ddefa` 的对应 diff 文本 |
| A3′ | 移植测试文件 | `aliccp_benchmark/tests/test_residual_prompt.py` 相对钉死 blob **恰好 4 处** AST/文本改动：(a) 模块 docstring；(b) 模块级 `DOC_PATH` → 本文件、`WHITELIST` → 7 项（seed1 的 6 项中 DOC_PATH 替换 + 新增本分支守卫测试文件；逐项集合钉死）；(c) `TestPreregConstants.test_thresholds_and_baseline_reference` 的 3 个期望值 → seed2 值；(d) `TestPreregConstants.test_prereg_doc_tokens_and_summary_ledger` 的 token 表 → 本文件 token 表（SUMMARY 行断言不变：`b2e17f9` 行恰 1 行含 `0.598839`/`0.578153`）。**其余逐类/逐方法/逐模块级语句不变**（含 `TestMechanismPin` 对 Census 祖本的全套 AST 钉死，逐字保留） | 守卫测试：重建等式（docstring + `DOC_PATH` 行 + `WHITELIST` 块 + 2 个方法 source segment 替换；`WHITELIST` 字面量集合逐项钉死） |
| A4′ | 新增守卫测试 | `aliccp_benchmark/tests/test_residual_prompt_seed_replication.py`（本分支新增；§8） | 本文件即为守卫 |

**被禁止的适配**：机制函数类/公式/初始化/门控语义/注入点/活性带/`BOUND_TOL`/G7 比值/U1 阈值/分类规则/protocol.py/metrics.py/model.py —— 零改动（静态守卫：`git diff --name-only 8133d32` ⊆ §改动清单）。

### 1.5 预注册前参照核验（只读；先于本文件提交；`verify_reference_seed2_residual_prompt.py`）

对固定产物 `s1-5c060b9c-m1688723740-e3-4e1b5c6f` + 基线 run `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07` 做只读复算（**21/21 通过**，报告 `artifacts/aliccp_bench/audit/reference-seed2-rp/verify_reference.json`；不写任何 run/stage1 产物、不训练）：

| 组 | 核对 | 结果 |
|---|---|---|
| 产物内容寻址 | `stage1_id` 重算 == 记录 == 期望；`config_hash` 重算 == `4e1b5c6f…`；`fingerprint_sha256 == 5c060b9c…`；`backbone_sha256` 重算 == `e5e7e610…`；`env_ids` sha 重算 == `5cd198f1…`；`env_ids` 形状 (2000000,)∈{0,1}（env_0=1532 / env_1=1998468） | PASS |
| 前缀指纹 A2 级 | `verify_fingerprint`：前缀字节 sha256 ×3 + 原始扫描标签计数 + 表头 + 自哈希；三文件字节数与指纹一致（2473647855 / 274769757 / 2711167840） | PASS |
| run 记录交叉 | config（stage1_id/seed/budgets/batch/lr/`spec_attenuation=1.0`/commit `79b5e07`/dirty=false）、metrics（epochs 5 / patience 2 / sha 链 / grads_none）、gate_report（A1–A6 PASS+S3 SKIP；B1/B2/B3 PASS、B4 FAIL、hard_pass=false） | PASS |
| 重评测（本分支评测路径 == 参考实现路径） | head `strict=True` 载入本分支 `NewTask(80/64/[32,32])` → **val BSI = 0.5809347091990792、test BSI = 0.5974422649550507、gate_mean = [0.789178, 0.2108221875] 与 run 记录逐位相等**（TOL 1e-9，实测差 0） | PASS |

- 核验脚本首版存在一处**检查器缺陷**（漏调 `protocol.freeze_backbone`，backbone 停留 train 模式、dropout 生效 → 重评测 AUC 偏低）；修正检查器后 21/21 通过。**产物与 run 未被任何修改**；该缺陷属检查器而非参照（gate_mean 在缺陷版即逐位一致，佐证 dnn_input 路径无关 dropout）。
- 钉死参照常量（后续判定全部引用）：
  - `BASELINE_AUC_TEST = 0.5974422649550507`、`BASELINE_AUC_VAL = 0.5809347091990792`（基线 run 记录，重评测逐位复核）；
  - `REFERENCE_PRED_STD = 0.005217193225189258`（参照头 val pred_std，fp64，n=500000，与运行期 `reference_head_stats` 同口径实测）；参照离散度全量：`pred_mean=0.9919248798495531`、`min=0.7844627499580383`、`max=0.9994847774505615`、`q05/q50/q95=0.9839656352996826/0.9929643869400024/0.9968113362789154`。
- **重要读法**（沿用 seed1 设计 §1.6）：BSI 正例率 ~98.6% ⇒ 基线预测天然集中；"无预测坍缩"（G7）必须用**相对参照**判据；参照值已在预注册前实测并钉死；运行期复算必须与之逐位一致（M0）。
- seed2 参照 pred_std（0.005217）显著大于 seed1 参照（0.003373）——为不同 backbone/head 的自然差异，如实记录、不解读为机制差异。

---

## 2. 预注册设计（实现即 §1.4 适配后的 `aliccp_benchmark/residual_prompt.py`）

机制（公式、初始化、`isolated_cpu_rng`、`prompt_deltas`、`forward` 注入与委托、`PromptAudit`、`PromptStats`、`evaluate_prompt_diagnostics`、`reference_head_stats`、`param_report`）与 `arm_verdict` 的 **M0+G1–G8+U 判定逻辑**逐字沿用 seed1；唯一改动为 §1.4-A1′ 的 run-reference 常量与 docstring。固定不搜索：hidden=16；α 初值 0；注入三路；RNG 隔离；新增参数 2385/8129。

**已知代价（如实记录）**：α=0 时生成器参数第一步梯度恒为 0（`∂δ/∂θ_generator ∝ α`），α 自身首步即有非零梯度；G3 把它写成显式探针口径、不当作缺陷。

**接线与影响面**：与 seed1 §2.3 相同——`bench.run_stage2` 关键字参数 `variant=RP.BASELINE_VARIANT`、`prompt_reference_newtask=None`（默认使基线臂逐位不变）；CLI `--variant` / `--prompt-reference-newtask` / `-rpg` 后缀；处理臂落盘 `prompt_report.json` + `metrics.json:rp_arm` + `gate_report.json:residual_prompt`（不并入 `hard_pass`）；处理臂未提供参照 → `ValueError` 拒绝出数。

---

## 3. 效用判据（U；看到结果之前写定；**与 seed1 完全相同的门限，不放宽**）

对固定基线 run `20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07`（同 `stage1_id`、同模型种子、同预算；`AUC-Val-BSI(best)=0.5809347091990792`、`AUC-Test-BSI=0.5974422649550507`）取**精确差值**：

| 编号 | 判据 | 落盘字段 |
|---|---|---|
| **U1** | `Δtest = test_bsi_arm − 0.5974422649550507 ≥ +0.0055`（用户指定 provisional 阈值；比评测协议 10.1 的 +0.005 更严） | `metrics.json:rp_arm.U.U1` |
| **U2** | `Δval = val_bsi_arm − 0.5809347091990792 > 0`（val 方向为正，防单侧噪声） | `metrics.json:rp_arm.U.U2` |

- `U.pass ⇔ U1 ∧ U2`。单 seed 单次判定；不做多 seed 平均。
- **报告项（不判定）**：评测协议 10.1 provisional 双条件（`Δtest ≥ +0.005` ∧ `Δval > 0`）为 U1∧U2 所蕴含，原样报告；禁止用 CVR/CTR 差值宣称改进（评测协议 10.1）。

## 4. 机制门禁（M0 + G1–G8；与 seed1 逐字相同；全部硬判据）

| 编号 | 判据 | 实测口径 |
|---|---|---|
| **M0 参照身份** | 运行期 `reference_head_stats` 复算：`|ref_val_auc − 0.5809347091990792| ≤ 1e-9` 且 `|ref_pred_std − 0.005217193225189258| ≤ 1e-9` | 处理臂专用（只读参照头、零训练）；不满足 = 参照/产物/装载不一致，本次不作效用解释 |
| **G1 构造恒等** | 共享参数/buffer 与基线 `NewTask`（同维、从构造前 RNG 现场重建）**逐位相同**；构造后全局 RNG 端点逐位一致；新增键**恰为** 5 键 | 运行期 `construction_identity` |
| **G2 初值恒等（gate zero）** | 构造时 `α == 0.0`；**真实第一个训练 batch** 上处理头前向与基线头前向 `torch.equal` | 运行期探针（优化器第一 step 之前） |
| **G3 门控/生成器梯度活** | ① 首 batch 反传后 α 梯度 ≠ 0；② 末个 epoch 首个 batch 反传后 α ≠ 0 且生成器梯度范数 > 0；③ `best_state` 载入后 `α_final ≠ 0` | 逐 epoch `grad_probe` |
| **G4 范数受控界** | val 上所有样本、所有三路：`ratio_max ≤ |α_final| + 1e-6` | `prompt_report.json:val_stats`（fp64 流式） |
| **G5 活性带** | val 上每一路平均 `ratio ∈ [0.005, 0.5]`（含端点） | 同上 |
| **G6 逐样本门控方差** | val 逐样本 `g_eff`：`std > 0` ∧ `max > 0` ∧ `min ≥ 0` | 同上 |
| **G7 无预测坍缩** | val：`pred_std > 0` ∧ `pred_std ≥ 0.5 × ref_pred_std`（相对参照；参照缺失/不合规 → FAIL） | `dispersion` + `reference` |
| **G8 参数预算** | 键集恰 5 键；`new_params_total == 2385`；`head_params == 8129` | `prompt_report.json:params` |
| **A 类** | A1/A2/A4/A5/A6 **必须 PASS**（A3 SKIP）。任一 FAIL ⇒ 本臂比较作废 | `gate_report.json`（协议 machinery，不改） |

**B 类说明**：B4 的 FAIL 为**继承事实**（§8）；B1/B2/B3 原样记录（注意：seed2 产物上 B1 = **PASS**，与 seed1 线的继承 B1 FAIL 不同——如实披露，不归因于本臂）。

## 5. 臂级结局分类（先定后跑；与 seed1 逐字相同的三标签）

| 触发情形 | 分类 | `subreason` | 处置 |
|---|---|---|---|
| M0 FAIL | `MECHANISM_FAIL` | `REFERENCE_IDENTITY` | 停止；不作效用解释 |
| G1/G2/G4/G8 任一 FAIL | `MECHANISM_FAIL` | `INVALID_IMPLEMENTATION` | 停止；仅当能证实具体代码缺陷时允许修复后重跑一次（留痕、不改判据） |
| G3 FAIL | `MECHANISM_FAIL` | `MECHANISM_INACTIVE` | 停止 |
| G5 低于下界 | `MECHANISM_FAIL` | `MECHANISM_SILENT` | 停止 |
| G5 高于上界 | `MECHANISM_FAIL` | `MECHANISM_OVER_PERTURB` | 负结果不得归因于"prompt 假设无效"；停止 |
| G6 FAIL | `MECHANISM_FAIL` | `GATE_DEGENERATE` | 停止 |
| G7 FAIL | `MECHANISM_FAIL` | `PREDICTION_COLLAPSE` | 停止 |
| A 类任一 FAIL | `MECHANISM_FAIL` | `PROTOCOL_INVALID` | 比较作废；停止 |
| M0+G1–G8 全过 ∧ A 类过 ∧ U1∧U2 | **`VALID_POSITIVE`** | — | 记录 |
| M0+G1–G8 全过 ∧ A 类过 ∧ ¬(U1∧U2) | **`VALID_NEGATIVE`** | — | 有效阴性：机制在工作但未过效用门槛 |

## 6. 复现状态（pooled；**先于 run 写死**；机制必须双 seed 通过）

**分量定义**：

- `M_this` ⇔ 本次 run M0+G1–G8 全 PASS ∧ A 类 PASS（A1/A2/A4/A5/A6；A3 SKIP）。
- `M_both` ⇔ `M_this` ∧ seed1 机制记录全 PASS（@`3e2f083`：M0+G1–G8+A 类全 PASS，已记录）。
- `P1` ⇔ seed1 U1∧U2（已记录：U1 **FAIL**，`Δtest=+0.005394543882337954 < +0.0055`；U2 PASS）⇒ **`P1 = false`（固定事实）**。
- `P2` ⇔ 本次 U1∧U2（§3；门限与 seed1 完全相同）。
- `D1` ⇔ sgn(seed1 Δtest)>0 ∧ sgn(seed1 Δval)>0（已记录：**true**）；`D2` ⇔ sgn(本次 Δtest)>0 ∧ sgn(本次 Δval)>0。

**状态（优先级自上而下；标签集固定为以下三者）**：

| 优先级 | 状态 | 定义 | 含义（写死） |
|---|---|---|---|
| 1 | **`REPLICATION_SUPPORTED`** | `M_both ∧ P2` | 机制在两个 canonical seed 上均全门禁复现，且**本复现 seed 自身**越过未放宽的 primary 效用门（+0.0055 ∧ val>0）。seed1 的窄幅未过线以 per-seed 表原样披露（不隐藏、不重判） |
| 2 | **`DIRECTION_REPLICATED_THRESHOLD_NOT_CLEARED`** | `M_both ∧ ¬P2 ∧ D2 ∧ ¬P1` | **本预注册显式定义的"窄幅未过线复现"状态**：机制双 seed 复现、效用方向双 seed 同为正，但 **+0.0055 在任一 seed 均未清除**（`¬P1` 由已记录的 seed1 结果满足）。状态名即声明阈值未清除；不构成提升主张，不得与 1 号混读 |
| 3 | **`NOT_SUPPORTED`** | 其余情形 | 分量披露：`¬M_this` ⇒ 本 seed 机制未复现（臂级 `MECHANISM_FAIL`，按 §5 处置）；或 `¬D2` ⇒ 方向不一致 |

**pooled 报告（report-only；不得改变上述状态）**：

- **per-seed 表**（逐 seed 与其同 seed 基线配对，跨 seed 不混）：`Δtest`、`Δval`、U1、U2、`α_final`、三路 `ratio_mean`、三路 `cos_mean`、`pred_std`/`ref_std`、臂级 classification。
- **符号一致性**：`sgn(Δtest)` 跨 seed 是否一致、`sgn(Δval)` 跨 seed 是否一致（布尔，逐项披露）。
- **算术均值**：`Δtest_mean = (Δtest_s1 + Δtest_s2)/2`、`Δval_mean` 同式——**描述性仅此**；n=2、无 run-to-run 噪声估计（A3 SKIP）、不做假设检验；**均值在任何情形下不得替代 P1/P2 的逐 seed 门限判定**（即便 mean ≥ +0.0055 而 P2=false，状态仍按 2 号处理）。
- **机制 pooled**：`α_final`、ratio、cos、geff、pred_std（含参照）双 seed 并列比较（描述）。
- 两数据集预测尺度不同（BSI 正例 ~98.6%）不可横比（沿用 seed1 §8 读法）。

## 7. 运行规程

### 7.1 唯一一次 arm run（前台；结果 run 恰 1 次）

```powershell
# cwd = 本 worktree 根；Stage-1 产物与基线 run 已就位（只读、内容寻址、§1.5 已核验）
# 本 worktree 无 .venv，用主树 venv 的解释器：
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --model-seed 1688723740 `
    --stage1-id s1-5c060b9c-m1688723740-e3-4e1b5c6f `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07/newtask.pt
```

- 预算 2M/500k/1M、`env_seed=20261003`（经产物 env_ids，A5）、epochs=5、patience=2、batch 2000、lr 1e-4 全部为协议默认，**不新增/不覆盖任何超参**；`run_id` 由 runner 追加 `-rpg` 后缀。`model_seed=1688723740` 为本次唯一显式 seed 参数（复现设计即如此；seed1 run 用其自身默认 `1688723512`）。
- **前台执行并等待至完成**（seed1 §9.0 记载两次后台执行被会话终止杀死；本预注册明确要求前台）。
- 运行前：全量单元测试绿（`-m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests`）+ 静态守卫 + 字节/AST 钉死守卫；工作树已提交干净（`git_state.dirty=False`，未跟踪文件不计入）。
- 运行后：**恰好一次**；不因结果好坏重跑、不调参、不换 seed/产物；`SUMMARY.md` 只追加。

### 7.2 无效执行处置

仅当 run 因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可**重跑一次**；因结果原因不构成无效。无任何事后调参救结果（hidden/初始化/α 初值/注入流集合/活性带/阈值/参照文件均不得动）。

### 7.3 基线臂不重跑

基线臂精确值取自已记录的 run `20261003-0624-…-79b5e07`（§1.5 已核验；其 `spec_attenuation=1.0` 分支对 stage2 计算路径为整句跳过——`8133d32→79b5e07` 的 `model.py/bench.py` 改动仅在 `spec_attenuation != 1.0` 时生效，重评测逐位一致已证）。

## 8. 继承事实（不得归因于本臂）

- **B4 = FAIL，继承自固定产物**：Stage-1 `cluster_events = [(epoch 2, diff_num 1000418, env_0 1532, env_1 1998468)]` ⇒ env_0 占比 0.0766% < 5%（raw 聚类在该 seed 上的已知退化形态；本分支**不做**归一化聚类修复——那是另一实验线，禁止混入）。
- **B1 = PASS，继承自固定产物**（与 seed1 线的 B1 FAIL 不同）：CTR `0.5534062000766764 ≥ 0.55`、CVR `0.5420522697385088 ≥ 0.5`、BSI 腿 = 本臂 test AUC `≥ 0.53`。
- 因此本臂 `hard_pass = false` 为**预期值**（仅由 B4 决定），与本臂无关；本臂有效性定义 = M0 + G1–G8 + A 类（§4/§5），**不依赖 `hard_pass`**。
- **A3 = SKIP**（协议：按需复跑，不阻塞首轮）。

## 9. 必录诊断（落盘 `runs/<run_id>/prompt_report.json`；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| α / 梯度轨迹 | `grad_probe`（逐 epoch 首 batch 的 `alpha`、`alpha_grad_norm`、`generator_grad_norm`）、`alpha_final` | G3 + 与 seed1（0→−0.0623→…→−0.0605）对照 |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean`（流序 `("gen","spec_0","spec_1")`） | G4/G5 + 与 seed1（h_norm 38.60/28.82/20.78；ratio 0.0448 三路一致）对照 |
| 余弦方向 | 逐流 `cos_mean` | 与 seed1（+0.3583/+0.4824/+0.4692）对照 |
| 逐样本门控 | `gate.geff_mean/std/min/max` | G6 + 对照 seed1（0.04480377/0.00391039/0.01527102/0.05277382） |
| 预测离散度 | `dispersion` + `reference_dispersion`（M0 口径复算） | G7 + 对照（seed2 参照 std 0.005217193225189258） |
| 源/头门 | `gate_mean`（2 维均值，B3 量）+ Stage-1 环境上下文（`cluster_events`、`env_acc`） | 落盘；B3 原样披露 |
| 参数清单 | `params`：逐参数 `name/shape/numel` + `new_params_total=2385`、`head_params=8129` | G8 |
| 构造恒等审计 | `construction_identity`（G1 三项） | G1 |
| 臂级判定 | `arm`（M0+G1–G8+U1/U2+classification+subreason） | §5 机械分类 |
| **cross-seed** | §6 的 per-seed 表 + 符号一致性 + 均值 + 机制 pooled | §6 复现状态与 pooled 报告 |

## 10. 结果（运行后回填，不回写第 1–9 节）

### 10.0 非结果核验

- **§1.5 参照核验**（预注册前，只读）：21/21 通过（含 `verify_fingerprint` A2 级复算与 val/test/gate_mean 逐位重评测）；检查器缺陷（漏 `freeze_backbone`）已在 §1.5/§13 披露，修正后通过，产物未动。
- **运行后独立复核**（§10.5）：27/27 ALL_PASS；不复核任何训练/重评测，只从 JSON 重推。
- 无其它非结果 run；无 canonical 产物写入。

### 10.1 唯一有效 run 与臂级分类

- `run_id = 20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg`；`commit = 013e105`（C2）；`stage1_id = s1-5c060b9c-m1688723740-e3-4e1b5c6f`；**前台**执行、wall 156.4s、peak VRAM 55.7MB；`git.dirty=false`；budgets/seed/epochs(5)/patience(2)/batch(2000)/lr(1e-4) 全部为协议默认。
- **臂级分类：`VALID_POSITIVE`**（M0 + G1–G8 全 PASS ∧ A 类 PASS ∧ U1∧U2；§5 机械分类）；`arm.pass = true`；`hard_pass = false`（仅 B4 继承 FAIL，§8 预期值）。
- 按 §7.1：**恰好一次**运行、无重跑、无中断、无调参（§10.6）。

### 10.2 效用（U1/U2 实测）

| 量 | 实测 | 阈值 | 结果 |
|---|---|---|---|
| Δtest | **+0.008103113327467715** | ≥ +0.0055 | **U1 PASS**（余量 +0.0026031） |
| Δval | **+0.008622497456618139** | > 0 | **U2 PASS** |

- `test_bsi = 0.6055453782825184`（基线 0.5974422649550507）；`best_val_auc_bsi = 0.5895572066556973`（基线 0.5809347091990792）；`best_epoch = 5/5`；逐 epoch val：0.4651 / 0.4905 / 0.5231 / 0.5598 / 0.5896（train_loss 0.0819 → 0.0478）。
- 报告项（不判定，原样落盘）：评测协议 10.1 provisional 双条件（`Δtest ≥ +0.005` ∧ `Δval > 0`）均满足。

### 10.3 机制门禁与诊断（M0 + G1–G8 全 PASS）

| 门禁 | 实测 |
|---|---|
| M0 参照身份 | `ref_val_auc = 0.5809347091990792`、`ref_pred_std = 0.005217193225189258`，与预注册常量逐位一致（≤1e-9；与 §1.5 预注册前实测亦逐位一致） |
| G1 构造恒等 | 共享参数逐位相同、全局 RNG 端点一致；新增键恰 5 键 |
| G2 初值恒等 | `α@构造 = 0.0`；真实首 batch 前向逐位一致（`max_abs_diff = 0.0`，n=2000） |
| G3 门控活 | 逐 epoch（α, α_grad, gen_grad）：(0, 0.05111, **0.0**) / (0.01644, 0.10020, 0.02577) / (0.03746, 0.22530, 0.08280) / (0.05092, 0.15703, 0.11414) / (0.06157, 0.06819, 0.12780)；`α_final = +0.07101669907569885`。首步 gen_grad=0 为 §2 预注册已知代价 |
| G4 范数受控 | 三路 `ratio_max = 0.04905545…` ≤ 0.07101670 + 1e-6 |
| G5 活性带 | 三路 `ratio_mean = 0.02813938582…` ∈ [0.005, 0.5]（三路 spread 3.3e-12） |
| G6 门控方差 | `geff` mean/std/min/max = 0.02813938582434281 / 0.006022934967912283 / 0.010255216134496864 / 0.04905545797600925 |
| G7 无坍缩 | `pred_std = 0.00494759789652526` = 0.948 × 参照（≥ 0.5 × 0.005217193225189258） |
| G8 参数预算 | 恰 5 键；`new_params_total = 2385`、`head_params = 8129`（29.34%，total 10514） |

逐流读数（val，n=500000，fp64 流式），流序 `("gen","spec_0","spec_1")`：

| 流 | ratio_mean | delta_norm_mean | h_norm_mean | cos_mean |
|---|---|---|---|---|
| gen | 0.02813939 | 4.8989 | 168.97 | **+0.01287** |
| spec_0 | 0.02813939 | 0.6768 | 23.98 | **+0.08432** |
| spec_1 | 0.02813939 | 0.6360 | 22.45 | **+0.05899** |

预测离散度：`pred_mean/std/min/max = 0.9923957079 / 0.0049475979 / 0.7908765078 / 0.9995243549`；`q05/q50/q95 = 0.9849603593 / 0.9933705926 / 0.9970371783`。源/头门：`gate_mean = [0.7686165625, 0.23138340625]`（B3 量）；Stage-1 环境上下文 `cluster_events = [(epoch 2, 1000418, 1532, 1998468)]`、`env_acc = 0.9992525`（与产物 meta 逐位相同，原样披露）。

A 类：A1/A2/A4/A5/A6 PASS，A3 SKIP。B 类：B1 PASS（CTR 0.5559 / CVR 0.5028 继承；BSI 腿 = 本臂 0.6055 ≥ 0.53）、B2 PASS（|val−test| = 0.0160 ≤ 0.05）、B3 PASS、B4 FAIL（继承，`events=[(1532, 1998468)]` 与基线逐字相同）。

### 10.4 与 seed1 对照 / pooled 复现状态（§6 定义）

| 量 | seed1（m1688723512，记录 @`3e2f083`） | **seed2（m1688723740，本 run）** |
|---|---|---|
| Δtest | +0.005394543882337954（U1 FAIL，差 1.05e-4） | **+0.008103113327467715（U1 PASS）** |
| Δval | +0.00448174078674779（U2 PASS） | **+0.008622497456618139（U2 PASS）** |
| test_bsi / val_bsi | 0.6042337617134492 / 0.5826350822240143 | 0.6055453782825184 / 0.5895572066556973 |
| α_final | -0.060542766004800797 | **+0.07101669907569885**（符号相反） |
| ratio_mean（三路） | 0.04480377 | 0.02813939 |
| cos_mean（三路） | +0.3583 / +0.4824 / +0.4692 | **+0.01287 / +0.08432 / +0.05899** |
| geff mean/std/min/max | 0.04480377 / 0.00391039 / 0.01527102 / 0.05277382 | 0.02813939 / 0.00602293 / 0.01025522 / 0.04905546 |
| pred_std（参照） | 0.0033045740（0.0033728455） | 0.0049475979（0.0052171933） |
| 臂级分类 | VALID_NEGATIVE | **VALID_POSITIVE** |

- **符号一致性**：sgn(Δtest) 跨 seed 一致（+,+）= **True**；sgn(Δval) 跨 seed 一致（+,+）= **True**。
- **pooled 均值（描述性，不判定）**：`Δtest_mean = +0.0067488286049028345`、`Δval_mean = +0.006552119121682964`；n=2、A3 SKIP、无 run-to-run 噪声估计、不做假设检验；均值未参与任何状态判定（本 run 的 P2 由自身单独过线）。
- **分量**：`M_this = True`（M0+G1–G8+A 类）；`M_both = True`（seed1 记录全 PASS）；`P1 = False`（已记录事实）；`P2 = True`；`D1 = True`；`D2 = True`。
- **复现状态（§6 机械判定，优先级 1）：`REPLICATION_SUPPORTED`** —— 机制在两个 canonical seed 上均全门禁复现，且本复现 seed 自身越过未放宽的 primary 效用门（+0.0055 ∧ val>0）。
- 读法（不构成改进主张）：跨 seed 差异集中在 (a) α 符号相反（+0.0710 vs −0.0605）、ratio_mean 小约 1.6 倍；(b) cos 三路近零（+0.013/+0.084/+0.059，seed1 为 +0.36/+0.48/+0.47）；(c) 效用方向两 seed 同为正，量级 seed2 更大（+0.0081/+0.0086 vs +0.0054/+0.0045），seed2 越过 +0.0055 而 seed1 窄幅未过——读作 seed 敏感的一次性判定，不读作提升主张。两 seed 参照 pred_std 不同（0.00522 vs 0.00337）为不同 backbone/head 的自然差异。

### 10.5 独立复核（`verify_stage2_residual_prompt_seed2.py`，运行后）

- 独立脚本（**不 import 机制模块**，纯 JSON 重推 M0/G1–G8/U1/U2/臂级分类/§6 复现状态）：**27 项检查全过（exit 0）**，`verify_report.json` 落盘 run 目录；记录臂级分类 `VALID_POSITIVE`/None 与独立重算一致；pooled 状态独立重推 = `REPLICATION_SUPPORTED`（`M_this/M_both/P1/P2/D1/D2 = True/True/False/True/True/True`；mean Δtest `0.0067488286049028345`、mean Δval `0.006552119121682964`；符号一致性 True/True）。
- 三份 JSON（`metrics.json` / `gate_report.json` / `prompt_report.json`）逐字互洽（`rp_arm == residual_prompt == arm`）；冻结三件套、env_ids、指纹、stage1 身份、B1 的 CTR/CVR 腿继承与 BSI 腿本臂渲染、B4 继承逐字、`hard_pass=false`、SUMMARY 行、run 元数据全部一致。
- 检查器纪律：seed1 verifier 修订版的两项修正（R1 B1 口径 / R2 U1/U2 一致性语义）**自始并入**本脚本（脚本头写明）；本脚本无需事后修订。

### 10.6 止损执行（§7/§11 落实）

- 恰一次**有效** arm run（前台；无工具性中断；无重跑）；未扩第三 seed、未执行 A3 复跑、未跑 `full`、未做 FLOPs、未做任何事后调参。
- 未触碰 Stage-1（不重训/不微调/不重新初始化）；未改机制超参（hidden=16、初始化、α 初值、注入流集合、活性带、`BOUND_TOL`、G7 比值、U1 阈值、参照文件、分类规则、§6 状态定义均未动）；基线臂未重跑（§7.3）。
- B 类失败（B4）原样披露；继承事实未归因于本臂；继承的 B1 PASS（与 seed1 线不同）同样如实披露。

## 11. 非目标与止损

- 不主张方法新颖性；不写"我们提出"；不改协议文件与模型语义；不动 Stage-1（不重训、不微调、不重新初始化 backbone）；不跑 `full` tag；不做 FLOPs。
- 不做：归一化聚类修复、TC-Prompt、CGR、affinity gate、KL-Prompt、温度/损失改动（评测协议排除清单有效）。
- 不因结果修改：hidden=16、初始化、α 初值、注入流集合、活性带、`BOUND_TOL`、G7 比值、U1 阈值、参照文件、分类规则、复现状态定义。
- **恰好一次** arm run；`DIRECTION_REPLICATED_THRESHOLD_NOT_CLEARED` 或 `NOT_SUPPORTED` 均如实记录、停止；不扩第三 seed、不做 A3 复跑（除非用户显式指示）。

## 12. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一），逐条先行工作清单沿用 seed1 设计 §11（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库这套冻结协议下的双 seed 一次性判定。

## 13. 偏离披露（预声明 + 运行后补）

- **分支工作流**：自 `infra/aliccp-fair-benchmark` @ `8133d32` 拉出（不从 seed1 分支拉出）；机制经 §1.4 字节/重建等式守卫钉死等价。协议未合并入 master，与同批 exp 分支先例一致。
- **分支本地 SUMMARY 系谱**：`8133d32` 的 `SUMMARY.md` 含 smoke 行与 seed1 基线行（`20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`）；raw seed2 基线行（`79b5e07`）记录在其它分支（`f589e61`），本分支**不复制**其它分支行，只由本实验 run 追加自己的行。
- **参照核验检查器缺陷**：§1.5 已披露（漏 freeze；修正检查器后 21/21；产物未动）。
- **非结果文件**：`artifacts/aliccp_bench/audit/reference-seed2-rp/`（未跟踪）；预注册/复核脚本置于 worktree 根（未跟踪；同 seed1 先例）。
- **非结果核验**：§1.5 参照核验不入 canonical 产物、不计为结果 run。
- **执行记录（运行后补）**：C1（`e931cd7` 预注册）→ C2（`013e105` 移植 + 守卫，88/88 测试绿、`git.dirty=false`）→ 前台单次 arm run（`20261004-0325-…-013e105-rpg`，wall 156.4s，**无工具性中断、无重跑**）→ C3（本 §10/§13 回填 + SUMMARY 行）。运行命令与 §7.1 逐字一致（cwd=本 worktree，主树 venv 解释器）。
- **脚本落盘（未跟踪，同 seed1 先例）**：`verify_reference_seed2_residual_prompt.py`（预注册前参照核验，报告在 `artifacts/aliccp_bench/audit/reference-seed2-rp/verify_reference.json`）、`verify_stage2_residual_prompt_seed2.py`（运行后独立复核，报告在 run 目录 `verify_report.json`）。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md` | 本文件（C1 单独提交） |
| `aliccp_benchmark/residual_prompt.py` | 新增（§1.4-A1′ 适配后的移植机制 + 诊断 + 门禁 + 接线函数） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（§1.4-A3′ 适配后的移植测试） |
| `aliccp_benchmark/tests/test_residual_prompt_seed_replication.py` | 新增（本分支守卫测试；§8 之外、§1.4-A4′） |
| `run_aliccp_benchmark.py` | 逐字节移植（§1.4-A2′） |
| `aliccp_benchmark/bench.py` | 逐字节移植（§1.4-A2′） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加一行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
