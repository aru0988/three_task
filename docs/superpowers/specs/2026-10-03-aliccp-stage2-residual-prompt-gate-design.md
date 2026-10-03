# AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）：跨数据集复现预注册与设计

- **状态**：预注册已写死（本文件在本分支任何实现、任何 run 之前**单独提交**）。结果只在第 9 节以新增小节回填；第 1–8 节的判据与数字不得在看到结果后改动。
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage2-residual-prompt-gate`（自 `infra/aliccp-fair-benchmark` @ `8133d32` **独立拉出**；`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、`multitaskrec/*`、`config.py`、master 既有脚本与 `baseline/*` 零修改；不合并 `master`、不触碰其它 worktree）。
- **上游协议（只引用、不修改）**：
  1. `docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（下称"AliCCP 评测协议"）——前缀预算、种子、Stage-1 产物、真冻结三件套、A/B/C 门禁、SUMMARY 台账**全部沿用**；
  2. Census 侧被复现对象：`exp/stage2-residual-prompt-gate` 的 `docs/superpowers/specs/2026-10-01-stage2-residual-prompt-gate-design.md`（下称"Census 设计"），预注册 commit `e87c945`、实现 commit `99b9510`、结果 commit `e0ad6af`。
- **定位声明**：**跨数据集复现**一项**工程消融（practical ablation）**：Census 上该机制"机制有效但效用为负"（`VALID_NEGATIVE`，Δtest −0.00233）；本实验检验**同一预注册形式**在 AliCCP 新任务（BSI）上是否有**不同的适用性画像**。**无新颖性主张、无改进主张**；可报告内容仅限：在 AliCCP 严格冻结协议下该形式是否过效用阈值、机制是否在工作、与两个参照（AliCCP 基线 run / Census 同机制读数）的差。非新颖性声明见第 11 节。

**提交时序（预注册纪律）**：本文件先于实现与运行**单独提交**（prereg commit）→ 实现 + 测试提交（TDD，本次运行的 `commit` 字段即该提交）→ **恰好一次** short 运行 → 结果只在第 9 节与 `SUMMARY.md` 以**新增小节 / 新行**回填。

---

## 1. 机制审计与移植（先审计，再动手）

### 1.1 被复现对象的钉死（内容寻址）

| 项 | 值 |
|---|---|
| 分支 | `exp/stage2-residual-prompt-gate` |
| 机制实现 | `99b9510:census_benchmark/residual_prompt.py`（385 行；git blob `107221b26382da7fd44990e167d9a61da2680d92`；LF-normalized sha256 `8139e067a57fd3e144846ca5d39015a1d51657d55da67b9da9e4e06c007658df`） |
| 实现测试 | `99b9510:census_benchmark/tests/test_residual_prompt.py`（456 行；git blob `847097b7cb48a56c8593dd76d63a71a21a4c9e4f`） |
| 运行接线 | `99b9510:run_census_benchmark.py`（相对其父提交的 diff：`--variant` / `--prompt-reference-newtask` / run_id 后缀 `-rpg` / `prompt_report.json` / `metrics.json:rp_arm` / `gate_report.json:residual_prompt` 子对象） |
| 结果 | `e0ad6af`：run `20261001-0123-s20260929-m1685480945-short-99b9510-rpg`，`VALID_NEGATIVE`；G1–G5 全过；α_final 0.314076；三路 ratio_mean 0.127718；Δtest −0.0023308435 |

### 1.2 Census 机制要点（一句话版）

逐样本方向 $P(x)=\tanh(W_2\,\mathrm{ReLU}(W_1x+b_1)+b_2)$（$x$=`dnn_input`，逐维有界）；有效门控 $g_{\text{eff}}(x)=|\alpha|\cdot\lVert P\rVert/\sqrt d$；残差 $\delta_s=\alpha\cdot(\lVert h_s\rVert/\sqrt d)\cdot P$ 注入**融合前的三路表征**（`gen_rep` + 两个 `spec_rep`），随后**原样委托** `super().forward`。结构性事实：S1 范数受控（$ratio_s=\lVert\delta_s\rVert/\lVert h_s\rVert=g_{\text{eff}}\le|\alpha|$）、S2 逐样本跨流一致、S3 α=0 ⇒ 与基线逐位一致。α 为标量 `nn.Parameter(torch.zeros(()))`；生成器构造在 `isolated_cpu_rng()` 内（不消耗共享随机流）。

### 1.3 维度 / 输入映射（Census → AliCCP Stage2）

| 符号 | Census（123/128） | AliCCP（本实验） | 来源与依据 |
|---|---|---|---|
| `dnn_input` $x$ | `[B,123]` | **`[B,80]`** | 16 特征 × embedding 5（评测协议 2.2） |
| `gen_rep` | `[B,128]` | **`[B,64]`** | `expert_dnn_hidden_units=(128,64)[-1]`（评测协议 8.1） |
| `spec_reps` | 2 × `[B,128]` | **2 × `[B,64]`** | Stage-1 `num_tasks=2`（CTR/CVR）⇒ 同流数 |
| `env_embs` | 2 × `[128]` | **2 × `[64]`** | 只经 `super().forward` 透传，机制不读取 |
| 注入流集合 | `gen` + `spec_0` + `spec_1` | **同左（3 路）** | `PromptStats.stream_names` 恰为 `("gen","spec_0","spec_1")`，无需扩张 |
| 生成器 $W_1$ | `16×123` | **`16×80`** | hidden=16 不变（Census 超参，不搜索） |
| 生成器 $W_2$ | `128×16` | **`64×16`** | 输出维 = `rep_dim` |
| 新增参数 | 4161（head 27063 的 15.376%） | **2385（head 8129 的 29.34%）** | 实测核算：1296+1088+1 / 8129（测试钉死） |
| 优化器 / 损失 | Adam lr 1e-3 + BCE + `get_l2_reg()` | **Adam lr 1e-4 + BCE + `get_l2_reg()`** | **宿主协议继承**，见 1.4-7 |
| 注入点 / 委托 | 融合前，`super().forward` | **同左** | 语义逐字移植 |

### 1.4 数据集特定假设逐项裁决

| # | Census 机制中的潜在假设 | AliCCP 事实 | 裁决 |
|---|---|---|---|
| 1 | 生成器输入维 123（embedding+稠密拼接） | `dnn_input`=80（纯 embedding，无稠密） | **无假设**：生成器只接收浮点向量；`input_size` 本为参数 |
| 2 | 表征维 128 | `rep_dim`=64 | **无假设**：`rep_dim` 本为参数；`prompt_deltas` 内含维度一致性检查（不一致即拒绝出数） |
| 3 | 流数 = 3（gen + 2 spec） | Stage-1 任务数同为 2 ⇒ 3 流 | **成立**：`PromptStats` 三流结构与 `len(spec_reps)==2` 的运行时校验均不变 |
| 4 | `h.detach()` 取模长 | backbone 冻结、`get_infos` 在 `no_grad` 下 | **成立**：detach 语义在两侧一致 |
| 5 | 环境数 / env 语义 | 均为 2，且机制不使用 `env_embs` | **无关** |
| 6 | 指标模块 API：`metrics.auc(y_true,y_hat)` | 本仓为 `metrics.auc_score(y_true,y_hat)`（实现逐字相同：`roc_auc_score(y_true.int().cpu(), y_hat.detach().cpu())`） | **机械改名适配 A2**（见 1.5） |
| 7 | Stage-2 优化器 lr=1e-3（Census 协议常量） | AliCCP 评测协议 Stage-2 lr=1e-4（`protocol.LR`） | **非机制超参**：机制不携带自己的优化器设置；两臂一律继承宿主协议 Stage-2 训练配置（Adam、`protocol.LR`、batch 2000、`BCE + get_l2_reg()`）——这是"同 backbone 配对、单变量（头变体）"的受控条件，不是移植适配 |
| 8 | 标签语义（Education 二分类） | BSI（raw==2→0 其余→1，正例 98–99%） | **无关**：机制不触标签；BCE 为宿主协议既有损失 |

**裁决：机制可在不改变其假设的前提下移植**——被移植的是"上下文生成 + 逐维有界 + 范数受控 + 零初始化标量门控 + 融合前三路注入"这一整体；两个数据集仅在**维度**（123/128 → 80/64）与**宿主协议常量**（阈值、lr）上不同，机制函数类、幅度机制、注入位置、初值恒等语义全部保持。**不触发"须停止并报告"的条件**。

### 1.5 移植适配清单（**恰好**以下 6 项；其余 AST 逐字一致，由测试守卫）

| 编号 | 适配 | 内容 | 守卫 |
|---|---|---|---|
| A1 | 模块与导入路径 | `census_benchmark` → `aliccp_benchmark`（`from aliccp_benchmark import metrics`） | AST 守卫（导入语句豁免、单独列出） |
| A2 | 指标改名 | `reference_head_stats` 内 `metrics.auc(...)` → `metrics.auc_score(...)`（纯改名，实现逐字相同） | AST 守卫：把移植文件的该调用名归一化回 `auc` 后，`reference_head_stats` 的 AST 必须与钉死 blob **完全相等** |
| A3 | 常量块 | 保留 `PROMPT_HIDDEN=16`、`RATIO_BAND=(0.005,0.5)`、`BOUND_TOL=1e-6`、`VARIANTS`、`RUN_ID_SUFFIX="-rpg"`、`EXTRA_PARAM_NAMES`；**删除** Census 的 `AUC_TEST_MIN=0.8521`；**新增**效用常量 `AUC_TEST_DELTA_MIN=0.0055`、`AUC_VAL_DIRECTION_MIN=0.0`、`BASELINE_AUC_TEST=0.5988392178311113`、`BASELINE_AUC_VAL=0.5781533414372665`、`PRED_STD_MIN_RATIO=0.5`、`REFERENCE_PRED_STD=0.0033728455401762676`、`REFERENCE_IDENTITY_TOL=1e-9`、`EXPECTED_NEW_PARAMS_TOTAL=2385`、`EXPECTED_HEAD_PARAMS=8129` | 常量逐一按名/值钉死测试 |
| A4 | `arm_verdict` 体替换 | E1（绝对阈值）→ U1/U2（对基线差值 + val 方向）；G1–G5 → G1–G8（新增 G6 逐样本方差、G7 无预测坍缩、G8 参数预算 + M0 参照身份）；结局分类 → 三标签（第 5 节） | 分类矩阵测试（含边界） |
| A5 | 文档字符串 / 模块头 | 指向本文件；机制公式段保留 | 人工审阅（不设 AST 守卫） |
| A6 | `evaluate_prompt_diagnostics` 新增 `reference` 装配职责转移 | 诊断函数本身 AST 一致；参照统计仍由 `reference_head_stats` 产出，由接线传入 `arm_verdict` | AST 守卫（函数本体严格一致） |

**AST 守卫口径**（测试 `TestMechanismPin`）：`git show 99b9510:census_benchmark/residual_prompt.py` 取出钉死 blob（不可得时**响亮失败**，不静默跳过）；断言：模块级函数名集合与类名集合相等；`isolated_cpu_rng`、`effective_gate`、`build_newtask`、`param_report`、`evaluate_prompt_diagnostics`、`ResidualPromptNewTask` / `PromptAudit` / `PromptStats` 的**全部方法** `ast.dump` 逐字相等；`reference_head_stats` 归一化 `auc_score`→`auc` 后逐字相等；豁免集合**恰为** {模块 docstring、import 语句、模块级常量赋值、`arm_verdict`}。

### 1.6 预注册前参照核验（只读，先于本文件提交）

对固定产物 `s1-5c060b9c-m1688723512-e3-3a30e2c0` + 基线 run 的 `newtask.pt` 做只读复算（不写任何 run 目录、不进 SUMMARY、不训练）：

- 复算 val AUC = `0.5781533414372665`，与基线 run `metrics.json:best_val_auc_bsi` **逐位一致**（差 ≤ 1e-9）⇒ backdrop、val 装载、参照头三者身份成立；
- 参照头 val 预测离散度（fp64，n=500000）：`pred_mean=0.9909601567002535`、`pred_std=0.0033728455401762676`、`min=0.9436303973197937`、`max=0.9984057545661926`、`q05/q50/q95=0.9845150649547577/0.9916611909866333/0.99490886926651`。

**重要读法**：BSI 正例率 ~98.6% ⇒ 基线预测天然**高度集中**（std 仅 0.0034）。因此"无预测坍缩"（G7）**不得**用绝对下限（Census 的 0.1294 量级在此毫无意义），必须用**相对参照**的判据（见第 4 节 G7）。该参照值 `REFERENCE_PRED_STD=0.0033728455401762676` 在预注册前实测并钉死；运行期复算必须与之逐位一致（M0）。

---

## 2. 预注册设计（实现即 `aliccp_benchmark/residual_prompt.py`）

### 2.1 机制（逐字移植 1.1 钉死 blob；唯一改动为 1.5 的 A1–A6）

公式、初始化、`isolated_cpu_rng`、`prompt_deltas`（含 `h.detach()` 模长与维度一致性拒绝）、`forward` 的注入与委托、`PromptAudit`（G1/G2/G3 探针）、`PromptStats`（流式 fp64 统计）、`evaluate_prompt_diagnostics`、`reference_head_stats`、`param_report` 全部保持 Census 语义。

### 2.2 参数、初始化与 RNG（固定，不搜索）

| 项 | 取值 | 理由 |
|---|---|---|
| 生成器隐层 | `16`（`Linear(80,16) → ReLU → Linear(16,64)`） | Census 超参不变；不搜索 |
| 生成器初始化 | PyTorch 默认；构造在 `isolated_cpu_rng()` 内 | 共享参数与全局 RNG 端点与基线逐位一致（G1） |
| 门控 `α` | `nn.Parameter(torch.zeros(()))`，精确 0 | S3 恒等；"保守"由结构性上界 S1 在训练全程保证 |
| 新增参数 | 恰为 `prompt_generator.0.weight [16,80]`、`prompt_generator.0.bias [16]`、`prompt_generator.2.weight [64,16]`、`prompt_generator.2.bias [64]`、`prompt_gate []`；合计 **2385**（head **8129** 的 29.34%；总 10514） | 测试钉死（G8） |
| 优化器 / 损失 | 宿主协议 Stage-2 逐字：`Adam(newtask.parameters(), lr=1e-4)` + `BCELoss(pred,y) + newtask.get_l2_reg()`，batch 2000 | 单变量受控（1.4-7）；不新增超参 |
| 注入流 | 三路全部注入（单一共享生成器 + 单一标量门控） | 与 Census 同范围 |

**已知代价（如实记录）**：α=0 时生成器参数第一步梯度恒为 0（`∂δ/∂θ_generator ∝ α`），α 自身首步即有非零梯度；一步后恢复。G3 把它写成显式探针口径、不当作缺陷。

### 2.3 接线与影响面（最小）

- 实现为 `NewTask` 子类 `ResidualPromptNewTask`，`forward` 只做"计算 δ、修改三路表征"，随后委托 `super().forward`（融合管线零复制、零漂移）。
- `multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`aliccp_benchmark/metrics.py`、既有测试三件、master 脚本、`baseline/*`：**零改动**（静态守卫：对 `8133d32` 的 `git diff --name-only` 必须为空）。
- `aliccp_benchmark/bench.py`：**最小接线**——`run_stage2` 新增关键字参数 `variant=RP.BASELINE_VARIANT`、`prompt_reference_newtask=None`（默认值使既有调用方行为**逐位不变**）；训练循环改为 `enumerate` 以挂审计探针（epoch 1 首 batch 前向恒等检查、逐 epoch 首 batch 反传后梯度探针）；`best_state` 载入后跑诊断；按下表写 JSON。基线臂（`variant="baseline"`）除恒为 `"baseline"` 的 `variant` 字段外**键集不变**（测试锁定）。
- `run_aliccp_benchmark.py`：`stage2` 新增 `--variant {baseline, residual-prompt}`（默认 `baseline`）与 `--prompt-reference-newtask`（`Path`，可选但**处理臂必需**）；处理臂 `run_id` 追加后缀 `-rpg`（日志文件名同后缀）；处理臂未提供参照 → **直接 `ValueError` 拒绝出数**。
- 处理臂落盘：`prompt_report.json`（完整机制诊断）；`metrics.json` 增 `variant`、`rp_arm`（臂级判定）、处理臂另增 `prompt_hidden`；`gate_report.json` 处理臂另增 `residual_prompt` 子对象（**不并入** `hard_pass` 判定）；`config.json` 增 `variant`（两臂）与 `prompt_hidden`（处理臂）。
- 运行产物目录与只追加纪律沿用评测协议 11 节；`SUMMARY.md` 由 runner 追加一行（列结构不变）。

---

## 3. 效用判据（U；看到结果之前写定）

对固定基线 run `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`（同 `stage1_id`、同模型种子、同预算；`AUC-Val-BSI(best)=0.5781533414372665`、`AUC-Test-BSI=0.5988392178311113`）取**精确差值**：

| 编号 | 判据 | 落盘字段 |
|---|---|---|
| **U1** | `Δtest = test_bsi_arm − 0.5988392178311113 ≥ +0.0055`（用户指定 provisional 阈值；比评测协议 10.1 的 +0.005 更严） | `metrics.json:rp_arm.U.U1` |
| **U2** | `Δval = val_bsi_arm − 0.5781533414372665 > 0`（val 方向为正，防单侧噪声） | `metrics.json:rp_arm.U.U2` |

- `U.pass ⇔ U1 ∧ U2`。单 seed 单次判定；不扩多 seed、不做多 seed 平均。
- **报告项（不判定）**：评测协议 10.1 provisional 双条件（`Δtest ≥ +0.005` ∧ `Δval > 0`）为 U1∧U2 所蕴含，原样报告；Census 同机制 Δ=−0.0023308435 与 AliCCP 衰减线参照值一并列出（第 8 节）。
- 禁止用 CVR/CTR 差值宣称改进（评测协议 10.1）。

## 4. 机制门禁（M0 + G1–G8；全部硬判据，任一 FAIL 即按第 5 节分类）

| 编号 | 判据 | 实测口径 |
|---|---|---|
| **M0 参照身份** | 运行期 `reference_head_stats` 复算：`|ref_val_auc − 0.5781533414372665| ≤ 1e-9` 且 `|ref_pred_std − 0.0033728455401762676| ≤ 1e-9` | 处理臂专用（只读参照头、零训练）；不满足 = 参照/产物/装载不一致，本次不作效用解释 |
| **G1 构造恒等** | 共享参数/buffer 与基线 `NewTask`（同维、从构造前 RNG 现场重建）**逐位相同**；构造后全局 RNG 端点逐位一致；新增键**恰为** 2.2 所列 5 键 | 运行期审计 `construction_identity`（`PromptAudit`） |
| **G2 初值恒等（gate zero）** | 构造时 `α == 0.0`；**真实第一个训练 batch** 上处理头前向与基线头（共享权重逐位相同）前向 `torch.equal` | 运行期探针（`no_grad`，优化器第一 step 之前） |
| **G3 门控/生成器梯度活** | ① 首 batch 反传后 `α` 梯度 ≠ 0；② 末个 epoch 首个 batch 反传后 `α ≠ 0` 且生成器梯度范数 > 0；③ `best_state` 载入后 `α_final ≠ 0` | 逐 epoch `grad_probe` |
| **G4 范数受控界** | val 上**所有**样本、**所有三路**：`ratio_max ≤ |α_final| + 1e-6`（S1 的结构性推论；违反 = 实现缺陷） | `prompt_report.json:val_stats`（fp64 流式） |
| **G5 活性带** | val 上**每一路**平均 `ratio ∈ [0.005, 0.5]`（含端点；低于下界 = 静默，高于上界 = 已非"残差"量级） | 同上 |
| **G6 逐样本门控方差** | val 逐样本 `g_eff`：`geff_std > 0` ∧ `geff_max > 0` ∧ `geff_min ≥ 0`（门控必须**逐样本变化**而非常数） | 同上 |
| **G7 无预测坍缩** | val：`pred_std > 0` ∧ `pred_std ≥ 0.5 × ref_pred_std`（**相对参照**；1.6 实测 BSI 预测天然集中，绝对下限不可用）；参照缺失/不合规 → 该门禁 FAIL | `dispersion` + `reference` |
| **G8 参数预算** | `param_report`：键集恰为 5 键；`new_params_total == 2385`；`head_params == 8129`；无其它可训练参数 | `prompt_report.json:params` |
| **A 类（冻结完整性等）** | A1/A2/A4/A5/A6 **必须 PASS**（AliCCP 评测协议原样；A3 SKIP）。A 类任一 FAIL ⇒ 本臂比较作废 | `gate_report.json`（协议 machinery，不改） |

**B 类说明**：B1/B4 的 FAIL 为**继承事实**（第 7 节），不参与本臂有效性判定，也不得归因于本臂；B2/B3 原样记录。

## 5. 结局分类（先定后跑）

| 触发情形 | 分类（三标签） | `subreason`（诊断字段） | 处置 |
|---|---|---|---|
| M0 FAIL | `MECHANISM_FAIL` | `REFERENCE_IDENTITY` | 停止；不作效用解释 |
| G1 / G2 / G4 / G8 任一 FAIL | `MECHANISM_FAIL` | `INVALID_IMPLEMENTATION` | 停止。仅当能**证实**具体代码缺陷时允许修复后重跑一次（两次 run 都留痕、不改判据）；否则结论作废 |
| G3 FAIL | `MECHANISM_FAIL` | `MECHANISM_INACTIVE` | 门控没学起来 → AUC 不构成对机制的检验；停止 |
| G5 低于下界（任一路） | `MECHANISM_FAIL` | `MECHANISM_SILENT` | 门控把 prompt 学到近零；停止 |
| G5 高于上界（任一路） | `MECHANISM_FAIL` | `MECHANISM_OVER_PERTURB` | 已不能称"残差"；负结果**不得**归因于"prompt 假设无效"；停止 |
| G6 FAIL | `MECHANISM_FAIL` | `GATE_DEGENERATE` | 门控退化为常数；停止 |
| G7 FAIL | `MECHANISM_FAIL` | `PREDICTION_COLLAPSE` | 预测坍缩；停止 |
| A 类任一 FAIL | `MECHANISM_FAIL` | `PROTOCOL_INVALID` | 比较作废；停止 |
| M0+G1–G8 全过 ∧ A 类过 ∧ U1∧U2 | **`VALID_POSITIVE`** | — | 记录；是否扩多 seed 由用户显式决定 |
| M0+G1–G8 全过 ∧ A 类过 ∧ ¬(U1∧U2) | **`VALID_NEGATIVE`** | — | 有效阴性：机制在工作但没换来泛化收益 → 如实记录、停止 |

判定优先级按表自上而下（先实现有效性、再活性、再效用）。`arm.pass ⇔ classification == "VALID_POSITIVE"`。

## 6. 运行规程

### 6.1 唯一一次 short 运行（处理臂）

```powershell
# cwd = 本 worktree 根；Stage-1 产物与基线 run 已就位（只读、内容寻址）
# 本 worktree 无 .venv，用主树 venv 的解释器：
D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 `
    --tag short --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0 `
    --variant residual-prompt `
    --prompt-reference-newtask artifacts/aliccp_bench/runs/20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9/newtask.pt
```

- 预算 2M/500k/1M、`model_seed=1688723512`、epochs=5、patience=2、batch 2000、lr 1e-4 全部为协议默认，**不新增/不覆盖任何超参**；`run_id` 由 runner 追加 `-rpg` 后缀。
- 运行前：全量单元测试绿（`-m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests`）+ 静态守卫 + AST 钉死守卫；工作树已提交干净（`git_state.dirty=False`）。
- 运行后：**恰好一次**；不因结果好坏重跑、不调参、不换 seed/产物；`SUMMARY.md` 只追加。

### 6.2 无效执行处置

仅当 run 因**工具性原因**（崩溃/中断/环境故障、产物未完整落盘）无效时，保留现场、记录原因后可**重跑一次**；因结果原因不构成无效。无任何事后调参救结果（hidden/初始化/α 初值/注入流集合/活性带/阈值/参照文件均不得动）。

### 6.3 基线臂不重跑

基线臂精确值取自已记录的 run `20261003-0130-…-b2e17f9`（其代码与 `8133d32` 逐字节一致：`git diff b2e17f9 8133d32 -- aliccp_benchmark/ run_aliccp_benchmark.py multitaskrec/ config.py` 为空；本分支只新增/接线，不改协议 machinery）。

## 7. 继承事实（不得归因于本臂）

- **B1 = FAIL，继承自固定产物**：CTR `0.5493130789281618 < 0.55`（差 0.0007）、CVR `0.5132119172500262 ≥ 0.5` PASS、BSI `0.5988392178311113 ≥ 0.53` PASS。该量由 `evaluate_b_gates` 直接读 Stage-1 meta，与阶段 2 头无关。
- **B4 = FAIL，继承自固定产物**：Stage-1 `cluster_events = [(epoch 2, env_0 566, env_1 1999434)]` ⇒ env_0 占比 0.0283% < 5%（raw 聚类在该 seed 上的已知退化形态；本分支**不做**归一化聚类修复——那是另一实验线，禁止混入）。
- 因此本臂 `hard_pass = false` 为**预期值**，与本臂无关；本臂有效性定义 = M0 + G1–G8 + A 类（第 4/5 节），**不依赖 `hard_pass`**。
- **A3 = SKIP**（协议：按需复跑，不阻塞首轮）。

## 8. 必录诊断（落盘 `runs/<run_id>/prompt_report.json`；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| α / 梯度轨迹 | `grad_probe`（逐 epoch 首 batch 的 `alpha`、`alpha_grad_norm`、`generator_grad_norm`）、`alpha_final` | G3 + 与 Census（0→0.2116→…→0.314076）对照 |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`、`delta_norm_mean`、`h_norm_mean`（流序 `("gen","spec_0","spec_1")`） | G4/G5 + 与 Census（h_norm 128.3/7.9/12.2；ratio 0.1277 三路一致）对照 |
| 余弦方向 | 逐流 `cos_mean` | 加性新方向（≈0）还是缩放原方向（≈±1）；对照 Census（−0.37/−0.07/−0.22） |
| 逐样本门控 | `gate.geff_mean/std/min/max` | G6 + 对照 Census（0.12771817/0.02738926/0.06613362/0.28415609） |
| 预测离散度 | `dispersion`（`pred_mean/std/min/max/q05/q50/q95`）+ `reference_dispersion`（M0 口径复算） | G7 + 对照（基线参照 std 0.0033728455401762676） |
| 源/头门 | `gate_mean`（head `gate_network` 2 维均值，B3 量）+ Stage-1 环境上下文（`cluster_events`、`env_acc`） | "source/head gates" 落盘；B3 原样披露 |
| 参数清单 | `params`：逐参数 `name/shape/numel` + `new_params_total=2385`、`head_params=8129` | G8 |
| 构造恒等审计 | `construction_identity`（G1 三项） | G1 |
| 臂级判定 | `arm`（M0+G1–G8+U1/U2+classification+subreason） | 第 5 节机械分类 |

对照目标（Census 同机制实测，来自 `e0ad6af` §9）：Δtest −0.0023308435；α_final 0.314076；三路 ratio_mean 0.12771816；cos −0.3656/−0.0710/−0.2160；pred_std 0.1294（参照 0.1338）。本实验结论必须**同时**对 AliCCP 基线 run 与 Census 行为报告差异；两数据集不可比之处（预测尺度、标签率）如实标注。

## 9. 结果（运行后回填，不回写第 1–8 节）

（待运行后追加：run_id、commit、U1/U2 实测与 Δ、M0/G1–G8/A/B 实测、逐源读数、与基线及 Census 的对照表、分类结论、止损执行情况。）

## 10. 非目标与止损

- 不主张方法新颖性；不写"我们提出"；不改协议文件与模型语义；不动 Stage-1（不重训、不微调、不重新初始化 backbone）；不跑 `full` tag；不扩多 seed；不做 FLOPs。
- 不做：归一化聚类修复、TC-Prompt、CGR、affinity gate、KL-Prompt、温度/损失改动（评测协议排除清单有效）。
- 不因结果修改：`hidden=16`、初始化、α 初值、注入流集合、活性带、`BOUND_TOL`、G7 比值、U1 阈值、参照文件、分类规则。
- 单次运行；`VALID_POSITIVE` 之前不扩多 seed、不执行 A3 复跑。

## 11. 非新颖性声明

构件全部为既有方法族（prompt tuning、FiLM 族条件化调制、adapter/LoRA 残差、门控残差、范数归一），逐条先行工作清单沿用 Census 设计第 8 节（写作引用前必须逐条核实，本会话未做系统检索）。本文件**不**声称独立于先行工作；可报告内容仅限本仓库这套冻结协议下的跨数据集一次性判定。

---

**改动清单（白名单；静态守卫以 `8133d32` 为基准）**

| 文件 | 处置 |
|---|---|
| `docs/superpowers/specs/2026-10-03-aliccp-stage2-residual-prompt-gate-design.md` | 本文件（prereg commit 先行） |
| `aliccp_benchmark/residual_prompt.py` | 新增（1.5 适配后的移植机制 + 诊断 + 门禁 + 接线函数） |
| `aliccp_benchmark/tests/test_residual_prompt.py` | 新增（CPU、秒级、不读真实数据集 + AST 钉死守卫） |
| `run_aliccp_benchmark.py` | 最小接线（2.3） |
| `aliccp_benchmark/bench.py` | 最小接线（2.3；基线臂行为不变） |
| `artifacts/aliccp_bench/SUMMARY.md` | 只追加一行（runner 自动） |
| 其余全部（`multitaskrec/*`、`config.py`、`aliccp_benchmark/protocol.py`、`metrics.py`、既有测试三件、`AliCCP_*.py`、`CensusIncome_*.py`、`baseline/*`、`mask/*`、`analysis/*`） | **零改动**（静态守卫断言 `git diff --name-only 8133d32` ⊆ 白名单） |
