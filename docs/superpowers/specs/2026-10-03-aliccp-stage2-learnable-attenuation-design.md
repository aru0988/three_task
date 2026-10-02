# AliCCP Stage-2 可学习衰减（单标量、恒等初始化）：post-hoc 最优系数的可发现性检验

- **状态**：预注册已写死（本文件在本臂任何 run 之前提交）；实现与结果见 §10（运行后追加，不回填）
- **日期**：2026-10-03
- **适用分支**：`exp/aliccp-stage2-learnable-attenuation`（自 `infra/aliccp-fair-benchmark` @ `8133d32` 独立拉出；**不从** `exp/aliccp-stage2-specific-attenuation-control`（固定对照臂，顶端 `3314420`）与 `exp/aliccp-stage2-null-expert`（null 臂，顶端 `aba8ee9`）拉出，不继承其任何代码改动；协议文件零修改）
- **协议依赖**：`docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md`（AliCCP 公平评测协议）与 `aliccp_benchmark/protocol.py`。本分支**只引用**，不修改其任何文件与常量（SUMMARY 列、A/B 门禁阈值、预算、指纹定义均不变）
- **假设来源（先例审计，只读）**：
  1. null 臂 run `20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx`：机制判据不成立（`null_top1_rate = 0`），但测试增益 Δtest = `+0.013846535450687258`（val 同向 `+0.013786499876619396`）；其行为退化为对 specific 混合的近似常数阻尼（`null_mean = 0.3027766269669533`，std = 0.0085，即 ×≈0.697）。
  2. 固定对照臂 run `20261003-0322-p2M-v500k-t1M-m1688723512-short-f88dca4-sattn`：把 specific 混合乘以**冻结常数** c = `0.6972233730330467`（= 1 − null_mean，post-hoc 推导）后，重现 null 增益的 91.8%（test）/ 96.3%（val），判定 `ATTENUATION_SUPPORTED`。即"常数阻尼 + 其诱发的头 gate 重平衡"是该增益的主要驱动。
- **本臂的问题（H）**：上述 0.697 是**事后**从 null 臂行为反推并**人工冻结**的系数。它是否**可被训练目标本身发现**？——在冻结 backbone 的 stage-2 训练中，一个从**精确恒等**（c = 1，输出与基线逐位相同）出发的、**唯一一个**可学习标量衰减，能否自行走进衰减区间，并在不接触任何 post-hoc 系数的前提下达到 PI 效用阈值？
- **定位声明（重要）**：可学习门控/标量伸缩是**既有工程手段**（learnable gate / scalar gating 属标准做法），本臂**不主张新颖性**；它是一个最小化的机制消融（discoverability ablation）：检验前两臂确立的"阻尼解释"对应的操作点是否能由一个标量从恒等点被优化器找到。全部判定口径在见结果前写死。
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；需变更先改本文件再改代码。

---

## 1. 设计（先于实现写死；含恒等初始化 × 非零梯度的数学可行性论证）

### 1.1 目标性质（构造约束，按优先级）

| 编号 | 性质 | 理由 |
|---|---|---|
| P1 | **恒等初始化**：初始前向输出与基线（未启用臂）**逐位相同** | 不把任何初始偏差混入对照；c = 1.0 时 `x * 1.0` 在 IEEE 754 下逐位恒等（±0、denormal、inf 均验证） |
| P2 | **初始即有非零学习信号**：∂L/∂θ 在初始化点一般非零 | 否则"可发现性"无从谈起 |
| P3 | **有界、衰减单侧**：c ∈ [c_min, 1]，c_min = 0.05 | 与前臂/固定对照验证过的族（0 < c ≤ 1）一致；防符号翻转与分支完全静默 |
| P4 | **最小参数化**：新增**恰一个**标量（0 维）；无逐样本生成器、无 router 类别、无其它参数 | 最小消融 |

### 1.2 数学可行性论证（结论先写：光滑参数化不可能，必须用 kink；已实证）

**命题**：不存在 C¹ 参数化 c(θ)（θ ∈ ℝ 开邻域）同时满足 P1 与 P2。

**证明**：若 c 在 θ_init 处取到上确界 1（P3 ⇒ c ≤ 1）且 θ_init 为内点，则由 Fermat 内点极值定理 c′(θ_init) = 0，从而 ∂L/∂θ = (∂L/∂c)·c′(θ_init) = 0——恒等点必为驻点。受限重参数化同样不可行：设 θ = φ(u)，u 无约束，φ(u*) = θ*（边界点）；若 φ′(u*) ≠ 0，由反函数定理 φ 将 u* 的开邻域映为 θ* 的开邻域，必越出 c ≤ 1 的定义域——矛盾；而任何在有限点触达边界的 φ 必在触点导数为零（如 θ = −u²），梯度再次归零。**结论：P1 + P2 只能由非光滑（kink）构造同时满足。**

**选定构造（kink 在恒等点）**：

```
c(θ) = clamp(θ, min = c_min, max = 1.0)        # θ 即系数本身（内点斜率 dc/dθ = 1）
θ_init = 1.0                                     # ⇒ c_init = 1.0（P1 精确成立）
```

- **P1**：`new_spec_rep * c` 在 c = 1.0 时逐位恒等（已实证：±0 / inf / denormal 均 bit-equal）。
- **P2**：PyTorch `clamp` 的反向对边界为**含等号**掩码（`x <= max` 通过梯度；已实证：θ=1.0 处 dy/dθ = −1.0 ≠ 0）；θ > 1 的饱和区梯度为 0（已实证 dy/dθ = 0）。
- **单侧性（如实披露）**：θ > 1 侧（"放大"方向）被 clamp 封死 ⇒ 若训练目标在恒等点局部偏好"不衰减/放大"（∂L/∂c < 0），参数上升一步后进入死区（∂L/∂θ = 0）驻留，c 恒为 1.0，整臂**逐位等价于基线**——"门控拒绝衰减"是合法且可判读的结果（见 §5.3 分类 `IDENTITY_PRESERVED`），不是实现缺陷。
- **c_min = 0.05（披露）**：护栏常量，非调参；比先例阻尼带（0.686–0.709）低约 14×，正常轨迹下不 binding；若最终 c 触底 ⇒ 退化结局，如实判失败（`DEGENERATE_FLOOR`），不重跑。

**被否决的备选（记录在案）**：① sigmoid/softplus 等光滑有界门：恒等不可达（c = 1 仅在极限点）或"最不偏"初始化 c₀ = 1 − δ 同时牺牲恒等性与梯度幅度（∂L/∂θ ∝ δ → 0，二者不可兼得）；② exp(θ) + 单侧 clamp：同为 kink 构造，但"参数 ≠ 系数"（c = e^θ），审计直观性不如直接 clamp；③ 双参数门/逐样本生成器：违反 P4。

### 1.3 参数化冻结（实现口径）

| 项 | 值 |
|---|---|
| 构造参数 | `NewTask(..., learnable_spec_attenuation: bool = False)` |
| 参数 | `self.spec_attenuation_raw = nn.Parameter(torch.tensor(1.0))`（float32 0 维；`__init__` **末尾**创建，**零初始化 RNG 消耗**） |
| 有效系数 | `c = torch.clamp(self.spec_attenuation_raw, min=SPEC_ATTENUATION_MIN, max=1.0)`，`SPEC_ATTENUATION_MIN = 0.05`（模块级常量，`multitaskrec/model.py` 单一事实来源） |
| 作用位置 | **与固定对照臂完全同一处**：`new_spec_rep = matmul(stack(spec_reps), W).squeeze()` 之后、`env_aware_rep = new_spec_rep * new_env_emb` 之前，`new_spec_rep = new_spec_rep * c` |
| 初始化 | θ = 1.0（**恒等**；**不以任何形式初始化在 0.697 附近**） |
| 优化器 | 既有 `Adam(params=newtask.parameters(), lr=1e-4)` **自动纳入**；单一 lr、无 param group、无 weight decay、无梯度裁剪 |
| L2 正则 | 不加：标量不进 `get_l2_reg()`（与 null 臂 `null_key` 先例一致，损失形式不变） |
| 默认关断 | `False` 时不创建参数：参数集合、state_dict、算子序列、RNG 消耗与 `8133d32` 基点**逐位一致**（测试 L1/L2 强制） |
| 禁用 post-hoc 系数 | **本臂代码任何路径不得出现 0.6972233730330467 / 0.6972233653068542**；该值仅作为 §5.4 的 descriptive 距离参照出现于文档与判定报告的**记录字段**中 |

### 1.4 梯度与轨迹记录（in-run，本臂专属）

- 训练循环中每步（`loss.backward()` 之后、`optimizer.step()` 之前）记录 `(step, epoch, θ_pre, c_pre, ∂L/∂θ)`；全序列落盘 `runs/<run_id>/attenuation_trace.json`（gitignore 产物）。
- 训练结束后记录 `θ_final`；**判定用 `c_final` = best-epoch state_dict（`newtask.pt`，即实际产生 test AUC 的头）中该标量的 clamp 值**；同时记录训练末值（两者在 best_epoch = 最后一轮时相同，属预期）。
- 汇总入 `metrics.json`：`grad_step1`、`grad_abs_max_first10`（M2 判定量）、`grad_abs_max` / `grad_abs_mean` / `grad_last`、`grad_nonzero_frac`（|g| > 1e-12 步占比）、`clamp_upper_frac`（θ ≥ 1 死区步占比，恒等驻留诊断）、`floor_frac`、逐 epoch 末 c。

---

## 2. 改动面（最小；白名单）

| 位置 | 改动 |
|---|---|
| `multitaskrec/model.py` → `NewTask`（且仅此类） | 模块级常量 `SPEC_ATTENUATION_MIN = 0.05`；构造参数 `learnable_spec_attenuation`；启用时创建标量参数；forward 条件乘（默认关断逐位一致）；方法 `spec_attenuation_value()`（唯一 clamp 出处） |
| `aliccp_benchmark/metrics.py` | 预注册常量（§6 全部字面量；含先例记录值）、`pred_dispersion`（float64 离散度，公式与 null/对照臂记录口径一致）、`SourceGateStats`（源任务 gate 均值）、`learnable_attenuation_verdict`（纯函数，§6 分类） |
| `aliccp_benchmark/bench.py` | `run_stage2(..., learnable_attenuation=False)`；run_id 后缀 `-lattn`；训练循环梯度/轨迹捕获（仅启用时）；val 探针（启用时）；config/metrics 落盘臂信息 |
| `run_aliccp_benchmark.py` | stage2 新增 `--learnable-attenuation`（默认 false） |
| `aliccp_benchmark/tests/test_learnable_attenuation.py`（新文件） | §3 不变量 |
| 本文件 + `artifacts/aliccp_bench/SUMMARY.md`（只追加行） | 文档与运行记录 |

**明确不改**：`aliccp_benchmark/protocol.py`（零改动）、`config.py`、`AliCCP_MPTRec.py`、`AliCCP_NewTask.py`、`baseline/*`、`multitaskrec/train.py`、`multitaskrec/dataset.py`、`multitaskrec/model.py` 中 `NewTask` 以外的任何类、CensusIncome / ByteRec 任何代码路径。**前两臂的代码（null expert、固定衰减）不移植进本分支。**

---

## 3. 不变量（由 `test_learnable_attenuation.py` 强制，不接受人工目测）

参考实现从 git `8133d32:multitaskrec/model.py` 动态加载（本分支基点 = 基线 run 实际使用的模型代码）。

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| L1 | 默认关断臂前向/反向与基点实现**逐位一致** | `torch.equal` 比对输出与每个参数的梯度 |
| L2 | 默认关断臂参数集合/state_dict = 基点；启用臂 state_dict 恰多 `{spec_attenuation_raw}`；构造零 RNG 消耗（同种子两臂共享参数逐位一致、标量 == 1.0） | 参数集合差集 + `load_state_dict(strict=True)` + 种子对照 |
| L3 | **恒等保持**：启用臂在初始化（θ=1.0，未训练）前向与默认臂**逐位一致** | `torch.equal` |
| L4 | 恒等点梯度：启用臂首次反向中标量梯度存在；构造夹具上（∂L/∂c ≠ 0）非零；**其余全部参数梯度与默认臂逐位一致**；另立微测试钉死 PyTorch clamp 边界梯度约定（max 边界通过、饱和区为零）——即 §1.2 可行性前提 | 梯度逐位比对 + 微测试 |
| L5 | 启用臂前向等于本文件写死的公式 `new_spec_rep' = clamp(θ) × Σ_k W_k spec_rep_k`（同温度 softmax） | 独立参考实现比对（`torch.equal`），θ ∈ {1.0, 0.8, 0.5, 0.02(触底), 1.5(触顶)} |
| L6 | 系数语义：c(1.0)=1.0、c(0.99)=0.99、c(0.05)=0.05、c(0.02)=0.05、c(1.5)=1.0；内点/边界/饱和区梯度符号正确 | 数值断言 |
| L7 | 参数预算：启用 = 默认 + 1；名字差集恰 `{spec_attenuation_raw}`；0 维；不进 `get_l2_reg()`（同权重下 L2 相等） | 计数 + 差集 + 回归对照 |
| L8 | 指标：预注册常量钉死（与 §6 字面量一致）；`learnable_attenuation_verdict` 六类分类边界正确；`pred_dispersion`（对照 numpy float64）与 `SourceGateStats`（流式 == 单批、构造数据精确值）数值正确 | 构造数据精确断言 |
| L9 | 运行器（CPU 极小夹具端到端）：默认 run 不出现本臂字段（仅顶层 `learnable_spec_attenuation: false` 标记）；启用 run 落盘 trace/探针/判定，`trainable_params` = 默认 + 1，A 类门禁照常 PASS | `TestRunnerLearnableArm` |
| L10 | 静态守卫：`aliccp_benchmark/protocol.py` 相对基点零改动；相对 `8133d32` 的全部跟踪文件改动 ⊆ §2 白名单；`SUMMARY_COLUMNS` 不变 | `git diff --name-only` |
| L11 | 文档↔代码字面量一致（初始化 1.0、下限 0.05、M 阈值 0.01 / 1e-12 / 10 步窗、PI `0.6043392178311113`、基线 `0.5988392178311113` / `0.5781533414372665`） | `test_doc_preregisters_same_numbers`（读取本文件） |

CPU 极小夹具不构成任何性能证据，只验证语义与接线。

---

## 4. 运行方式（恰好一次）

```powershell
# 0) 测试（全部不变量；先写测试后写实现）
.venv\Scripts\python.exe -m unittest discover -s aliccp_benchmark/tests -t aliccp_benchmark/tests -v

# 1) 本实验唯一要跑的 run（固定 Stage-1 产物 = 基线/null/固定对照三臂使用的同一份）
.venv\Scripts\python.exe run_aliccp_benchmark.py stage2 --tag short `
  --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0 --learnable-attenuation
```

- 基线臂**不重跑**：对照 = 协议 SUMMARY 已有 run `20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9`（同 `stage1_id`、同预算、同种子、同顺序，协议 §12.1 允许）；null 臂与固定对照臂的**记录值**用于 §6.4 descriptive 比较（不重跑任何旧臂）。
- 预算 / 种子 / 顺序全部走协议默认（train 2M / val 500k / test 1M 前缀；model seed `1688723512`；env seed `20261003`；batch 2000、shuffle=False；stage2 epochs=5、patience=2）——**不搜参、不换结构、不调 lr、不改初始化、不调 c_min**。
- 处理臂 `run_id` 形如 `YYYYMMDD-HHmm-p2M-v500k-t1M-m1688723512-short-<commit7>-lattn`。
- **前置条件**：开跑前先提交本批改动（否则 run_id / metrics.json 的 commit 记的不是本实验代码）；运行期间 `git.dirty` 必须为 false。

---

## 5. 预注册门禁与接受标准（写死；看到结果后不得修改）

### 5.1 机械门禁（本臂专属；全部由 `metrics.learnable_attenuation_verdict` 落盘）

| 编号 | 判据 | 含义 |
|---|---|---|
| M1 变化 | `\|c_final − 1.0\| ≥ 0.01` | 标量从恒等**实质性移动** |
| M2 梯度 | 训练**前 10 步**内 `max \|∂L/∂θ\| > 1e-12` | 恒等点（θ 移动 ≤ 10·lr = 1e-3，c ≥ 0.999）存在数值非零学习信号（10 步窗防单批巧合为零；单步序列全量落盘备查） |
| M3 内点 | `c_min + 0.01 ≤ c_final ≤ 0.99` | 未驻留恒等边界、未触底 clamp |
| M4 参数预算 | 可训练参数 == `8130`（= `8129` + 1）且可训练名字集合含且仅含一个新名 `spec_attenuation_raw`；state_dict 新增键恰为此标量 | 除标量外**零新增表征/router 参数** |
| M5 冻结完整性 | A1 PASS（backbone sha 前后一致、`.grad` 全 None）且 A6 PASS（sha == 产物记录）且标量不属于 backbone | 真冻结三件套未被本臂破坏 |

`mechanism_pass = M1 ∧ M2 ∧ M3 ∧ M4 ∧ M5`；`M3 ⇒ M1`（因 c ≤ 1），两者仍分列记录。

### 5.2 效用门禁（U）与方向门禁（D）

| 编号 | 判据 | 依据 |
|---|---|---|
| **U1（PI，与前臂同一字面量口径）** | `AUC-Test-BSI ≥ 0.6043392178311113`（= 基线 `0.5988392178311113` + 0.0055）**且** `AUC-Val-BSI(best) > 0.5781533414372665` | provisional +0.0055（null 臂主判据同款）；协议级 Δ ≥ +0.005 口径单独记录 |
| D1 参数方向 | `c_final ≤ 1.0 − 0.01`（衰减方向，非恒等/非放大） | 本假设的单侧性；由 M1∧M3 蕴含，单列记录 |
| D2 效用方向 | `AUC-Val-BSI(best) > 0.5781533414372665` | val 同向（U1 的组成项，单列记录） |

### 5.3 机械分类（互斥且穷尽；判定函数按此顺序短路）

| 顺序 | 条件 | 分类 |
|---|---|---|
| 1 | ¬A类全PASS（A3 SKIP 视为通过）∨ ¬M4 ∨ ¬M5 | `INTEGRITY_FAIL`（比较作废，协议 §12.1） |
| 2 | `c_final ≤ c_min + 0.01` | `DEGENERATE_FLOOR`（触底退化） |
| 3 | `\|c_final − 1.0\| < 0.01` | `IDENTITY_PRESERVED`（未实质移动；含"门控拒绝衰减"的驻留情形） |
| 4 | ¬M2 | `GRADIENT_WINDOW_ANOMALY`（移动了但前 10 步无信号——异常，如实记录） |
| 5 | U1 满足 | `LEARNED_ATTENUATION_SUPPORTED`（机制 + 效用 + 方向全过） |
| 6 | 其余 | `LEARNED_ATTENUATION_NO_UTILITY`（机制过、效用未过） |

### 5.4 descriptive（预注册为**记录量**，不参与分类）

- `posthoc_distance`：`|c_final − 0.6972233730330467|`（f64 参照）与 `|c_final_f32 − 0.6972233653068542|`（float32 有效值，c_final 为 float32）——回答"标量是否自行落在 post-hoc 最优点附近"，纯描述。
- 重现比：`Δtest(本臂)/Δtest(null)`、`/Δtest(control)`（test 与 val 各一）；先例数字见 §5.5。
- 头 gate 均值、val 预测离散度、`source_gate_mean` 与三臂记录值（§6）并排。
- 若出现 `IDENTITY_PRESERVED`：报告"是否与基线 run 记录值逐位相等"（该情形下前向恒等是**运行内**代数性质；跨进程逐位复现未经 A3 验证——A3 SKIP——如实标注为"观察到的"而非"保证的"）。

### 5.5 先例常量（逐位取自各自 run `metrics.json`，供 §5.4 引用）

| 量 | 基线 `…-b2e17f9` | null `…-6224c0f-nullx` | 固定对照 `…-f88dca4-sattn` |
|---|---|---|---|
| AUC-Test-BSI | `0.5988392178311113` | `0.6126857532817985` | `0.6115453624066993` |
| AUC-Val-BSI(best) | `0.5781533414372665` | `0.5919398413138859` | `0.5914326183692061` |
| Δtest / Δval | — | `+0.013846535450687258` / `+0.013786499876619396` | `+0.012706144575588052` / `+0.013279276931939643` |
| 头 gate 均值（val） | `[0.851149, 0.1488510625]` | `[0.48325546875, 0.5167445625]` | `[0.49435615625, 0.50564384375]` |
| val `pred_std` | `0.0033728455401762676` | `0.004380239336642408` | `0.004451373214009348` |
| trainable params | — | 8193 | 8129 |

（`source_gate_mean` 三臂共享同一 frozen backbone：`[0.176159685842067, 0.2537776756367646]`。）

### 5.6 结果解读的预注册口径

- `LEARNED_ATTENUATION_SUPPORTED` ⇒ 证据支持"该操作点可被训练目标从恒等点自行发现"：post-hoc 系数不是必需的，常数阻尼这一机制**可学习**。
- `LEARNED_ATTENUATION_NO_UTILITY` ⇒ 标量进入了衰减内点但效用未达 PI：说明固定对照的增益**不（仅）来自"任意内点衰减"**，与系数具体取值/优化动力学有关；本臂不进一步归因。
- `IDENTITY_PRESERVED` ⇒ 在冻结 backbone 的 stage-2 训练目标下，恒等点对该标量**局部不偏好衰减**（一阶）；"阻尼收益"不能由该最小学习信号取得——干净负结果，不重跑、不改初始化弥补。
- `DEGENERATE_FLOOR` / `GRADIENT_WINDOW_ANOMALY` / `INTEGRITY_FAIL` ⇒ 如实记录，按 §7 止损。

---

## 6. 止损规则（与协议 §10.2、前两臂一致）

1. **恰好一次 run**：无论结果如何**不得重跑**（实现缺陷例外，见 4）；不得挑 seed、不得扩多 seed、不得跑全量。
2. **不得调参**：θ 初始化、`c_min`、lr、epochs、patience、作用位置一律不得事后调整；本臂代码不得使用 post-hoc 系数。
3. 阈值/门禁只允许在**看到本臂任何 run 结果之前**修改；每次修改单独 commit 并在本文件记录理由；已判定的 run 不得用新阈值重判。
4. 运行期间发现实现缺陷 → 允许修复后重跑，但必须同时保留失败 run 的产物与说明。
5. 效用（U1）与机制（M1–M5）**各自独立如实报告**；不得只报通过的维度。

---

## 7. 门禁与继承披露（B1/B4）

- 本臂执行协议 §10 全部 A 类与 B 类判定（`--tag short` ⇒ `enforce_b=True`），`learnable_arm` 为**臂级**判定，不并入 `hard_pass`（A/B 门禁语义与 `protocol.py` 不变）。
- **继承的失败（非本臂引入，四臂按构造完全相同）**：基线 short run 已记录 `B1 FAIL`（`AUC-Val-CTR 0.5493130789281618 < 0.55`）与 `B4 FAIL`（cluster 后 env_0 仅 566/2,000,000）。两者由**共享 Stage-1 产物**决定，本臂与其引用同一 `stage1_id` ⇒ 判定逐位相同。如实披露为**继承基线缺陷**，不归因于本臂、不据此否定本臂、也不据此宣称任何东西。
- A 类门禁必须全 PASS（A3 SKIP 视为通过），否则本臂判定按 `INTEGRITY_FAIL` 记录、比较作废。
- B2（val-test 差）/ B3（头 gate 未坍缩）为臂相关门禁，照常判定与披露。

---

## 8. 报告与分支处置

- `SUMMARY.md` 只追加、不重写：处理臂即使失败也必须有行（协议 §9.3）。
- 结果在运行后**追加**到本文件 §10：**确证性发现（预注册口径：分类、M1–M5、U1、D1/D2、A/B 门禁）与 post-hoc 解读（描述量、与三臂对照的叙述、机理猜测）分节书写、显式标注**；§1–§9 预注册内容不得回填、不得修改。
- 分支处置（用户硬规则）：无论成功失败，**只保留 `exp/aliccp-stage2-learnable-attenuation` 本分支，不合并回 `master`**；推送需用户显式批准。
- 与本实验无关的既有探索（TC-Prompt / CGR / affinity gate / KL-Prompt / T4 / null expert / 固定衰减对照代码等）一律不引用、不重建（仅引用其**记录值**与文档）。

---

## 9. 局限（预声明）

1. **单 seed、单 run**：分类与效用的点估计；不引入事后置信区间叙事。
2. **单侧性**：kink 构造禁用了"放大"方向；"恒等点一阶不偏好衰减"的负结论范围仅限**本构造**（例如双参数门或不同初始化可能不同），不外推为"衰减永远不可学"。
3. **恒等 × 非零梯度的数学不可能性（§1.2）是本设计的**前提**：kink 是唯一同时满足 P1/P2 的构造族；clamp 边界梯度依赖框架约定（已实证并对抗性测试钉死），换框架需重新验证。
4. **归因上限**：SUPPORTED 时归因 = "一个从恒等出发的标量衰减被训练目标发现且有用（同作用位置，无 post-hoc 系数）"；不区分 c 的路径依赖与固定对照的差异来源，也不主张方法新颖性。
5. **继承**：B1/B4 失败继承自共享 Stage-1 产物（§7）；null/对照臂的对照值为其各自 run 的记录值（不重跑）。

---

## 10. 结果（实测；运行后追加，不得回填预期值）

**（运行后在此追加：确证性发现与 post-hoc 解读分节，见 §8。本节在 run 之前为空。）**

