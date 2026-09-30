# CensusIncome 阶段 2 范数受控残差 Prompt（可学习门控）预注册与设计

- **状态**：**预注册已冻结**（第 3、4 节全部阈值与判据在看到任何本次运行结果之前写定；第 9 节结果待回填）。
- **日期**：2026-10-01
- **适用分支**：`exp/stage2-residual-prompt-gate`（自 `infra/fair-stage2-benchmark` @ `87afe03` **独立拉出**，未携带任何其它实验分支的代码）。
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。本文件**只引用**它，不修改它；划分、三个种子、真冻结三件套、A/B 门禁、SUMMARY 台账一律沿用。
- **数据集**：只跑 CensusIncome（评测协议 2.3）。
- **定位**：**工程消融（practical ablation）**，不是新颖性主张（第 8 节）。它回答一个可证伪的问题：
  **在冻结 Stage-1 表征上，加一路"由用户/上下文表征逐样本生成的、有界的、范数受控的残差 prompt，并用保守的近零可学习门控"能否把新任务 Education 的 AUC 推过预注册阈值？**

**提交时序（本条即预注册纪律的一部分）**：本文件先于实现与运行**单独提交**（prereg commit）→ 实现 + 测试提交（impl commit，本次运行的 `commit` 字段即该提交）→ 运行 → 结果只在第 9 节与 `SUMMARY.md` 以**新增小节 / 新行**回填。第 3、4 节的任何数字不得在看到结果后改动。

---

## 1. 机制审计（先审计，再动手）

### 1.1 阶段 2 的可达面（来自 `multitaskrec/model.py`、`run_census_benchmark.py` 的代码阅读）

阶段 2 的新任务头签名固定为 `NewTask.forward(dnn_input, gen_rep, spec_reps, env_embs)`，四个输入**全部来自冻结 backbone**（`get_infos()` 在 `torch.no_grad()` 下抽取）：

| 符号 | 来源 | 形状 | 量级（基线 run 在 val 上实测） |
|---|---|---|---|
| `dnn_input` | `EmbeddingNetwork(x)` | `[B, 123]` | —— |
| `gen_rep` | 共享专家 | `[B, 128]` | 范数均值 ≈ 128.3 |
| `spec_reps` | 两个任务专属专家 | 各 `[B, 128]` | 范数均值 ≈ 7.9 / 12.2（**低范数**） |
| `env_embs` | 环境嵌入 | 2 × `[128]` | —— |

`spec_reps` 与 `gen_rep` 的范数相差约 10–16 倍（来源：`exp/stage2-lora-adapter` 的 R1 诊断，见 1.2）。新任务头内部的融合顺序：`spec_reps` 经投影注意力加权 → `new_spec_rep` → 乘新环境嵌入 → 与 `gen_rep` 一起过 `gate_network` 2 路 softmax → `fused_rep` → `tower_network`。`metrics.evaluate_newtask` 只通过该签名调用头部 ⇒ **保持同签名的子类即可让 AUC 由与基线完全相同的代码算出**。

### 1.2 同族先行实验对照（全部已在各自分支完成并留痕）

| 分支（同源 `87afe03`） | 机制 | `AUC-Test-Education` | 相对基线 Δ | 判定 |
|---|---|---|---|---|
| 基线 run `20260929-1735-…-904f8d0` | 论文 `NewTask` | `0.8500685307175756` | —— | 预注册阈值 `0.8521` 的分母 |
| `exp/stage2-null-expert` | 路由候选追加零候选 | `0.851365` | +0.00130 | 未达标，停止 |
| `exp/stage2-attn-env-prior` | 新环境向量 = attention 加权旧 env_embs | `0.8489392168` | −0.00113 | 未达标，停止 |
| `exp/stage2-cgr-repro` | 历史 CGR 原样复现 | `0.847356` | −0.00271 | `CONFIRMED_DEGENERATE` |
| `exp/stage2-cgr-instance` | 修正后逐样本门控（混合 `W_attn` 与均匀权重） | `0.848288` | −0.00178 | `MECHANISM_OK` + `EFFECT_NOT_CONFIRMED` |
| `exp/stage2-affinity-gate-repro` | 历史 affinity gate 原样复现 | `0.846515` | −0.00355 | `STOP` |
| `exp/stage2-affinity-gate-corrected` | 修正后确定性逐样本软门控（`W_attn` vs 均匀） | `0.849130` | −0.00094 | `STOP_MECHANISM`（`gate_std 0.00713 < 0.01`） |
| `exp/stage2-cond-scale-bias` | router 条件化 FiLM（`γ(x)=W(x)Γ`、`β(x)=W(x)B` 加在已路由表征上） | `0.8501177932753766` | +0.0000493 | C1 FAIL（C2 PASS），方向停止 |
| `exp/stage2-lora-adapter` | 三路共享、绝对尺度 rank=4 残差 `h+scale·BA(h)` | `0.8491310170933439` | −0.000938 | S1 FAIL + **R1 FAIL**（`spec` 两路占比 1.00 / 0.58 > 0.5 上界） |

**先行工作给出的两条可复用诊断事实**（本实验的设计动机，如实署名来源）：

1. **来源专属表征是低范数弱势方**：`spec` 两路范数（7.9 / 12.2）比 `gen_rep`（128.3）小一个量级；绝对尺度的修正量对三路"一视同仁"时，对 `spec` 两路即构成过度扰动（lora 分支 R1 的直接读数）。
2. **"机制被激活"≠"机制有用"**：cond-scale-bias 的调制比 0.092 落在合规带内、γ/β 确实离开零点，但 Δ ≈ +4.9e-5，与训练噪声不可区分。

### 1.3 候选机制与"冗余性"裁决

**候选**（本题设）：逐样本 prompt 生成器从**用户/上下文表征**（`dnn_input`）生成**有界残差**；残差以**保守的近零可学习门控** `α`（零初始化）缩放、并**按目标流自身的范数归一**（范数受控），注入到进入融合之前的表征上。

**与最近两个已完成实验的逐项区分**（这是"是否冗余"审计的核心）：

| 维度 | cond-scale-bias（已完成） | lora-adapter（已完成） | **本候选** |
|---|---|---|---|
| 残差来源 | 既有 router `W(x)` 对 **2 个 per-env 参数向量**的凸组合（可达方向集 = 1 维仿射族） | 表征自身：`B(A(h))`（对 `h` 的固定低秩算子） | **上下文生成器** `P(x)`：`tanh` MLP，逐样本在 `R^128` 中**自由方向** |
| 有界性 | 无显式界（幅度由学到的 γ/β 范数隐含） | 无界（线性算子） | **逐维 `tanh` 有界**，`‖P‖ ≤ √d` |
| 幅度控制 | 无门控；幅度隐含在参数范数里 | 固定绝对尺度 `scale=1.0` | **标量门控 `α`（零初始化）+ 逐流范数归一**：相对扰动 `ratio ≤ |α|` **结构性成立** |
| 注入位置 | 已路由表征 `e_new` 上（融合中段） | `gen_rep` 与 `spec_reps`（融合前） | `gen_rep` 与 `spec_reps`（融合前，同 lora 注入点，**但残差函数、幅度机制完全不同**） |
| 初值恒等 | γ=β=0 ⇒ 逐位一致 | `up=0` ⇒ 逐位一致 | `α=0` ⇒ **逐位一致** |

裁决：**有效且不冗余**。三点理由：

1. **残差函数类不同**：`P(x)` 是上下文（`dnn_input`）的**自由逐样本方向**；cond-scale-bias 的加性项 `β(x)` 只能落在两个固定向量的连线上（K=2），lora 的 `B(A(h))` 是作用在 `h` 上的固定线性算子。三者可达方向集不同。
2. **幅度机制不同且是本实验唯一的新假设**：`ratio_s(x) = ‖δ_s‖/‖h_s‖ = g_eff(x) ≤ |α|`（见 2.1），门控值**就是**相对扰动的上界——这正是对 lora R1 诊断（绝对尺度压垮低范数流）的**结构性回应**，lora 分支文档列出的未执行后续（rank/scale 调参、逐流独立适配器）**均不是本方案**：本方案不调 rank/scale、不逐流独立（单一共享生成器 + 单一标量门控），而是把"幅度"从算子本身移到**一个可解释的标量**上。
3. **回答的问题不同**：本实验问"上下文里是否存在可被新任务头**加性利用**的迁移信号，且能被一个保守门控安全地按需取用"——六个先行实验无一回答此问题。

**弱冗余风险（如实声明）**：本候选与 lora-adapter **共用注入点**（融合前的三路表征），且动机同源于 R1 诊断。因此本实验**不得**被表述为"独立发现"，其可报告内容只是：在上述预注册形式下，Education AUC 是否过阈值、机制是否在工作、与基线的差是多少。

---

## 2. 预注册设计（实现即 `census_benchmark/residual_prompt.py`）

### 2.1 公式（唯一一种）

设 $x$ = `dnn_input` $\in\mathbb R^{123}$，$d=128$，残差注入到"进入 `NewTask` 融合之前"的三路表征
$h_s\in\{\text{gen\_rep},\ \text{spec\_rep}_0,\ \text{spec\_rep}_1\}$：

$$P(x)=\tanh\!\big(W_2\,\mathrm{ReLU}(W_1 x + b_1)+b_2\big)\in(-1,1)^{d}
\qquad (W_1{:}\ 16{\times}123,\ W_2{:}\ d{\times}16)$$

$$g_{\text{eff}}(x)=\alpha\cdot\frac{\lVert P(x)\rVert_2}{\sqrt d}\in[0,\,|\alpha|]
\qquad(\alpha\ \text{为可学习标量，零初始化})$$

$$\delta_s(x)=\alpha\cdot\frac{\lVert h_s(x)\rVert_2}{\sqrt d}\cdot P(x)
\;=\;g_{\text{eff}}(x)\cdot\underbrace{\frac{P(x)}{\lVert P(x)\rVert_2}}_{\text{单位方向}}\cdot\lVert h_s(x)\rVert_2,
\qquad h_s' = h_s + \delta_s$$

**读法**：残差 = 门控 × 单位方向（来自上下文）× 基向量自身范数。由此得到三条**结构性事实**（实现用乘积式，不做除法，`‖P‖=0` 时 δ=0 无 0/0）：

| 编号 | 事实 | 依据 |
|---|---|---|
| S1 | **范数受控**：$ratio_s(x) := \lVert\delta_s\rVert/\lVert h_s\rVert = g_{\text{eff}}(x) \le \lvert\alpha\rvert$，对**所有**样本与流成立（`‖h‖=0` 时 δ=0，ratio 记 0） | `‖P‖ ≤ √d`（tanh 逐维开区间） |
| S2 | **逐样本、跨流一致**：同一 $x$ 下三路的相对扰动**相同**（= $g_{\text{eff}}(x)$），与各流自身范数无关——这正是对 lora"绝对尺度压垮低范数流"的修正 | δ 的构造 |
| S3 | **保守初值**：$\alpha=0 \Rightarrow \delta\equiv 0 \Rightarrow$ 前向与基线**逐位一致**（共享权重相同时） | δ = α·(…) |

### 2.2 参数、初始化与 RNG（固定，不搜索）

| 项 | 取值 | 理由 |
|---|---|---|
| 生成器隐层 | `16`（`Linear(123,16) → ReLU → Linear(16,128)`） | 线性部分秩 ≤ 16（12.5% 的表示维），参数预算小；**不做任何秩/宽度搜索** |
| 生成器初始化 | PyTorch 默认（Kaiming uniform + uniform bias），行互不相同 | 对称可破缺（吸取 cgr 分支 D3 常量初始化的教训） |
| 门控 `α` | `nn.Parameter(torch.zeros(()))`，**精确 0** | S3 恒等；"保守近零"由**结构性上界** S1 在训练全程保证，不靠初始值大小 |
| 构造顺序 | `super().__init__()`（= 基线 `NewTask` 全部模块及其 RNG 消耗）→ **在 `isolated_cpu_rng()` 内**追加生成器与 `α` | 共享参数与基线**逐位一致**；构造后全局 RNG 端点与基线**逐位一致**（同 lora 分支 A6 的口径） |
| 新增参数 | 恰为 `prompt_generator.0.weight [16,123]`、`prompt_generator.0.bias [16]`、`prompt_generator.2.weight [128,16]`、`prompt_generator.2.bias [128]`、`prompt_gate []`；合计 **4161**（头 27063 的 15.38%） | 测试钉死 |
| 优化器 / 损失 | `Adam(newtask.parameters(), lr=1e-3)` + `BCELoss + get_l2_reg()`（只正则 tower），逐字同基线 | 不新增超参、不新增正则项 |
| 注入流 | 三路全部注入（单一共享生成器 + 单一标量门控） | 与 lora 同范围以直接对照 R1 诊断；单变量 |

**已知代价（如实记录）**：`α=0` 时生成器参数的第一优化步梯度**恒为 0**（`∂δ/∂θ_generator ∝ α`），`α` 自身首步即有非零梯度；一步之后生成器恢复梯度。这是零初始化门控的数学后果（同 cgr 分支零初始化输出层的已知代价），G3 把它写成显式判据、不当作缺陷。

### 2.3 实现形式与影响面

- 实现为 `NewTask` 的**子类** `ResidualPromptNewTask`：`forward(dnn_input, gen_rep, spec_reps, env_embs)` 只做"计算 δ、修改三路表征"，随后**委托 `super().forward(dnn_input, gen_rep', spec_reps', env_embs)`**——融合管线零复制、零漂移。
- `multitaskrec/model.py` / `config.py` / `census_benchmark/protocol.py` / `census_benchmark/metrics.py` **零改动**（测试静态守卫）。
- `run_census_benchmark.py` 只做与 cond-scale-bias 分支同款的最小接线：`stage2` 新增 `--variant {baseline, residual-prompt}`（默认 `baseline`）、`--prompt-reference-newtask`（可选，只读诊断用）；处理臂 `run_id` 追加后缀 `-rpg`；`config.json` 两臂都记 `variant`；`metrics.json` 两臂都记 `variant`、处理臂另记 `rp_arm` 与 `mechanism.prompt`；`gate_report.json` 处理臂另记 `residual_prompt` 子对象（**不并入** `overall_pass`）；处理臂另写 `prompt_report.json`。基线臂除恒为 `"baseline"` 的 `variant` 字段外键集不变。

---

## 3. 预注册判据（看到结果之前写定）

### 3.1 效应判据（E1）

| 编号 | 判据 | 落盘字段 |
|---|---|---|
| **E1** | `AUC-Test-Education >= 0.8521`。基线 = `0.8500685307175756`（run `20260929-1735-…-904f8d0`，stage1 `s1-096f8f16-m1685480945-e2-cb2094b3`） | `metrics.json:rp_arm.E1` |

阈值 `0.8521` 为用户指定的接受阈值，与 cond-scale-bias / lora-adapter 两分支同一口径；单 seed 单次判定，不做多 seed 平均后达标。

### 3.2 机制门禁（G1–G5；全部为硬判据，任一 FAIL 即按 3.3 分类处置）

| 编号 | 判据 | 实测口径 |
|---|---|---|
| **G1 构造恒等** | 共享参数与基线 `NewTask`（同一 model seed、从构造前 RNG 现场出发）**逐位相同**；构造后全局 RNG 端点逐位一致；新增键恰为 2.2 所列 5 个 | 运行期审计：`shared_params_bit_identical`、`global_rng_endpoint_identical`、`extra_keys`（对真实实例，构造前记录 RNG 现场 → 隔离 RNG 内重建基线头逐键比较） |
| **G2 初值恒等** | 构造时 `α == 0.0`；**真实第一个训练 batch** 上，处理头前向与基线头（共享权重逐位相同的实例）前向 `torch.equal` | 运行期探针（`no_grad`，优化器第一 step 之前） |
| **G3 门控活性** | ① 第一个 batch 反传后 `α` 梯度 ≠ 0；② 末个 epoch 首个 batch 反传后：`α ≠ 0` 且生成器梯度范数 > 0；③ 载入 `best_state` 后 `α_final ≠ 0` | 逐 epoch 梯度探针 `grad_probe`（诊断落盘） |
| **G4 范数受控界** | val 上**所有**样本、**所有**三路：`ratio ≤ |α_final| + 1e-6`（该界是 S1 的结构性推论；违反 = 实现缺陷） | `prompt_report.json:val_stats`（fp64 CPU 流式累加） |
| **G5 活性带** | val 上**每一路**的平均 `ratio ∈ [0.005, 0.5]`（含端点；同 lora R1 带）。低于下界 = prompt 等同静默；高于上界 = 已非"残差"量级 | 同上 |

**G5 的带内不构成通过、只说明机制处于可解释区间**；E1 才是性能判据。

### 3.3 结局分类（先定后跑）

| 触发情形 | 分类 | 处置 |
|---|---|---|
| G1 / G2 / G4 任一 FAIL | `INVALID_IMPLEMENTATION` | 停止。仅当能**证实**具体代码缺陷时才允许修复后重跑一次（两次 run 都留痕、不改判据）；否则本次结论作废、不解释 AUC |
| G3 FAIL | `MECHANISM_INACTIVE`（INCONCLUSIVE） | 门控没学起来 → 本次 AUC 不构成对机制的检验；停止 |
| G5 低于下界（任一路） | `MECHANISM_SILENT`（INCONCLUSIVE） | 门控把 prompt 学到近零 → 同上；停止 |
| G5 高于上界（任一路） | `MECHANISM_OVER_PERTURB` | 相对扰动 > 0.5 已不能称"残差"；负结果**不得**归因于"prompt 假设无效"；停止 |
| G1–G5 全过 且 E1 PASS | `CONFIRMED` | 记录；是否多 seed 由用户显式决定，本文件不预设 |
| G1–G5 全过 且 E1 FAIL | `VALID_NEGATIVE` | **有效阴性**：机制在工作但没换来泛化收益 → 如实记录、停止 |

### 3.4 止损与阈值纪律（与评测协议同款）

1. 阈值 / 判据 / 分类规则只允许在**看到本次运行结果之前**修改；每次修改单独记录理由。本文件先于运行单独提交即此纪律的凭据。
2. **禁止事后调参救结果**：不因结果好坏改 `hidden=16`、初始化、`α` 初值、注入流集合、活性带、E1 阈值；不换 seed / 划分 / stage1 产物。
3. 不论结果如何，`SUMMARY.md` **只追加不重写**该行；不得只报告通过的 run。
4. 单次运行；`CONFIRMED` 之前不扩多 seed、不跑 `full` tag、不做 A3 复跑。

---

## 4. 必录诊断（全部落盘 `runs/<run_id>/prompt_report.json`；class C，不判定）

| 量 | 字段 | 用途 |
|---|---|---|
| 门控分布 | `gate.alpha_by_epoch`（每 epoch 首个 batch 反传后 `α`）、`gate.alpha_final`（`best_state`）；逐样本有效门控 `g_eff` 的 `mean/std/min/max` | "门控学到了多大、是否逐样本变化" |
| 残差/基范数比 | 逐流 `ratio_mean/std/max`（`("gen","spec_0","spec_1")` 流序）+ 验证 G4 界 | 范数受控是否成立、幅度是否在带内（G5） |
| 逐源效应 | 逐流 `h_norm_mean`、`delta_norm_mean`、`cos(delta,h)_mean` | 三路基范数悬殊（动机）与残差方向是"加新方向"（cos≈0）还是"缩放原方向"（cos≈±1） |
| 预测离散度 | 处理头在 val 上的 `pred_std/mean/min/max`；若给了 `--prompt-reference-newtask`，基线头（只读载入、零训练）同口径数值 + 复算 val AUC（对账 `0.8527881905614896`） | 防"预测坍缩成常数"；只读对账不参与判定 |
| 可训练参数清单 | `params`：逐参数 `name/shape/numel` + `new_params_total=4161`、`head_params=27063`、占比 | 参数预算核算 |
| 冻结/协议证据 | `backbone_sha256_before/after`、`env_ids`、`split_fingerprint` 沿用 run 目录既有文件 | A1/A2/A4/A5 |
| 梯度轨迹 | `grad_probe`：逐 epoch 首 batch 的 `alpha_grad_norm`、`generator_grad_norm` | G3 |
| 构造恒等审计 | `construction_identity`：G1 三项 + 两侧参数量 | G1 |

统计口径与 `GateStats`/`RepStats` 一致：fp64、CPU 累加、显存 O(1)、不保留中间张量；预测离散度在 val 上留存至多一个 epoch 的预测向量（val 专用，test 预测**绝不**落盘）。

---

## 5. 运行规程

### 5.1 首轮唯一一次运行（单 seed 短跑）

```powershell
# cwd = 仓库根；Stage-1 产物已存在（只读、内容寻址）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 `
    --variant residual-prompt --tag short --gpu 0 `
    --prompt-reference-newtask artifacts/census_stage2/runs/20260929-1735-s20260929-m1685480945-short-904f8d0/newtask.pt
```

- 数据 / split seed / model seed / env seed / batch size / lr / epochs / patience 全部继承 Stage-1 产物与评测协议常量，**不新增任何超参**。
- 单元测试（CPU、秒级、不读真实数据）：`census_benchmark/tests/test_residual_prompt.py`，并跑全量回归 `-m unittest discover -s census_benchmark/tests -t census_benchmark/tests`。

### 5.2 已知继承事实（不得归因于本臂）

- 基线 run 的 **B3 已是 FAIL**：`gate_mean = [0.952401451979339, 0.8543901456350915]`，第 0 维在冻结 backbone 上就越过 0.95 上界（该量由 `metrics.evaluate_newtask` 直接从冻结 backbone 读出，与阶段 2 头无关）。本臂 B3 无论通过与否都必须如实记录；**不得**归因于本臂，也不得因此重跑挑结果。
- 其余 A1/A2/A4/A5/B1/B2/B4 预期全过（沿用同一 stage1 / 划分 / 冻结语义）。

### 5.3 结果回填位置

`SUMMARY.md` 追加一行（列结构不变，`run_id` 带 `-rpg` 后缀）；本文件第 9 节新增小节；`prompt_report.json` 与 `metrics.json:rp_arm` 为机器可读留痕。**不改动**第 1–8 节任何预注册文本。

---

## 6. 与评测协议的接口（改动清单）

| 文件 | 处置 |
|---|---|
| `multitaskrec/model.py`、`config.py`、`census_benchmark/protocol.py`、`census_benchmark/metrics.py` | **零改动**（测试静态守卫：与 `87afe03` 逐字节一致） |
| `census_benchmark/residual_prompt.py` | 新增（机制 + 诊断 + 门禁 + 接线函数） |
| `census_benchmark/tests/test_residual_prompt.py` | 新增（CPU、秒级、不读真实数据集） |
| `run_census_benchmark.py` | 最小接线（第 2.3 节；基线臂键集除 `variant` 外不变，由测试锁定） |
| `docs/superpowers/specs/2026-10-01-stage2-residual-prompt-gate-design.md` | 本文件（prereg commit 先行） |
| `artifacts/census_stage2/{stage1,runs,splits}` | 只读 / 只追加；`SUMMARY.md` 追加一行 |

---

## 7. 明确的非目标

- 不主张方法新颖性；不写"我们提出"（第 8 节）。
- 不改协议文件与模型语义；不动 Stage-1（不重训、不微调、不重新初始化 backbone）。
- 不做：null expert、TC-Prompt、CGR、affinity gate、KL-Prompt、T4、温度/损失改动（评测协议第 10 节的排除清单继续有效）。
- 不跑 AliCCP / ByteRec；不做 FLOPs；不做多 seed；不产出论文级性能结论。
- 不为过 G5 / E1 而调 `hidden` / 初始化 / `α` 初值 / 注入流集合 / 活性带 / 阈值。

---

## 8. 先行工作与非新颖性声明

本实验用到的构件**没有一个是新的**，属既有方法族的直接组合：

| 构件 | 先行工作（据既有知识；**本会话未做系统检索，写作引用前必须逐条核实**） |
|---|---|
| Prompt / 提示向量调制冻结表征 | Prompt-tuning（Lester et al. 2021）、Visual Prompt Tuning（Jia et al. 2022）、L2P（Wang et al. 2022） |
| 逐样本条件化调制（FiLM 族） | Perez et al., *FiLM*, AAAI 2018；条件化归一化（De Vries et al. 2017） |
| 冻结主干 + 小残差模块（adapter / LoRA） | Houlsby et al. 2019；Hu et al., *LoRA*, ICLR 2022；Rebuffi et al. 2017 |
| 残差门控 / 近恒等初始化 | 残差网络（He et al. 2016）系的一般做法；门控残差在推荐/CTR 中也常见 |
| 范数归一 / 相对尺度控制 | 各归一化族工作的常规手段（LayerNorm 等），本实验只把它用于**残差幅度上界** |

**可报告内容仅限**：在**本仓库这套严格冻结协议**下，上述组合（上下文生成的、有界的、范数受控的、零初始化标量门控的残差 prompt）对 Education 新任务是否有效、参数代价多少、机制是否真在工作。与已完成的 `exp/stage2-cond-scale-bias`（router 条件化 FiLM）与 `exp/stage2-lora-adapter`（绝对尺度共享低秩残差）的差异见 1.3——**不得**表述为"独立于先行工作"。

---

## 9. 结果（待运行后回填）

（本节在运行完成后以新增小节形式回填：运行标识与配置、E1 实测、G1–G5 实测、结局分类、逐源读数、预测离散度、与基线/先行实验的差异对照、止损执行情况。第 1–8 节一字不改。）
