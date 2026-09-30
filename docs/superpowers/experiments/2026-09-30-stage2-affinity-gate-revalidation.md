# exp/stage2-affinity-gate-repro：历史 Affinity Gate 忠实复现与硬门控退化复核（实验说明）

- **状态**：**真实验已执行**（单 seed 短跑，2026-09-30 18:37）。臂状态 **`STOP` /
  `CONFIRMED_DEGENERATE`**（机制 STOP 判据 1、2 命中；保真判据未命中），`effect.evaluated = false`。
  完整结果与判读见第 8 节；**第 6 节的阈值、规则与优先级在结果落地前后一字未改**。
  本次收尾**只回填文档**：未提交、未推送、未合并 `master`（由此产生的可审计性缺口如实记在 8.3 第 9 条）。
- **日期**：2026-09-30
- **分支**：`exp/stage2-affinity-gate-repro`（从 `infra/fair-stage2-benchmark` @ `87afe03` 独立拉出）
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。
  本文件**只引用**它；划分、种子、冻结三件套、A/B 门禁一律沿用，`census_benchmark/protocol.py` /
  `census_benchmark/metrics.py` / `multitaskrec/model.py` **零改动**。
- **数据集**：只跑 CensusIncome（协议 2.3）。
- **定位**：这不是新方法，是**历史实现的忠实复现 + 退化判据的前置诊断**（第 9 节"非新颖性"）。

---

## 1. 要回答的问题

历史 Affinity Gate 在 `archive/exploration` 上跑出过 CensusIncome `0.8601 ± 0.0026`
（相对 prompt `−0.0028`，即**未超越** prompt）、AliCCP `0.6714 ± 0.0014`。当时给出的解释是：

> "**Affinity Gate 跌最少（0.0016）**，因为它的逐样本 cosine similarity 特征能有效检测 OOD 样本
> **并切换到 FW**。代价是正常 seed 上均值偏低（过于保守）。"
> —— `archive/exploration:logs/all_experiment_results.md`
>
> "**Affinity Gate**: OOD-aware 硬切换，最稳定（σ=0.0014）但均值偏低，过于保守"
> —— `archive/exploration:arch_notes.md`

本轮**不提出任何改进**，只做两件事：

1. **忠实复现**：把历史 Affinity Gate 的公式、初始化、构造顺序**逐项照搬**（含被审计出来的缺陷），
   在**当前公平评测协议**（固定划分、真冻结、val 选点 / test 只评一次）下重跑一次单 seed 短跑。
2. **前置诊断**：先判定硬门控**是否退化**（是否恒等于 attention / 是否与输入无关 / 是否从未学到梯度）。
   **退化是主导判据**——一旦退化即 `CONFIRMED_DEGENERATE` / `STOP`，此时"是否达到 `0.8521`"这一问题
   不再解释；只有在**无 STOP** 时，`AUC-Test-Education >= 0.8521` 才有意义。

> 换句话说：本分支的产物是**一份可审计的否证材料**，而不是一个新方法。第 3 节的 D1 已经给出
> **解析证明**：历史实现在 eval 期 `g_hard ≡ 1`（对任意参数、任意输入）——所以"切换到 FW"这件事
> 在历史评测路径上**从未发生过**。本轮要做的是把这条证明用落盘读数坐实。

---

## 2. 审计溯源（audit provenance）

| 项 | 来源 | 关键内容 |
|---|---|---|
| 历史训练脚本 | `archive/exploration`：`run_newtask_from_ckpt.py` | `ALL_MODES` 含 `"affinity_gate"`；`epochs=30`、`patience=5`、Adam(lr=1e-3)、`loss = BCE(pred, y) + get_l2_reg()`；`newtask.train()` 每轮；val 选点、best_weight 回载后评 test |
| 历史模型 | `archive/exploration`：`multitaskrec/model.py` → `NewTask(fusion_mode="affinity_gate")` | `affinity_mlp = Linear(4,16) → ReLU → Linear(16,1) → Sigmoid`（**默认初始化**，与 CGR 的常量初始化不同）；`W_attn / W_fw / g_soft / g_hard / g / W` 表达式原文 |
| 历史结果 | `archive/exploration`：`logs/all_experiment_results.md` | CensusIncome affinity_gate 三 seed val/test = `0.8613/0.8583`、`0.8626/0.8632`、`0.8610/0.8590`（mean test `0.8601 ± 0.0026`，Δ vs prompt `−0.0028`）；AliCCP `0.6714 ± 0.0014` |
| 历史**没有**的读数 | `git grep -ln affinity_gate archive/exploration` 只命中 `model.py` / `run_newtask_from_ckpt.py` / 两份笔记 | affinity_gate **没有专用诊断脚本**（不像 CGR 有 `analysis/cgr_diagnosis.py`）；历史 runner 对 affinity_gate 只打印 `lambda=--- kl=---`，**没有任何门控读数落盘** —— 这正是 D6"不可识别"的直接来源。本轮第 5 节的机制读数全部是**新加**的 |
| 历史解释（**被复核的主张**） | 同上 + `arch_notes.md` | "逐样本 cosine similarity 特征能有效检测 OOD 样本并切换到 FW"；"OOD-aware 硬切换…过于保守" |
| 当前协议 | `docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md` + `artifacts/census_stage2/SUMMARY.md` | 基线参照行 `20260929-1735-s20260929-m1685480945-short-904f8d0`，`AUC-Test-Education = 0.850069`；同 stage1 产物 `s1-096f8f16-m1685480945-e2-cb2094b3` |

**口径差异（如实声明）**：

1. 历史 runner 的划分是 `train_test_split(test_ds, test_size=0.5, random_state=seed)`——**每个 seed 一套
   划分**。因此历史 `±0.0026 / ±0.0014` 的方差里混着**划分方差**，"最稳定（σ 最小）"这一读数不能直接
   与"门控更稳"划等号。本轮改用协议的固定 `split_seed=20260929`，划分方差被消掉。
2. 历史 runner 只做 eval 模式评测；本轮的退化诊断读的是 **eval 与 train 两条路径**的差异（第 3 节 D1）。
3. 本轮**不重跑** stage1：处理臂与基线臂引用**同一个** stage1 产物（`s1-096f8f16-…`），
   这是协议 6.3"对照一致"的要求。
4. 历史 runner 的损失在 CPU 上算（`loss_func(pred.cpu(), y.float())`），本轮的 stage2 路径按评测协议
   在设备上算（`loss_func(pred, y.float().to(device))`）。这一差异**对处理臂与基线臂完全相同**，
   不引入新的混杂；此处如实记录，以免与历史脚本逐行对照时被误读为"复现偏离"。
5. 历史 runner 用 `epochs=30 / patience=5`；本轮按协议 6.1 的短跑配置 `epochs=5 / patience=2`
   （与基线臂完全相同，见第 4 节末）。

---

## 3. 已知缺陷：复现时**故意保留**（不得"顺手修好"）

> 性质区分：**D1 / D3 / D4 / D5** 是实现与接线层的缺陷，逐项照搬、不修；**D2** 是设计层问题，且其中
> "归一化维度错误"一条经复核**已撤回**（见下）；**D6** 是本轮审计新发现，只加修正**读数**、不改模型语义。

### D1 双重 sigmoid（主线；本轮的核心否证点）

历史代码把 `affinity_mlp` 的**末端 `nn.Sigmoid()` 输出**当成 logit 使用：

```
g_logit = Sigmoid(z)          # z = MLP 的前 sigmoid 输出，g_logit ∈ (0, 1)
g_soft  = Sigmoid((g_logit + G) / 0.5)      # G ~ Gumbel(0,1)（仅训练期）
g_hard  = (g_soft > 0.5)
```

**（a）eval 期 `g_hard ≡ 1`（可证，与参数、输入都无关）。**
`g_logit ∈ (0,1)` ⇒ `g_logit / 0.5 ∈ (0,2)` ⇒ `g_soft = Sigmoid(·) ∈ (0.5, Sigmoid(2))`，
其中 `Sigmoid(2) = 0.8808`。由于 `g_soft > 0.5` **恒成立**，`g_hard ≡ 1`，于是

```
W = 1·W_attn + 0·W_fw = W_attn        （逐样本成立；"逐位"见下面的 float32 留痕）
```

即：**评测路径上的 Affinity Gate 就是纯 attention**，硬切换从未切到 FW。
唯一例外是 float32 下 `Sigmoid(z)` **下溢为 0.0**（z ≲ −104）时 `g_soft = Sigmoid(0) = 0.5` ⇒ 判 0 ⇒ FW；
本轮把它作为边界情形写进测试留痕（`test_eval_float32_underflow_is_the_only_route_to_fw`）。

**（a′）float32 精度留痕（STE 舍入，**不得**读成"逐位恒等"）。**
`g = g_hard.detach() + g_soft − g_soft.detach()` 在 `g_hard = 1` 时并不总精确等于 1：
`1 + g_soft ∈ (1.5, 1.8808]` 的 ulp 是 `2⁻²³`，`fl(1 + g_soft)` 只在 `1 + g_soft` 的 `2⁻²⁴` 单位系数为
偶数时精确；系数 `c ≡ 1 (mod 4)` 时是平局，round-half-even 向下取偶 ⇒ `fl(1 + g_soft) = 1 + g_soft − 2⁻²⁴`
⇒ `g = 1 − 2⁻²⁴`（`c ≡ 3` 时向上 ⇒ `g = 1`）。后果：

* `W` 与 `W_attn` 的逐位差**不恒为 0**：逐样本 `‖W − W_attn‖₁ ≲ 2⁻²⁴·‖W_attn − W_fw‖₁ + 舍入 ~1e-7`；
  判据 2（`routing_l1 < 1e-6`）**仍然命中**（余量约一个数量级），退化结论不变。
* 落盘读数因此**不得**写成"`routing_l1_mean ≡ 0.0`、`routing_identity_frac ≡ 1.0`"；正确口径是
  `routing_l1_max < 1e-6` 与 `ste_exact_frac`（`g == g_hard` 的样本占比）——见 5.1 的对应行。
* `g_hard ≡ 1` 本身仍是**精确**的（`g_soft > 0.5` 严格成立），故判据 1 不受影响。

**（b）训练期路由概率被夹在解析窗口内。**
`g_hard = 1 ⟺ g_soft > 0.5 ⟺ g_logit + G > 0 ⟺ G > −g_logit`，`G ~ Gumbel(0,1)`（CDF `exp(−e^{−x})`）⇒

```
P(attention | x) = 1 − exp(−exp(g_logit)) = 1 − exp(−exp(Sigmoid(z)))
                 ∈ (1 − e^{−1}, 1 − e^{−e}) = (0.6321, 0.9340)
```

于是：**门控在训练期也永远不会偏向 FW**（下界 0.6321），特征能移动的幅度 ≤ `0.9340 − 0.6321 = 0.3019`。
训练期的硬切换把 `1 − P ∈ [0.0659, 0.3680]` 的样本**整支丢弃 attention**，而这个决定由 Gumbel 噪声主导：
同一样本换一份噪声，判据翻面的概率是 `2p(1−p) ∈ [0.123, 0.465]`。

**（c）train / eval 路由错配。** 训练期 `P(attention) ≤ 0.9340`、eval 期恒为 1 ⇒ 错配 `≥ 0.066`，
典型 ~0.19。被评的模型与训练时的模型**不是同一个路由策略**。

### D2 "OOD 判据"不自洽（设计层；本轮审计曾误判一条并**撤回**）

**（a）撤回："归一化维度错误"不成立。** 本文件早期草稿（以及当时的分支代码/测试）把

```
E_norm = F.normalize(exist_env_embs, dim=0)
```

记为"按**任务维**归一化的错误维（应为 `dim=-1`）"，并据此认为 `C` 不是余弦。**该结论已撤回**，因为它
建立在一个错误的形状假设上。实际数据路径（三处独立佐证）：

1. `MPTRec.get_infos` 逐个任务取 env 嵌入：`env_embedding_network(env_indices[i])`，`env_indices` 是
   一维 buffer ⇒ 索引结果是 **0 维**张量 ⇒ 每个元素形状 `[rep_dim]`（**一维**）；
2. 因此 `torch.stack(env_embs, dim=1)` 得到 `[rep_dim, K]`——**`dim=0` 就是特征维**，
   `normalize(E, dim=0)` 是"逐任务列归一化"，正是标准做法；
3. 另两处历史代码只在该布局下成立：`exist_env_embs[:, :num_source_tasks]` 切的是 dim=1（任务维），
   `torch.mm(H_out, exist_env_embs)` 要求 `[B,d] @ [d,K]`。

结论：`C = normalize(H, dim=-1) @ normalize(E, dim=0)` **是真余弦**（历史注释 "values in [-1, 1]" 属实），
"cosine similarity 特征"这一命名**没有**名不副实。本轮不改公式，改为在实现里加一条 fail-fast 布局守卫
（`env_embs` 元素必须是 `[rep_dim]`，否则显式抛错），并把 `env_layout` / `env_normalize_dim` 随读数落盘
自证；测试 `test_env_layout_is_feature_first_and_cos_sim_is_a_true_cosine` 与
`test_layout_guard_rejects_wrong_env_layout` 锁定这条口径。

**边界（读数口径，非缺陷）**：`projection_network` 末端是 LayerNorm，若某样本把 ReLU 全灭则 `H(x)` 恰为
**零向量**，此时余弦在数学上无定义——历史口径（`F.normalize`）给 `0`（`0 / clamp_min(‖·‖, 1e-12)`），而独立
复算的 `0 / (0·‖E‖)` 是 `NaN`（"逐行 allclose"在零投影样本上不是良定义的比较）。因此 I4 的真余弦比对
只在 `H(x) ≠ 0` 的样本上进行，零投影样本单独断言"约定值 `0`、不是 `NaN`"；留痕读数 `cos_entry_min/max`
里出现的 `0` 可能来自这种样本，**不**代表两个方向真的正交。

> 这条撤回本身是审计产物的一部分：**审计结论也要可证伪**，误判必须留痕而不是静默删掉。

**（b）保留："OOD 判据"与它要判的对象同源，且 K=2 时信息量只有 2 个数。**
4 维门控特征 `[max_k C, min_k C, max_k C − min_k C, mean_k |C|]` 全部由 `projection_network` 的输出
`H(x)` 派生，而 `W_attn = softmax(H Eᵀ / T)` 用的是**同一个** `H(x)` 与**同一组** `E`。于是
"这一样本与源任务不匹配（OOD）"与"attention 路由给出的权重"出自同一投影，无法相互独立地证伪；
历史把它读作"能有效检测 OOD 样本"，这一步在实现里没有支撑点（CGR 的根因分析恰恰是因为这点才把门控
输入**解耦**成 task-embedding + 统计量）。

更硬的限制：K=2 时 `{max, min} = {C_1, C_2}`，故

```
mean_k |C| = (|max| + |min|) / 2        （恒等式；第 4 维是前三维的函数）
```

⇒ 4 维特征完全由 `(C_1, C_2)` 决定，门控**实际只看到 2 个数**。本模块不修，只把 `cos_entry_min/max`
与逐源任务列均值（`cos_col_mean` / `routing_col_mean`）落盘，让读者自行判断；上述恒等式是**结构性**事实
（对任意输入成立），故由测试 `test_gate_features_are_redundant_for_k2` 锁定而不是当作读数（见 4.1 节 I4）。

### D3 全局 RNG 消耗（两处）

1. **训练期每步** `torch.rand_like(g_logit)` 从**全局** RNG 抽 B 个 Gumbel 噪声 ⇒ affinity 臂与其它臂
   **不共享随机流**（同一 model seed 下，后续 dropout / 初始化抽样全部错位）。
2. **构造期** `affinity_mlp` 建在 `projection_network` 与 `gate_network` **之间**，两个 `nn.Linear`
   的默认初始化多消耗全局 RNG ⇒ 同 seed 下 `gate_network` / `tower_network` 的初值与基线**不同**。

⇒ 历史 "affinity_gate vs prompt" 的对比**不是单变量对比**：差异里混着"不同随机流 + 不同初始化"。
本复现**原样保留**这两点（测试 `test_rng_construction_order_confound_is_reproduced` 锁定），
并在 `metrics.json:affinity_arm.provenance` 留痕；比较时必须把这一位移计入不确定性。

### D4 硬 STE + 噪声污染的梯度

* 前向用 **hard 0/1**：训练期 `1 − P ∈ [0.0659, 0.3680]` 的样本**完全丢弃 attention 分支**；
* 反向传播的是**带噪样本**的导数（`g = g_hard.detach() + g_soft − g_soft.detach()` 只对 `g_soft` 求导，
  而 `g_soft` 是被噪声污染的随机变量）；
* 再叠加两次 sigmoid 的局部灵敏度上限：`∂g_logit/∂z = σ′(z) ≤ 0.25`（z=0 处取到），
  而修正路径（z 当 logit、τ=0.5）是 `(1/τ)·σ′(z/τ) ≤ 2`。

⇒ 门控的学习信号被**双重压制**（噪声饱和 + logit 被压扁），衰减比由 `gradient_probe` 落盘。

### D5 门控不进 L2

`NewTask.get_l2_reg()` 只正则 `tower_network`（继承 master）；`affinity_mlp` 与 `projection_network`
都不进 L2。与 CGR 同源，本模块**不修**（测试 `test_get_l2_reg_excludes_affinity_mlp` 锁定：
把门控权重放大 1000 倍，正则值逐位不变）。

### D6 判据不可识别 + 修正口径（本轮审计新发现）

承接 D1：**eval 期的硬门控规则是参数的常函数**（`g_hard ≡ 1`，与 θ、x 都无关）。历史只评过 eval 模式，
所以"门控学会了按 OOD 切换"这一主张**在任何历史留痕读数上都不可识别**——看到的必然是"纯 attention"。
历史把 `σ=0.0014` 读成"门控让结果更稳"，但那个稳定性与门控**无关**（它等于同 seed 下的 attention 路径）。

**修正（只读数、不改模型语义）**：用 MLP 的**前 sigmoid 输出 `z`** 作为 logit 的反事实读出

```
corrected_gate_mean          = mean_x  σ(z/τ)          (τ = 0.5，无噪声)
corrected_routing_share_attn = mean_x [ z > 0 ]        （修正后的硬判据）
```

该读出**是**输入相关的，因而可检验：若它落在 0.05–0.95 之间，说明参数**确实**学到了逐样本差异，
缺陷只在历史的 eval 规则；若它同样逼近 0 或 1，说明门控本身就没学到东西。
`gradient_probe` 的 `grad_norm_corrected` 给出同一修正路径下的梯度量级，用于量化 D4 的压制比。

---

## 4. 忠实复现的定义（`census_benchmark/affinity_gate.py`）

设 $x$ = `dnn_input`，$H(x)$ = `projection_network(x)`（末端 LayerNorm），$E=[\mathbf e_0,\mathbf e_1]$，
$T = 150$，$K=2$，$\tau = 0.5$：

$$h = \frac{H}{\lVert H\rVert_2},\qquad E_{\text{norm}} = \mathrm{normalize}(E,\ \dim=0),\qquad C = h\,E_{\text{norm}}^{\top}$$

$$a(x) = \big[\,\max_k C,\ \min_k C,\ \max_k C - \min_k C,\ \overline{|C|}\,\big] \in \mathbb R^4$$

$$z(x) = W_2\,\mathrm{ReLU}(W_1 a + b_1) + b_2,\qquad g_{\text{logit}} = \sigma(z),\qquad
g_{\text{soft}} = \sigma\!\big((g_{\text{logit}} + G)/\tau\big)$$

$$g_{\text{hard}} = \mathbb 1[g_{\text{soft}} > 0.5],\qquad
g = g_{\text{hard}}\text{.detach()} + g_{\text{soft}} - g_{\text{soft}}\text{.detach()}$$

$$W_{\text{attn}} = \mathrm{softmax}\big(H E^{\top}/T\big),\qquad W_{\text{fw}} = \mathbf 1/K,\qquad
W = g\,W_{\text{attn}} + (1-g)\,W_{\text{fw}}$$

$$\text{new\_spec\_rep}=\textstyle\sum_k W_k\,\text{spec\_rep}_k,\quad
\text{env\_aware}=s\odot\text{new\_spec\_rep},\quad
\text{pred}=\text{tower}\big([\text{env\_aware},\text{gen\_rep}]\cdot \text{gate\_network}(x)\big)$$

| 历史项 | 历史取值 | 复现位置 | 是否逐字相同 |
|---|---|---|---|
| 公式（`C / a / z / g_logit / g_soft / g_hard / g / W / 融合管线`） | 见上 | `HistoricalAffinityGateNewTask.gate_terms` / `routing_pipeline` | ✅ 同表达式、同运算顺序（`torch.equal` 逐位锁定） |
| MLP 结构 / 隐层宽度 / 温度 / τ | `4→16→1` + 末端 Sigmoid / `16` / `150` / `0.5` | `AFFINITY_INPUT_DIM` / `AFFINITY_HIDDEN` / `TEMPERATURE` / `GUMBEL_TAU` | ✅ |
| 初始化 | **默认 torch 初始化**（无常量初始化，与 CGR 不同） | `__init__`（不调用 `nn.init.constant_`） | ✅（测试与"按历史顺序消耗 RNG"的参照逐位比对） |
| 构造顺序（含 RNG 混入） | env → projection → **affinity_mlp** → gate → tower | `__init__`（`nn.Module.__init__` + 手工搭树） | ✅（D3 故意保留） |
| 门控输入归一化维 | `dim=0`（`E` 为 `[rep_dim, K]` ⇒ **就是特征维**，逐任务列归一化） | `ENV_NORMALIZE_DIM = 0` + 布局守卫 `ENV_LAYOUT` | ✅（**非缺陷**；早期"错误维"之说已撤回，见 D2(a)） |
| 正则 | 只正则 tower（`affinity_mlp` **不进** L2） | 继承 `NewTask.get_l2_reg`，未重写 | ✅（D5 故意保留） |
| Gumbel 抽样 | `-(-rand_like(x).clamp(1e-10).log()).clamp(1e-10).log()`，**全局 RNG** | `gumbel_noise_like(..., generator=None)` | ✅（D3 故意保留） |
| 损失 / 优化器 / 选点 | `BCE + get_l2_reg`、Adam(lr=1e-3)、val 选点、test 只评一次 | 评测协议的 stage2 路径（未改） | ✅ |

**stage2 配置**：`epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`（协议 6.1；与基线臂完全相同）。

### 4.1 不变量（由测试强制，不接受人工目测）

`census_benchmark/tests/test_affinity_gate.py`（新增）：

| 编号 | 不变量 | 强制手段 |
|---|---|---|
| I1 | 与**归档类**（`git show archive/exploration:multitaskrec/model.py` 里的真 `NewTask`）在同 seed、同参数下**前向与梯度逐位一致** | `TestArchiveFidelity`（AST 抽类定义后 exec；eval / train 两种模式都逐位比对） |
| I2 | 忠实构造顺序 + 默认初始化 + `state_dict` 键集 = 基线 + `affinity_mlp.*` | `TestFaithfulConstruction` |
| I3 | eval 期 `g_hard ≡ 1`、`g_soft ∈ (0.5, σ(2)]`（对任意参数成立）；`W` 与 `W_attn` 的逐位差 ≤ STE 舍入量级（`routing_l1_max < 1e-6`，**不断言"逐位相等"**）；训练期路由概率 ∈ 解析窗口 | `test_eval_hard_gate_is_always_attention_for_any_parameters`、`test_ste_rounding_reading_is_recorded`、`test_exact_share_sweep_hits_analytic_bounds` |
| I4 | D2 复核：`env_embs` 元素一维 ⇒ `stack(dim=1) = [rep_dim, K]`、`cos_sim` **就是真余弦**（"错误维"已撤回）、错误布局触发守卫、K=2 时第 4 维 `mean\|C\|` 是前三维的函数（恒等式） | `test_env_layout_is_feature_first_and_cos_sim_is_a_true_cosine`、`test_layout_guard_rejects_wrong_env_layout`、`test_gate_features_are_redundant_for_k2`、`test_d2_corrected_readings_and_cos_entry_bounds` |
| I5 | D3：训练前向消耗全局 RNG、eval 不消耗；探针与轨迹记录**不消耗**全局 RNG | `test_eval_forward_consumes_no_rng_and_training_forward_does`、`test_trace_consumes_no_extra_rng`、`test_probe_is_deterministic_and_consumes_no_global_rng` |
| I6 | D4/D5：三条路径的梯度范数与衰减比；局部灵敏度上限 `0.25` vs `1/τ = 2`；`get_l2_reg` 不含门控 | `TestGradientProbe`、`test_get_l2_reg_excludes_affinity_mlp` |
| I7 | D6 修正读数 = `P(z>0)` 与 `mean σ(z/τ)`；解析读数 = 闭式解（并与蒙特卡洛交叉验证） | `TestAffinityStats`、`TestNoiseSignalProbe` |
| I8 | 预注册阈值与判据（含"STOP 主导于 AUC"、"保真 STOP 优先"）；文档与代码同数、且同步记录 D2 撤回与 D1 精度留痕 | `TestPreregisteredCriteria`（含 `test_doc_preregisters_same_numbers`、`test_doc_records_the_d2_retraction_and_ste_precision`） |
| I9 | 接线：`run_id` 后缀 `-affinity`、config / metrics 记录；基线臂不受影响（哨兵 mock 下仍跑通） | `TestRunnerIntegration` |

> 测试用 CPU 极小夹具（`input_size=8`、`rep_dim=4`、64 样本、2 epoch），**不构成**任何性能证据。

---

## 5. 记录指标（口径固定，全部 JSON 可序列化）

落盘位置：`artifacts/census_stage2/runs/<run_id>/metrics.json` 的 `affinity_arm` 段（**不写入 `mechanism`**，
两臂的 `mechanism` 键集保持完全一致）。

### 5.1 退化 / 保真诊断（`affinity_arm.diagnostics`，best_state 载入后、只用 val 前向一遍）

| 字段 | 定义 | 读法 |
|---|---|---|
| `eval_routing_share_attn` | eval 期硬判据为 1（选 attention）的样本占比 | **D1 签名读数：忠实实现恒为 `1.0`** |
| `routing_l1_mean` / `_max` | 逐样本 `‖W − W_attn‖₁` 的均值 / 最大 | **D1+D6 签名读数：`_max < 1e-6` ⇔ 评测路径就是纯 attention**（STE 舍入留痕 ⇒ 不恒为 `0.0`，见 D1(a′)） |
| `routing_identity_frac` | `W` 与 `W_attn` **逐位**相等的样本占比 | 不恒为 `1.0`（STE 舍入，见 D1(a′)）；机制结论由 `routing_l1_max < 1e-6` 承担 |
| `ste_exact_frac` | STE 值 `g` 与 `g_hard` **逐位**相等的样本占比 | D1(a′) 的直接留痕：`g = 1 − 2⁻²⁴` 的样本不计入 |
| `g_logit_min` / `_max`、`g_logit_in_unit_interval` | 双 sigmoid 的**结构签名**：`g_logit = σ(z) ∈ (0,1)`（float32 饱和时为 `1.0` / `0.0`） | 忠实实现落在开区间内；饱和是同一签名的更强形式 |
| `routing_fw_l1_mean` | 逐样本 `‖W − W_fw‖₁` 的均值 | 与 FW 的"距离"，> 0 表示确实没用 FW |
| `g_soft_min` / `_max` | eval 期软门控的极值 | 应落在 `(0.5, 0.8808]`（D1 的直接读数） |
| `presigmoid_abs_mean` / `_max`、`presigmoid_saturation_frac` | `\|z\|` 的均值 / 最大 / 饱和占比（阈值 `8.0`） | 门控是否一出生就死在饱和区 |
| `noise_share_exact_mean/min/max/window` | 训练期路由概率的**闭式解** `1 − exp(−exp(σ(z)))` 的统计量与极差 | 必须落在 `(0.6321, 0.9340)`，`window ≤ 0.3019` |
| `attention_drop_frac_exact` | `1 − mean(P)`：训练期被硬判据整支丢弃 attention 的样本占比 | `∈ [0.066, 0.368]` |
| `flip_rate_exact_mean` | `mean 2p(1−p)`：同一样本换一份噪声就翻面的概率 | `∈ [0.123, 0.465]`——判据是硬币 |
| `corrected_routing_share_attn` | **D6 修正口径**：`P(z > 0)`（z 当 logit、无噪） | 明显偏离 0/1 ⇒ 参数确实学到了逐样本差异（只是历史 eval 规则把它藏起来了） |
| `corrected_gate_mean` | `mean σ(z/τ)`（D6 修正口径的软读数） | 同上 |
| `cos_max_mean` / `cos_min_mean` / `cos_range_mean` / `cos_absmean_mean` | 4 维门控输入特征的样本均值 | 门控到底看到了什么 |
| `cos_col_mean[k]` / `routing_col_mean[k]` | **逐源任务**读数：第 k 个源任务的余弦列均值 / 路由权重列均值 | 逐源任务是否被区别对待（D2(b) 的证据来源） |
| `cos_entry_min` / `_max` | 逐条目余弦的全局极值 | 真余弦 ⇒ 必落在 `[-1, 1]`（D2(a) 的直接读数） |
| `env_normalize_dim` / `env_layout` | 归一化维（`0` = 特征维）与布局串 | 口径自证：`[rep_dim, K]` ⇒ `dim=0` 是特征维，**非缺陷**（D2(a)） |
| `temperature` / `affinity_input_dim` / `affinity_hidden_units` / `eval_g_soft_lower` / `_upper` / `gumbel_tau` | 常量回显 | 口径自证 |

### 5.2 噪声 / 信号探针（`affinity_arm.noise_signal_probe`，固定 seed、私有 `torch.Generator`）

| 字段 | 定义 |
|---|---|
| `noise_share_mc_mean/min/max/window` | 每样本重复抽 `draws=256` 次 Gumbel 噪声得到的**经验**路由概率及其极差 |
| `noise_share_exact_mean/min/max/window` | 同一批样本的**闭式解**（用于交叉验证） |
| `mc_exact_max_abs_error` | 蒙特卡洛与闭式解的逐样本最大偏差（抽样误差量级） |
| `flip_rate_mc` / `flip_rate_exact_mean` | 同一样本相邻两次抽取就翻面的比例 / 其闭式解 |
| `train_eval_disagreement_mc` | `1 − mean(P)`：训练期判据与 eval 判据（恒 1）不一致的样本占比 |
| `g_soft_within_sample_std` | **噪声轴**：每样本在 256 次抽样上的软门控标准差，再对样本取均值 |
| `g_soft_between_sample_std` | **样本轴**：每样本软门控均值在样本间的标准差 |
| `noise_to_signal_ratio` | 上两者之比（噪声离散度 ÷ 信号离散度） |
| `gumbel_share_lower/upper`、`gumbel_share_window_max_exact`、`flip_rate_min/max_exact` | 解析边界回显 |

### 5.3 梯度探针（`affinity_arm.gradient_probe`，`torch.autograd.grad` 不写 `.grad`）

| 字段 | 定义 |
|---|---|
| `grad_norm_historical` | 历史路径（双 sigmoid + Gumbel 噪声，私有 generator）：affinity_mlp 全参数梯度 L2 范数 |
| `grad_norm_noiseless` | 去噪（eval 分支、硬判据恒 1）后的同一量 |
| `grad_norm_corrected` | **D6 修正路径**（z 当 logit、无噪）的同一量 |
| `attenuation_noise` / `attenuation_double_sigmoid` / `attenuation_total` | `noiseless/historical`、`corrected/noiseless`、`corrected/historical`（分母为 0 时为 `null`） |
| `local_sensitivity_logit_max` / `local_sensitivity_corrected_max` | `0.25` vs `1/τ = 2`（可证的局部灵敏度上限） |

### 5.4 训练期轨迹（`affinity_arm.train_gate_trace`，逐 epoch 一条）与总量（`affinity_arm.affinity_grad_norm`）

逐 epoch：`epoch` / `steps` / `grad_nonzero_steps` / `grad_norm_mean` / `grad_norm_max`
（**backward 之后、`optimizer.step()` 之前**读取）+ 与 5.1 同名的门控读数。
总量 = 全期门控读数（与 5.1 同名字段，覆盖整个训练集） + `epochs` / `steps` / `grad_norm_mean` /
`grad_norm_max` / `grad_nonzero_steps`；`grad_norm_max` 就是机制 STOP 判据 3 的输入。

> **接线要求**：轨迹读取的是 `forward` 存下的 `last_terms`（真实前向的中间量），**不重算**——
> 训练期重算会多抽一次 Gumbel 噪声，破坏"忠实复现"的随机流（测试 `test_trace_consumes_no_extra_rng` 锁定）。

### 5.5 溯源与判据快照

`affinity_arm.provenance`（历史来源/主张/结果、`env_normalize_dim=0`、`gumbel_noise_rng="global"`、
`eval_hard_gate="constant_1"`、`affinity_mlp_in_l2=false`、构造顺序）、`affinity_arm.prereg`（判据与优先级原文）、
`affinity_arm.status` / `stop_reason` / `pass`、`config.json` 的 `variant` / `fusion_mode` / `affinity_tau` /
`affinity_hidden_units`。

**冻结复验（补 A1 的时序缺口）**：诊断/探针在 A1 取 `backbone_sha256` **之后**才运行，
故 `affinity_arm` 另记 `backbone_sha256_after_probes`（必须等于 `backbone_sha256_after`）与
`grads_all_none_after_probes`（必须为 `true`）。这两项**不改变 A1 的判定语义**，
只把"诊断没有扰动冻结 backbone"这一事实也留痕（测试锁定）。

---

## 6. 预注册判据（**只允许在看到结果之前修改**）

### 6.1 优先级（主导关系）

```
保真 STOP  >  机制 STOP  >  效应阈值
```

任何 STOP 一旦命中：`effect.evaluated = false`、`effect.pass = null`，臂状态 `STOP`——
**不得**用该 run 的 AUC 去谈"是否达标"。这条规则的意义在于：本实验要回答的是
"历史的门控机制是否真的在工作"，而不是"这个数字好不好看"。

### 6.2 机制 STOP（三条任一命中 ⇒ `CONFIRMED_DEGENERATE`）

```
eval_attn_share >= 0.999   或   routing_l1 < 1e-6   或   grad_norm == 0
```

| 规则 | 输入字段 | 理由 |
|---|---|---|
| `eval_attn_share >= 0.999` | `diagnostics.eval_routing_share_attn` | eval 期硬门控几乎恒选 attention ⇒ "OOD-aware 切换"在评测路径上不存在（D1/D6）。忠实实现预期 = `1.0` |
| `routing_l1 < 1e-6` | `diagnostics.routing_l1_mean` | `W = W_attn`（1e-6 内，STE 舍入留痕见 D1(a′)）⇒ 评测路径就是纯 attention，机制不存在（D1/D6 的直接后果） |
| `grad_norm == 0` | `train_gate_trace` 的 `grad_norm_max` | 门控**从未收到过任何梯度** ⇒ 它不是"没学好"，而是根本没在学（D4；空轨迹同样判 0） |

### 6.3 保真 STOP（优先于上面三条）

```
noise_share_window <= 0.32      （解析上界 0.3019；超出 ⇒ FIDELITY_BREAK ⇒ STOP）
```

忠实复现下 `noise_share_window ≤ 0.3019` 是**恒等式**（`g_logit = σ(z) ∈ (0,1)`）。
一旦观测超出 0.32，说明复现偏离历史（例如"顺手修好了" D1），此时机制结论与效应结论都不可信 ⇒
优先 STOP 并先排查实现偏离。

### 6.4 效应阈值（**仅在无任何 STOP 时**才解释）

```
AUC-Test-Education >= 0.8521
```

机读规则文本（与 `census_benchmark/affinity_gate.py` 的 `RULE_EFFECT` 逐字一致）：

```
auc_test_education >= 0.8521
```

- 参照：协议基线行 `20260929-1735-s20260929-m1685480945-short-904f8d0` 的 `0.850069`（同 stage1、同划分）。
- STOP 时 `effect.evaluated = false`、`effect.pass = null`。
- 判定写入 `metrics.json:affinity_arm`，**不并入** `judge()` 的 `overall_pass`（A/B 门禁语义不变）；
  `SUMMARY.md` 照协议追加一行（`run_id` 带 `-affinity` 后缀，与基线行天然可区分）。

### 6.5 阈值纪律

1. 阈值只允许在**看到本次 run 结果之前**修改；每次修改单独记录理由（先进本文档，再改 `affinity_gate.py`）。
2. **禁止事后调参救判据**：不得为了避开退化而改 `τ`、改批大小、改初始化、换 model seed、换划分。
   退化就是退化——这正是预注册的全部意义。
3. 单次判定：本轮只跑单 model seed `1685480945`；"多 seed 平均后才达标"不算通过。
4. 运行期间发现实现缺陷 → 允许修复后重跑，但必须同时保留失败 run 的产物与说明。
5. **复核更正留痕（预注册完整性）**：本文档在**看到任何 run 结果之前**做过两处更正——(1) 撤回 D2 的
   "归一化维度错误"结论（3 节 D2(a)）；(2) 补记 D1 的 STE float32 舍入留痕（3 节 D1(a′)）。
   两处都**不改**任何阈值、规则或口径（`0.8521` / `0.999` / `1e-6` / `0.32` / 解析上界 `0.3019`
   全部原样），只更正叙述与读数解读，并同步改了代码、测试与 `provenance.d2_audit_correction`；
   因此预注册仍然成立。若将来再发现类似误判，按同一方式处理：改文档 + 改代码 + 留撤回归因，
   **不得**为了让判据好看而调整叙事。

---

## 7. 接线与运行（基线臂零影响）

| 位置 | 改动 |
|---|---|
| `census_benchmark/affinity_gate.py` | **新增**：忠实复现头 + 读数累加器 + 训练轨迹 + 诊断 + 噪声/梯度探针 + 判据 |
| `run_census_benchmark.run_stage2` | 新增 `variant="baseline"` 参数；新任务头改用 `affinity_gate.build_newtask(variant, …)`；处理臂逐 epoch 记录轨迹、best_state 后做诊断与探针 |
| `run_id` | 处理臂追加后缀 `-affinity`（协议 `make_run_id` 未改） |
| `config.json` | 两臂记 `variant`；处理臂另记 `fusion_mode` / `affinity_tau` / `affinity_hidden_units` / `noise_probe_draws` / `noise_probe_seed` |
| `metrics.json` | 两臂记 `variant` 且 `mechanism` 键集不变；处理臂另记 `affinity_arm` |
| CLI | `stage2 --variant {baseline,affinity}`（默认 `baseline`）与等价别名 `--affinity` |
| `protocol.py` / `metrics.py` / `model.py` / `config.py` | **零改动**（测试 `git diff` 静态守卫锁定） |

**基线臂不受影响的三条证据**（测试锁定）：

1. `build_newtask("baseline", …)` 返回的**就是** `NewTask`（`type(...) is NewTask`），同 seed 下参数 /
   buffer / **全局 RNG 终点**与直接构造逐位相同；
2. 基线 run 的 `run_id` 无 `-affinity`、payload 无 `affinity_arm`、`config.json` 无处理臂专属字段；
   测试用 `mock` 把 `HistoricalAffinityGateNewTask` 换成"一旦被构造就报错"的哨兵，基线 run 仍必须跑通；
3. 基线臂与处理臂的 `mechanism` 键集完全一致（`["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"]`）。
   *唯一*新增的是 `config.json` / `metrics.json` 里恒为 `"baseline"` 的 `variant` 标签与 `run_id` 的后缀约定
   （与 null-expert / attn / csb / cgr 分支同一约定）。

**运行命令**（单 seed 短跑；与基线同一 stage1 产物）：

```powershell
# 0) 测试（全部不变量）
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v

# 1) 处理臂（本实验唯一要跑的东西）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --variant affinity --tag short --gpu 0
# 等价写法：--affinity

# 2) 基线对照（可选；同一 stage1、同一划分，预期复现 SUMMARY.md 的 0.850069）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --tag short --gpu 0
```

产物：`artifacts/census_stage2/runs/<run_id>/`（`metrics.json:affinity_arm` 含判据、诊断、探针与逐 epoch 轨迹），
`SUMMARY.md` 追加一行。**A3 复跑**按协议按需触发，不阻塞本轮。

**前置条件**：开跑前先把本批改动提交，否则 `run_id` / `metrics.json` 里的 `commit` 记的是 `87afe03`
（不含本实验代码），可审计性有缺口。**（结果留痕：本次执行时该前置条件**未**被满足，缺口已如实记录在
8.3 第 9 条；此处只补记录，不改动任何判据。）**

---

## 8. 结果（真实验已执行：单 seed 短跑）

> 执行于 2026-09-30 18:37（本机时区），产物目录
> `artifacts/census_stage2/runs/20260930-1837-s20260929-m1685480945-short-87afe03-affinity/`
> （`metrics.json:affinity_arm` / `config.json` / `gate_report.json` / `stdout.log`）。
> **不论好坏一律留痕**（协议 7.3：不得只报告成功的 run，不得在看到结果后改判据）；第 6 节的阈值、
> 规则与优先级**一字未改**，`metrics.json:affinity_arm.prereg` 是与第 6 节同数的落盘快照。

### 8.1 预判（**写于 run 之前**，仅作对照留痕；不是结果）

> 以下三条是第 3 节解析结论的直接推论，写在看到任何 run 结果之前：

1. `eval_attn_share = 1.0`（精确）、`routing_l1_max < 1e-6`（STE 舍入留痕 ⇒ `routing_l1_mean` 是 ~1e-8…1e-7
   而不是 `0.0`，`routing_identity_frac` 也不恒为 `1.0`——见 D1(a′)）
   ⇒ 判据 1、2 **预期命中**，臂状态预期 `STOP`（`stop_reason = "degenerate"`），
   `effect.evaluated = false`；
2. `noise_share_exact_*` 落在 `(0.6321, 0.9340)`、`window ≤ 0.3019` ⇒ 保真判据**预期不触发**；
   若触发，说明复现偏离历史，先排查实现再谈结论；
3. `grad_norm_max > 0`（历史梯度被压制但**不为 0**）⇒ 判据 3 **预期不命中**；
   若命中（精确为 0），说明门控连梯度都没收到，退化程度比解析预期更极端。

**因此本轮的预期结论是**：历史 "affinity gate 通过逐样本 cosine 特征检测 OOD 并切换到 FW" 这一解释
**不成立**；被评测到的模型就是纯 attention（`W = W_attn` 在 1e-6 内逐样本成立，`g_hard ≡ 1` 精确成立），
历史 Δ=−0.0028 只能归因于 D3 的随机流/初始化位移与训练随机性，**不能**读成"门控机制带来了稳定性"。

### 8.2 实测回填

| 项 | 取值 |
|---|---|
| `run_id` / `commit` / `stage1_id` | `20260930-1837-s20260929-m1685480945-short-87afe03-affinity` / `87afe03`（**该字段不含本实验代码**，见 8.3 第 9 条）/ `s1-096f8f16-m1685480945-e2-cb2094b3`（与基线行同一 stage1 产物） |
| 配置（`config.json`） | `variant=affinity`、`epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`、`input_size=123`、`rep_dim=128`、`fusion_mode=affinity_gate`、`affinity_tau=0.5`、`affinity_hidden_units=16`、`noise_probe_draws=256`、`noise_probe_seed=20260930`、`split_seed=20260929`、`model_seed=1685480945`、`env_seed=20260929` |
| `AUC-Val-Education`（best，epoch 5） | `0.8492416205632258` |
| `AUC-Test-Education` | `0.8465147339710887` |
| 基线参照 `20260929-1735-…-904f8d0` / Δ | `0.8500685307175756`（其 best val `0.8527881905614896`）/ **Δ = −0.0035537967464869** |
| 机制 STOP 三条 `observed` | ① `eval_attn_share = 1.0`（≥ 0.999，**命中**）② `routing_l1_mean = 8.425798429712178e-09`（< 1e-6，**命中**）③ `grad_norm_max = 7.530469790294838e-04`（≠ 0，**未命中**） |
| 保真 STOP `observed`（`noise_share_window`） | `1.6556509383086393e-02` ≤ 0.32 ⇒ **未命中**（解析上界 `0.3018914053261298`） |
| 臂状态 | `status = STOP`、`stop_reason = "degenerate"`、`degeneracy.status = CONFIRMED_DEGENERATE`、`triggered_rules = ["eval_attn_share >= 0.999", "routing_l1 < 1e-6"]`、`effect.evaluated = false`、`effect.pass = null`（`effect.observed = 0.8465147339710887` 仅留痕、不解释） |
| A/B 门禁（`gate_report.json`） | A1 / A2 / A4 / A5 / B1 / B2 / B4 全 PASS，**B3 FAIL**（`gate_mean = [0.952401451979339, 0.8543901456350915]`，取自冻结 backbone 的 `gate_networks` ⇒ **继承性失败**，与处理臂无关）；`overall_pass = false`、`failures = ["B3"]`。B2 `gap = 0.0027268865921371566`（limit 0.03）；B4 `env_shares = [0.49411847255704855, 0.5058815274429515]` |

关键读数（`metrics.json:affinity_arm`；eval 列 = best_state 载入后对 val 前向一遍，`n = 49881`、`n_batches = 195`；
训练列 = 全期累加，`n = 997615`、`n_batches = 3900`）：

| 读数 | eval | 训练全期 | 读法 |
|---|---|---|---|
| `g_hard_mean`（= `eval_routing_share_attn`） | `1.0` | `0.9202768603118437` | D1 签名读数 |
| `g_soft_mean` / `_min` / `_max` | `0.8788061158649092` / `0.8615279197692871` / `0.8807885646820068` | `0.8466924225018643` / `0.028793198987841606` / `1.0` | eval 全样本落在 `(0.5, σ(2)=0.8807970…]` 内 ⇒ D1 解析窗口实测成立 |
| `g_logit_min` / `_max`、`g_logit_in_unit_interval` | `0.9140196442604065` / `0.9999592304229736`、`true` | `0.4433450996875763` / `0.9999562501907349` | 双 sigmoid 结构签名 |
| `routing_l1_mean` / `_max` | `8.425798429712178e-09` / `7.450580596923828e-08` | `0.02804456861853814` / `0.6586210429668427` | **判据 2 的读数**（eval 全样本 < 1e-6） |
| `routing_identity_frac` / `ste_exact_frac` | `0.8434073093963633` / `0.7467171869048335` | `0.7727850924454824` / `0.776168161064138` | **都不是 `1.0`** ⇒ D1(a′) 的精度留痕如预期 |
| `g_min`（eval） | `0.9999999403953552` | `0.0` | 恰为 `1 − 2⁻²⁴` ⇒ D1(a′) 推导的 STE 舍入值被**逐位**观测到 |
| `routing_fw_l1_mean` | `0.34554091299720857` | `0.3419700327432658` | 距 FW 很远 ⇒ 确实没走 FW |
| `presigmoid_abs_mean` / `_max` / `saturation_frac`（\|z\| ≥ 8） | `6.441874981270655` / `10.108620643615723` / `0.28289328602072933` | `4.411667594832753` / `10.038057327270508` / `0.061805405893054936` | 门控把自己推进饱和区 |
| `noise_share_exact_mean` / `_min` / `_max` / `_window` | `0.9323055603545068` / `0.9174481494552673` / `0.9340046588383537` / `0.016556509383086393` | `0.9198616499784975` / `0.7894242560407985` / `0.934004124731609` / `0.14457986869081052` | 全落在 Gumbel 窗口 `(0.6321, 0.9340)` 内 |
| `attention_drop_frac_exact` / `flip_rate_exact_mean` | `0.06769443964549315` / `0.12620331543563004` | `0.08013835002150249` / `0.1458041663066922` | 训练期判据是硬币 |
| `corrected_routing_share_attn` / `corrected_gate_mean`（D6） | `1.0` / `0.9995836911974161` | `0.9725806047423105` / `0.9679280587136581` | **修正口径也恒选 attention** |
| `cos_range_mean` / `cos_col_mean` / `routing_col_mean` | `0.6833576007405309` / `[0.1527319168548551, −0.11490096384107919]` / `[0.5657911409571699, 0.4342088522651484]` | `0.7602644347758108` / `[0.12733123453516348, −0.09889813350620519]` / `[0.5530823408976003, 0.4469176534011034]` | 输入有结构、逐源任务列有别 |
| `cos_entry_min` / `_max` | `−0.672027587890625` / `0.6775820255279541` | `−0.6820110082626343` / `0.6832059621810913` | 落在 `[−1, 1]` ⇒ 真余弦（**D2(a) 实测自证**） |
| `env_normalize_dim` / `env_layout` | `0` / `"feature_first_(rep_dim, K)"` | 同 | 口径自证（**非缺陷**） |
| `noise_signal_probe.g_soft_within_sample_std` / `_between_sample_std` / `noise_to_signal_ratio` | — | `0.18074005454334963` / `0.011528862729794469` / **`15.677179855412485`** | 噪声轴离散度 ÷ 信号轴离散度 |
| `noise_signal_probe.train_eval_disagreement_mc` | — | `0.06744384765625` | 训练判据 vs eval 判据（恒 1）的不一致率 |
| `noise_signal_probe.flip_rate_mc` / `flip_rate_exact_mean` | — | `0.12565870098039217` / `0.12642953617098868` | 换一份噪声的翻面率（MC 与闭式解一致） |
| `noise_signal_probe.mc_exact_max_abs_error` | — | `0.05023384773911155`（`draws = 256`、`n_samples = 256`） | 蒙特卡洛 vs 闭式解的抽样误差 |
| `gradient_probe.grad_norm_historical` / `_noiseless` / `_corrected` | — | `2.2778190839199903e-04` / `3.219138546419542e-04` / `1.0900023106489135e-05` | 三条路径的同一量 |
| `gradient_probe.attenuation_noise` / `_double_sigmoid` / `_total` | — | `1.4132547089207794` / **`0.0338600620921165`** / `0.04785289219603362` | 见 8.3 第 6 条 |
| `train_gate_trace[*].grad_norm_mean`（逐 epoch 1→5） | — | `1.1640647520256034e-04` → `6.716134081617064e-05` → `3.6546248467813156e-05` → `2.438754157573268e-05` → `1.7082675646559795e-05` | 单调衰减 |
| `affinity_grad_norm.grad_nonzero_steps` / `steps` | — | `3900` / `3900` | 门控**收到过**梯度（判据 3 不命中） |
| `backbone_sha256_after_probes` / `grads_all_none_after_probes` | `a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85`（= `backbone_sha256_after`）/ `true` | — | 诊断与探针未扰动冻结 backbone |

逐 epoch 轨迹里 D6 修正读数的**坍缩过程**也一并留痕（这是"门控并非从未有过输入依赖"的证据）：

| epoch | `corrected_routing_share_attn` | `corrected_gate_mean` | `presigmoid_abs_mean` | `saturation_frac` | `g_min` |
|---|---|---|---|---|---|
| 1 | `0.862903023711552` | `0.8506502187486294` | `1.6515126922274845` | `0.0` | `0.0` |
| 2 | `1.0` | `0.9935146228426359` | `3.7861458139147874` | `0.0` | `0.0` |
| 3 | `1.0` | `0.997174782699567` | `4.672577488294714` | `0.0` | `0.0` |
| 4 | `1.0` | `0.9987341098781629` | `5.455776995030516` | `0.03743929271312079` | `0.0` |
| 5 | `1.0` | `0.9995665593992986` | `6.492324984696264` | `0.27158773675215386` | `0.0` |

### 8.3 判读（结论）

**一句话**：历史 Affinity Gate "用逐样本 cosine 特征检测 OOD 样本并切换到 FW" 这一机制，
在**评测路径上不存在**——它不只是"效果不好"，而是**结构上没有发生过**；训练期那些看起来像路由的
随机性**没有存活到 eval**，且其本身由 Gumbel 噪声主导（噪声:信号 ≈ 15.7:1）。

1. **硬门控在 eval 期是结构常函数 ⇒ 被评测到的模型就是纯 attention。**
   `g_logit = σ(z) ∈ (0,1)` ⇒ `g_soft = σ((g_logit+G)/τ) > 0.5` 对**任意参数、任意输入**恒成立
   ⇒ `g_hard ≡ 1` ⇒ `W = g·W_attn + (1−g)·W_fw = W_attn`。实测坐实：`eval_routing_share_attn = 1.0`
   （49881/49881）、`routing_l1_mean = 8.43e-09`、`routing_l1_max = 7.45e-08`（判据阈值 1e-6，余量约 13×）。
   ⇒ **"切换到 FW"从未在历史评测路径上发生**；这条既是解析结论（§3 D1），现在也是落盘读数。
2. **`routing_l1` 不恒为 0 是 D1(a′) 预测的 STE float32 舍入，不是"门控在工作"。**
   `routing_identity_frac = 0.8434`、`ste_exact_frac = 0.7467` 都**不是** `1.0`；eval `g_min =
   0.9999999403953552 = 1 − 2⁻²⁴` 与 D1(a′) 的推导值**逐位相同**；`routing_l1_max` 也落在 D1(a′)
   预估的 ~1e-7 包络内。⇒ 早期"`W` 与 `W_attn` 逐位相等"的口径是错的（§5.1 已改），机制结论只能由
   `routing_l1_max < 1e-6` 承担。
3. **训练期路由是噪声主导的硬币，且不存活到 eval。**
   * **噪声 ÷ 信号**：`noise_to_signal_ratio = 15.677179855412485`（同一样本重抽噪声造成的软门控波动
     `g_soft_within_sample_std = 0.1807`，是样本间差异 `g_soft_between_sample_std = 0.01153` 的 15.7 倍）。
     ⇒ 门控的"逐样本差异"信号比它被混合的噪声小一个数量级以上。
   * **翻面率**：`flip_rate_exact_mean = 0.1262`（MC `0.1257`，闭式解与经验值一致）——同一样本换一份
     噪声就有 ~12.6% 的概率改变硬判据。
   * **训练/eval 错配**：`train_eval_disagreement_mc = 0.0674`（≈ eval 口径的
     `attention_drop_frac_exact = 0.0677`）——训练期按噪声判为 FW 的那批样本，在 eval 期一律被判回 attention。

   ⇒ 训练期 `1 − P ∈ [0.0659, 0.3680]` 那部分"整支丢弃 attention"的样本，其去留由 Gumbel 噪声而非输入
   决定；**训练时被评的模型与评测时的模型不是同一个路由策略**（D1(c)）。故训练期的随机性**没有任何
   可迁移到评测的机制含义**——它既不是 OOD 检测，也不是"稳定性来源"。
4. **D6 修正口径也没能救回机制（预注册的可检验分支落在"没学到"这一支）。**
   把前 sigmoid 输出 `z` 当 logit 的反事实读出：eval `corrected_routing_share_attn = 1.0`、
   `corrected_gate_mean = 0.9995836911974161`。按 §3 D6 **写在看到结果之前**的读法——
   "若它同样逼近 0 或 1，说明门控本身就没学到东西"——本 run 命中该分支。
   *准确性补充（不改变结论）*：轨迹显示 **epoch 1 尚有输入依赖**（`corrected_routing_share_attn =
   0.8629`、`g_min = 0.0`、`presigmoid_abs_max = 4.57`），此后逐 epoch 坍缩到 `1.0`，同时
   `presigmoid_abs_mean` 从 `1.65` 升到 `6.49`、饱和占比从 `0` 升到 `0.283`。⇒ 更精确的表述是
   "**训练早期存在微弱输入依赖，随后被自身动力学推到恒 attention 角**"，而不是"从来没有过"。
   无论哪种表述，**被评测的那组参数上不存在逐样本路由**。
5. **"输入有信号" ≠ "门控学到了信号"（必须分开）。** 门控看到的输入确实有结构：
   `cos_range_mean = 0.6834`、`cos_col_mean = [+0.1527, −0.1149]`、`cos_entry_min/max = −0.672/+0.678`
   （真余弦值域内）。所以"没学到"**不是**因为输入无信息，可归因的是 §3 D2(b)（这 4 维与 `W_attn` 出自
   同一个 `H(x)`、**同源**；K=2 时第 4 维 `mean_k|C| = (|max|+|min|)/2` 是前三维的函数 ⇒ 实际只有
   2 个自由数）与 D4（梯度路径被压扁 + 噪声污染）。
   **D2 的审计更正据此如实记录**：归一化维 `dim=0` 是**特征维、正确**（`env_layout =
   "feature_first_(rep_dim, K)"`，`cos_entry_* ∈ [−1,1]` 实测自证；"应为 `dim=-1`"之说已撤回）；
   **存活下来的 D2 问题是"冗余 + 同源"**——门控特征既与它要解释的路由出自同一投影，K=2 时又只有
   2 个自由数。
6. **D4 的"双重压制"在实测工作点上方向反转（如实记录，不改任何判据）。**
   `attenuation_double_sigmoid = corrected / noiseless = 0.0338600620921165 < 1`：在训练最终落到的
   饱和工作点（`presigmoid_abs_mean = 6.44`、`28.3%` 样本 `|z| ≥ 8`）上，D6 修正路径的梯度反而比
   历史无噪路径**小约 29.5 倍**。原因：D4 的 `σ′(z) ≤ 0.25` vs `(1/τ)σ′(z/τ) ≤ 2` 是 **z ≈ 0 处**
   取到的全局上界，而训练把 `z` 推到深饱和区——双 sigmoid 的**内层** `σ(z)` 反而把外层 sigmoid 从
   饱和区拉了出来（`g_soft ∈ (0.8615, 0.8808]`，贴近但未达 `σ(2)`），而修正路径的 `σ(z/τ) = σ(2z)`
   已彻底饱和。噪声衰减 `attenuation_noise = 1.4132547089207794`（无噪/历史）⇒ 噪声把梯度压低约 29%。
   ⇒ 这一条只说明 **D4 的叙述适用范围是 z ≈ 0 附近**，不构成对判据 1/2 或 D6 结论的影响。
7. **梯度确实到达门控，只是极小且逐 epoch 衰减（判据 3 未命中，符合 8.1 预判 3）。**
   `grad_nonzero_steps = 3900 / 3900`、`grad_norm_max = 7.53e-4`、`grad_norm_mean = 5.23e-5`，逐 epoch
   均值 `1.16e-4 → 1.71e-5` 单调下降。⇒ 退化**不是**"梯度断了"（那是预注册里更极端的形态），
   而是"**梯度在，但它把门控推向常数**"。
8. **保真成立：本轮复现没有"顺手修好"D1。** `noise_share_window = 0.016556509383086393 ≤ 0.32`
   （解析上界 `0.3018914053261298`），`noise_share_exact_*` 全部落在 `(0.6321, 0.9340)`，
   `g_logit_in_unit_interval = true`。观测窗口远小于解析上界的原因也一并记下：`σ(z)` 饱和到
   `[0.9140, 0.99996]`，只覆盖解析区间的**顶端子区间**——这从另一个侧面印证"门控把自己推进了恒
   attention 角"。
9. **可审计性缺口（如实记录，不粉饰）**：`metrics.json:commit = 87afe03`，但本实验的代码
   （`census_benchmark/affinity_gate.py`、`census_benchmark/tests/test_affinity_gate.py`、
   `run_census_benchmark.py` 的改动）在开跑时**尚未提交**——正是 §7"前置条件"预警的那条缺口。
   支持"跑的就是当前工作树"的证据是时间戳（三个源文件 mtime `18:26:36` / `18:27:07` / `16:02:43`，
   均早于产物 `18:37:19`），但**时间戳不构成内容指纹**。本次收尾按指示不提交、不推送、不合并，
   故该缺口保留在此，供后续复核者按工作树内容对齐。
10. **A/B 门禁与公平性未受影响**：A1 / A2 / A4 / A5 / B1 / B2 / B4 全 PASS；唯一 FAIL 的 B3 是
    **继承性失败**（`gate_mean` 取自冻结 backbone 的 `gate_networks`，与处理臂无关——基线行
    `20260929-1735-…-904f8d0` 同样 B3 FAIL）。冻结复验 `backbone_sha256_after_probes` 与
    `backbone_sha256_after` 逐位相同、`grads_all_none_after_probes = true` ⇒ 诊断与探针未扰动 backbone。
11. **效应阈值不作解释。** `AUC-Test-Education = 0.8465147339710887`、Δ = −0.00355 **只作留痕**：
    按 §6.1 的优先级，机制 STOP 一旦命中即 `effect.evaluated = false`，**不得**用该 run 的 AUC 去谈
    "是否达标"或"门控是否伤害了性能"。何况该 Δ 是单 seed，且按 §3 D3 混着"不同随机流 + 不同初始化"
    的位移，与历史 Δ = −0.0028 一样**不能**读成"门控机制带来了稳定性"。
12. **测试**：`census_benchmark/tests/test_affinity_gate.py` 共 **73** 个 test 方法（13 个 `TestCase`
    类；本轮静态计数复核为 73），由执行者独立运行**全部通过**——复现件本身的不变量（I1–I9）没有被
    真实验推翻。

### 8.4 预判 vs 实测（8.1 写在 run 之前，逐条对照）

| 8.1 预判 | 实测 | 是否成立 |
|---|---|---|
| 判据 1、2 **预期命中**，`stop_reason = "degenerate"`、`effect.evaluated = false` | `eval_attn_share = 1.0`、`routing_l1_mean = 8.43e-09` 双命中；`status = STOP`、`stop_reason = "degenerate"`、`effect.evaluated = false` | ✅ 成立 |
| `routing_l1_mean` 是 ~1e-8…1e-7 而非 `0.0`、`routing_identity_frac` 不恒为 `1.0`（D1(a′)） | `8.425798429712178e-09` / `0.8434073093963633`；`g_min = 1 − 2⁻²⁴` | ✅ 成立（且逐位吻合） |
| 保真判据**预期不触发** | `noise_share_window = 0.0166` ≤ 0.32 | ✅ 成立 |
| 判据 3 **预期不命中**（`grad_norm_max > 0`） | `7.530469790294838e-04`，3900/3900 步非零 | ✅ 成立 |
| 预期结论：历史解释不成立、被评测模型即纯 attention | 命中，且 D6 修正读数独立地指向同一结论 | ✅ 成立 |

**预判之外、本轮新观测到的两点**（如实补记，不改判据）：
(a) D6 修正读数在**最终参数上也是常数**（`corrected_routing_share_attn = 1.0`），但**训练早期曾有**
微弱输入依赖（epoch 1 = `0.8629`）——"没学到"是**训练动力学的结果**，不是初始状态；
(b) `attenuation_double_sigmoid < 1`（8.3 第 6 条），即 D4 的压制叙述在工作点上方向反转。

### 8.5 本轮处置

- **判定**：`STOP` / `CONFIRMED_DEGENERATE` ⇒ **历史解释被否证**（"退化"而非"效果不足"）。
  三种结局都按第 6 节判据落盘；**不重跑挑好的、不换阈值、不换 seed、不换划分**（§6.5）。
- **不做多 seed / full**：机制 STOP 是**结构性**的（判据 1 对任意参数、任意输入恒成立），单 seed 短跑
  足以定论；多 seed 只会重复同一结论而消耗算力。
- **保留**：`exp/stage2-affinity-gate-repro` 连同失败结果一并保留（产物目录 + `SUMMARY.md` 行 + 本文件）；
  **不合并 `master`**。本次收尾未提交、未推送（§9.2 的"保留并推送"是常驻策略，由执行者另行决定）。

---

## 9. 非新颖性与结果处置（保留策略）

### 9.1 非新颖性

- 硬门控 / 门控融合 / Gumbel-Softmax 路由（`gated fusion`、MoE 路由、conditional computation 同族）
  都是**已有大类**；本分支**不主张任何新颖性**，也不提出新方法。
- 被复核的 Affinity Gate 在历史上也**从未超越** prompt（CensusIncome `−0.0028`；AliCCP `−0.0030`，
  相对 FW 为 `−0.0016`）。
- 本分支的产出只有三样：**(a)** 一份可审计的忠实复现（含与归档类逐位一致的测试）、
  **(b)** 一套"退化是否成立"的预注册判据与读数、**(c)** 对历史解释的否证或支持。

### 9.2 结果处置

- **成功 / 失败 / 退化**：三种结局都按第 6 节判据落盘；失败与退化**不得**重跑挑好的、不得换阈值。
- **不合并 `master`**：本分支是 `exp/*`，只引用评测协议，不改协议文件。
- **保留并推送**：按用户硬规则，无论结果如何，`exp/stage2-affinity-gate-repro` 连同失败结果一并保留并推送。
- 权重的处置沿用协议 9.3：`*.pt` / `*.json` / `*.log` 不入库（`.gitignore` 天然拦截），
  `SUMMARY.md` 行与本文件入库。
