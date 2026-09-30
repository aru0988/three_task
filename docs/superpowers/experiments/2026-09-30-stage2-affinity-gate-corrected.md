# exp/stage2-affinity-gate-corrected：修正版逐样本 Affinity 软门控（预注册 + 设计）

- **状态**：**已运行（单 seed 短跑）⇒ `STOP_MECHANISM`**——机制 6 条中 5 条通过，唯 `gate_std >= 0.01`
  未过（实测 `0.00713`）；效应按主导关系**不评估**；对齐读数已记录（**负值**）但不参与判定。
  **工程缺陷全部修好，失败的是被修正后的机制本身**——读数、解释与逐条对照见第 11 节。
  第 1–10 节是本文档**尚未运行**时冻结的预注册与设计文本，一字未改（阈值/规则/优先级按 6.5 保持原样，
  结果只以**新增小节**回填）
- **日期**：2026-09-30
- **分支**：`exp/stage2-affinity-gate-corrected`（从 `infra/fair-stage2-benchmark` @ `87afe03` 独立拉出）
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。
  划分、三个种子、真冻结三件套、A/B 门禁、报告纪律一律沿用；`census_benchmark/protocol.py` /
  `census_benchmark/metrics.py` / `multitaskrec/model.py` / `config.py` **零改动**（测试静态守卫锁定）。
- **被修正对象**：`exp/stage2-affinity-gate-repro` @ `88d787c` 的历史 Affinity Gate 忠实复现
  （`docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-revalidation.md`，实验结论
  `status = STOP` / `CONFIRMED_DEGENERATE`）。
- **数据集**：只跑 CensusIncome（协议 2.3）。
- **定位**：这不是新方法（见第 9 节）。它是对一条**已被否证**的历史机制的"缺陷逐条修正后重测"，
  回答一个可证伪的问题：**修正后的确定性逐样本软门控，能在公平协议下同时通过机制判据与效应阈值吗？**

---

## 1. 历史复现给出的、必须修掉的缺陷（审计溯源）

历史复现（`88d787c`，单 seed 短跑，`AUC-Test-Education = 0.8465147339710887`，Δ = −0.0035537967464869
vs 协议基线 `0.8500685307175756`）在真实验里逐条坐实了下列缺陷。**本分支逐条修正，并在第 3 节给出
"改了什么 / 没改什么"的对照表**：

| 编号 | 历史缺陷（已被落盘读数坐实） | 本分支的处置 |
|---|---|---|
| **D1** | `affinity_mlp` 末端 `nn.Sigmoid` 的输出被当作 logit 再 sigmoid 一次 ⇒ eval 期 `g_soft ∈ (0.5, σ(2)]` 恒 > 0.5 ⇒ 硬判据 `g_hard ≡ 1` **对任意参数、任意输入**成立 ⇒ `W = W_attn`：评测路径上**根本没有路由**（实测 `eval_routing_share_attn = 1.0`、`routing_l1_mean = 8.43e-09`） | 单 sigmoid：`logit` 无约束，`g = sigmoid(logit)` **就是输出**；无硬判据、无 STE |
| **D1(c)/D6** | 训练期硬切换是 Gumbel 噪声主导的硬币（`noise_to_signal_ratio = 15.677179855412485`、`flip_rate_exact_mean = 0.1262`、`attention_drop_frac_exact = 0.0677`），且不存活到 eval（训练/评测路由错配 ≥ 0.066） | 无 Gumbel、无噪声、无 STE：train / eval **同一公式**（机制判据 5 在运行时逐位验证） |
| **D3** | 训练期每步 `torch.rand_like` 消耗全局 RNG（与其它臂不共享随机流）；`affinity_mlp` 建在 projection 与 gate **之间** ⇒ 同 seed 下共享参数初值与基线不同 | 构造在基线模块**之后**追加（共享参数与其 RNG 消耗与基线逐位一致）；门控前向**不抽任何随机数**（机制判据 6） |
| **D4** | 硬 STE + 噪声污染梯度 ⇒ 门控被自身动力学推向恒 attention 角（`presigmoid_abs_mean` 1.65 → 6.49、饱和占比 0 → 0.2716、`corrected_gate_mean` → 0.99957） | 梯度直接来自 sigmoid 软输出；末层中性初始化（bias 恒零、权重 ×0.1）⇒ 初始 gate ≈ 0.5、非饱和、第 1 步特征通路就有非零梯度 |
| **D2(b)** | 4 维门控特征（`[max_k C, min_k C, max−min, mean_k|C|]`）全部由**同一个** `H(x)` 派生（与 `W_attn` 同源）；K=2 时第 4 维 `mean|C| = (|max|+|min|)/2` 是前三维的函数 ⇒ 门控实际只看到 **2 个自由数** | 3 维特征：`[注意力归一化熵, cos(gen_rep, attention 融合源表征), cos(gen_rep, 均匀融合源表征)]`。第 2、3 维依赖 `gen_rep` / `spec_reps` —— 历史特征集**结构上看不到**的信息（测试用"扰动 `gen_rep`：历史特征逐位不变、本特征变化"锁定）；特征矩阵满列秩（与历史 4 维的精确恒等式对照） |
| **D5** | 门控不进 L2（`get_l2_reg` 只正则 tower） | **保留不改**（与历史臂同口径；门控 MLP 仅 41 个参数，L2 影响可忽略）。`provenance()["gate_in_l2"] = false` 留痕 |

> **审计事实（不得复述的错误结论）**：`env_embs` 的元素是一维 `[rep_dim]`（`get_infos` 逐个 index 取
> env 嵌入），`torch.stack(env_embs, dim=1)` 得 `[rep_dim, K]`（特征在前、任务在后）；因此
> **归一化维 `dim=0` 就是特征维**（逐任务列归一化），历史复现件已**撤回**早前那条"归一化维错误"的
> 结论——**不是缺陷**。本模块不使用历史 C 特征、不归一化 E，但仍保留同款布局守卫（fail-fast），
> 并把布局写进 `provenance()["env_layout"]` 自证。

---

## 2. 设计（`census_benchmark/affinity_gate_corrected.py`）

设 $x$ = `dnn_input`，$H(x)$ = `projection_network(x)`，$E$ = `stack(env_embs, dim=1)` 形状 `[rep_dim, K]`，
$s_k$ = `spec_reps[k]`，$T = 150$（基线 `NewTask.temperature`，不改），$K$ = 源任务数（CensusIncome：2）：

$$W_{\text{attn}} = \mathrm{softmax}(H E^{\top} / T) \in \mathbb R^{K},\qquad
\hat s_{\text{attn}} = \textstyle\sum_k W_{\text{attn},k}\, s_k,\qquad
\hat s_{\text{fw}} = \tfrac1K \textstyle\sum_k s_k$$

**门控特征（3 维，逐样本、可解释）**：

$$a(x) = \Big[\underbrace{\tfrac{1}{\log K}\,H_{\text{ent}}(W_{\text{attn}})}_{\text{注意力自信度}},\;
\underbrace{\cos\!\big(g, \hat s_{\text{attn}}\big)}_{\text{与 attention 融合的一致性}},\;
\underbrace{\cos\!\big(g, \hat s_{\text{fw}}\big)}_{\text{与均匀融合的一致性}}\Big],\qquad g = \texttt{gen\_rep}$$

- 第 1 维 ∈ `[0, 1]`：`1` = 路由完全均匀（最不自信）、`0` = 完全 one-hot（最自信）；
- 第 2、3 维 ∈ `[-1, 1]`：通用表征与两种候选融合的**方向一致性**。它们的差 `cos_attn − cos_fw` 正是
  "这次 attention 融合是否比均匀融合更贴近通用表征"的可解释判据——但它由 MLP 自己从两个输入里学，
  不写死（`gate_features` 只给 3 个数，不含冗余的第 4 维）。

**门控与融合（恰好一次 sigmoid，无约束 logit）**：

$$z(x) = W_2\,\mathrm{ReLU}(W_1 a(x) + b_1) + b_2,\qquad
\boxed{g(x) = \sigma\big(z(x)\big)},\qquad
W = g\,W_{\text{attn}} + (1-g)\,W_{\text{fw}},\qquad W_{\text{fw}} = \mathbf 1/K$$

$$\text{new\_spec\_rep} = \textstyle\sum_k W_k s_k,\quad
\text{env\_aware} = \text{env\_emb} \odot \text{new\_spec\_rep},\quad
\text{pred} = \text{tower}\big([\text{env\_aware},\, g]\cdot \text{gate\_network}(x)\big)$$

**初始化（中性、非饱和、非死启动）**：`Linear(3,8) → ReLU → Linear(8,1)`，末层 **bias 置零**、
权重按 `0.1` 缩放（默认 kaiming 上界 `1/√8` 的 0.1 倍）⇒ 初始 `|z|` 很小、`g ≈ 0.5`、不饱和；
权重**不置零**：zero-init 会让第 1 步特征通路的梯度恒为 0（死启动），缩放则让 4 个参数在第 1 步
都拿到非零梯度（测试锁定）。缩放用 `mul_` / `zero_`，**不消耗全局 RNG**。

**构造顺序（D3 的修正）**：`super().__init__()`（= 基线 `NewTask` 的全部模块与它们的 RNG 消耗）
→ 再 `build_gate_mlp()`。⇒ 同 seed 下共享参数与基线**逐位一致**；门控之后没有基线模块，
训练随机流（dropout / batch 顺序）与基线完全同源。

**逐样本性（I4）**：所有归约只发生在**特征轴**与**源轴（K）**上，从不跨 batch；由
"batch 组成无关"行为断言 + "门控路径源码不得含 `dim=0`"静态守卫双重锁定。

**与基线共享的部分（不动的语义）**：attention 路由本身（同一 $H$、$E$、$T$）、`projection_network`、
`gate_network`、`env_embedding_network`、`tower_network`、`get_l2_reg`（只正则 tower）、
`W` 之后的融合管线（逐行同式，含 `.squeeze()` 在批大小为 1 时的语义——协议路径批大小为 256，
批次尾部为 217/… ⇒ 不触发；该语义是基线既有行为，本分支不"顺手改"）。

---

## 3. 没改 / 不改的东西（防止"只改一半"）

| 项 | 处置 |
|---|---|
| `multitaskrec/model.py`、`config.py`、`CensusIncome_*.py` | **零改动**（测试用 `git diff master` 静态守卫） |
| `census_benchmark/protocol.py`、`census_benchmark/metrics.py` | **零改动**（测试用 `git diff HEAD` 静态守卫）；A/B 门禁语义不变 |
| 数据划分 / 三个种子 / stage1 产物 | 完全沿用协议；处理臂与基线臂**引用同一个 stage1 产物** |
| stage2 超参 | `epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`（协议 6.1；与基线臂完全相同） |
| 损失 / 优化器 / 选点 | `BCE(pred, y) + newtask.get_l2_reg()`、Adam(lr=1e-3)、val 选点、test 只评一次 |
| 历史 D5（门控不进 L2） | **保留**（见第 1 节表末行） |

---

## 4. 不变量（由 `census_benchmark/tests/test_affinity_gate_corrected.py` 强制）

| 编号 | 不变量 |
|---|---|
| I1 | 确定性：无 Gumbel / 无 dropout / 无随机分支；门控前向不消耗全局 RNG；train / eval 输出**逐位一致** |
| I2 | 单 sigmoid：`g == sigmoid(gate_logit)` 逐位成立；`routing_gate_mlp` 内无 `nn.Sigmoid`；末层是 `nn.Linear`（logit 无约束，可离开 `[0,1]`） |
| I3 | 中性初始化：末层 bias 恒零、权重 ≤ `0.1/√8` 且 > 0；初始 `g` 均值 ∈ `(0.49, 0.51)`、`std < 0.05`、`|logit| < 1`、饱和占比 0；第 1 步 4 个门控参数梯度全非零 |
| I4 | 逐样本：batch 组成无关（子批 / 重排一致到 `1e-6`）；门控路径源码无 `dim=0` 归约 |
| I5 | 基线公平：构造 = 基线构造 + 门控构造（RNG 流逐位对齐）；共享参数与基线逐位相同；`state_dict` 键集 = 基线 + `routing_gate_mlp.*`；`get_l2_reg` 不含门控（放大门控 1000× 后正则值逐位不变） |
| I6 | 特征：熵的端点（均匀 → 1、one-hot → 0）；扰动 `gen_rep` 时**历史 4 维特征逐位不变**而本特征变化（证明"增加信息"）；本特征矩阵满列秩、历史 4 维含精确恒等式 `c3 = c1 − c2` 与 `mean\|C\| = (\|max\|+\|min\|)/2` |
| I7 | blend：端点上逐位等于纯 attention / 纯均匀；中间值是两者的凸组合 |
| I8 | 诊断：读数可手工复算；`Σ_k share_k = 1`；pin 到端点时 `pred_delta_vs_*` **恒等于 0.0**；L1O 无定义样本走均匀回退并被计数；相关性工具对退化输入返回 `None`（不返回 nan）；诊断无副作用、不消耗全局 RNG、拒绝基线头 |
| I9 | 预注册：阈值/规则文本/主导关系与本文档**逐字同数**（文档测试读本文档并比对代码常量） |
| I10 | 接线：`run_id` 后缀 `-affcorr`；config / metrics 记录；基线臂逐位不受影响（哨兵 mock 下仍跑通、键集不变） |

> 测试用 CPU 极小夹具（`input_size=8`、`rep_dim=4`、64/16 样本、2 epoch），**不构成**任何性能证据。

---

## 5. 诊断：源贡献与对齐（val、训练后、零泄漏）

`source_contribution_probe`（只用 val；在训练与选点**全部结束**之后运行）：

1. **留一源（L1O）边际预测效应**：把源 $k$ 的权重置 0 并在其余源上重新归一得 $W^{(-k)}$，
   $\Delta_k(x) = p_W(x) - p_{W^{(-k)}}(x)$；报告 `source_reliance_mean_abs_delta[k]`（幅度）与
   `source_reliance_mean_delta[k]`（方向）。其余源权重和为 0 时该反事实无定义 ⇒ 回退为其余源上的均匀
   权重并计数（`l1o_undefined_count`；K=2 时回退值与极限一致，不影响数值）。
2. **逐源依赖份额** $\text{share}_k(x) = |\Delta_k| / \sum_j |\Delta_j|$（尺度无关，逐样本和为 1）。
3. **gate 移动了多少预测**：`pred_delta_vs_attn_mean_abs`、`pred_delta_vs_fw_mean_abs`
   （pin 到端点时必须**恒等于 0.0**，测试锁定）。
4. **对齐读数（预注册，见第 6 节）**：逐样本效用
   $u(x) = \mathrm{BCE}(p_{\text{fw}}, y) - \mathrm{BCE}(p_{\text{attn}}, y)$（> 0 ⇒ 该样本上 attention
   路由更优；概率夹逼到 `[1e-7, 1−1e-7]` 防 `log(0)`），报告 `gate_utility_spearman` / `_pearson` /
   `_auc`（`u = 0` 的样本在 AUC 里**排除**并单独计数）。效用读 val 标签，但只在训练与选点结束之后计算，
   **不进入任何梯度 / 选点 / test 路径**。

**逐源相关性的可识别性限制（必须如实声明，不得伪造）**：标量 `g` 决定的是"attention 路由 vs 均匀路由"
这**两个整体路由**的混合比例，**没有逐源偏好** ⇒ "gate 偏好源 $k$" 在 K=2 与标量 gate 下**不可识别**。
因此：

- `routing_weight_vs_reliance_*`（路由权重 $W_k$ 与依赖份额 $\text{share}_k$ 的相关）**只作 diagnostic**，
  且带**力学耦合**（$W_k$ 本身进入融合、直接放大该源的边际效应），不设阈值、不参与任何判定；
- gate 的**可识别**对齐读数只有 `gate_utility_*`（第 6 节的对齐阈值作用在它上面）。
  该字段与理由随读数一起落盘（`per_source_correlation_status = "diagnostic_only"` +
  `per_source_correlation_note`）。

---

## 6. 预注册判据（**只允许在看到结果之前修改**）

### 6.1 优先级（主导关系）

```
机制  >  效应  >  对齐
```

机制不过 ⇒ `effect.evaluated = false`、`effect.pass = null`、`status = STOP_MECHANISM`——
**不得**用该 run 的 AUC 去谈"是否达标"。效应不过 ⇒ `STOP_EFFECT`，同样**不扩展**到多 seed。
对齐不过（机制与效应都过）⇒ `PASS_ALIGNMENT_FAIL`：**效应结论成立**，但"门控的逐样本变化携带可用信息"
这一更强主张不成立——如实标注，不翻转效应结论。

### 6.2 机制（6 条全过才 `MECHANISM_OK`；读数取 **val、best_state 载入之后**）

| 规则（预注册原文，逐字） | 阈值来源 / 理由 |
|---|---|
| `gate_std >= 0.01` | 逐样本 gate 的**总体**标准差（ddof=0）。塌成常数（历史结局）会直接命中失败 |
| `0.05 <= gate_mean <= 0.95` | 未坍缩窗口：既不贴近纯均匀、也不贴近纯 attention |
| `routing_l1_mean >= 1e-4` | `mean_x ‖W − W_attn‖₁`：确实**偏离**纯 attention（历史实测 `8.43e-09`，即纯 attention） |
| `gate_grad_norm_min_step > 0` | **每个**训练步的门控全参数梯度 L2 范数最小值严格 > 0（历史判据是"最大值 == 0"，本分支更严：门控必须**每步**都在学）。空轨迹按 0 处理 ⇒ 命中失败 |
| `train_eval_identical` | 同输入下 train / eval 前向**逐位一致**（`torch.equal`，运行时逐 run 检查并落盘） |
| `no_global_rng_consumed` | 门控前向不消耗全局 RNG（CPU + CUDA 状态逐位比对） |

### 6.3 效应（**仅在 `MECHANISM_OK` 时才解释**）

```
auc_test_education >= 0.8521
```

- 参照：协议基线行 `20260929-1735-s20260929-m1685480945-short-904f8d0` 的
  `AUC-Test-Education = 0.8500685307175756`（**同一 stage1 产物、同一划分**）。
- 单次判定：本轮只跑**一个** model seed `1685480945`、`epochs=5`（协议 6.1 的短跑配置）；
  "多 seed 平均后才达标"不算通过（协议 6.3 第 3 条）。

### 6.4 对齐（第三层）

```
spearman(gate, utility) >= 0.05
```

- 阈值依据（**预注册，写在看到任何结果之前**）：n ≈ 5×10⁴ 时零假设下 Spearman 的标准误
  `1/√(n−1) ≈ 4.5e-3` ⇒ `0.05` ≈ **11 个标准误**，属于保守取值（纯噪声门控的经验相关应 ≈ 0）。
- 无定义（gate 或 utility 无方差）⇒ `ALIGNMENT_UNDEFINED`，**按保守处理**（不算通过）。
- 见第 5 节的限制声明：**不**对逐源相关性设阈值。

### 6.5 阈值纪律

1. 阈值只允许在**看到本次 run 结果之前**修改；每次修改单独记录理由（先进本文档，再改
   `census_benchmark/affinity_gate_corrected.py`）。
2. **禁止事后调参救判据**：不得为了过判据改学习率、批大小、初始化、温度、特征集、seed 或划分。
3. **止损**：先跑**一个** 5-epoch 的 CensusIncome seed；**机制或效应任一失败即止损**——
   不加 seed、不上全量、不改配置重跑（要改就新开分支并重新预注册）。
4. 运行期间发现实现缺陷 ⇒ 允许修复后重跑，但必须同时保留失败 run 的产物与说明。
5. 失败结果一律留痕（协议 7.3：不得只报告成功的 run）。

---

## 7. 预判与已知风险（**写在看到任何结果之前**；不是结果）

以下四条是设计推导，先落在此处，供结果回填时逐条对照（无论对错都如实记录）：

1. **机制判据 3（`routing_l1_mean >= 1e-4`）最容易过，判据 1（`gate_std >= 0.01`）最可能卡住。**
   K=2 时 `‖W − W_attn‖₁ = (1−g)·|2·W_attn,1 − 1|`；历史实测 `|2·W_attn,1 − 1|` 的量级约 0.13，
   故只要 `1 − g ≳ 8e-4` 即可越过 `1e-4`。而 `gate_std >= 0.01` 要求门控的逐样本差异达到
   `|Δz| ≈ 0.04` 量级——**若门控学不到任何有用的逐样本信号，它会停在 `g ≈ 0.5` 附近，标准差 < 0.01
   ⇒ `MECHANISM_FAIL`**。这是**合法结局**（正是本实验要检出的东西），不得读成"实现有 bug"。
2. **已知风险（不修，只声明）：温度 `T = 150` 让 attention 路由本身接近均匀。**
   `W_attn ≈ 1/K ± O(0.13/K)` ⇒ 熵特征 `∈ [0.98, 1.0]`（变化很小）、两个一致性特征高度相关。
   在 float32、无 batch 统计（逐样本性要求禁止标准化）的前提下，这 3 个特征**本身**的变化幅度小，
   门控要靠自身权重把它们放大（`|z|` 的一阶灵敏度可以被学大）。若机制失败，**这是首要候选解释**；
   如实报告，**不得**为了过判据而事后改特征集或引入 batch 统计。
3. **效应阈值（`>= 0.8521`）可能偏难。** attention ≈ 均匀 ⇒ `W_attn ≈ W_fw` ⇒ `W` 在 `g ∈ [0,1]`
   之间移动时对预测的影响本身有限（测量口径：`source_contribution.pred_delta_vs_attn_mean_abs` /
   `pred_delta_vs_fw_mean_abs`）。处理臂与基线臂的差异因此可能小于单 seed 的噪声量级（历史复现件上
   单 seed Δ 的量级 ~3.5e-3）。该风险**不改变阈值**（阈值纪律：不得事后调整）。
4. **对齐判据（第三层）最可能给出 `PASS_ALIGNMENT_FAIL`。** 训练目标里**没有**任何"让 `g` 与效用对齐"
   的项，`g` 只是被 BCE 间接塑形；因此 `spearman(g, u) >= 0.05` 是一个**额外的**、更严格的主张。
   它失败时结论是"机制能跑、效应达标，但逐样本变化尚未被证明携带可用信息"——如实标注即可。

---

## 8. 接线与运行

| 位置 | 改动 |
|---|---|
| `census_benchmark/affinity_gate_corrected.py` | **新增**（隔离模块）：门控头 + 读数累加器 + 训练轨迹 + val 诊断 + 源贡献探针 + 预注册判据 |
| `census_benchmark/tests/test_affinity_gate_corrected.py` | **新增**：I1–I10 的不变量 |
| `run_census_benchmark.run_stage2` | 新增 `variant="baseline"` 参数；新任务头改用 `AGC.build_newtask(variant, …)`；处理臂逐 epoch 记轨迹、best_state 后做诊断与探针、按预注册判据落盘 |
| `run_id` | 处理臂追加后缀 `-affcorr`（协议 `make_run_id` 未改） |
| `config.json` | 两臂记 `variant`；处理臂另记 `fusion_mode` / `gate_feature_dim` / `gate_feature_names` / `gate_hidden_units` / `gate_init_weight_scale` |
| `metrics.json` | 两臂记 `variant` 且 `mechanism` 键集不变；处理臂另记 `affinity_corrected_arm`（不并入 `judge()` 的 `overall_pass`） |
| CLI | `stage2 --variant {baseline,affinity_corrected}`（默认 `baseline`）与等价别名 `--affinity-corrected` |
| `protocol.py` / `metrics.py` / `model.py` / `config.py` | **零改动**（测试静态守卫锁定） |

**基线臂不受影响的三条证据**（测试锁定）：① `build_newtask("baseline", …)` 返回的**就是** `NewTask`
（`type(...) is NewTask`），同 seed 下参数与直接构造逐位相同；② 基线 run 的 `run_id` 无 `-affcorr`
后缀、payload 无 `affinity_corrected_arm`、config 无处理臂专属字段，且在"一旦构造修正头就报错"的哨兵
mock 下仍必须跑通；③ 两臂的 `mechanism` 键集完全一致。唯一新增的是 `config.json` / `metrics.json` 里
恒为 `"baseline"` 的 `variant` 标签与 `run_id` 后缀约定（与其它 `exp/*` 分支同一约定）。

### 运行命令（单 seed 短跑；与基线同一 stage1 产物）

```powershell
# 0) 测试（全部不变量；CPU 极小夹具）
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v

# 1) 处理臂（本实验唯一要跑的东西）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --variant affinity_corrected --tag short --gpu 0
# 等价写法：--affinity-corrected

# 2) 基线对照（可选；同一 stage1、同一划分，预期复现 SUMMARY.md 的 0.850069）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --tag short --gpu 0
```

产物：`artifacts/census_stage2/runs/<run_id>/`（`metrics.json:affinity_corrected_arm` 含判据、诊断、
源贡献读数与逐 epoch 轨迹），`SUMMARY.md` 追加一行。A3 复跑按协议按需触发，不阻塞本轮。

**审计前置条件（必须先做，否则 `commit` 字段不含本实验代码——历史复现件在这条上留过缺口）**：
开跑前先把本批改动**提交**到 `exp/stage2-affinity-gate-corrected`，再执行上面的命令。
本次交付按用户指示**不提交、不推送、不合并**，因此执行者需自行完成这一步。

---

## 9. 非新颖性声明（不主张任何新颖性）

- 门控融合 / **soft gating** / **MoE**（mixture-of-experts）路由 / 条件计算（conditional computation）
  都是**已有大类**；"用一个小 MLP 从少量统计量产生逐样本混合系数、再在两个路由之间插值"是其中的
  标准构件。本分支**不主张任何新颖性**，也不提出新方法。
- 逐源贡献的 L1O / 边际效应、以及"门控分数 vs 效用"的相关性诊断同样是标准做法（可解释性与
  路由分析里的常规工具）。
- 本分支的产出只有四样：**(a)** 一份把历史缺陷逐条修正的**确定性**门控实现；**(b)** 一套**预注册**
  的机制 / 效应 / 对齐判据；**(c)** 一套零泄漏的源贡献与对齐诊断（含"逐源相关性不可识别"的显式声明）；
  **(d)** 一次单 seed 短跑的**可证伪**结论（等执行者运行后回填）。
- 历史 Affinity Gate **从未超越** prompt（CensusIncome −0.0028；AliCCP −0.0030），本分支也不承诺提升；
  它检验的是"缺陷修完之后机制是否还成立"。

---

## 10. 结果处置

- **三种结局都按判据落盘**（`PASS` / `PASS_ALIGNMENT_FAIL` / `STOP_MECHANISM` / `STOP_EFFECT`），
  失败**不得**重跑挑好的、不得换阈值、不得换 seed 或划分。
- **不合并 `master`**：本分支是 `exp/*`，只引用评测协议、不改协议文件；是否合并由用户在看过结果后决定。
- 权重 / JSON / 日志沿用协议 9.3：`*.pt` / `*.json` / `*.log` 不入库（`.gitignore` 天然拦截），
  `SUMMARY.md` 行与本文档入库。
- 本文档在结果落地后**只允许新增"结果"小节**（含逐条对照预判），第 6 节的阈值与规则一字不改。

---

## 11. 结果（单 seed 短跑实跑回填）

> 本节是第 10 节允许的**唯一**新增内容：只回填读数与逐条对照。第 6 节的**阈值 / 规则 / 优先级一字未改**，
> 第 1–10 节的其他文字也未改。

**一句话结论**：**工程缺陷全部修好了，失败的是被修正后的机制本身。** 历史缺陷（D1 / D2(b) / D3 / D4 /
D1(c)+D6）逐条修正后，评测路径上**确实有路由**、门控**每一步都在学**、train/eval **逐位一致**、
**不消耗全局 RNG**（第 11.3 节五条正向证据）——但学到的门控在训练中**持续向 attention 漂移**、
**逐样本变化幅度不足**（`gate_std = 0.00713 < 0.01`，机制 6 条中**唯一**未过项），并且与实测的逐样本效用
**负相关**（Spearman `−0.2499`）。按 6.1 的主导关系，本 run = **`STOP_MECHANISM`**；效应的 AUC 阈值**不评估**。

### 11.1 运行身份与主读数

| 项 | 值 |
|---|---|
| `run_id` | `20260930-1912-s20260929-m1685480945-short-87afe03-affcorr` |
| `stage1_id` | `s1-096f8f16-m1685480945-e2-cb2094b3`（与协议基线行**同一** stage1 产物、同一划分） |
| 臂 | `variant = affinity_corrected`；`fusion_mode = affinity_gate_corrected`；`run_id` 带 `-affcorr` 后缀 |
| 配置 | `epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`（协议 6.1；与基线臂**逐项相同**） |
| 选点 | best epoch **5**，`AUC-Val-Education = 0.8521534636988015` |
| 主报告值 | `AUC-Test-Education = 0.8491303844406325` |
| 基线对照 | `0.8500685307175756`（`20260929-1735-…-904f8d0`）⇒ Δ = **−0.0009381462769431** |
| 臂判定 | **`status = STOP_MECHANISM`**；`stop_reason = mechanism_fail`；`dominance = mechanism > effect > alignment` |

读数来源：`artifacts/census_stage2/runs/<run_id>/metrics.json` 的 `affinity_corrected_arm.*`
（`*.json` / `*.pt` 按协议 9.3 不入库；`SUMMARY.md` 行与本文件是入库记录）。

### 11.2 机制 6 条逐条（val、`best_state` 载入之后；n = 49881 / 195 batches）

| 判据（预注册原文） | 阈值 | 实测 | 判定 |
|---|---|---|---|
| `gate_std >= 0.01` | 0.01 | `0.007132461220191373` | **FAIL** |
| `0.05 <= gate_mean <= 0.95` | `[0.05, 0.95]` | `0.9241543347299588` | PASS |
| `routing_l1_mean >= 1e-4` | `1e-4` | `0.03572193051547912` | PASS |
| `gate_grad_norm_min_step > 0` | 0 | `2.405436024949036e-06`（训练 **3900/3900** 步非零） | PASS |
| `train_eval_identical` | true | true | PASS |
| `no_global_rng_consumed` | true | true | PASS |

⇒ `MECHANISM_FAIL`，`failed_rules = ["gate_std >= 0.01"]`，`status = STOP_MECHANISM`。
**失败项只有一条，且它是"逐样本差异"判据**——不是任何一条工程/确定性判据。

### 11.3 工程缺陷：确实修好了（本次 run 的正向证据）

| 缺陷 | 修正 | 本次 run 的正向证据 |
|---|---|---|
| **D1** 双 sigmoid + eval 硬门控 ⇒ 评测路径上恒为纯 attention | 单 sigmoid、无硬判据 | 评测路径上**有真实路由**：`routing_l1_mean = 3.572193051547912e-2`（历史实测 `8.43e-09`，即纯 attention），`routing_l1_max = 6.666263937950134e-2`；logit 无约束且未饱和（`gate_logit_abs_max = 2.768991708755493`、`gate_logit_saturation_frac = 0.0`，饱和阈值 `|logit| ≥ 8`） |
| **D1(c)/D6** train/eval 路由错配（Gumbel 噪声不存活到 eval） | 无噪声、无硬切换，train/eval 同一公式 | `train_eval_identical = true`（`torch.equal`，批大小 256 上逐位） |
| **D3** 前向消耗全局 RNG + 构造顺序位移 | 构造在基线模块之后追加；前向不抽随机数 | `no_global_rng_consumed = true`（CPU + CUDA 逐位比对，`cuda_rng_checked = true`）；探针跑完 backbone 仍冻结（`backbone_sha256_after_probes == backbone_sha256_before`，`grads_all_none_after_probes = true`） |
| **D4** 硬 STE + 噪声污染梯度 ⇒ 门控被推死 | 梯度直接来自 sigmoid 软输出 | 门控**每一步都在学**：`grad_nonzero_steps = 3900` / `grad_zero_steps = 0`，`grad_norm_min = 2.405436024949036e-06`、`grad_norm_mean = 1.6205967465632255e-03`、`grad_norm_max = 8.024501784062872e-03`；第 1 个 epoch 的 `gate_mean = 0.6517452072566197`、`gate_std = 0.08130241086927656`（中性初始化、非死启动、非饱和） |
| **D2(b)** 门控特征与 `W_attn` 同源（K=2 时只有 2 个自由数） | 3 维特征走 `gen_rep` 通路 | 特征名 `attn_entropy_norm` / `cos_gen_attn` / `cos_gen_fw` 与 I6 测试锁定一致（依赖 `gen_rep` / `spec_reps`，历史特征集结构上看不到） |

**⇒ 机制失败不能归因于任何一条历史缺陷。** 修正是有效的：它把"评测路径上根本没有路由"变成了
"评测路径上有路由，但**学到的**路由向 attention 漂移、且逐样本差异不足"——这是一个**机制层面的**结论，
不是实现层面的。

### 11.4 漂移：门控持续向 attention 靠（collapse toward attention）

训练轨迹（train 集，每 epoch 780 步 / 199523 样本；末行是评测读数）：

| epoch | `gate_mean` | `gate_std` | `routing_l1_mean` |
|---|---|---|---|
| 1 | 0.6517452072566197 | 0.08130241086927656 | 0.1487484824435898 |
| 2 | 0.7923098999472196 | 0.02203894940662219 | 0.10723951317292901 |
| 3 | 0.8446492464340456 | 0.01529098399587477 | 0.07602987853791249 |
| 4 | 0.8830561166022379 | 0.011757194616227349 | 0.056968075049668956 |
| 5 | 0.9117554509068497 | 0.010521397433086445 | 0.043139282088236516 |
| **eval（best = 5）** | **0.9241543347299588** | **0.007132461220191373** | **0.03572193051547912** |

- 三个量**单调**：`gate_mean` 0.6517 → 0.9242（越来越像纯 attention）、`gate_std` 0.0813 → 0.0071
  （逐样本差异越来越小）、`routing_l1_mean` 0.1487 → 0.0357（越来越贴近 `W_attn`）。
- 预测层面同样如此：`pred_delta_vs_attn_mean_abs = 0.0019004457149378248` 比
  `pred_delta_vs_fw_mean_abs = 0.03075532555399655` **小约 16 倍** ⇒ 路由权重也已接近纯 attention
  （`w_col_mean = [0.6143953944507964, 0.385604605579077]` vs `w_attn_col_mean = [0.6241694325595943, 0.37583056736631965]`）。
- 但它**不是**历史的"恒等 attention"：起点在 0.5 附近、每步有非零梯度、train/eval 一致、logit 未饱和
  （`saturation_frac = 0.0`）。行为是**缓慢漂移**；在协议固定的 5 个 epoch 内**看不到**停下来的迹象
  （**不外推**：没有更长跑的读数，不得声称它会稳定到某个值）。

### 11.5 逐样本变化不足（insufficient sample variation）

评测集上门控几乎是个常数、且落在近 attention 的一侧：`gate ∈ [0.9059646129608154, 0.9409770369529724]`
（带宽 0.035），`gate_std = 0.00713` —— 这**就是**唯一未过的判据。两点佐证：

- **门控的输入在样本间几乎不分离**：`attn_entropy_mean = 0.796502818050315`（区间 `[0.5934737920761108, 1.0]`，
  `1` = 路由完全均匀 ⇒ attention 路由本身接近均匀）；`cos_gen_attn_mean = 0.3981396480452261` vs
  `cos_gen_fw_mean = 0.4298167391544724`，差 `cos_gen_diff_mean = −0.03167709110924624`。
  即两个候选融合与 `gen_rep` 的一致性只差约 0.03，且**平均而言均匀融合更一致**。
- **输入几乎不动，输出却动了 0.27**：跨 5 个 epoch，`attn_entropy_mean` 仅 0.8313 → 0.7866、
  `cos_gen_attn_mean` 稳定在 ≈ 0.394–0.403，而 `gate_mean` 从 0.6517 走到 0.9118 ⇒ 漂移由**门控自身的
  权重**（把近常数的输入放大）驱动，而**不是**由逐样本信号驱动。这与 `gate_std` 同步收缩是同一件事的两面。

与第 7 节第 2 条预判（`T = 150` 让 attention 近均匀 ⇒ 特征变化幅度小；若机制失败，这是首要候选解释）
**方向一致**。但本 run **没有对照臂**（未改温度、未换特征集、未加 batch 统计），因此"温度是原因"只是
**一致的解释**，**不是**被证明的结论——不得写成因果。

### 11.6 对齐：负相关（记录，但不参与判定）

- `gate_utility_spearman = −0.24991698792103564`（阈值 `>= 0.05`；n = 49881 ⇒ 零假设下 SE ≈ 4.5e-3，
  该读数约 **−56 个标准误**，不是噪声）；`gate_utility_pearson = −0.06628754938220026`；
  `n_utility_zero = 0`、`gate_utility_auc = null`（无 `u = 0` 样本需要排除）。
- **方向为负**：`gate` 越大（越偏 attention）的样本上，attention 相对均匀的逐样本效用
  `u(x) = BCE(p_fw, y) − BCE(p_attn, y)` **越差**——与预注册期望的方向**相反**。
- 与特征读数自洽：平均而言 `cos_gen_attn < cos_gen_fw`（差 −0.0317，即均匀融合与 `gen_rep` 更一致），
  而门控却漂到 `g = 0.924`（主要走 attention）。
- **判定口径必须说清**：按 6.1，机制已失败 ⇒ 对齐**不进入主导链**，本 run 的 `status` 仍是 `STOP_MECHANISM`；
  落盘的 `ALIGNMENT_FAIL` 只是**读数状态**。**不得**据此宣称"效应结论成立但对齐不成立"——那是
  `PASS_ALIGNMENT_FAIL` 的语义，本 run 没有走到那一层。

### 11.7 效应：按主导关系不评估

- `effect.evaluated = false`、`effect.pass = null`；实测 `AUC-Test-Education = 0.8491303844406325`
  低于预注册阈值 `0.8521`，也低于基线 `0.8500685307175756`（Δ = −0.0009381462769431）。
- 按 6.1，机制不过就**不得**用该 run 的 AUC 去谈"是否达标"——上面这行只是为了留痕（7.3：不得省略失败的 run）。
  单 seed 短跑本来也不构成效应证据（6.3 第 3 条）。

### 11.8 公平性门禁：全部 PASS，除**继承的** B3

- A1 / A2 / A4 / A5 / B1 / B2 / B4 **全部 PASS**（B1：`auc_val_income = 0.9373971773636002`、
  `auc_val_marital = 0.9912246018108644`、`auc_test_education = 0.8491303844406325`，下限 0.6；
  B2：`gap = 0.003023079258169026 ≤ 0.03`）。
- **B3 的 FAIL 是继承的，与本分支无关**：本 run 的 B3 明细 `gate_mean = [0.952401451979339,
  0.8543901456350915]` 与基线 run `20260929-1735-…-904f8d0` 的 B3 明细**逐位相同**。B3 读的是**冻结
  backbone** 的 `fused_preds` 分支门（协议 M3 的 `gate_w`），两臂共享同一 stage1 产物 ⇒ 该失败是共享
  backbone 的属性；`overall_pass = false` / `failures = ["B3"]` 与基线行完全一致，**未因本分支而变化**。
- ⚠ **同名不同物（读表防混淆）**：`gate_report.json` 与 `metrics.json:mechanism.gate_mean` 里的
  `gate_mean` 是 **M3 的 backbone 分支门**；`affinity_corrected_arm.diagnostics.gate_mean` 是**本模块的
  逐样本软门控**。两者含义完全不同，不得混用。
- A3 仍为 `on_demand`，未触发（协议 8：不阻塞首轮）。

### 11.9 与第 7 节预判的逐条对照（无论对错都记录）

| # | 第 7 节的预判 | 实际结果 | 判定 |
|---|---|---|---|
| 1 | 判据 3（`routing_l1_mean`）最易过；判据 1（`gate_std`）最可能卡住 | `routing_l1_mean` 以 **357×** 余量通过（`3.57e-2` vs `1e-4`）；`gate_std` 是**唯一**失败项（`0.00713` vs `0.01`，差 **28.7%**） | **命中** |
| 2 | `T = 150` ⇒ attention 近均匀、3 个特征自身变化幅度小；若机制失败，这是首要候选解释 | 读数与之一致：熵 ∈ `[0.59, 1.0]`（近均匀）、两个 cos 只差 −0.0317；且输入均值跨 epoch 几乎不动而输出动了 0.27 | **一致；但无对照臂 ⇒ 未被证明（不得写成因果）** |
| 3 | 效应阈值（`>= 0.8521`）可能偏难：`W_attn ≈ W_fw` ⇒ 门控对预测的影响有限 | `pred_delta_vs_attn_mean_abs = 0.0019`、`pred_delta_vs_fw_mean_abs = 0.0308`（差约 16×）；实测 AUC 低于基线 9.4e-4 | **方向一致；但因机制先失败而"不评估"（阈值未改）** |
| 4 | 对齐最可能不过，且预期状态串是 `PASS_ALIGNMENT_FAIL` | 对齐确实没过，**且方向为负**（−0.2499，比"不相关"更强）；但本 run 因机制先失败而**没走到对齐层** | **"不过"命中；预测的"状态串"不适用** |

### 11.10 纪律与边界（本次**没有**做的事）

- **阈值 / 规则 / 优先级：一字未改**（第 6 节保持原样；`prereg` 块与本文档同数由 I9 测试锁定）。
- **未**扩展多 seed、**未**跑全量、**未**换配置重跑、**未**调参救判据（6.5 第 2、3 条）。单 run、
  单 model seed `1685480945`、5 epoch 短跑。
- **未**合并 `master`（第 9 节 / 第 10 节）；协议文件（`protocol.py` / `metrics.py` / `model.py` /
  `config.py`）**零改动**。
- 失败 run 的读数按 7.3 全量留痕（本节 + `SUMMARY.md` 行）。
- 聚焦测试：`census_benchmark/tests/test_affinity_gate_corrected.py` 的 **79 个用例（10 个测试类）
  全部通过**（运行者在本 run 前后执行；CPU 极小夹具，不构成性能证据）。**这是 run 当时的口径**：
  该文件随后被下面的测试侧清理改动（现定义 **81 个用例、仍 10 个测试类**），新守卫的通过情况**不属于**
  本 run 的证据。
- **（结果回填之后追加）测试侧清理**：旧守卫 `test_doc_exists_and_is_marked_not_yet_run` 只做字面匹配——
  本节回填后它匹配到的是"第 1–10 节写于运行之前"的阶段说明，**已成为空转断言**（原 ⚠ 所指）。该用例
  **已退役**，换成**阶段感知**守卫：`split_doc_phases()` 按 `## 11. ` 把文档切成（冻结的预注册段 / 结果段），
  标题缺失即硬失败；① `test_doc_prereg_records_it_was_frozen_before_the_run`——"尚未运行"须**被限定**在
  "第 1–10 节"那一行，且同行含"冻结""一字未改"；②
  `test_doc_header_status_is_ran_with_stop_mechanism_result`——**状态**行须为"已运行 + `STOP_MECHANISM`"、
  不得再含"尚未运行"；③ `test_doc_result_section_records_exact_run_id_and_result_markers`——run_id 与结果
  读数**只许**出现在结果段，**出现在冻结段即失败**（防预注册被事后回写）。该清理**不改第 1–10 节的预注册
  与设计，也不改本 run 的判据、读数与结论（`STOP_MECHANISM` 依旧）**。
- **非新颖性（第 9 节）不变**：本结果**不**支持任何方法主张。它是对一条**已被否证**的历史机制、
  在"缺陷逐条修正之后"的**再否证**（公平协议、单 seed 短跑）。

### 11.11 审计留痕：本 run 的代码身份（**重要**）

- `config.json` / `metrics.json` / `SUMMARY.md` 里记的 `commit` 是 **`87afe03`**（= `git rev-parse HEAD`），
  但产生本 run 的代码——`census_benchmark/affinity_gate_corrected.py`（未跟踪）、`run_census_benchmark.py`
  的改动（未提交）、`census_benchmark/tests/test_affinity_gate_corrected.py`（未跟踪）——**在运行时尚未提交**。
- ⇒ **`87afe03` 不包含本实验代码，本 run 无法从该 commit 复现**。这正是第 8 节"审计前置条件"警告过的
  缺口（历史复现件在此留过同样的缺口），本次**再次发生**。
- 处置：本分支按用户指示**不提交、不推送、不合并**；**后续任何复跑（含门禁 A3）之前必须先提交本批改动**，
  并把新 commit 记入 `SUMMARY.md`。上表读数与"`commit = 87afe03`"字段的对应关系以本节说明为准。

---

## 附录 A：与历史实现（`exp/stage2-affinity-gate-repro`）的差异清单

| 维度 | 历史（忠实复现） | 本分支（修正版） |
|---|---|---|
| 门控 MLP | `Linear(4,16) → ReLU → Linear(16,1) → **Sigmoid**` | `Linear(3,8) → ReLU → Linear(8,1)`（无末端激活） |
| 门控输出 | `g_logit = σ(z)` 被当作 logit 再 σ 一次（双 sigmoid） | `g = σ(z)`，**恰好一次** |
| 硬判据 / STE | `g_hard = (g_soft > 0.5)` + straight-through | 无 |
| 随机性 | 训练期每步 Gumbel 噪声（全局 RNG） | 无（确定性；train/eval 同一公式） |
| 特征 | 4 维，全部由 `H(x)` 派生（与 `W_attn` 同源，K=2 时 ≤ 2 个自由数） | 3 维，含 `gen_rep` 与两种融合的一致性（非同一来源、满列秩） |
| 构造顺序 | `affinity_mlp` 建在 projection 与 gate 之间（共享参数初值 ≠ 基线） | 基线模块**之后**追加（共享参数逐位 = 基线） |
| blend | `W = g·W_attn + (1−g)·W_fw` | 同式（保留） |
| 末端融合管线 / attention 路由 / `gate_network` / tower / L2 | 与基线同 | 与基线同（不改） |
| 判据 | 退化 STOP（`eval_attn_share ≥ 0.999` 等）+ 保真 STOP + 效应 | 机制 6 条 + 效应 + 对齐（三层主导） |
| 诊断 | 噪声/信号探针、梯度探针、退化读数 | val 门控读数、L1O 源贡献、gate↔效用对齐、逐源相关（diagnostic） |
