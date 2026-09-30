# exp/stage2-cgr-instance：修正后的逐样本 CGR 工程消融（实验说明）

- **状态**：**实跑已完成**：`MECHANISM_OK` + `EFFECT_NOT_CONFIRMED`（§8.0 的结局 B）。
  本文件先于结果写成（预注册）；第 5 节阈值与规则原文**一字未改**，第 8 节已按实跑回填。
- **日期**：2026-09-30
- **分支**：`exp/stage2-cgr-instance`（从 `infra/fair-stage2-benchmark` @ `87afe03` 独立拉出）
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。
  划分、三个种子、真冻结三件套、A/B 门禁、SUMMARY 一律沿用；`census_benchmark/protocol.py`、
  `census_benchmark/metrics.py`、`multitaskrec/model.py`、`config.py` **零改动**（测试静态守卫锁定）。
- **数据集**：只跑 CensusIncome（协议 2.3）。
- **定位**：这不是新方法，是**修正后的工程消融**（第 9 节"非新颖性"）。

---

## 1. 要回答的问题

`exp/stage2-cgr-repro` 把 `archive/exploration` 的历史 CGR 逐项照搬后重跑，判定
**`CONFIRMED_DEGENERATE`**（run `20260930-1355-s20260929-m1685480945-short-87afe03-cgr`），
三条预注册退化判据**全部**命中：

| 判据 | 实测 | 根因 |
|---|---|---|
| `unique_frac <= 0.01` | `0.003909850230414747`（= `1/批大小` 的批加权平均，逐位吻合） | D1 批级门控 |
| `weight_l1 < 1e-6` | `0.0` | D3 硬饱和 ⇒ `W ≡ W_attn` |
| `grad_norm == 0` | `0.0`（3900 步无一例外） | D3 ⇒ `\|z\| ≈ 130` ⇒ σ′ 在 float32 下溢为 0 |

**注意 D1 是架构性的**：只要门控输入仍是批级统计量，`unique_frac` 就恒等于 `1/批大小`，
与训练多久、什么 seed、什么 lr 全无关；D3 则是数值性的（出生即在饱和区，梯度恒 0 ⇒ 再也出不来）。

本分支提出的问题因此是：**把这两条根因（以及审计出的其余缺陷）都修好之后，
"逐样本置信门控混合 attention 与均匀权重"这件事还剩下多少可测效应？**

本分支**只做一次**单 seed 短跑（协议 6.1、6.3），先过机制门、再看效应门，不搜参、不换结构。

---

## 2. 缺陷 → 修复映射（审计来源：`exp/stage2-cgr-repro` 的复核结论）

| 编号 | 历史缺陷 | 本分支的修复 | 固定在哪 |
|---|---|---|---|
| **D1** | 门控输入 `z` 只含批级/环境级统计量（`e_new` + env 矩 + 批内 gen_rep 矩）⇒ `g` 在批内逐位相同、换批即换 g | 门控输入改为**逐样本**：`[H_out(x), p_max, H_norm, agree]`；无任何跨样本归约 | `gate_features`；`test_same_sample_keeps_the_same_g_*`（B=1 与换批组成两种扰动）+ 源码静态守卫 |
| **D2** | `confidence_mlp` 建在 `projection_network` 与 `gate_network` **之间**，多抽两组 `nn.Linear` 的 RNG ⇒ 同 seed 下 `gate_network`/`tower_network` 初值与基线**不同** | `super().__init__()` **先建全部共享模块**（消耗与基线逐位相同的 RNG），门控参数在其后追建 | `CGRInstanceNewTask.__init__`；`test_shared_parameters_are_bit_identical_to_baseline_same_seed`、`test_extra_parameters_are_appended_after_the_shared_ones` |
| **D3** | 常量初始化（`0.01/0.01`、`0.1/2.0`）⇒ 32 个隐层单元逐位相同（对称永不破缺）+ 起点硬饱和（`\|z\| ≈ 130`，`g_raw ≡ 1.0`） | 隐层用 `nn.Linear` **标准初始化**（行互不相同）；门控输出层**零初始化** ⇒ `z ≡ 0` ⇒ `g ≡ 0.5`（精确），`σ′(0) = 0.25`，不饱和 | `__init__`；`test_hidden_rows_are_not_tied`、`test_presigmoid_is_exactly_zero_at_init`、`test_g_is_exactly_one_half_at_init_with_full_sigmoid_slope` |
| **D4** | warmup：`alpha = min(1, epoch/5)` 由 runner 逐 epoch 写入且不是 buffer ⇒ 载入 `best_state` 后被评测的是**凸混合**而非"纯 CGR"；`alpha` 停在末轮取值 | **无 warmup**（第 3.4 节给理由） | 模块内无任何 alpha 机制（源码静态守卫 `test_no_warmup_machinery_in_source`） |
| **D5** | 梯度信号恒 0（D3 的直接后果，实测 3900 步全 0） | 零初始化 ⇒ `∂L/∂W_out ≠ 0`（初始步即有梯度）；一步之后隐层也恢复梯度 | `TestGateGradient` 三个测试 |
| **D6** | 门控输入用 `gen_rep.var(dim=0)`（ddof=1），批大小 = 1 时自由度 ≤ 0 ⇒ `g = NaN` | 逐样本量不存在批内自由度；forward 用 `squeeze(-1)` 而非 `squeeze()` | `TestBatchSizeOne`；`test_baseline_head_breaks_at_batch_size_one` 作为对照留痕 |

> **为什么这次可以修 D2**：`exp/stage2-cgr-repro` 必须原样保留 D2，因为它的目的是复核**历史那一版**；
> 把 D2 修掉就变成一次新实验。本分支正是那次新实验——所以 D2 不但可以修，而且必须修，
> 否则"门控 vs 无门控"的对比里会混着 `gate_network`/`tower_network` 的初始化位移。

---

## 3. 设计

### 3.1 公式

设 $x$ = `dnn_input`（`[B, 123]`），$H$ = `projection_network`（末端 LayerNorm），
$E = [\mathbf e_0, \dots, \mathbf e_{K-1}]$（`nn.Embedding` 的环境向量，`[128, K]`），$K = 2$，$T = 150$：

$$H_{\text{out}}(x) = H(x)\in\mathbb R^{128},\qquad
W_{\text{attn}}(x)=\mathrm{softmax}\big(H_{\text{out}}(x)\,E\,/\,T\big)\in\mathbb R^{K}$$

**逐样本门控特征**（全部只依赖该样本自己）：

| 特征 | 定义 | 取值域 | 含义 |
|---|---|---|---|
| $H_{\text{out}}$ | 128 维 | LayerNorm ⇒ 逐样本零均值单位方差 | attention 的投影本身（最丰富的逐样本信号） |
| $p_{\max}$ | $\max_k W_{\text{attn},k}$ | $[1/K, 1]$ | attention 置信度（尖峰程度） |
| $H_{\text{norm}}$ | $-\sum_k W_{\text{attn},k}\log W_{\text{attn},k} / \log K$ | $[0, 1]$ | 归一化熵（0 = 尖峰，1 = 均匀） |
| $\text{agree}$ | $\cos\big(\sum_k W_{\text{attn},k}\,\text{spec\_rep}_k,\ \text{gen\_rep}\big)$ | $[-1, 1]$ | attention 融合后的源表征与通用表征的**一致性** |

$$\text{gate\_in}(x)=\big[\,H_{\text{out}};\ p_{\max};\ H_{\text{norm}};\ \text{agree}\,\big]\in\mathbb R^{131}$$

$$z(x)=\mathrm{MLP}_{131\to 32\to 1}\big(\text{gate\_in}(x)\big),\qquad g(x)=\sigma\big(z(x)\big)\in(0,1)$$

$$W(x)=g(x)\,W_{\text{attn}}(x)+\big(1-g(x)\big)\,W_{\text{fw}},\qquad W_{\text{fw}}=\mathbf 1/K$$

$W$ 之后的管线与基线 `NewTask.forward` **逐行相同**：
$\text{new\_spec\_rep}=\sum_k W_k\,\text{spec\_rep}_k$ → $\text{env\_aware}=s\odot\text{new\_spec\_rep}$
→ $\text{fused}=[\text{env\_aware},\text{gen\_rep}]\cdot\text{gate\_network}(x)$ → `tower_network`。

**为什么用这四类特征**（可解释性）：$H_{\text{out}}$ 是 attention 权重的来源，给门控最大的表达力；
$p_{\max}$ 与 $H_{\text{norm}}$ 是"attention 有多确定"的两个互补标量（同一个 softmax 的峰值与熵）；
`agree` 是"两个分支是否一致"的单一有界标量。三者都不是批级量，因此 D1 在结构上不可能复发。

**为什么只有 `agree` 而没有单独的 L2 距离**：`cos` 与"归一化距离"在固定范数下互为单调函数，
而 $H_{\text{out}}$ 已由 LayerNorm 定标；再塞一个尺度敏感通道只增加自由度、不增加可解释性
（"保持最小、可解释"是本分支的硬要求）。K=1 时熵的分母 $\log 1 = 0$ 显式退化为 0（有测试）。

### 3.2 初始化（D3 的正面设计）

| 层 | 初始化 | 目的 |
|---|---|---|
| `Linear(131, 32)` | PyTorch 默认（Kaiming uniform + uniform bias） | 32 行**互不相同** ⇒ 对称可破缺（D3 的反面） |
| `ReLU` | — | 非线性 |
| `Linear(32, 1)` | **权重与偏置全零** | $z \equiv 0$ ⇒ $g \equiv 0.5$（精确），$\sigma'(0)=0.25$（斜率最大） |

$g \equiv 0.5$ 的语义：起点正好是"完全信任 attention"与"完全用均匀权重"的**中性混合**
$W = 0.5\,W_{\text{attn}} + 0.5\,W_{\text{fw}}$，没有任何先验偏向；且**不饱和**（历史死在 $|z| \approx 130$）。

**已知代价（如实记录，不藏）**：零初始化末层 ⇒ 第一步 $\partial z/\partial(\text{hidden}) \propto W_{\text{out}} = 0$，
隐层在**第一个优化步**拿不到梯度；末层自身从第一步起就有非零梯度，一步之后隐层恢复。
测试 `test_hidden_layer_gradient_is_zero_at_init_then_nonzero_after_one_step` 把这条**双向**锁住。

### 3.3 与基线的关系（可检验的等价性）

* **共享参数逐位相同**：同一 model seed 下，`env_embedding_network` / `projection_network` /
  `gate_network` / `tower_network` 与基线 `NewTask` **逐位相同**（D2 修复的直接后果，测试锁定）。
* **端点等价**：$g \equiv 1$ 时 $W$ 与 $W_{\text{attn}}$ **逐位相同**，此时本头的 forward 与基线 forward
  **逐位相同**（`torch.equal`，测试锁定）⇒ 管线没有静默漂移；$g \equiv 0$ 时退化为均匀权重。
* **正则不变**：`get_l2_reg` 不重写，只正则 `tower_network`（与基线、与历史同口径）。
* **门控不进 L2**：与历史实现一致；这是"机制消融"而非"加正则"。
* **门控读 $H_{\text{out}}$ 的副作用（如实声明）**：`projection_network` 因此多了一条经门控的梯度通路
  （基线下它只经 $W_{\text{attn}}$ 收到梯度）。这是"门控读 attention 投影"这一设计的固有耦合，
  不是缺陷；它意味着本臂与基线**不是**单变量对照，而是"整条融合权重通路"的对照。

### 3.4 为什么不做 warmup

历史 warmup 的动机是"门控起点不可信"（常量初始化下 $g_{\text{raw}} \equiv 1$，只能靠 $\alpha$ 把它拉回 0.5）。
零初始化已经让起点**精确**落在 $g = 0.5$，再加 warmup 只会把 $g$ 再乘一次 $(1-\alpha)$ 拉向 0.5：
既无依据，又会在 `best_state` 载入后留下"被评测的其实是凸混合"的历史怪癖（D4）。
因此**不加**：无 warmup 是本分支的默认，且模块内不存在 alpha 机制（静态守卫锁定）。

### 3.5 规模

| 项 | 本分支 | 历史 CGR |
|---|---|---|
| 门控输入 | $128 + 3 = 131$ | $5 \times 128 = 640$（全为批级量） |
| 隐层宽度 | 32 | 32 |
| 门控参数量 | $131\cdot32 + 32 + 32 + 1 = \mathbf{4257}$ | $640\cdot32+32+32+1 = 20545$ |

门控仍是"小 MLP"（协议 2.2.4 不做 FLOPs，此处只记参数量）。

---

## 4. 记录指标（口径固定，全部 JSON 可序列化）

落盘位置：`artifacts/census_stage2/runs/<run_id>/metrics.json` 的 `cgr_instance_arm` 段
（**不写入 `mechanism`**，两臂的 `mechanism` 键集保持完全一致）。

### 4.1 门控读数（`cgr_instance_arm.diagnostics`，best_state 载入后、只用 val 前向一遍）

字段名与 `exp/stage2-cgr-repro` 的 `CGRStats` **逐名对齐**，便于两臂直接对照。
无 warmup ⇒ 本臂的 `g` 即历史读数的 `g_raw`（历史另有 alpha 插值后的 `g`）。

| 字段 | 定义 | 读法 |
|---|---|---|
| `g_mean` / `g_min` / `g_max` | 样本级 `g` 的均值 / 最小 / 最大 | 机制 3 的输入；历史为 `1.0/1.0/1.0` |
| `g_within_batch_std_mean` / `_min` / `_max` | 每批 `g` 的**批内样本维总体标准差**（ddof=0），再对批取均值 / 极值 | **D1 的签名读数**：历史恒为 0；机制 1 的输入 |
| `g_within_batch_unique_frac_mean` / `_max` | 每批 `unique(g)/批大小`，再取均值 / 最大 | D1 的辅助读数：历史 = `1/批大小`（256 → 0.0039） |
| `weight_l1_mean` / `_max` | 每批逐样本 $\lVert W_b-W_{\text{attn},b}\rVert_1$ 的批内均值 / 最大 | **机制 4 的输入**；恒为 0 ⇔ 门控就是纯 attention |
| `presigmoid_abs_mean` / `_max` | $\lvert z\rvert$ 的样本均值 / 最大 | 饱和的直接读数；历史 `129.18 / 136.81` |
| `presigmoid_saturation_frac` | $\lvert z\rvert \ge 8$ 的样本占比（阈值随字段 `presigmoid_saturation_abs` 一起落盘） | $\sigma'(8)\approx3.4\text{e-}4$；历史为 `1.0` |
| `p_max_mean` / `entropy_norm_mean` / `cos_agreement_mean` | 三个标量门控特征的 val 均值 | 门控**读到了什么**的可解释读数（新臂专属） |
| `hidden_row_symmetry_deviation(_max)` / `hidden_rows_identical` | 隐层权重各行与行均值的平均 / 最大绝对偏差 | **D3 的签名读数**：历史恒为 0；本臂必须 > 0 |
| `gate_output_weight_absmax` / `gate_output_bias` | 训练后门控输出层的权重绝对最大 / 偏置 | "门控确实离开了零初始化"的证据 |
| `gate_hidden` / `gate_input_dim` | 32 / 131 | 结构快照 |
| `n_samples` / `n_batches` | 诊断覆盖的样本数 / 批数 | val 集规模 |

### 4.2 训练期轨迹（`cgr_instance_arm.train_gate_trace`，逐 epoch 一条）

| 字段 | 定义 |
|---|---|
| `epoch` / `steps` / `grad_nonzero_steps` | epoch 序号 / 该 epoch 优化步数 / 其中门控梯度范数 > 0 的步数 |
| `grad_norm_mean` / `grad_norm_max` | 门控**全参数**梯度 L2 范数（**backward 之后、`optimizer.step()` 之前**读取）的均值 / 最大 |
| 其余字段 | 与 4.1 同名的批级 / 样本级读数，按 epoch 聚合 |

总体量在 `cgr_instance_arm.confidence_grad_norm`（`epochs` / `steps` / `grad_norm_mean` / `grad_norm_max` / `grad_nonzero_steps`）。

### 4.3 溯源与判据快照

`cgr_instance_arm.provenance`（被修正对象 / 历史来源 / D1–D6 修复清单 / 门控输入性质 / spec 路径）、
`cgr_instance_arm.prereg`（判据阈值与规则原文）、`cgr_instance_arm.status|pass|mechanism|effect`；
`config.json` 的 `variant` / `gate_hidden` / `gate_input_dim` / `gate_scalar_features` / `warmup` /
`attention_temperature`。

---

## 5. 预注册判据（**只允许在看到结果之前修改**）

### 5.1 主导判据：机制（四条**全部**满足才 `MECHANISM_OK`；任一不满足 ⇒ `MECHANISM_FAILED` ⇒ `STOP`）

```
g_within_batch_std_mean >= 0.01   且   grad_norm > 0
0.05 <= g_mean <= 0.95            且   weight_l1_mean >= 1e-4
```

| # | 规则 | 输入字段 | 理由（被修正的缺陷） |
|---|---|---|---|
| 1 | `g_within_batch_std_mean >= 0.01` | `diagnostics.g_within_batch_std_mean` | 门控必须有**逐样本分化**；历史恒为 0（D1）。0.01 是"远大于浮点噪声、又远小于有效分化"的量级 |
| 2 | `grad_norm > 0` | `train_gate_trace` 的 `grad_norm_max` | 门控必须**收到过梯度**；历史 3900 步精确为 0（D3+D5） |
| 3 | `0.05 <= g_mean <= 0.95` | `diagnostics.g_mean` | 门控不得坍缩到单一分支（与协议 B3 同一区间的用意）；历史为 `1.0` |
| 4 | `weight_l1_mean >= 1e-4` | `diagnostics.weight_l1_mean` | $W$ 必须真的**偏离** $W_{\text{attn}}$；历史为 `0.0`（D3+D4 的恒等式） |

**机制失败即 STOP，与 AUC 无关**——`effect.evaluated = false`、`effect.pass = null`，
臂状态 `STOP`。这一条是本次预注册的核心：先证明机制活着，再谈效应。

> 判据 4 的一个结构性含义（先说清楚，避免事后争论）：$\lVert W-W_{\text{attn}}\rVert_1
> = \lvert 1-g\rvert\cdot\lVert W_{\text{attn}}-W_{\text{fw}}\rVert_1$。
> 若 attention 本身已近似均匀（$W_{\text{attn}}\approx W_{\text{fw}}$），则**任何** $g$ 都几乎不改变 $W$，
> 机制在语义上就是空的 ⇒ `STOP` 是正确判定，而不是误报。

### 5.2 效应阈值（**仅在机制通过时**才解释）

```
AUC-Test-Education >= 0.8521
```

- 参照：协议基线行 `20260929-1735-s20260929-m1685480945-short-904f8d0` 的
  `AUC-Test-Education = 0.8500685307175756`（同 stage1 `s1-096f8f16-m1685480945-e2-cb2094b3`、同一划分）。
- 判定写入 `metrics.json:cgr_instance_arm`，**不并入** `judge()` 的 `overall_pass`（A/B 门禁语义不变）；
  `SUMMARY.md` 照协议追加一行（`run_id` 带 `-cgrinst` 后缀，与基线行天然可区分）。
- 已知继承项：B3（冻结 backbone 的 `gate_networks` 平均权重）在基线行即为 FAIL，与本臂无关；
  **不得**拿 `overall_pass` 当本臂的判据（本臂判据只有 `cgr_instance_arm.*`）。

### 5.3 阈值纪律

1. 阈值只允许在**看到本次 run 结果之前**修改；每次修改单独记录理由（先进本文档，再改
   `census_benchmark/cgr_instance.py` 的 `RULE_*` / 常量）。
2. **禁止事后调参救判据**：不得为了避开失败而改批大小、改 $T$、改初始化、换 model seed、换划分、
   改门控宽度或特征集。机制失败就是失败——这正是预注册的全部意义。
3. **单次判定**：本轮只跑单 model seed `1685480945` 的 **5 epoch 短跑 1 次**（协议 6.1、6.3）；
   "多 seed 平均后才达标"不算通过。
4. 未达效应阈值 ⇒ 如实记录，**不得**跳到多 seed 或全量跑；分支按协议 3.3 处置。

---

## 6. 接线与运行（基线臂零影响）

| 位置 | 改动 |
|---|---|
| `census_benchmark/cgr_instance.py` | **新增**：修正头 + 读数累加器 + 轨迹 + 诊断 + 判据 + 溯源 |
| `run_census_benchmark.run_stage2` | 新增 `variant="baseline"` 参数；新任务头改用 `cgr_instance.build_newtask(variant, …)`；处理臂逐 epoch 记录轨迹、best_state 后做机制诊断 |
| `run_id` | 处理臂追加后缀 `-cgrinst`（协议 `make_run_id` 未改） |
| `config.json` / `metrics.json` | 处理臂新增处理臂专属键（`variant` 等）；**基线臂键集逐键不变** |
| CLI | `stage2 --variant {baseline,cgr-instance}`（默认 `baseline`）与等价别名 `--cgr-instance` |
| `protocol.py` / `metrics.py` / `model.py` / `config.py` | **零改动**（测试 `git diff` 静态守卫锁定） |
| `census_benchmark/tests/test_cgr_instance.py` | **新增**：本文件的测试（stdlib `unittest`；先写测试后写实现） |

**基线臂不受影响的三条证据**（测试锁定）：

1. `build_newtask("baseline", …)` 返回的**就是** `NewTask`（`type(...) is NewTask`），
   同 seed 下参数逐位相同、**全局 RNG 终点**与直接构造逐位相同（不多抽一次）；
2. 基线臂的 `metrics.json` / `config.json` 键集与改动前**逐键相同**（无 `variant` 等处理臂键）、
   `run_id` 无 `-cgrinst`、返回值 `arm is None`；测试用 `mock` 把 `CGRInstanceNewTask` 换成
   "一旦被构造就报错"的哨兵，基线 run 仍必须跑通；
3. 基线臂的 `mechanism` 键集与处理臂完全一致（`["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"]`）。

**运行命令**（单 seed 短跑；与基线同一 stage1 产物）：

```powershell
# 处理臂：修正后的逐样本 CGR
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --variant cgr-instance --tag short --gpu 0
# 等价写法：--cgr-instance
```

产物：`artifacts/census_stage2/runs/<run_id>/`（`metrics.json:cgr_instance_arm` 含判据、诊断与逐 epoch 轨迹），
`SUMMARY.md` 追加一行。**A3 复跑**按协议按需触发，不阻塞本轮。

**测试**（本分支新增，全部 CPU、无 GPU 依赖）：

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -p "test_cgr_instance.py" -v
# 全量（含协议/指标/smoke 回归）
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

---

## 7. 本轮明确不做

- 不做多 seed、不做 `full` tag、不跑 AliCCP / ByteRec、不做 FLOPs（协议 6.1、6.3、2.2.4）。
- 不做调参续命（改 $T$ / 批大小 / 宽度 / 特征集 / 初始化都是在救判据，§5.3 禁止）。
- 不合并 `master`（结果出来后再按协议 3.3 处置）。
- 不主张新颖性（§9）。

---

## 8. 结果（已回填）

> 实跑已于 2026-09-30 完成。本节按协议 7.3"不论好坏一律留痕"回填：run_id、配置、逐 epoch 曲线、
> 四条机制判据的 `observed` 与命中情况、`AUC-Test-Education` 与基线差、A/B 门禁表、判定结论。
> **第 5 节的任何阈值与规则均未改动**；下面全部按预注册原文机械执行。
>
> **一句话结论**：落 §8.0 的**结局 B** —— `MECHANISM_OK` + `EFFECT_NOT_CONFIRMED`。
> 机制（D1–D6 修复后）活了，效应没有；单 seed 短跑**止损**，不扩展多 seed / `full`，不合并 `master`。

### 8.0 执行前的三种可能结局（先写清楚，避免事后解释）

| # | 结局 | 判据 | 含义 |
|---|---|---|---|
| A | `MECHANISM_FAILED` / `STOP` | 四条机制门任一不满足 | 机制又一次没活过来（例如门控塌成全局标量 ⇒ 规则 1 命中；或 g 冲到 1 ⇒ 规则 3 命中）。**此时 AUC 不解释**；`effect.evaluated = false` |
| B | `MECHANISM_OK` + `EFFECT_NOT_CONFIRMED` | 机制全过，`AUC-Test-Education < 0.8521` | 机制活着但**无可测效应**——这是一个干净、可报告的负结果（也是与基线同一 stage1/划分下的单变量对照结论） |
| C | `MECHANISM_OK` + `EFFECT_CONFIRMED` | 机制全过且 `AUC >= 0.8521` | 机制活着且有正向效应；**仍需**多 seed 复现才谈得上任何结论（§5.3 第 3 条），单次 run 不构成证据 |

**风险声明（不预判结果）**：本臂的起点是 $g \equiv 0.5$（attention 与均匀权重的中性混合），
而基线用的是纯 $W_{\text{attn}}$。若数据上"均匀混合"持续有害，最省力的下坡方向是把 $g$ **整体**推向 1
（而不是学出逐样本分化）——那正是规则 1 / 规则 3 要抓的塌缩。因此结局 A 是有实际可能性的，
**它同样是有效结果**：说明"逐样本置信门控"这一族在 CensusIncome 的 5 epoch 短跑里学不出逐样本信号。

### 8.1 run 与配置

| 项 | 值 |
|---|---|
| `run_id` | `20260930-1435-s20260929-m1685480945-short-87afe03-cgrinst` |
| `commit` | `87afe03`（执行时的 HEAD） |
| `stage1_id` | `s1-096f8f16-m1685480945-e2-cb2094b3`（与基线行**同一个** stage1） |
| 划分指纹 | `096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c`（val `6f2c5868…`、test `68e02792…`） |
| 协议配置 | tag `short`、epochs 5、patience 2、lr 1e-3、batch 256、input_size 123、rep_dim 128、frozen `true` |
| 臂专属配置 | `variant=cgr-instance`、`gate_hidden=32`、`gate_input_dim=131`、`gate_scalar_features=[p_max, entropy_norm, cos_agreement]`、`warmup=false`、`attention_temperature=150` |
| 真冻结（A1） | `backbone_sha256` 前/后同为 `a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85` |
| 运行命令 | `.venv\Scripts\python.exe run_census_benchmark.py stage2 --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 --variant cgr-instance --tag short --gpu 0` |
| 产物 | `artifacts/census_stage2/runs/<run_id>/`（`metrics.json:cgr_instance_arm`、`config.json`、`newtask.pt`、`gate_report.json`、`stdout.log`） |

**溯源提示（如实记录）**：run 的 `commit` 字段是执行时的 HEAD（`87afe03`），而本臂的实现
（`census_benchmark/cgr_instance.py` 新增 + `run_census_benchmark.py` 接线）在该次执行时仍是**未提交的
工作区改动**。复现这一行需要这份工作区代码，而不是 `87afe03` 的树。

### 8.2 训练曲线（逐 epoch，val）

| epoch | loss（本臂） | AUC-Val-Edu（本臂） | AUC-Val-Edu（基线行） | Δ（本臂 − 基线） |
|---|---|---|---|---|
| 1 | 0.2581106746999117 | 0.8404614397134184 | 0.8412748819561441 | −0.0008134422427257 |
| 2 | 0.2460492736636064 | 0.8467170877076555 | 0.8473774764767228 | −0.0006603887690673 |
| 3 | 0.2436870086651582 | 0.8495212047268985 | 0.8496795536233096 | −0.0001583488964111 |
| 4 | 0.24217908397699014 | 0.8505536705870181 | 0.8516411115669168 | −0.0010874409798987 |
| 5 | 0.2408276552764269 | **0.8510949134858995** | **0.8527881905614896** | −0.0016932770755901 |

基线行 = `20260929-1735-s20260929-m1685480945-short-904f8d0`（同 stage1、同划分，§5.2 的参照行）。
两臂 `best_epoch` 均为 5（5 epoch 短跑未触发 early stop）；test AUC：本臂 `0.8482881971511005`
vs 基线 `0.8500685307175756`，**Δ = −0.0017803335664751**。

读法（仅描述本次曲线）：逐 epoch 差值在前 3 轮很小（≤ 8.2e-4）且收窄，第 4、5 轮放大到
−1.1e-3 / −1.7e-3——差异全程为负，且随训练变长而扩大（只有 5 轮，不外推）。

### 8.3 机制判据（主导，机械执行 §5.1）

| # | 规则（预注册原文） | 阈值 | observed | 命中 |
|---|---|---|---|---|
| 1 | `g_within_batch_std_mean >= 0.01` | 0.01 | `0.14419801514551248` | **PASS**（14 倍余量） |
| 2 | `grad_norm > 0` | 0.0 | `0.02980218017421471`（轨迹 `grad_norm_max`） | **PASS** |
| 3 | `0.05 <= g_mean <= 0.95` | [0.05, 0.95] | `0.9009520232979779` | **PASS**（上界余量 0.049） |
| 4 | `weight_l1_mean >= 1e-4` | 1e-4 | `0.02337668896501307` | **PASS**（≈234 倍余量） |

⇒ `mechanism.status = MECHANISM_OK`、`failed_rules = []`、`pass = true`（`metrics.json:cgr_instance_arm.mechanism`）。

**与退化臂的签名读数对照**（退化臂 = `20260930-1355-s20260929-m1685480945-short-87afe03-cgr`，同 stage1、同划分）：

| 签名读数（val 诊断） | 退化臂 | 本臂 | 判读 |
|---|---|---|---|
| `g_within_batch_std_mean` | `0.0` | `0.14419801514551248` | D1 修复：门控真的逐样本分化 |
| `g_within_batch_unique_frac_mean` | `0.003909850230414747`（= 1/256） | `0.718978753101737` | 同上 |
| `weight_l1_mean` | `0.0` | `0.02337668896501307` | D3+D4 修复：$W$ 真的偏离 $W_{\text{attn}}$ |
| `presigmoid_abs_mean` / `_max` | `129.1841770734294` / `136.81227111816406` | `6.699636325120343` / `20.144716262817383` | 不再出生即饱和 |
| `presigmoid_saturation_frac` | `1.0` | `0.3782201639902969` | 仍部分饱和，但非全局（见 §8.4） |
| `grad_nonzero_steps / steps` | `0 / 3900` | `3900 / 3900` | D5 修复：门控每一步都收到梯度 |
| `hidden_rows_identical` | `true` | `false` | D3 修复：隐层对称已破缺 |
| `g_mean` | `1.0` | `0.9009520232979779` | 未坍缩（但偏上，见 §8.4） |

门控**读到了什么**（val，本臂专属可解释读数）：`p_max_mean = 0.7402729215212993`、
`entropy_norm_mean = 0.7835984706642065`、`cos_agreement_mean = 0.3965869654173964`
⇒ attention 只是**轻度**尖峰（均匀时 $p_{\max}=0.5$、熵=1.0），两路表征的余弦一致度约 0.40。

### 8.4 训练轨迹（逐 epoch，`cgr_instance_arm.train_gate_trace`）

| epoch | steps | 梯度非零步 | `grad_norm_mean` | `grad_norm_max` | `g_mean` | `g_within_batch_std_mean` | `weight_l1_mean` | `g_max` |
|---|---|---|---|---|---|---|---|---|
| 1 | 780 | 780 | 0.0008793343599251483 | 0.006048444420320248 | 0.8563814733769313 | 0.17299515294115314 | 0.04301556356600124 | 0.9999991655349731 |
| 2 | 780 | 780 | 0.0020951705687719977 | 0.012500314860364115 | 0.8803099188707274 | 0.1774942460521811 | 0.039088843009534656 | 0.9999992847442627 |
| 3 | 780 | 780 | 0.004689294126771627 | 0.02980218017421471 | 0.8862827657448757 | 0.15529860810136048 | 0.04104976994267768 | 0.9999998807907104 |
| 4 | 780 | 780 | 0.005810009619358167 | 0.024961004888678414 | 0.8983250785372134 | 0.134655138560536 | 0.033309890506780196 | 1.0 |
| 5 | 780 | 780 | 0.004733079185650529 | 0.01736159181348297 | 0.9138255456216952 | 0.13211618618290713 | 0.02229718542660808 | 1.0 |
| 总体 | 3900 | **3900** | 0.0036413775720954902 | 0.02980218017421471 | 0.8870249564302886 | 0.15451186636762762 | 0.03575225049032037 | 1.0 |

趋势读法（如实记录，**不影响 §8.3 / §8.5 的判定**）：

* `g_mean` **单调上升** 0.8564 → 0.9138（全程 0.8870），朝"完全信任 attention"（$g \to 1$）漂移；
  规则 3 的上界 0.95 在 val 诊断上只剩 0.049 余量。这正是 §8.0 风险声明里写的"最省力的下坡方向"。
* `g_within_batch_std_mean` 与 `weight_l1_mean` **同向下降**（0.1730 → 0.1321；0.0430 → 0.0223），
  与 §5.1 判据 4 的结构恒等式 $\lVert W-W_{\text{attn}}\rVert_1 = \lvert 1-g\rvert\cdot
  \lVert W_{\text{attn}}-W_{\text{fw}}\rVert_1$ 一致：$g$ 整体上移 ⇒ $\lvert 1-g\rvert$ 收缩 ⇒ L1 收缩。
  逐样本分化本身并未消失（`unique_frac ≈ 0.72`、std ≫ 0.01），只是**幅度在收窄**。
* `grad_norm_mean` 逐 epoch 上升（8.8e-4 → 4.7e-3），`grad_nonzero_steps` 每轮都是 780/780。
* `presigmoid_saturation_frac` 在 0.29–0.42 间波动（val `0.3782`）：零初始化消除了**出生**饱和，
  但训练把约 1/3 的样本重新推进 $\lvert z\rvert \ge 8$ 区，`g_max` 从第 4 轮起精确等于 1.0
  （float32 下 $\sigma$ 饱和）。这与"逐样本分化仍然活跃"不矛盾：饱和的是一部分样本，
  梯度在整个训练过程中始终非零。

事后解释（**不是判据**，只为读懂数字）：attention 本身只是轻度非均匀（熵 0.784、$p_{\max}$ 0.740），
而 $g \approx 0.90$ 意味着只把权重往均匀方向拉约 10% ⇒ 对 $W$ 的扰动本来就小
（`weight_l1_mean = 0.0234`，K=2 时平均每维偏差约 1.2 个百分点）。这正是 §5.1 判据 4 后面那条
结构性提示所描述的情形。

### 8.5 效应判据（仅在机制通过时解释——机制通过了，故必须解释）

| 项 | 值 |
|---|---|
| 规则（预注册原文） | `auc_test_education >= 0.8521` |
| observed | `0.8482881971511005` |
| `evaluated` | `true` |
| `pass` | **`false`** |
| 与基线差 | `0.8482881971511005 − 0.8500685307175756` = **`−0.0017803335664751`** |
| 臂状态 | **`EFFECT_NOT_CONFIRMED`**（`cgr_instance_arm.status`，`pass = false`） |

两点如实记录、**均不改变判定**：

1. 阈值 `0.8521` 高于基线行自身的 `0.8500685307175756`（这是 §5.2 预注册时就并列写明的）。
   因此本臂是"两头都没够着"：既没到绝对阈值（差 `0.0038118028488995`），
   也在配对意义上低于同 stage1 / 同划分的基线 `0.0018`。
2. 阈值纪律（§5.3）禁止在看到结果后调阈值、或改用"相对基线"口径重判本次。若将来要把判据改成
   相对基线，必须作为**下一次** run 的新预注册，不得回溯本 run。

### 8.6 实现期发现：混合权重的均匀不动点边界

加固测试夹具时确认了一条**结构性质**（不是实现缺陷，也不是新判据）：

$$\frac{\partial W}{\partial g} = W_{\text{attn}} - W_{\text{fw}}
\;\Longrightarrow\;
W_{\text{attn}} \equiv W_{\text{fw}} \;\Rightarrow\; \frac{\partial L}{\partial g} \equiv 0
\;\Rightarrow\; \text{门控全部参数梯度精确为 } 0$$

即"attention 恰好均匀"是混合权重的**不动点**：门控在此既收不到一阶信号、也无法靠自己离开。
夹具实证：随机初值下投影瓶颈全灭（$H_{\text{out}} \equiv 0 \Rightarrow W_{\text{attn}}$ 逐位 $= W_{\text{fw}}$）
并不罕见——B=1 夹具扫描 60 个全局 RNG 状态下有 20 个如此（约 1/3）。因此任何**无条件**断言
"门控梯度非零"的测试都会随机失败。

处置（已完成，见 `census_benchmark/tests/test_cgr_instance.py` 第二版）：

* **夹具确定性化**：`seeded_head` / `deterministic_reps` / `force_live_attention`（构造上必然存活：
  瓶颈第 0 行写成批内首个样本的单位方向，第二层只留一个 1 ⇒ $H_{\text{out}}=\mathrm{LayerNorm}([r,0,\dots,0])$）；
  凡断言"梯度非零"必先验收前置条件 `assert_attention_is_live`（失败信息显式指向夹具而非实现）。
* **边界本身正面锁定**：`test_collapsed_attention_is_a_gradient_fixed_point`——
  瓶颈置零 ⇒ $W_{\text{attn}}$ 逐位 $= W_{\text{fw}}$ ⇒ 门控仍在计算图上（`.grad is not None`）
  但梯度逐位为 0、`confidence_grad_norm == 0.0`。
* 附带纪律：`test_forward_differs_from_baseline_at_the_neutral_gate` 一类测试也改为先
  `force_live_attention` 再断言，否则夹具随机采到全灭瓶颈时，"混合生效"会被误报为"混合是恒等映射"。

**不使本次真跑失效**：本次 run `grad_nonzero_steps = 3900/3900`（每一步门控梯度范数都 > 0）、
`weight_l1_mean = 0.02337668896501307`（≫ 1e-4），训练**从未**落在这个边界上。
该边界也正是预注册判据 4 在结构上要防的东西：`weight_l1_mean` 一旦为 0，混合就是恒等映射、机制为空。

### 8.7 A/B 门禁与 SUMMARY

| 门禁 | 结果 | 关键读数 |
|---|---|---|
| A1 | PASS | `backbone_sha_equal = true`、`grads_all_none = true` |
| A2 | PASS | `split_fingerprint_consistent = true` |
| A4 | PASS | `disjoint_and_complete = true` |
| A5 | PASS | `env_ids_sha256_matches_stage1 = true` |
| B1 | PASS | val income `0.9373971773636002`、val marital `0.9912246018108644`、test edu `0.8482881971511005`（下限 0.6） |
| B2 | PASS | gap `0.0028067163347990087`（上限 0.03） |
| B3 | **FAIL** | `gate_mean = [0.952401451979339, 0.8543901456350915]` |
| B4 | PASS | env shares `[0.49411847255704855, 0.5058815274429515]` |
| A3 | 未触发 | `on_demand`（协议 6.1 / 8 节：不阻塞首轮） |

`overall_pass = false`、`failures = ["B3"]`。**B3 是继承失败**：本臂的 `mechanism.gate_mean` 与基线行
**逐位相同**（backbone 冻结、两臂共享同一 stage1，`mechanism` 四个键的取值全部逐位一致），
该失败由冻结 backbone 决定，与本臂无关——与 §5.2 的"已知继承项"一致。
本臂的判据只有 `cgr_instance_arm.*`，**不得**拿 `overall_pass` 当本臂结论。

SUMMARY 行已按协议追加并逐字段复核（`artifacts/census_stage2/SUMMARY.md` 第 4 行）：

| run_id | commit | auc_test_education | A1 | A2 | A4 | A5 | B1 | B2 | B3 | B4 | stage1_id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260930-1435-s20260929-m1685480945-short-87afe03-cgrinst | 87afe03 | 0.848288 | PASS | PASS | PASS | PASS | PASS | PASS | FAIL | PASS | s1-096f8f16-m1685480945-e2-cb2094b3 |

核对：`auc_test_education` = `f"{0.8482881971511005:.6f}"` = `0.848288` ✓；8 个门禁列与
`gate_report.json` 逐项一致 ✓；`stage1_id` 与基线行相同（同一冻结 backbone、同一划分）✓。

### 8.8 结论

按 §8.0 的预注册结局表，本次落 **结局 B**：

> **`MECHANISM_OK` + `EFFECT_NOT_CONFIRMED`**：把 `exp/stage2-cgr-repro` 审计出的 D1–D6 逐条修好之后，
> "逐样本置信门控混合 attention 与均匀权重"在机制上**活过来了**（四条机制门全过，且与退化臂的
> 签名读数全部反向），但在 CensusIncome 单 seed 5 epoch 短跑上**没有可测的正向效应**：
> `AUC-Test-Education = 0.8482881971511005 < 0.8521`，且比同 stage1 / 同划分的基线低 `0.0017803335664751`。

处置（按协议）：

* **止损**：不扩展多 seed、不跑 `full`、不搜参、不改 $T$ / 批大小 / 宽度 / 特征集 / 初始化
  （§5.3、协议 6.3 第 4 条）。
* **不合并 `master`**：协议 3.3 第 1 条只在"验证通过"时合并；本次未达阈值 ⇒ 分支保持现状，
  等用户显式指示（本分支的位置就是这次工程消融的完整留痕）。
* **可报告范围**：单 model seed、5 epoch、CensusIncome、`education` 新任务、单个冻结 backbone。
  任何"逐样本门控无效"的**一般性**主张都超出本 run 的证据范围；同理，`g_mean` 单调上行的趋势
  也只是 5 轮内的观察，不构成"必然塌缩"的结论。

### 8.9 验证留痕

| 项 | 结果 |
|---|---|
| 聚焦测试 | `census_benchmark/tests/test_cgr_instance.py` **82 项全部通过**（实跑会话执行；stdlib `unittest`、纯 CPU、无 GPU 依赖；含 §8.6 的确定性夹具与不动点测试） |
| 测试计数复核 | 本文件 `def test_` 静态计数 = **82**，与实跑报告的 82 项一致 |
| 基线臂回归 | 基线臂"键集/行为逐键不变"的四条集成测试（`TestRunnerIntegration`）在 82 项内，全部通过 |
| 静态守卫 | `protocol.py` / `metrics.py` / `multitaskrec/model.py` / `config.py` 零改动（守卫测试通过） |
| `git diff --check` | 干净（无空白错误） |
| SUMMARY 复核 | 已按 §8.7 逐字段核对 |

---

## 9. 非新颖性声明

置信门控 / 条件化路由 / 门控融合（gated fusion）是**已有大类**：MoE 的门控网络、FiLM 的特征调制、
条件化归一化、prompt 调制同属一族；"用一个小 MLP 从输入算一个标量去混合两路权重"没有任何新颖性可言。
历史 CGR 在 CensusIncome 上也**从未超越** prompt（`archive/exploration`：
mean test AUC `0.8618 ± 0.0048`，Δ vs prompt `−0.0011`）。

本分支因此**不提出方法**，只交付一样东西：**一次把审计缺陷逐条修好后的、可证伪的工程消融**——
它要么给出"机制活着、效应达标"的证据，要么给出"机制活着但效应不达标"或"机制又一次死了"的证据。
三种结局都写进第 8 节，都不合并 `master`，除非用户显式指示。
