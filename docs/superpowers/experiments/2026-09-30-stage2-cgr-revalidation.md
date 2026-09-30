# exp/stage2-cgr-repro：历史 CGR 忠实复现与门控退化复核（实验说明）

- **状态**：**已终结（方向停止）**。真实验（单 seed 短跑，第 7 节命令）**已执行**：
  `run_id = 20260930-1355-s20260929-m1685480945-short-87afe03-cgr`，
  退化判定 **`CONFIRMED_DEGENERATE`**（三条预注册判据**全部**命中）⇒ 臂状态 **`STOP`**，
  效应判据 `evaluated = false`。结果已回填第 8 节；按第 9.2 节处置：**不做多 seed、不做调参续命、
  不合并 `master`**，本分支连同失败结果保留并推送。
- **日期**：2026-09-30
- **分支**：`exp/stage2-cgr-repro`（从 `infra/fair-stage2-benchmark` @ `87afe03` 独立拉出）
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。本文件**只引用**它；划分、种子、冻结三件套、A/B 门禁一律沿用，`census_benchmark/protocol.py` / `census_benchmark/metrics.py` / `multitaskrec/model.py` **零改动**。
- **数据集**：只跑 CensusIncome（协议 2.3）。
- **定位**：这不是新方法，是**历史实现的忠实复现 + 退化判据的前置诊断**（第 9 节"非新颖性"）。

---

## 1. 要回答的问题

历史 CGR（Confidence-Gated Routing）在 `archive/exploration` 上跑出过 CensusIncome `0.8618 ± 0.0048`
（相对 prompt `−0.0011`，即**未超越** prompt）、AliCCP `0.6632 ± 0.0241`（seed3 崩到 `0.6354`）。
当时给出的根因是"门控是全局标量、没有逐样本信息"（commit `6b023d7`）。

本轮**不提出任何改进**，只做两件事：

1. **忠实复现**：把历史 CGR 的公式、初始化、warmup **逐项照搬**（含历史对比里的 RNG 构造顺序缺陷），
   在**当前公平评测协议**（固定划分、真冻结、val 选点 / test 只评一次）下重跑一次单 seed 短跑。
2. **前置诊断**：先判定门控**是否退化**（无样本级分化 / 恒等于 attention / 从未学到梯度）。
   **退化是主导判据**——一旦退化即 `CONFIRMED_DEGENERATE` / `STOP`，此时"是否达到 `0.8521`"这一问题
   不再解释；只有在**非退化**时，`AUC-Test-Education >= 0.8521` 才有意义。

> 换句话说：本分支的产物是**一份可审计的否证材料**，而不是一个新方法。

---

## 2. 审计溯源（audit provenance）

| 项 | 来源 | 关键内容 |
|---|---|---|
| 历史训练脚本 | `archive/exploration`：`run_newtask_from_ckpt.py` | `ALL_MODES` 含 `"cgr"`；`warmup_epochs = 5`；每个 epoch 开头 `newtask.gate_warmup_alpha = min(1.0, epoch / warmup_epochs)`；`loss = BCE(pred.cpu(), y) + get_l2_reg()`（**只正则 tower**）；Adam(lr=cfg.lr)；val 选点 + patience 早停；结束时载入 `best_weight` 再评 test |
| 历史模型 | `archive/exploration`：`multitaskrec/model.py` → `NewTask(fusion_mode="cgr")` | `confidence_mlp = Linear(5·rep_dim, 32) → ReLU → Linear(32, 1) → Sigmoid`；常量初始化 `0.01 / 0.01`、`0.1 / 2.0`；`gate_warmup_alpha` 默认 `1.0`；cgr 分支的 `W_attn / W_fw / g / W` 表达式原文 |
| 历史诊断 1 | `archive/exploration`：`analysis/cgr_diagnosis.py` | 逐 epoch 记录 `g_mean / g_std`（在模块外**复算**同一公式） |
| 历史诊断 2 | `archive/exploration`：`analysis/cgr_gradient_diagnosis.py` | 记录 **`projection_network`** 的逐 batch 梯度范数（prompt vs cgr 对比） |
| 历史结果与根因 | `archive/exploration`：`logs/all_experiment_results.md`（根因分析随 commit `6b023d7` 落盘） | CensusIncome cgr 三 seed val/test = `0.8645/0.8627`、`0.8650/0.8660`、`0.8602/0.8566`，mean test `0.8618 ± 0.0048`，Δ vs prompt `−0.0011`，参数量 ~900；AliCCP mean `0.6632 ± 0.0241`（seed3 `0.6354`）；根因原文："门控输入 …… 所有统计量均来自冻结的 MPTRec，**对全部样本完全相同**……g 是一个全局标量" |
| 当前协议 | `docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md` + `artifacts/census_stage2/SUMMARY.md` | 基线参照行 `20260929-1735-s20260929-m1685480945-short-904f8d0`，`AUC-Test-Education = 0.850069`；固定 stage1 产物 `s1-096f8f16-m1685480945-e2-cb2094b3` |

**口径差异（如实声明）**：历史诊断 2 量的是 `projection_network` 的梯度；本轮按审计要求改量
**`confidence_mlp`（门控自身）**的梯度范数——因为"门控有没有学到东西"只取决于门控自己的梯度。
`projection_network` 的梯度不在本轮记录范围内（如需回溯历史读数，用归档脚本）。

---

## 3. 已知缺陷：复现时**故意保留**（不得"顺手修好"）

| 编号 | 缺陷 | 证据 / 后果 | 本轮如何被观测 |
|---|---|---|---|
| **D1** | **批级门控**：`z` 的输入是 `[e_new, mean_k E_k, var_k E_k, mean_b gen_rep, var_b gen_rep]`，全部是**批级 / 环境级统计量**，没有逐样本信息 ⇒ `g` 在**批内逐位相同** | 同一批内所有样本共用一个 `g`；**同一样本换个批上下文就换一个 `g`**（历史称"全局标量"，严格说是"批级标量"） | `g_within_batch_std_*`（恒为 0）、`g_within_batch_unique_frac_*`（≈ 1/批大小） |
| **D2** | **RNG 构造顺序**：历史把 `confidence_mlp` 建在 `projection_network` 与 `gate_network` **之间**，两个 `nn.Linear` 的默认初始化多消耗全局 RNG ⇒ 同一 model seed 下 `gate_network` / `tower_network` 的初始化与基线**不同** | 历史 "CGR vs Prompt" 的对比**不是单变量对比**：差异里混着 gate/tower 初始化的位移 | 测试 `test_rng_construction_order_confound_is_reproduced`（锁定"多抽之前相同、之后不同"）；`metrics.json:cgr_arm.provenance.rng_construction_order = "historical"` |
| **D3** | **饱和常量初始化 + 行对称永不破缺**：`W1` 全元素 `0.01`、`W2` 全元素 `0.1` ⇒ 32 个隐层单元**逐位相同**，且对称性在训练中**不会**被打破（同梯度 ⇒ 同更新） | 门控退化成"输入的标量和"的**一个标量函数**（rank-1），起点 `g ≈ 0.88–0.99`（偏向信任 attention） | `hidden_row_symmetry_deviation(_max)`、`hidden_rows_identical`、`presigmoid_abs_*` / `presigmoid_saturation_frac` |
| **D4** | **warmup 的时序怪癖**：`alpha` 由 runner 逐 epoch 写入且**不是 buffer**（不进 `state_dict`）⇒ 载入 `best_state` 后 `alpha` 停在**最后一轮**的取值；5 epoch 协议下 `alpha = 0.2/0.4/0.6/0.8/1.0`，早停时更小 | 被评测的其实是 `g·W_attn + (1−g)·W_fw` 的**凸混合**（`g` 被强行拉向 0.5），而不是"纯 CGR" | `diagnostics.warmup_alpha_at_eval` + 逐 epoch `train_gate_trace[*].alpha` |
| **D5** | **梯度信号极弱**：`g` 每步只有一个标量，反传到门控的只有 1 维信号 | 历史根因分析："梯度通过全局 g 反向传播到 MLP 权重时信号极弱，模型无法自救" | `confidence_grad_norm`（mean/max/nonzero_steps）与判据 3 |
| **D6** | **单样本批 NaN**（本轮审计新发现）：门控输入里 `gen_rep.var(dim=0)` 用 ddof=1，批大小 = 1 时自由度 ≤ 0 ⇒ `g = NaN` ⇒ 该步整批预测与损失全毁 | CensusIncome **恰好躲开**：训练集 `199523 = 779×256 + 99`、val `49881 = 194×256 + 217`，末批都不是 1（实测于 `splits/20260929/split_fingerprint.json`）。任何 batch_size / 样本数变化都可能踩中 | `test_single_sample_batch_gives_nan_gate`（带告警抑制的 NaN 断言，锁住历史行为） |

**为什么必须原样保留 D2**：本分支的目的是"复核历史结论是否站得住"，因此被复核的对象必须是**历史那一版**。
把 D2 修掉（例如用 `super().__init__()` 再补建 `confidence_mlp`）会让门控臂与基线只差一个变量——
那是一次**新实验**（需要新的预注册），不是复现。D2 的存在已在 `metrics.json:cgr_arm.provenance` 留痕，
比较时必须把"初始化位移"计入不确定性。

---

## 4. 忠实复现的定义（`census_benchmark/cgr.py`）

设 $x$ = `dnn_input`，$H(x)$ = `projection_network(x)`（末端 LayerNorm），$E=[\mathbf e_0,\mathbf e_1]$，
$T = 150$，$K=2$：

$$W_{\text{attn}}(x)=\mathrm{softmax}\big(H(x)E/T\big),\qquad W_{\text{fw}} = \mathbf 1/K$$

$$z(x) = \mathrm{MLP}_{0.01/0.01,\,0.1/2.0}\big([\,\mathbf e_{\text{new}},\ \overline{E},\ \mathrm{var}(E),\
\overline{\mathrm{gen\_rep}},\ \mathrm{var}(\mathrm{gen\_rep})\,]\big),\qquad g_{\text{raw}}=\sigma(z)$$

$$g = \alpha\,g_{\text{raw}} + (1-\alpha)\cdot 0.5,\qquad W = g\,W_{\text{attn}} + (1-g)\,W_{\text{fw}}$$

$$\text{new\_spec\_rep}=\textstyle\sum_k W_k\,\text{spec\_rep}_k,\quad
\text{env\_aware}=s\odot\text{new\_spec\_rep},\quad
\text{pred}=\text{tower}\big([\text{env\_aware},\text{gen\_rep}]\cdot \text{gate\_network}(x)\big)$$

| 历史项 | 历史取值 | 复现位置 | 是否逐字相同 |
|---|---|---|---|
| 公式（`W_attn / W_fw / z / g / W / 融合管线`） | 见上 | `CGRNewTask.gate_terms` / `CGRNewTask.forward` | ✅ 同表达式、同运算顺序（测试用逐位 `torch.equal` 锁定） |
| 常量初始化 | `0.01 / 0.01`、`0.1 / 2.0` | `CGRNewTask.__init__` | ✅ |
| 隐层宽度 / 温度 | `32` / `150` | `HIDDEN_UNITS` / `TEMPERATURE` | ✅ |
| 构造顺序（含 RNG 混入） | projection → **confidence_mlp** → gate → tower | `CGRNewTask.__init__`（`nn.Module.__init__` + 手工搭树） | ✅（D2 故意保留） |
| warmup | `alpha = min(1.0, epoch / 5)`，runner 逐 epoch 写入 | `warmup_alpha` / `set_warmup_alpha`，接线在 `run_census_benchmark.run_stage2` | ✅ |
| 正则 | 只正则 tower（`confidence_mlp` **不进** L2） | 继承 `NewTask.get_l2_reg`，未重写 | ✅ |
| 损失 / 优化器 / 选点 | `BCE + get_l2_reg`、Adam(lr)、val 选点、test 只评一次 | 评测协议的 stage2 路径（未改） | ✅ |

**stage2 配置**：`epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`（协议 6.1；与基线臂完全相同）。

---

## 5. 记录指标（口径固定，全部 JSON 可序列化）

落盘位置：`artifacts/census_stage2/runs/<run_id>/metrics.json` 的 `cgr_arm` 段（**不写入 `mechanism`**，
两臂的 `mechanism` 键集保持完全一致）。

### 5.1 门控读数（`cgr_arm.diagnostics`，best_state 载入后、只用 val 前向一遍）

| 字段 | 定义 | 读法 |
|---|---|---|
| `g_mean` / `g_min` / `g_max` | 样本级 `g` 的均值 / 最小 / 最大 | 忠实实现的 `g` 批内相同、跨批不同 |
| `g_within_batch_std_mean` / `_max` | 每批 `g` 的**批内样本维总体标准差**（ddof=0），再对批取均值 / 最大值 | **D1 的签名读数：忠实实现恒为 0** |
| `g_within_batch_unique_frac_mean` / `_max` | 每批 `unique(g) / 批大小`，再取均值 / 最大值 | **D1 的签名读数：忠实实现 ≈ 1/批大小**（256 → 0.0039） |
| `weight_l1_mean` / `_max` | 每批逐样本 $\lVert W_b - W_{\text{attn},b}\rVert_1$ 的批内均值，再对批取均值 / 最大值 | `= (1-g)\lVert W_{\text{attn}}-W_{\text{fw}}\rVert_1`；**恒为 0 ⇔ CGR 就是纯 attention** |
| `presigmoid_abs_mean` / `_max` | $|z|$ 的样本均值 / 最大 | 饱和的直接读数 |
| `presigmoid_saturation_frac` | $|z| \ge 8$ 的样本占比（阈值随字段 `presigmoid_saturation_abs` 一起落盘） | `sigmoid'(8) ≈ 3.4e-4`，门控实际上动不了 |
| `g_raw_mean/min/max` | sigmoid 之后的原始门控（warmup 之前） | 区分"饱和"与"warmup 拉平" |
| `warmup_alpha_at_eval` | 诊断时生效的 `alpha`（D4 留痕） | 早停时 < 1 |
| `hidden_row_symmetry_deviation(_max)`、`hidden_rows_identical` | `confidence_mlp[0].weight` 各行与行均值的平均 / 最大绝对偏差 | **D3 的签名读数：常量初始化 + 常量第二层 ⇒ 恒为 0（对称永不破缺）** |
| `n_samples` / `n_batches` | 诊断覆盖的样本数 / 批数 | val 集规模 |

### 5.2 训练期轨迹（`cgr_arm.train_gate_trace`，逐 epoch 一条；总体量在 `cgr_arm.confidence_grad_norm`）

| 字段 | 定义 |
|---|---|
| `epoch` / `alpha` | epoch 序号 / 该 epoch 生效的 warmup alpha（历史式 `min(1, epoch/5)`） |
| `steps` / `grad_nonzero_steps` | 该 epoch 的优化步数 / 其中梯度范数 > 0 的步数 |
| `grad_norm_mean` / `grad_norm_max` | `confidence_mlp` 全参数梯度 L2 范数（**backward 之后、`optimizer.step()` 之前**读取）的均值 / 最大值 |
| 其余字段 | 与 5.1 同名的批级 / 样本级读数，按 epoch 聚合 |

### 5.3 溯源与判据快照

`cgr_arm.provenance`（历史来源 / 构造顺序 / 门控输入性质 / 初始化 / spec 路径）、
`cgr_arm.prereg`（判据阈值与规则原文）、`cgr_arm.variant`、`config.json` 的 `variant` /
`warmup_epochs` / `temperature`。

---

## 6. 预注册判据（**只允许在看到结果之前修改**）

### 6.1 主导判据：退化（三条任一命中 ⇒ `CONFIRMED_DEGENERATE` ⇒ `STOP`）

```
unique_frac <= 0.01   或   weight_l1 < 1e-6   或   grad_norm == 0
```

| 规则 | 输入字段 | 理由 |
|---|---|---|
| `unique_frac <= 0.01` | `diagnostics.g_within_batch_unique_frac_mean` | 批内 ≤1% 的样本有不同 `g` ⇒ 门控**没有样本级分化**（D1）。批大小 256 时忠实实现 = 0.0039 |
| `weight_l1 < 1e-6` | `diagnostics.weight_l1_mean` | `W ≡ W_attn` ⇒ CGR **恒等于纯 attention**，机制不存在（D3 硬饱和的直接后果） |
| `grad_norm == 0` | `train_gate_trace` 的 `grad_norm_max` | 门控**从未收到过任何梯度** ⇒ 它不是"没学好"，而是根本没在学（D5；空轨迹同样判 0） |

### 6.2 效应阈值（**仅在非退化时**才解释）

```
AUC-Test-Education >= 0.8521
```

- 参照：协议基线行 `20260929-1735-s20260929-m1685480945-short-904f8d0` 的 `0.850069`（同 stage1、同划分）。
- 退化时 `effect.evaluated = false`、`effect.pass = null`，臂状态为 `STOP`——**不得**用退化 run 的 AUC
  去谈"是否达标"。
- 判定写入 `metrics.json:cgr_arm`，**不并入** `judge()` 的 `overall_pass`（A/B 门禁语义不变）；
  `SUMMARY.md` 照协议追加一行（`run_id` 带 `-cgr` 后缀，与基线行天然可区分）。

### 6.3 阈值纪律

1. 阈值只允许在**看到本次 run 结果之前**修改；每次修改单独记录理由（先进本文档，再改 `cgr.py`）。
2. **禁止事后调参救判据**：不得为了避开退化而改批大小、改 `T`、改初始化、换 model seed、换划分。
   退化就是退化——这正是预注册的全部意义。
3. 单次判定：本轮只跑单 model seed `1685480945`；"多 seed 平均后才达标"不算通过。

---

## 7. 接线与运行（基线臂零影响）

| 位置 | 改动 |
|---|---|
| `census_benchmark/cgr.py` | **新增**：忠实复现头 + 读数累加器 + 轨迹 + 诊断 + 判据 |
| `run_census_benchmark.run_stage2` | 新增 `variant="baseline"` 参数；新任务头改用 `cgr.build_newtask(variant, …)`；处理臂逐 epoch 写 warmup、记录轨迹、best_state 后诊断 |
| `run_id` | 处理臂追加后缀 `-cgr`（协议 `make_run_id` 未改） |
| `config.json` | 两臂记 `variant`；处理臂另记 `warmup_epochs` / `temperature` |
| `metrics.json` | 两臂记 `variant` 且 `mechanism` 键集不变；处理臂另记 `cgr_arm` |
| CLI | `stage2 --variant {baseline,cgr}`（默认 `baseline`）与等价别名 `--cgr` |
| `protocol.py` / `metrics.py` / `model.py` / `config.py` | **零改动**（测试 `git diff` 静态守卫锁定） |

**基线臂不受影响的三条证据**（测试锁定）：
1. `build_newtask("baseline", …)` 返回的**就是** `NewTask`（`type(...) is NewTask`），
   同 seed 下参数 / buffer / **全局 RNG 终点**与直接构造逐位相同；
2. 基线 run 的 `run_id` 无 `-cgr`、payload 无 `cgr_arm`、`config.json` 无处理臂专属字段；
   测试用 `mock` 把 `CGRNewTask` 换成"一旦被构造就报错"的哨兵，基线 run 仍必须跑通；
3. 基线臂的 `mechanism` 键集与处理臂完全一致（`["cos_gen_spec", "env_acc_stage1", "gate_mean", "gen_std"]`）。
   *唯一*新增的是 `config.json` / `metrics.json` 里恒为 `"baseline"` 的 `variant` 标签和 `run_id` 里的
   `-cgr` 区分后缀（与 null-expert / attn / csb 分支同一约定）。

**运行命令**（单 seed 短跑；与基线同一 stage1 产物）：

```powershell
# 处理臂：忠实复现的历史 CGR
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --variant cgr --tag short --gpu 0
# 等价写法：--cgr

# 如需基线对照（同一 stage1、同一划分；预期复现 SUMMARY.md 的 0.850069）
.venv\Scripts\python.exe run_census_benchmark.py stage2 `
  --stage1-dir artifacts\census_stage2\stage1\s1-096f8f16-m1685480945-e2-cb2094b3 `
  --tag short --gpu 0
```

产物：`artifacts/census_stage2/runs/<run_id>/`（`metrics.json:cgr_arm` 含判据、诊断与逐 epoch 轨迹），
`SUMMARY.md` 追加一行。**A3 复跑**按协议按需触发，不阻塞本轮。

---

## 8. 结果（已回填：2026-09-30 真实验）

> 实验已由用户执行（命令见第 7 节）。结果**不论好坏一律留痕**（协议 7.3）：以下是这次 run 的全部判定
> 与关键读数，**没有**只记录通过的 run，**没有**在看到结果后改判据（阈值、规则、指标定义均与第 6 节一致）。

### 8.1 结论

**`CONFIRMED_DEGENERATE` ⇒ 臂状态 `STOP`。** 三条预注册退化判据**全部**命中；按 §6.2，
效应判据 `evaluated = false`、`pass = null`——**不得**用本 run 的 AUC 谈"是否达标"。
按 §9.2 处置：**方向停止**（不做多 seed、不做调参续命、不合并 `master`）。

### 8.2 运行概况

| 项 | 取值 |
|---|---|
| `run_id` | `20260930-1355-s20260929-m1685480945-short-87afe03-cgr` |
| commit / `stage1_id` | `87afe03` / `s1-096f8f16-m1685480945-e2-cb2094b3`（与基线参照行**同一个** stage1） |
| 配置（`config.json`） | `variant=cgr`、`epochs=5`、`patience=2`、`batch_size=256`、`lr=1e-3`、`warmup_epochs=5`、`temperature=150`、`rep_dim=128`、`input_size=123` |
| `AUC-Val-Education`（best） | `0.8496247474820915`（**epoch 5**；逐 epoch `0.8411 → 0.8466 → 0.8483 → 0.8493 → 0.8496`，未触发早停） |
| `AUC-Test-Education` | `0.84735625087073` |
| 基线参照 `20260929-1735-…-904f8d0`（同 stage1、同划分） | `0.8500685307175756` ⇒ **Δ = −0.0027122798468456** |
| 效应判据 `auc_test_education >= 0.8521` | `observed = 0.84735625087073`、`evaluated = false`、`pass = null`（退化优先） |
| 退化判定 | **`CONFIRMED_DEGENERATE`**，`triggered_rules` 三条**全部**命中 |
| 臂状态 | **`STOP`**（`cgr_arm.status`、`cgr_arm.pass = false`） |

### 8.3 三条退化判据（逐条 `observed`）

| 规则 | 阈值 | `observed` | 命中 | 落盘字段 |
|---|---|---|---|---|
| `unique_frac <= 0.01` | `0.01` | `0.003909850230414747` | ✅ | `cgr_arm.degeneracy.checks.unique_frac_le_0.01` |
| `weight_l1 < 1e-6` | `1e-06` | `0.0` | ✅ | `cgr_arm.degeneracy.checks.weight_l1_lt_1e-6` |
| `grad_norm == 0` | `0.0` | `0.0` | ✅ | `cgr_arm.degeneracy.checks.grad_norm_eq_0` |

`cgr_arm.confidence_grad_norm`：`steps = 3900`（= 5 epoch × 780 步）、**`grad_nonzero_steps = 0`**、
`grad_norm_mean = grad_norm_max = 0.0`、`n_samples = 997615`（= 5 × 199523）。

### 8.4 签名读数（`cgr_arm.diagnostics`，best_state 载入后、只用 val 前向一遍）

| 读数 | 实测 | 对应缺陷 | 读法 |
|---|---|---|---|
| `g_mean` / `g_min` / `g_max` | `1.0` / `1.0` / `1.0` | D3+D4 | 全 val 集（49881 样本）门控恒为"完全信任 attention" |
| `g_raw_mean` / `_min` / `_max` | `1.0` / `1.0` / `1.0` | **D3** | **warmup 之前**就已饱和在 `1.0` |
| `g_within_batch_std_mean` / `_max` | `0.0` / `0.0` | **D1** | 批内**零**方差——不是"几乎相同"，是同一个标量 |
| `g_within_batch_unique_frac_mean` / `_max` | `0.003909850230414747` / `0.004608294930875576` | **D1** | 见 8.5(0) 的逐位吻合 |
| `weight_l1_mean` / `_max` | `0.0` / `0.0` | D3+D4 | `W ≡ W_attn` **逐样本**成立 ⇒ CGR 就是纯 attention |
| `presigmoid_abs_mean` / `_max` | `129.1841770734294` / `136.81227111816406` | **D3** | 死区深处（`\|z\| ≈ 130`） |
| `presigmoid_saturation_frac` | `1.0`（阈值 `8.0`） | **D3** | **100%** 样本饱和 |
| `warmup_alpha_at_eval` | `1.0` | **D4** | best epoch = 5 = 末轮 ⇒ alpha 停在 `1.0` |
| `hidden_row_symmetry_deviation` / `_max` | `0.0` / `0.0` | **D3** | 32 个隐层行逐位相同 |
| `hidden_rows_identical` | `true` | **D3** | 对称**从未**破缺 |
| `n_samples` / `n_batches` | `49881` / `195` | — | val 集规模（`49881 = 194×256 + 217`） |

逐 epoch 轨迹（`cgr_arm.train_gate_trace`，每 epoch 780 步、全部 `grad_nonzero_steps = 0`）：

| epoch | `alpha` | `g_mean` | `g_raw_mean` | `weight_l1_mean` | `presigmoid_abs_mean` |
|---|---|---|---|---|---|
| 1 | `0.2` | `0.6000000238418579` | `1.0` | `0.17877413446883178` | `131.37035455775097` |
| 2 | `0.4` | `0.7000000476837158` | `1.0` | `0.15059509225563128` | `131.33246949622128` |
| 3 | `0.6` | `0.800000011920929` | `1.0` | `0.09816984489010283` | `131.2985738078212` |
| 4 | `0.8` | `0.9000000357627869` | `1.0` | `0.049290217522036994` | `131.26876834629957` |
| 5 | `1.0` | `1.0` | `1.0` | `0.0` | `131.24511284456153` |

各 epoch 的批内读数与上表一致：`g_within_batch_std_* = 0.0`、
`unique_frac_mean = 0.0039141920001295`、`unique_frac_max = 0.010101010101010102`、`saturation_frac = 1.0`。

### 8.5 解释

**(0) 判据 1 的观测值与"批级常量"的结构预测逐位吻合**——这是本次最硬的一条证据。
val 集 `49881 = 194×256 + 217`。若每批内 `g` 是同一个标量，则每批 `unique(g)/批大小 = 1/批大小`，于是

- `unique_frac_mean = (194 × 1/256 + 1/217) / 195 = 0.003909850230414747`（与落盘**逐位相同**）
- `unique_frac_max  = 1/217 = 0.004608294930875576`（与落盘**逐位相同**）

训练期同理：`199523 = 779×256 + 99` ⇒ `mean = (779/256 + 1/99)/780 = 0.0039141920001295`、
`max = 1/99 = 0.010101010101010102`（均与落盘逐位相同）。
门控在批内**没有任何逐样本分化，连一位都没有**——D1 的结构判断被实测坐实。

**(1) 门控在第一个 epoch 之前就死在饱和区，且梯度是浮点意义上的恒 0。**
`g_raw ≡ 1.0` 从 epoch 1 第一步起成立（3900 步无一例外，`g_raw_min = g_raw_max = 1.0`），
`|z|` 的样本均值 `129.18`、最大 `136.81`、`saturation_frac = 1.0`。
float32 下 `σ′(129) = e^{−129}/(1+e^{−129})² ≈ 1e−56`，而 `e^{−129}` 已低于最小**次正规**数
（≈ `1.4e−45`）⇒ 直接下溢为 `0` ⇒ `σ′` 与 `σ(1−σ)` 两种写法**都精确等于 0**。
反传到 `confidence_mlp` 的梯度因此**精确为 0**，不是"很小"。判据 3 的
`grad_norm_max = 0.0` / `grad_nonzero_steps = 0` 是这一点的直接读数。
D5 的历史措辞是"梯度信号极弱"；本复现给出的更极端的结论是：**信号在 float32 里不存在**。

> **对 §3 / D3 描述的一处实测修正**（不改变任何判定）：D3 把"起点"估为 `g ≈ 0.88–0.99`，
> 第 10 节的测试夹具也断在"`g_raw ≥ 0.85`、**未**浮点饱和"。真实数据上饱和**更极端**：
> `|z| ≈ 130` ⇒ `g_raw` **精确等于 `1.0`**。原因是 `confidence_mlp` 的输入（`e_new`、两个 env
> 的 embedding 矩、`gen_rep` 的矩，共 640 维）在真实量级下远大于夹具量级，而 `W1 = 0.01`、
> `W2 = 0.1`、`b2 = 2.0` 的常量初始化把 32 个同构单元的和放大到百量级。
> 由于梯度恒 0，`g_raw` 在整个训练过程中**一位都没动过**——它不是"漂到"饱和，而是**出生即在饱和**。

**(2) 逐 epoch 的 `g` 轨迹 100% 由 warmup 解释，门控自身零贡献。**
`g_mean = 0.6 / 0.7 / 0.8 / 0.9 / 1.0` 与 `alpha = 0.2 / 0.4 / 0.6 / 0.8 / 1.0` 满足第 4 节公式
`g = α·g_raw + (1−α)·0.5`；代入 `g_raw ≡ 1.0` 得 `g = 0.5 + 0.5α`——五个值**全部逐位吻合**。
跨 epoch 的变化没有一个比特来自门控"学到"的东西，全部来自 runner 写 `alpha`（D4）。

**(3) `weight_l1` 的衰减同样是 warmup 的算术，不是机制。**
`weight_l1 = (1−g)·‖W_attn − W_fw‖₁ = 0.5(1−α)·‖W_attn − W_fw‖₁`。由逐 epoch `weight_l1_mean`
反解出 `‖W_attn − W_fw‖₁ ≈ 0.447 / 0.502 / 0.491 / 0.493`（epoch 1–4）：除 epoch 1→2 的一次跳变
（被训练的 attention 分支自身在收敛）外，整条衰减曲线由 `(1−α)` 决定；到 `α = 1.0` 时
`weight_l1 ≡ 0` 是**恒等式**（`(1−α) = 0` 乘任何有限数），不是"门控学会了选 attention"。

**(4) 在被评测的那个模型上，CGR 臂的 forward 与基线臂等价。**
`weight_l1_max = 0.0` 意味着**逐样本** `W ≡ W_attn`（不只是平均为 0）；而 `W` 之后的
`env_aware_rep → gate_network → all_reps → fused_rep → tower_network` 与基线 `NewTask.forward`
**逐行相同**（`census_benchmark/cgr.py:161`–`169` vs `multitaskrec/model.py:915`–`930`）。
再加上 `confidence_mlp` 梯度恒 0 ⇒ Adam 对它的更新亦恒 0（`m = v = 0` ⇒ `lr·0/(0+ε) = 0`）
⇒ 该模块**完全惰性**，既不改变自身也不影响其他参数（Adam 逐参数独立）。

> **因此 Δ = −0.0027 只能归因于 D2 的初始化位移**（`confidence_mlp` 插队多抽了两组
> `nn.Linear` 的 RNG ⇒ `gate_network` / `tower_network` 与基线拿到不同的初值）**与训练随机性**；
> **不能**读成"CGR 比基线差 0.0027"，更不能读成"CGR 未达 0.8521"或"达到了某水平"。
> 这正是 §3 与 D2 早已警告的：历史"CGR vs Prompt"的对比**不是单变量对比**。
>
> *证据强度声明*：第 (4) 点的前半（`W ≡ W_attn` 逐样本成立、`confidence_mlp` 惰性）由落盘诊断
> 与代码逐行对照直接支持；"两臂 forward 等价"是由此推出的结论，本轮**没有**做逐位前向等价性
> 测试来直接验证（那需要额外的 A/B 夹具，属新工作）。

**(5) 三条判据不是三份独立证据，而是两条根因链的侧面。**
主链：`常量初始化 (D3) → |z| ≈ 130 硬饱和 → σ′ 在 float32 下溢为 0 → 梯度恒 0（判据 3）
→ 门控冻结在 g_raw = 1.0 → α = 1 时 W ≡ W_attn（判据 2）`。
旁链：`批级门控输入 (D1) → 批内 g 唯一（判据 1）`。

**关键区别：判据 1 是架构性的，判据 2/3 是实现/数值性的。** 只要门控输入仍是批级统计量，
`unique_frac` 就**恒等于** `1/批大小`，与训练多久、什么 seed、什么 lr 全部无关。
所以本臂的退化**不可能**通过"多训几轮 / 换 seed / 调 lr / 调 T"绕开——§6.3 第 2 条禁止的
正是这类操作，而判据 1 的存在使这种禁止有了结构性依据，不只是纪律要求。

### 8.6 与本节上一版预判的对照（留痕）

上一版 §8 的预判原文：

> 忠实实现的 `g` 是批级常量 ⇒ `g_within_batch_std_mean = 0`、
> `g_within_batch_unique_frac_mean ≈ 1/256 = 0.0039 ≤ 0.01` ⇒ 判据 1 预期命中，臂状态预期为 `STOP`。

**实际**：判据 1 命中，且观测值与 `1/批大小` 的批加权平均**逐位吻合**（8.5(0)）；
**另外两条判据也一并命中**，`hidden_rows_identical = true`、`saturation_frac = 1.0`。
预判的**方向正确但低估了退化程度**：D3/D5 造成的硬饱和（判据 2/3）比"批级常量"这一条更致命，
而上一版只预期了后者。上一版设的"若未命中则优先排查实现偏离"的分支**未被触发**。

### 8.7 A/B 门禁（`gate_report.json`）

| A1 | A2 | A4 | A5 | B1 | B2 | B3 | B4 | `overall_pass` |
|---|---|---|---|---|---|---|---|---|
| PASS | PASS | PASS | PASS | PASS | PASS | **FAIL** | PASS | **`false`** |

- **A3 = `on_demand`**：按协议按需触发，本轮未跑（不阻塞，见第 7 节）。
- **B3 是继承性失败，与本臂无关。** B3 判的是 `mechanism.gate_mean` 每一维 ∈ `[0.05, 0.95]`，
  本 run 为 `[0.952401451979339, 0.8543901456350915]`——与基线 run `20260929-1735-…-904f8d0`
  **逐位相同**（csb / pattn 两条臂的 `gate_mean` 亦为同值）。该读数来自**冻结 backbone 的
  `gate_networks`**（stage1 产物），由构造决定，与处理臂无关；基线行在 `SUMMARY.md` 里同样是 B3 FAIL。
- 其余门禁：A1（backbone sha 前后相同、backbone 无梯度）、A2（划分指纹一致）、A4（划分互斥且完备）、
  A5（`env_ids` 指纹匹配 stage1）均 PASS；B2 的 val/test gap = `0.0022684966113615257`（限 `0.03`）；
  B1 三项均远高于 `0.6` 地板；B4 两个 env 占比 `0.494 / 0.506`。
- 臂级判定（`cgr_arm`）**未**并入 `overall_pass`，与 §6.2 一致。**注意**：本 run 的
  `overall_pass = false` 完全由**继承的** B3 造成，因此**不能**把 `overall_pass` 当作本臂的失败判据——
  本臂的判据是 `cgr_arm.degeneracy` / `cgr_arm.status`。

### 8.8 未做的事（按 §9.2 处置，不是疏漏）

- **不做多 seed / 不做 `full` tag / 不跑 AliCCP·ByteRec / 不做 FLOPs**（§6.3 第 3 条、§11）。
- **不做调参续命**：不换 `batch_size` / `T` / 初始化 / model seed / 划分（§6.3 第 2 条）。
- **不合并 `master`**；失败结果、run 记录、判据结论随本分支保留并推送（§9.3）。
- 权重的处置沿用协议 9.3：`*.pt` / `*.json` / `*.log` 不入库（`.gitignore` 天然拦截），
  `SUMMARY.md` 行与本文件入库。

---

## 9. 非新颖性与结果处置（保留策略）

### 9.1 非新颖性

- 置信门控 / 条件化路由 / 门控融合（gated fusion）是**已有大类**（MoE 门控、FiLM、条件化归一化、
  prompt 调制同族）；CGR 这个名字与这个实现都**不是**本项目的贡献，本分支**不主张任何方法新颖性**。
- 历史数据本身已经说明问题：CensusIncome 上 cgr `0.8618 ± 0.0048` **低于** prompt `0.8629`；
  AliCCP 上 `0.6632 ± 0.0241` 也是最差之一。本轮只是把"为什么"变成**可复核的判据与读数**。
- 因此：本分支的结论**只能**写成"复现与诊断"，不得写进论文的方法贡献，也不得当作新 baseline。

### 9.2 结果处置（预注册）

| 情形 | 处置 |
|---|---|
| `CONFIRMED_DEGENERATE` | 记录退化证据（逐条 `observed` + 签名读数）→ **方向停止**：不做多 seed、不做调参续命、不合并 `master` |
| 非退化且 `AUC-Test-Education >= 0.8521` | 如实记录该单 seed 结果；**仍不合并** `master`（是否扩多 seed 由用户显式决定，属新一轮预注册） |
| 非退化且 `< 0.8521` | 同上：记录"忠实复现后未达效应阈值"→ 方向停止 |

### 9.3 保留策略（用户明确要求）

1. **失败结果必须留在 `exp/stage2-cgr-repro` 分支并推送**：run 记录、判据结论、本文件一律随分支提交
   （协议 7.3：不得只报告成功的 run）。**不得**因结果为负而删除分支或删改结论。
2. **永不合并 `master`**：无论成功失败，本复现都不进 `master`（`master` 是干净基线）。
3. **不得事后改口径**：判据、阈值、指标定义在看到结果后一律冻结；如需新定义，另开分支重新预注册。
4. 权重的处置沿用协议 9.3：`*.pt` / `*.json` / `*.log` 不入库（`.gitignore` 天然拦截），
   `SUMMARY.md` 行与本文件入库。
5. 不删除、不改写已有 run 目录；`run_id` 含时间戳，复跑天然新目录。

---

## 10. 测试（TDD 契约，先写测试后写实现）

实现：`census_benchmark/cgr.py`；测试：`census_benchmark/tests/test_cgr.py`（stdlib `unittest`）。

```powershell
.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests -v
```

| 组 | 用例（要点） |
|---|---|
| 静态守卫 | 源码可被本解释器编译；`multitaskrec` / `config.py` / `protocol.py` / `metrics.py` 无 diff |
| 预注册判据 | 阈值常量；三条规则的**含端点**边界（`<= 0.01`、`< 1e-6`、`== 0`）；退化 ⇒ `STOP` 且 `effect.evaluated = false`；payload 可 JSON 落盘 |
| 忠实构造 | 常量初始化逐位；**RNG 混入证据**（多抽之前的模块与基线逐位相同、之后的不同）；`state_dict` 键集只多 `confidence_mlp`；隐层行在初始化时**完全对称** |
| 忠实公式 | `W_attn / W_fw / z / g_raw / g / W` 与手写复算**逐位**一致；整条管线逐位一致；`alpha=0 ⇒ g ≡ 0.5` |
| 结构缺陷证据 | `g` **批内唯一**、改动任一样本改变整批 `g`、同一样本换批即换 `g`；**D6**：单样本批 ⇒ `g = NaN` |
| 饱和行为 | 真实量级（`rep_dim=128`）下历史初始化给出**饱和地板**（`g_raw ≥ 0.85`、未浮点饱和）；硬饱和夹具（`sigmoid(30) = 1.0`）下 `W ≡ W_attn`、`weight_l1 = 0` ⇒ 判据 2 命中。**不**对任意小夹具断言 sigmoid 恰好等于 1 |
| 累加器口径 | 手算逐字段核对；样本级聚合与分批无关（批级读数按批计，显式断言其依赖分批）；空累加器零安全；`json.dumps` 可落盘 |
| 梯度轨迹 | 与手算梯度范数一致；归零 ⇒ 0；空轨迹 ⇒ `grad_norm_max = 0` ⇒ 判据 3 命中；epoch 记账错误显式报错；基线头被拒 |
| 诊断函数 | 与独立复算一致、参数/backbone 逐位不变、无残留梯度、`warmup_alpha_at_eval` 留痕；基线头被拒 |
| 接线与隔离 | 变体工厂逐位同 master；CLI 默认与 `--cgr` 别名；处理臂 `-cgr` 后缀 + `cgr_arm` 全字段；**基线臂**无后缀、无 `cgr_arm`、`mechanism` 键集不变、`mock` 哨兵证明基线路径不构造 CGR 头 |

**执行状态**：测试已在实现之前写成（TDD：先失败），并**已执行通过**（见下）。真实实验（第 7 节命令）
**已执行**，结果见第 8 节（`CONFIRMED_DEGENERATE` ⇒ `STOP`）。

| 命令 | 结果 |
|---|---|
| `.venv\Scripts\python.exe -m unittest discover -s census_benchmark/tests -t census_benchmark/tests` | **`Ran 65 tests ... OK`**（`test_cgr` 42 + `test_protocol` 8 + `test_metrics` 10 + `test_smoke` 5） |

实现过程中被测试**当场拦下**并修正的三处（记录在此，便于复核测试确有鉴别力）：
① `tower_network` 的真实键名是 `mlp.linear0/1/2.*`（不是数字索引）；
② 真实量级夹具必须补 11 个 dense 特征才凑满 123 维（`EmbeddingNetwork` 直接拼接非 vocab 列）；
③ 常量初始化把门控推到饱和，微小扰动下 `Δg` 只有 1e-5~1e-6 量级——用大扰动（+50）断"结构依赖"、
用 1e-5 阈值断"依赖存在"，并显式记录该敏感度限制，而不是写一个靠夹具运气过的强断言。

---

## 11. 明确的非目标

- 不改 `multitaskrec/model.py` / `config.py` / `census_benchmark/protocol.py` / `census_benchmark/metrics.py`。
- 不做任何"改进 CGR"的尝试（逐样本门控、温度调整、初始化修复、正则化、低秩化……）——那属新预注册。
- 不跑 AliCCP / ByteRec；不做多 seed；不做全量 `full` tag；不做 FLOPs；不产出论文级性能结论。
- 不主张新颖性；不把本分支结论写成方法贡献。
