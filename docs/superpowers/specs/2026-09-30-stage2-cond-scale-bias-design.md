# 阶段 2 条件化 Scale+Bias（消融）设计与前置门槛

- **状态**：第 0 阶段（零训练前置诊断）**已执行且门槛 PASS**（见第 7 节）；第 1 阶段（条件化 Scale+Bias）**已执行，臂判定 FAIL（C1 未过 / C2 通过）→ 方向停止**，本分支**不合并 `master`**（见第 6、7 节）
- **日期**：2026-09-30
- **适用分支**：`exp/stage2-cond-scale-bias`（从 `infra/fair-stage2-benchmark` @ `87afe03` 独立拉出）
- **唯一事实来源**：本文件。任何与本文件冲突的实验做法以本文件为准；如需变更先改本文件再改代码。
- **上游协议**：`docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md`（下称"评测协议"）。本文件**只引用**它，不修改它；划分、种子、冻结三件套、门禁 A/B 类一律沿用。

---

## 1. 定位：只属消融 / 工程改进

**本方向不是论文创新点，只是 MPT-Rec 的一个消融实验与工程改进**，用来回答一个具体问题：

> 新任务头里那个把 `spec_reps` 按 env 加权融合的 **router W**，在基线阶段 2 训练完之后
> （`runs/<run_id>/newtask.pt`，即实际参与评测的那个头）到底有没有样本级差异？
> 如果没有，任何"条件化 router"的设计都无从谈起。

因此：

| 编号 | 约束 |
|---|---|
| P1 | 本分支的结论**只用于消融**；不写进论文的方法贡献，不宣称任何性能提升。 |
| P2 | **不改模型语义**：`multitaskrec/model.py`、`config.py`、`baseline/`、`CensusIncome_*.py` 一律不动（与评测协议第 10 节同口径）。 |
| P3 | **零训练前置诊断**：第 0 阶段只做前向复算，不训练、不反传、不更新任何参数、不写回 checkpoint。 |
| P4 | 第 0 阶段先于第 1 阶段：条件化 Scale+Bias 本体（第 5 节）在门槛通过前不写一行代码。**门槛已 PASS（第 7 节），P4 已解除**——第 1 阶段现按第 5 节实现。 |
| P5 | 无论成功或失败，本分支**只留本分支**，不合并回 `master`（第 6 节）。 |

---

## 2. 第 0 阶段：零训练前置诊断

实现：`census_benchmark/router_probe.py`；测试：`census_benchmark/tests/test_router_probe.py`。

### 2.1 诊断对象与公式（与 `NewTask.forward` 逐行同构）

设

| 符号 | 含义 | 取值 |
|---|---|---|
| `dnn_input` | 冻结 backbone 的 `EmbeddingNetwork` 输出 | $\mathbb{R}^{d}$，$d=123$ |
| $H$ | `projection_network(dnn_input)`，末端是 `LayerNorm` | $\mathbb{R}^{r}$，$r=128$ |
| $E$ | $[\mathbf{e}_0, \mathbf{e}_1]$，$\mathbf{e}_k=$ 冻结 backbone 的 `env_embedding_network(k)` | $\mathbb{R}^{r\times K}$，$K=2$ |
| $T$ | `NewTask.temperature` | $150$ |

则 router（**本次诊断复算的量**，不做任何修改）：

$$z = H E / T, \qquad W = \mathrm{softmax}(z) \in \Delta^{K-1}$$

其中 softmax 沿 env 维（`dim=-1`）。$W[b,k]$ 即"样本 $b$ 把多少权重分给 env $k$"。

**为什么值得先测**：$H$ 末端是 `LayerNorm`（每行零均值、单位方差），$E$ 是训练出来的 env embedding，
而 $T=150$ 把内积压小两个数量级——$W$ 很可能在 **0.5 附近几乎不动**。若真如此，"按 $W$ 条件化"
就是给一个常量做条件，整个方向不成立。这正是"先诊断、再动手"的原因。

**关键约束（诊断对象与口径保真）**：诊断的是**基线 run 训练后的新任务头**——即
`--newtask-checkpoint`（必填）指向的 `runs/<run_id>/newtask.pt`，不是随机初始化的头
（随机头代表不了基线阶段 2 实际在用的 router）。取用方式：

```
seed_model(model_seed) → build_mptrec(device) → to(device) → load_state_dict → freeze → NewTask(...)
                                                                                        ↓
                                                                     strict 载入 newtask.pt
```

前六步与 `run_census_benchmark.run_stage2` 的构造顺序**完全一致**（其中只有 `seed_model`、
`build_mptrec`、`NewTask(...)` 三步消耗全局 RNG，`to` / `load_state_dict` / `freeze` 都不消耗），
随后 strict 载入训练后的权重。保持同一构造顺序的意义是**结构保真**：只要 vocab / `input_size` /
`rep_dim` / tower 超参有任何漂移，载入就会因键或形状不符而**报错**，而不是静默地诊断到另一个头。
checkpoint 缺失 → `FileNotFoundError`；不是张量 state_dict、键不符、形状不符 → `ValueError`，一律拒绝出数。
测试用正反两例锁定构造顺序（正例断言逐参数 `torch.equal` 且 RNG 终点逐位一致，反例断言"把 NewTask
建在 backbone 之前 → 初始化必然改变"），并单独锁定：不兼容 checkpoint 必须报错、diagnose 的是
checkpoint 里的头而不是随机初始化的头。

### 2.2 硬约束与证据

| 约束 | 落地方式 | 落盘证据字段 |
|---|---|---|
| 零训练 | 无优化器、无 `backward`、`@torch.no_grad()` 抽表征 | `trained_steps: 0` |
| 真冻结 | 复用评测协议的 `freeze_backbone`（参数级 + 模式级） | `frozen: true` |
| 无残留梯度 | `assert_no_grads`（协议同一函数） | `grads_all_none: true` |
| backbone 未变 | 前后 `backbone_sha256` 比对（协议同一函数） | `backbone_sha256_before == _after` |
| checkpoint 只读 | 只读 `stage1/` 与 `--newtask-checkpoint`，产物只写 `probes/`；测试断言两个 `.pt` 字节不变、`runs/` 不被创建 | —— |
| 诊断对象可查 | `--newtask-checkpoint` 必填，strict 载入；缺失 / 非 state_dict / 键不符 / 形状不符一律拒绝运行 | `newtask_checkpoint`、`newtask_checkpoint_sha256`、`newtask_checkpoint_state_sha256`、`newtask_state_sha256`（后两者必须相等） |
| 新任务头口径归属可查 | checkpoint 旁若有同 run 的 `config.json`，校验 `stage1_id` / `model_seed` 与 `--stage1-dir` 一致（缺失即跳过，不一致即拒绝） | `newtask_run_config` |
| 同一固定划分 | 用 stage1 的 `split_seed` 重建并双向校验指纹（A2） | `split_ok`、`split_fingerprint_sha256`、`split_sha256` |
| 口径归属可查 | 目录名与 `meta.stage1_id` 不一致即拒绝运行 | `stage1_id`、`stage1_backbone_sha256(_matches)` |

**固定输入**：stage1 产物 `s1-096f8f16-m1685480945-e2-cb2094b3`（`model_seed=1685480945`、
`split_seed=20260929`、`epochs=2`、`input_size=123`、`rep_dim=128`），划分 = `splits/20260929`；
新任务头 = 基线 run `20260929-1735-s20260929-m1685480945-short-904f8d0` 的 `newtask.pt`。

---

## 3. 四项指标与预注册门槛

### 3.1 指标定义（$N$ 个样本，$K$ 个 env）

| 指标 | 定义 | 读法 |
|---|---|---|
| `router_w_std` | $\frac{1}{K}\sum_k \sqrt{\frac{1}{N}\sum_b W_{bk}^2 - \left(\frac{1}{N}\sum_b W_{bk}\right)^2}$ | 各 env 列在**样本维度**的总体标准差（ddof=0）之均值。**这就是"router 有没有样本差异"** |
| `normalized_entropy` | $\frac{1}{N}\sum_b \frac{H(W_b)}{\log K}$，$H(p)=-\sum_k p_k\log p_k$（约定 $0\log 0 = 0$） | 1 = 每样本都均匀（无信息）；0 = 每样本都独热（极尖） |
| `source_share` | $\frac{1}{N}\sum_b \mathbb{1}[\arg\max_k W_{bk} = k]$，长度 $K$ | 各 env 被选中的样本占比，和为 1；**并列时取最小下标**（torch 语义） |
| `top1_share` | $\frac{1}{N}\sum_b \max_k W_{bk}$ | 平均 top-1 权重（路由置信度）；均匀路由时为 $1/K$ |

另落盘 `router_w_std_per_env`（逐 env 的 std，便于检查是否只有单侧在动）与 `n_samples`。
`K<2` 时归一化熵无定义，直接 `ValueError` 拒绝出数。

### 3.2 预注册门槛

```
router_w_std >= 0.02   →  PASS：允许进入第 1 阶段（条件化 Scale+Bias，5 epoch）
router_w_std <  0.02   →  FAIL：方向停止，不再投入
```

- **为什么是 0.02**：$K=2$ 时 $W$ 的两列互补，标准差即"单个样本的 env 权重偏离 0.5 的典型幅度"。
  0.02 意味着典型样本约把 51%/49% 的权重分给两侧——低于这个量级，条件化模块能接收到的
  **样本间信号**与数值噪声同阶，任何下游增益都无法与随机性区分。
- **含等号**：`>= 0.02` 判 PASS（测试锁定边界）。

### 3.3 阈值纪律（与评测协议的阈值纪律同款）

1. 阈值只允许在**看到本次诊断结果之前**修改；每次修改单独记录理由。
2. **禁止事后调参救门槛**：不得为了过门槛而换 model seed、改 $T$、改 `projection_network`
   初始化、或换 val/test 划分。门槛未过就是未过——这正是"预注册"的全部意义。
3. 门槛未过时，仍须把诊断结果落盘留痕（**不得只报告通过的 run**）。

---

## 4. 产物

```
artifacts/census_stage2/probes/<probe_id>/router_probe.json
# probe_id = rp-<stage1_id>-<val|test>-<新任务头状态哈希前8>
```

- 内容：四项指标 + 逐 env std + 样本数、门槛判定（`rule` / `threshold` / `observed` / `proceed`）、
  诊断对象证据（`newtask_checkpoint` 路径 / 文件哈希 / 状态哈希 / `newtask_run_config`）、
  以及第 2.2 节全部证据字段与代码 commit。
- `*.json` 已被现有 `.gitignore` 天然拦截；另在 `.gitignore` 显式登记 `artifacts/census_stage2/probes`。
- **不写入 `SUMMARY.md`**：该表是评测协议的 AUC 门禁台账（列为 A/B 类门禁），前置诊断没有 AUC，
  混进去会污染协议台账。
- 重跑同一 `probe_id` 会覆盖同名 JSON。这是可接受的：诊断是
  `(stage1 产物, 新任务头 checkpoint, split seed, model seed, 代码)` 的纯确定性函数（测试用两次运行结果全等锁定），
  覆盖无法"洗白"结果——换 checkpoint 就是换 `probe_id`（状态哈希入名），不会用新头覆盖旧头的结论；
  门槛判定一旦作出即记入本文件第 7 节。

**运行命令**：

```bash
python -m census_benchmark.router_probe \
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 \
    --newtask-checkpoint artifacts/census_stage2/runs/20260929-1735-s20260929-m1685480945-short-904f8d0/newtask.pt \
    --gpu 0
# 可选：--split test（默认 val）、--root（默认 artifacts/census_stage2）
```

---

## 5. 第 1 阶段：条件化 Scale+Bias（已执行，C1 未过 → 方向停止）

**门槛已 PASS**（第 7 节）。唯一变量是新任务头里的 router 条件化调制；其余（冻结 backbone、5 epoch、
`lr=1e-3`、`batch_size=256`、`patience=2`、loss、val 选点、test 只评一次）全部沿用评测协议的 stage2 配置。
本阶段已于 2026-09-30 单 model seed 执行一次，**C1 FAIL / C2 PASS → 方向停止**；真实数字与读法见第 7 节，
处置按第 6 节"门槛 PASS 但 C1 未过"：不合并 `master`、不做二次调参续命。

实现：`census_benchmark/cond_scale_bias.py`；测试：`census_benchmark/tests/test_cond_scale_bias.py`；
接入：`run_census_benchmark.py stage2 --variant cond-scale-bias`（默认 `baseline`）。

### 5.1 公式

设 $x$ = `dnn_input`，$H(x)$ = `projection_network(x)`，$E=[\mathbf e_0,\mathbf e_1]\in\mathbb R^{r\times K}$
（冻结 backbone 的 `env_embedding_network(k)`，与第 2.1 节同一对象），$T=150$：

$$W(x)=\mathrm{softmax}\big(H(x)E/T\big)\in\Delta^{K-1},\qquad
e_{\text{new}}(x)=\sum_k W_k(x)\,\mathrm{spec\_rep}_k$$

$$\gamma(x)=\sum_k W_k(x)\,\gamma_k,\qquad \beta(x)=\sum_k W_k(x)\,\beta_k$$

$$\text{env\_aware}=s\odot\big(e_{\text{new}}(x)+\gamma(x)\big)+\beta(x)$$

| 记号 | 实现对应 | 处置 |
|---|---|---|
| $W(x)$ | `softmax(mm(projection_network(x), stack(env_embs,1)) / temperature)` | **既有源 router，原样保留**（与 `router_probe.router_weights` 同一表达式，测试锁定逐位一致） |
| $e_{\text{new}}(x)$ | `matmul(stack(spec_reps, 2), W)`（= 基线 `new_spec_rep`） | **既有 router 融合，原样保留** |
| $s$ | `env_embedding_network(new_env_idx)`（= 基线 `new_env_emb`） | **既有 env scale，原样保留**；不随样本变化（$s(x)\equiv s$） |
| $\gamma_k,\beta_k$ | `gamma` / `beta`：`nn.Parameter(zeros(K, r))` | **本消融的唯一新增量**，$K=2$、$r=128$ |

- **恒等回退（零初始化）**：$\gamma=\beta=0\Rightarrow\text{env\_aware}=s\odot e_{\text{new}}$，与基线**逐位一致**。
  测试锁定三层：同种子构造下共享参数逐位相同且全局 RNG 终点一致；同一组权重下前向逐位相同；
  首个训练步里共享参数的梯度逐位相同。
- 其余结构（`gate_network` 融合、`tower_network`、`get_l2_reg`、温度、`new_env_idx`）一律沿用基线，不含新增损失项。
- **$\gamma/\beta$ 的容量与正则**：$\gamma,\beta\in\mathbb R^{K\times r}$ 是**满宽**参数（$2\times2\times128=512$ 个）
  且**不进 L2 正则**（`get_l2_reg` 未改，只正则 tower——与基线同口径）。这条在判定时必须一起看：
  若 C1 通过但 $\|\gamma(x)\|$ 与 $\|e_{\text{new}}\|$ 同阶（见 5.2 上界），说明收益可能来自"多了一组可学参数"
  而非"条件化"本身，该结论只能算工程改进，**不得**当作条件化有效的证据。

### 5.2 有效调制比 `modulation_ratio` 与取值带

原第 5.2 节的 `prompt_ratio := r_p / r` 是**瓶颈宽度比**，适用于旧稿（在 router logits 上做低秩调制）。
本稿的 $\gamma/\beta$ 是满宽、逐 env 的参数，**不存在瓶颈宽度 $r_p$**，故 `prompt_ratio` 不再适用；
C2 改由**有效调制比**承担，判据形式与理由不变（下界=调制不能是惰性的，上界=调制必须仍是 prompt 量级）：

```
gamma_x_norm_mean := mean_b ||gamma(x)_b||_2,  beta_x_norm_mean := mean_b ||beta(x)_b||_2
e_new_norm_mean   := mean_b ||e_new(x)_b||_2
modulation_ratio  := gamma_x_norm_mean / e_new_norm_mean      # 相对幅度（无量纲）
```

**允许带 `[0.01, 1.0]`（含端点）**：低于 0.01 说明调制与数值噪声同阶（等价于什么都没做）；
高于 1.0 说明调制项淹没了它要调制的表征（不再是"prompt 量级"的调制器）。
`e_new_norm_mean = 0`（退化表征）时比率无定义：$\gamma$ 也为 0 记 0.0，否则记 `inf`（判失败），测试锁定。
落盘同时给出 `beta_x_norm_mean` / `bias_ratio` 与逐 env 参数范数 `gamma_param_norm_per_env` /
`beta_param_norm_per_env`，便于分辨"只有单侧在动"与"参数是否真的离开零点"。

### 5.3 成功判据（预注册，两个条件**同时**满足）

| 条件 | 判据 | 落盘字段 |
|---|---|---|
| C1 有效性 | `AUC-Test-Education >= 0.8521` | `metrics.json:csb_arm.C1` |
| C2 规模合规 | `modulation_ratio ∈ [0.01, 1.0]` | `metrics.json:csb_arm.C2` |

- 参照：`infra` 分支的基线 run `20260929-1735-s20260929-m1685480945-short-904f8d0` 在 `SUMMARY.md`
  记录的 `auc_test_education = 0.850069`。0.8521 是在该基线上的小幅提升门槛（用户预注册值）。
- C2 在 **val** 上测（与 M3/M4 同一次机制评估的口径）；它是**机制 sanity 带**，不是性能门禁：
  带内不构成通过，只说明"调制幅度在可解释区间内"，出带即判失败并记录。
- 判定沿用评测协议的门禁脚本与 A/B 类判据；本消融**不得**在 val 上反复挑点、**不得**改划分。
- **单次判定**：本轮只跑单个 model seed（`1685480945`）。任何"多 seed 平均后才达标"的结论不算通过 C1。
- **判定写入位置**：`metrics.json` 的 `csb_arm`（臂级结论），**不并入** `judge()` 的 `overall_pass`
  （A/B 门禁语义不变，`protocol.py` 零改动）；`SUMMARY.md` 的 AUC 台账照常按协议追加行
  （`run_id` 带 `-csb` 后缀，与基线 run 天然可区分）。

### 5.4 为什么以子类扩展而不是改 `model.py`

`multitaskrec/model.py` 一律不动（P2）：条件化头是 `NewTask` 的**子类**
（`census_benchmark/cond_scale_bias.py::CondScaleBiasNewTask`），只重写 `forward` 并新增 $\gamma/\beta$。
这样 `master`/其它分支的 `NewTask` 与基线阶段 2 路径**不可能**被本消融改动；
基线臂的 `config.json` / `metrics.json` / `gate_report.json` / `SUMMARY.md` 语义与键集保持不变
（只多一个恒为 `"baseline"` 的 `variant` 字段，见 5.5）。

### 5.5 与评测协议的接线（最小改动）

| 位置 | 改动 |
|---|---|
| `run_census_benchmark.run_stage2` | 新增 `variant="baseline"` 参数；用 `cond_scale_bias.build_newtask(variant, ...)` 构造新任务头 |
| `run_id` | 处理臂追加后缀 `-csb`（沿用 null-expert/attn 分支的"协议 run_id + 臂后缀"约定） |
| `config.json` | 两臂都记 `variant`；处理臂另记 `num_envs` |
| `metrics.json` | 两臂都记 `variant`；处理臂另记 `csb_arm`（C1/C2 判定）与 `mechanism` 里的调制诊断 |
| `mechanism` 键集 | **处理臂才追加** `gamma_x_norm_mean` / `beta_x_norm_mean` / `e_new_norm_mean` / `modulation_ratio` / `bias_ratio` / 逐 env 参数范数 / `n_samples`；基线臂键集与原来完全一致 |
| CLI | `stage2 --variant {baseline,cond-scale-bias}`，默认 `baseline` |
| `protocol.py` / `metrics.py` / `model.py` / `config.py` | **零改动** |

**调制诊断的公平性**：诊断在**选点完成、`best_state` 已载入之后**、只用 val 跑一遍 `no_grad` 前向
（`evaluate_modulation`），不训练、不反传、不参与 early stop / 选点 / test 评测，因此不可能改变
A/B 门禁或 C1 的结果；基线臂根本不跑这段。

**运行命令**（标准 short 命令 + 一个 variant 旗标）：

```bash
python run_census_benchmark.py stage2 \
    --stage1-dir artifacts/census_stage2/stage1/s1-096f8f16-m1685480945-e2-cb2094b3 \
    --variant cond-scale-bias --tag short --gpu 0
```

### 5.6 定位重申

$\gamma(x)=W(x)\Gamma$、$\beta(x)=W(x)B$ 属**条件化调制**这一大类（FiLM / 条件化归一化 / prompt 调制的
同族做法），本方向**不主张任何方法新颖性**；它是一次**消融 / 工程诊断**，回答的问题是
"在已经证明有样本级差异的 router 之上，加一路条件化 scale+bias 是否带来可复现的收益"。

---

## 6. 结果处置：只留本分支

| 情形 | 处置 |
|---|---|
| 门槛 FAIL | 方向停止。本分支保留诊断代码与结论后即可删除；**不合并 `master`**，不另开分支续做。 |
| 门槛 PASS 且 C1+C2 通过 | 结论写入本文件第 7 节 + 该 run 的 `metrics.json:csb_arm`。**仍不合并 `master`**（`master` 是干净基线）。是否升级为多 seed / 正式消融，由用户显式决定。 |
| 门槛 PASS 但 C1 或 C2 未过 | 同上：记录失败（`csb_arm.pass = false` 与逐条 observed 一并留痕），**不合并**，不做二次调参续命。 |

补充：第 1 阶段的判定按 5.3 节执行——**不因结果好坏而改口径**；若 C1 未过即方向停止，
$\gamma/\beta$ 的调参（初始化、正则、满宽→低秩）属新一次预注册，不得在本轮事后追加。

补充：`master` 只接受"论文所需代码 + 运行修复"，本方向是探索性消融，符合评测协议第 3.3 节
"不推送远端；推送与合并需用户显式批准"的约定。

---

## 7. 状态与结论（第 0、1 阶段均已回填）

| 项 | 状态 |
|---|---|
| 第 0 阶段代码（`router_probe.py` + 测试） | 已实现（严格 TDD：测试先写） |
| 第 0 阶段测试执行 | 已执行：`census_benchmark.tests.test_router_probe` 全绿（含 `test_probe_consumes_real_stage2_run_checkpoint`） |
| 门槛判定结果 | **PASS**（2026-09-30，`router_w_std = 0.207444 ≥ 0.02`），明细见下表 |
| 第 1 阶段（Scale+Bias） | 已实现（`census_benchmark/cond_scale_bias.py` + `--variant cond-scale-bias`）并**已执行**（2026-09-30，单 model seed / `tag=short`），臂判定 **FAIL**（C1 未过、C2 通过）→ 方向停止（第 6 节），**不合并 `master`** |

**第 0 阶段实测（`artifacts/census_stage2/probes/rp-s1-096f8f16-m1685480945-e2-cb2094b3-val-f8f0b149/router_probe.json`）**：

| 运行时间 | probe_id | `router_w_std` | `normalized_entropy` | `source_share` | `top1_share` | 门槛判定 |
|---|---|---|---|---|---|---|
| 2026-09-30T13:10:34 | `rp-s1-096f8f16-m1685480945-e2-cb2094b3-val-f8f0b149` | 0.20744372956191676 | 0.7907991508538772 | [0.7752250355846916, 0.22477496441530842] | 0.7356200531716264 | **PASS**（`>= 0.02`） |

- 诊断对象：基线 run `20260929-1735-s20260929-m1685480945-short-904f8d0` 的 `newtask.pt`
  （状态哈希 `f8f0b149…`，与 checkpoint 逐位一致），划分 val，`n_samples = 49881`，`trained_steps = 0`，
  `frozen = true`，`grads_all_none = true`，backbone 前后哈希同为 `a12a5f53…`，`split_ok = true`。
- 逐 env std `[0.20744372955691548, 0.20744372956691803]`：两侧对称在动，不是单侧漂移。
- **读法**：$W$ 的两列互补，std ≈ 0.207 意味着典型样本把权重分到约 71%/29%（`top1_share` 0.7356 与此一致），
  远高于 0.02 的"与数值噪声同阶"界；`source_share` 0.775/0.225 说明 env 0 是多数派但 env 1 仍被 22% 的
  样本选中。**router 确实携带样本级信号 → 条件化 Scale+Bias 值得进入第 1 阶段。**

**第 1 阶段真跑结果**（2026-09-30，第 5.5 节命令 + `--variant cond-scale-bias --tag short`；单 model seed
`1685480945`、5 epoch、`patience=2`，按 5.3 单次判定；无论通过与否都要留痕，第 6 节）：

| 运行时间 | run_id | `modulation_ratio` | `AUC-Test-Education` | C1（`>= 0.8521`） | C2（`∈ [0.01, 1.0]`） | 臂判定 | A/B 门禁 |
|---|---|---|---|---|---|---|---|
| 2026-09-30T13:24 | `20260930-1324-s20260929-m1685480945-short-87afe03-csb` | 0.09219527083871916 | 0.8501177932753766 | **FAIL**（observed 0.8501177932753766 < 0.8521） | PASS | **FAIL** | 除继承的 B3 外全部 PASS |

**读法：机制确实被激活，但没有换来可复现的泛化收益（方向停止）**

- **相对基线的真实 delta ≈ 0**：同 stage1、同划分的基线 run `20260929-1735-…-904f8d0`
  `auc_test_education = 0.8500685307175756`，本臂 `0.8501177932753766` →
  **Δ = +0.0000492626**（≈ +4.9e-5），比 C1 所需的小幅提升（+0.0021）小约 43 倍；
  单 seed、5 epoch 下该量级与训练噪声不可区分。val 侧同向但同样微弱：
  best `auc_val_education = 0.8534284503303139`（epoch 5）vs 基线 `0.8527881905614896`（epoch 5）；
  val 曲线（epoch 1→5）0.8424 / 0.8479 / 0.8511 / 0.8525 / 0.8534（4 位小数，全精度在 run 的 `metrics.json`）。
- **机制确实在工作（C2 PASS）**：`modulation_ratio = 0.09219527083871916 ∈ [0.01, 1.0]`；
  `gamma_x_norm_mean = 0.8053260672792273`、`beta_x_norm_mean = 0.7914070365093426`、
  `e_new_norm_mean = 8.735004083756271`（`bias_ratio = 0.09060179353333717`）；
  逐 env 参数范数 `gamma = [0.8916080847919324, 1.754524712128432]`、
  `beta = [0.8624159179765235, 1.7467638710058115]`——γ/β 确实离开零初始化，且两侧 env 都在动
  （不是单侧漂移）。调制幅度落在"仍是 prompt 量级"的可解释带内，但**没有**带来有意义的泛化提升：
  **"条件化调制被激活" ≠ "条件化调制有用"**——本消融对这个问题的答案是否定的。
- **B3 是继承失败，不是本臂引入**：B3 只看**冻结 backbone** 的 `gate_networks` 样本均值
  （`census_benchmark/metrics.py`，与 stage2 新任务头无关）；本臂与其基线 run 的
  `gate_mean` 逐位相同（`[0.952401451979339, 0.8543901456350915]`，第 0 维在 stage1 冻结时就已越过
  0.95 上界）。其余 A1/A2/A4/A5/B1/B2/B4 全 PASS：A1 前后 backbone 哈希一致（`a12a5f53…`）、
  A5 env_ids 哈希与 stage1 一致、B1 三项 AUC 远高于 0.60、B2 gap `0.0033106570549373 ≤ 0.03`、
  B4 env 份额 `0.49411847255704855 / 0.5058815274429515 ≥ 0.05`。
- **处置**（第 6 节"门槛 PASS 但 C1 未过"）：记录失败并停止——**不合并 `master`**、不做二次调参续命
  （γ/β 的初始化 / 正则 / 满宽→低秩属新一次预注册）；**未做**多 seed、未做 `full` tag、未做 A3 复跑。
- **留痕位置**：run 目录 `artifacts/census_stage2/runs/20260930-1324-…-csb/`（`metrics.json:csb_arm`
  含 C1/C2 逐条 `observed`；按 `.gitignore` 不入库），`SUMMARY.md` 追加一行（AUC + A/B 门禁：
  B3 FAIL、其余 PASS）。**commit 口径**：run 记录的 `commit = 87afe03` 是当时的 HEAD；本消融源码
  当时尚未提交（为工作区改动，源码 mtime 13:07–13:18 均早于 run 产物落盘 13:24，此后未再修改），
  本次收尾提交把源码 / 测试 / 本文件 / `SUMMARY.md` 一并入库，即本分支上与该 run 对应的完整提交。

**已执行的验证（本分支；代码级验证与留痕核对，真跑结果见上表）**：

| 验证 | 结果 |
|---|---|
| `test_cond_scale_bias`（27 例） | 全绿：零初始化逐位一致（参数 / 前向 / 首个训练步梯度 / RNG 终点）、公式手算复核、诊断口径、C1/C2 含端点边界、接线与 CLI |
| 全量回归（protocol / metrics / smoke / router_probe / cond_scale_bias） | 83 例全绿 |
| **基线臂不受影响（逐字节）** | 同一 tiny CPU 夹具、同一 model seed 下，HEAD 版 runner 与本版 runner 的基线臂产出：`run_id` / `gate_report.json` / `newtask.pt`（逐张量）/ `metrics.json`（除新增的恒为 `"baseline"` 的 `variant` 字段）/ `config.json`（同）/ `SUMMARY.md` **全部相等**——基线路径确为纯增量改动 |
| 收尾跨产物核对（2026-09-30，随本文件一并提交） | 本文档与 `SUMMARY.md` 的数字逐项对照 run 目录的 `metrics.json` / `gate_report.json` / `config.json` 及 `probes/…/router_probe.json`：全部一致（含 Δ、B2 gap、`gate_mean`、逐 env 参数范数）；`git diff --check` 干净 |

**变更记录**（先改本文件、再改代码）：

| 日期 | 变更 | 理由 |
|---|---|---|
| 2026-09-30 | 诊断对象由"随机初始化的新任务头"改为"基线 run 训练后的新任务头"：`--newtask-checkpoint` 必填、strict 载入；`probe_id` 追加新任务头状态哈希前 8 位 | 随机初始化头不代表基线阶段 2 实际在用的 router；且诊断不再是 `(stage1, split seed, model seed, 代码)` 的函数，probe_id 必须随诊断对象区分，否则不同头的结论会互相覆盖 |
| 2026-09-30 | 修正 `test_mean_and_population_std_match_hand_computation` 的期望值 | 期望式只算了第一批样本的熵（0.7219），漏掉第二批（0.9710）；指标定义（逐样本 $H(W_b)/\log K$ 的均值，总体 std ddof=0）与实现均未变，属测试期望算错 |
| 2026-09-30 | **第 5 节公式改稿**：由"在 router logits $z$ 上做 $s(x)\odot z+b(x)$ 再 softmax（旧稿 5.1）"改为"在已路由表征上做 $s\odot(e_{\text{new}}+\gamma(x))+\beta(x)$，$W$ 完全不被调制" | 用户预注册的定稿公式；调制放在表征空间后，源 router $W$ 与既有 scale $s$ 都保持原样，消融的因果链更干净（只测"给已路由表征加条件化 scale+bias"，不改变路由本身）。旧稿的 $A_s,A_b,c_s,c_b,\alpha,\beta,\tanh$ 与瓶颈 $r_p$ 一并作废 |
| 2026-09-30 | **C2 判据随之改口径**：`prompt_ratio = r_p/r`（瓶颈宽度比）→ `modulation_ratio = ‖γ(x)‖/‖e_new‖`（有效调制比），带仍为 `[0.01, 1.0]` | 新稿的 $\gamma/\beta$ 满宽、无瓶颈 $r_p$，`prompt_ratio` 无定义；有效调制比保留原判据的全部理由（下界=非惰性，上界=仍是 prompt 量级） |
| 2026-09-30 | 第 5 节由"只是设计，本轮不写代码"改为"已实现"（`census_benchmark/cond_scale_bias.py` + `--variant` 接线）；明确 `model.py` 不动、以子类扩展 | 门槛 PASS 后 P4 解除；子类扩展满足 P2 且不改变基线臂的键集与语义 |
| 2026-09-30 | **第 1 阶段真跑结果回填**（第 7 节）：C1 FAIL（`AUC-Test-Education = 0.8501177932753766 < 0.8521`，Δ = +0.0000492626）、C2 PASS、继承 B3 FAIL → 方向停止 | 第 6 节"门槛 PASS 但 C1 未过"的预注册处置；结果不论好坏一律留痕，**不改口径**、不做二次调参 |

---

## 8. 明确的非目标

- 不改 `multitaskrec/model.py` / `config.py` / `census_benchmark/metrics.py` / `census_benchmark/protocol.py`；
  `run_census_benchmark.py` 只做 5.5 节列出的最小接线（新增 `--variant` 旗标，默认 `baseline`）。
- 第 1 阶段**只引入 $\gamma/\beta$ 一个变量**：不引入其它模型模块、损失项、门控或温度参数
  （Null Expert / TC-Prompt / CGR / affinity gate / KL-Prompt / T4 一律不碰）。
- 不跑 AliCCP / ByteRec；不做多 seed（第 1 阶段先单 seed 判定）；不做 FLOPs；不产出论文级性能结论。
- 不为过门槛 / 过 C1 而调 seed / 温度 / 初始化 / 划分 / `modulation_ratio` 口径（第 3.3、5.3 节）。
- 不主张方法新颖性：条件化调制（FiLM / 条件化归一化 / prompt 调制）是已有大类，本方向只是消融与工程诊断。
