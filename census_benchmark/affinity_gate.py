"""历史 Affinity Gate（OOD-aware 硬切换 Attention/FW）的**忠实复现与门控退化诊断**（不是新方法）。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-revalidation.md

审计对象（`archive/exploration` 分支的历史实现，逐项对齐、不做"顺手修好"）：

    run_newtask_from_ckpt.py            mode="affinity_gate"（ALL_MODES 之一；Adam/BCE/val 选点/test 只评一次）
    multitaskrec/model.py               NewTask(fusion_mode="affinity_gate") 的整条 affinity_gate 分支
    logs/all_experiment_results.md      历史结论与根因分析（"逐样本 cosine similarity 特征能有效检测 OOD 样本
                                        并切换到 FW"；"OOD-aware 硬切换，最稳定（σ=0.0014）但均值偏低"）

历史公式（$x$ = `dnn_input`，$H(x)$ = `projection_network(x)`，$T=150$，$K$ = len(spec_reps)）：

    E        = stack(env_embs, dim=1)        # env_embs 的元素是**一维** [rep_dim]（MPTRec.get_infos 逐个 index）
                                             # ⇒ E 的布局是 **[rep_dim, K]**（d 在前、任务在后）
    h_p_norm = normalize(H, dim=-1)          # [B, d] 逐样本按特征维归一化
    E_norm   = normalize(E, dim=0)           # [d, K] 逐**任务列**归一化（dim=0 就是特征维，**正确**）
    C        = h_p_norm @ E_norm             # [B, K]，是真余弦（历史注释 "values in [-1, 1]" 属实）
    a(x)     = [ max_k C , min_k C , max_k C − min_k C , mean_k |C| ]        # [B, 4]
    z(x)     = W2 ReLU(W1 a + b1) + b2       # affinity_mlp 的**前 sigmoid** 输出
    g_logit  = sigmoid(z)                    # ← D1：末端 Sigmoid 之后又被当作 logit 使用
    G        ~ Gumbel(0, 1)                  # 仅训练期抽样：−log(−log U)
    g_soft   = sigmoid((g_logit + G) / 0.5)  # 训练期；eval 期去掉 G
    g_hard   = (g_soft > 0.5)                # ← D1：硬切换；eval 期恒为 1（见下）
    g        = g_hard.detach() + g_soft − g_soft.detach()                   # straight-through
    W_attn   = softmax(H @ E / T) ,  W_fw = 1/K ,  W = g·W_attn + (1−g)·W_fw
    new_spec_rep = Σ_k W_k spec_rep_k ,  env_aware = new_env_emb ⊙ new_spec_rep
    pred     = tower([env_aware, gen_rep] · gate_network(x))

以下历史缺陷**故意**复现、不修（编号与 experiments 文档第 3 节一致）：

  * **D1 双重 sigmoid（主线）**：`affinity_mlp` 末端有 `nn.Sigmoid()`，其输出 `g_logit ∈ (0,1)` 又被当成
    logit 送进 `sigmoid((g_logit + G)/0.5)`。两个可证后果：
      - **eval 期 `g_hard ≡ 1`**：`g_soft = sigmoid(2·g_logit) ∈ (0.5, sigmoid(2)=0.8808)` 恒 > 0.5
        ⇒ 硬切换**永远**选 attention，`W ≈ W_attn`，与参数、与输入都无关。
        （唯一例外：float32 下 `sigmoid(z)` 下溢为 0.0 时 `g_soft = 0.5`，此时判 0 ⇒ FW；见测试。）
      - **训练期路由概率被夹在解析窗口内**：`g_hard = 1 ⟺ G > −sigmoid(z)`，故
        `P(attention | x) = 1 − exp(−exp(sigmoid(z))) ∈ (1−e^{−1}, 1−e^{−e}) = (0.6321, 0.9340)`：
        门控**永远不会偏向 FW**，且特征能移动的概率幅度 ≤ 0.3019（= 窗口宽度）。
    *精度留痕（float32）*：STE 表达式 `g_hard.detach() + g_soft − g_soft.detach()` 在 `g_hard = 1` 时
    **并非总是精确等于 1**。`1 + g_soft ∈ (1.5, 1.8808]`，该区间 ulp = 2⁻²³，因此 `fl(1 + g_soft)` 只在
    `1 + g_soft` 的 2⁻²⁴ 单位系数为偶数时精确；系数 `c ≡ 1 (mod 4)` 时是平局，round-half-even 向下取偶
    ⇒ `fl(1 + g_soft) = 1 + g_soft − 2⁻²⁴`，再减 `g_soft` 得 `g = 1 − 2⁻²⁴`（`c ≡ 3` 时向上 ⇒ `g = 1`）。
    于是 eval 期 `W` 与 `W_attn` 的逐位差**不恒为 0**：`|g − 1| ≤ 2⁻²⁴`，逐样本
    `‖W − W_attn‖₁ ≲ 2⁻²⁴·‖W_attn − W_fw‖₁ + 乘法/加法的舍入 ~1e-7`（K=2 时 < 1e-6）⇒ 判据 2
    （`routing_l1 < 1e-6`）仍命中，余量约一个数量级。
    因此 `routing_identity_frac` **不应**写成"恒为 1"、`routing_l1_mean` **不应**写成"恒为 0.0"
    （正确说法：不恒为 0，但恒 < 1e-6）；本模块用 `ste_exact_frac` 与 `routing_l1_max` 把这条留痕。
  * **D2 "OOD 判据"不自洽（设计层，非代码 bug；本轮审计曾误判、已撤回一条）**：
      - **撤回**：早期草稿把"`F.normalize(E, dim=0)` 按任务维归一化"记为**错误维**——**不成立**。
        `get_infos` 以 0 维索引取 env 嵌入（`env_embedding_network(env_indices[i])`，元素形状 `[rep_dim]`），
        `torch.stack(env_embs, dim=1)` 得 `[rep_dim, K]`，`dim=0` **就是特征维** ⇒ 逐任务列归一化，
        `C = h E_normᵀ` **是真余弦**（历史注释 "values in [-1, 1]" 属实）。三处独立佐证：`num_source_tasks`
        切片用 `[:, :K]`（dim=1）、`torch.mm(H, E)` 需要 `[B,d]@[d,K]`、以及本模块的布局守卫。
      - **保留**：门控的 4 维特征全部由 `projection_network` 的输出 `H(x)` 派生，而 `W_attn` 也用同一个
        `H(x)`——"这一样本与源任务不匹配（OOD）"与"attention 路由给出的权重"出自同一投影，无法相互独立地
        证伪；且 K=2 时 4 个特征只是两个余弦 `(C_1, C_2)` 的函数，其中第 4 维
        `mean|C| = (|max|+|min|)/2` 还是前三维的函数 ⇒ 门控实际只看到 **2 个数**（像空间维数 ≤ 2）。
        历史把它读作"能有效检测 OOD 样本"，这一步在实现里没有支撑点（CGR 的根因分析恰恰是因为这点才把
        门控输入**解耦**）。
    本模块不修，只把 `cos_entry_min/max` 与逐源任务列均值落盘，让读者自行判断（4 维特征的第 4 维
    冗余性 `mean|C| = (|max|+|min|)/2` 是恒等式，由测试锁定，见 experiments 文档第 4.1 节 I4）。
  * **D3 全局 RNG 消耗**：训练期每步 `torch.rand_like(g_logit)` 吃**全局** RNG（每步 B 次抽样）
    ⇒ 与其它臂**不共享随机流**；构造期 `affinity_mlp` 建在 projection 与 gate 之间又多抽 2 组
    `nn.Linear` 初始化 ⇒ 历史 "affinity_gate vs prompt" 的对比不是单变量对比。
  * **D4 硬 STE + 噪声污染的梯度**：前向用 hard 0/1（训练期有 `1−p ∈ [0.066, 0.368]` 的样本**整支丢弃**
    attention），反向传播的却是**带噪样本**的导数；再叠加 `sigmoid` 两次带来的局部灵敏度上限
    `∂g_logit/∂z = σ′(z) ≤ 0.25`（修正路径为 `1/τ = 2`），门控的学习信号被双重压制。
  * **D5 门控无正则**：`NewTask.get_l2_reg()` 只正则 `tower_network`（继承 master），
    `affinity_mlp` / `projection_network` 都不进 L2（与 CGR 同源，本模块不修）。
  * **D6 判据不可识别 + 修正口径（本轮审计新发现）**：承接 D1——eval 期硬门控规则是参数的**常函数**
    （`g_hard ≡ 1`），因此"门控学会了按 OOD 切换到 FW"这一历史结论**在任何留痕读数上都不可识别**：
    历史只评过 eval 模式（且 affinity_gate 没有任何诊断脚本、runner 只打印 `lambda=--- kl=---`），
    看到的必然是"纯 attention"。**修正**：用前 sigmoid 输出 `z` 作为 logit 的反事实读出
    `sigmoid(z/τ)`（τ=0.5、无噪声、同一组参数），该读出**是**输入相关的，因而可检验。
    本模块把它作为 `corrected_routing_share_attn` / `corrected_gate_mean` 落盘（只读数，不改模型语义）。

硬约束：

  * **不改 `multitaskrec/model.py`**：本模块以 `NewTask` 的**子类**扩展；`forward` 只复刻历史 affinity_gate
    分支的表达式与运算顺序（测试用与归档类逐位 `torch.equal` 比对锁定）。
  * **不影响基线臂**：`build_newtask(BASELINE_VARIANT, ...)` 返回的就是基线 `NewTask` 本身（逐位同 master，
    构造不多消耗一次 RNG）。
  * **protocol.py / metrics.py 零改动**：A/B 门禁语义不变；臂级判定只写在 `metrics.json:affinity_arm`。
  * **诊断不扰动训练**：训练轨迹读取真实前向存下的 `last_terms`（**不重算** ⇒ 不额外消耗 RNG，D3 留痕）；
    噪声/梯度探针一律走**私有 `torch.Generator`**，不动全局 RNG。

预注册判据（**只允许在看到结果之前修改**）：

    STOP（机制，主导于 AUC）：
      eval_attn_share >= 0.999  或  routing_l1 < 1e-6  或  grad_norm == 0      ⇒ CONFIRMED_DEGENERATE
    STOP（保真，主导于退化）：
      noise_share_window > 0.32（解析上界 0.3019）                              ⇒ FIDELITY_BREAK
    效应阈值（**仅在无 STOP 时**才解释）：
      AUC-Test-Education >= 0.8521

用法（接线见 run_census_benchmark.run_stage2）：

    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --variant affinity --tag short
    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --affinity        --tag short
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitaskrec.model import MLP, NewTask

# ---- 臂与接线 ----
BASELINE_VARIANT = "baseline"
VARIANT = "affinity"
VARIANTS = (BASELINE_VARIANT, VARIANT)
RUN_ID_SUFFIX = "-affinity"                # 沿用 null-expert / attn / csb / cgr 分支的"协议 run_id + 臂后缀"约定
FUSION_MODE = "affinity_gate"              # 历史 fusion_mode 取值

# ---- 历史常量（archive/exploration 原值，不得改）----
TEMPERATURE = 150                          # NewTask.temperature
AFFINITY_INPUT_DIM = 4                     # [max_cos, min_cos, max_cos−min_cos, mean|cos|]
AFFINITY_HIDDEN = 16                       # nn.Linear(4, 16, ...) 隐层宽度
GUMBEL_TAU = 0.5                           # 历史硬编码 /0.5
CLAMP_EPS = 1e-10                          # 历史 clamp(1e-10)（两处）
# E = stack(env_embs, dim=1) 的布局是 [rep_dim, K]（env_embs 元素是一维 [rep_dim]）⇒ dim=0 就是**特征维**，
# `F.normalize(E, dim=0)` 即"逐任务列归一化"，是标准做法；该常量因此是**正确维**，不是缺陷（见 experiments 文档 D2）。
ENV_LAYOUT = "feature_first_(rep_dim, K)"
ENV_NORMALIZE_DIM = 0                      # = 特征维（对 [d, K] 而言）

# ---- 解析边界（D1 的可证后果；仅作读数基准，判定用阈值见下）----
# g_logit = sigmoid(z) ∈ (0,1) ⇒ P_train(attention) = 1 − exp(−exp(g_logit)) ∈ (1−e^{−1}, 1−e^{−e})
GUMBEL_SHARE_LO = 1.0 - math.exp(-math.exp(0.0))          # 0.6321205588285577（z → −∞）
GUMBEL_SHARE_HI = 1.0 - math.exp(-math.exp(1.0))          # 0.9340119641546875（z → +∞）
GUMBEL_SHARE_WINDOW_MAX = GUMBEL_SHARE_HI - GUMBEL_SHARE_LO   # 0.30189140532612976
G_LOGIT_LO, G_LOGIT_HI = 0.0, 1.0                         # 双 sigmoid 的**结构性**签名（开区间）
EVAL_G_SOFT_MIN = float(torch.sigmoid(torch.tensor(0.0 / GUMBEL_TAU)))    # 0.5（下确界）
EVAL_G_SOFT_MAX = float(torch.sigmoid(torch.tensor(1.0 / GUMBEL_TAU)))    # sigmoid(2) = 0.8807970779778823
FLIP_RATE_MIN = 2.0 * GUMBEL_SHARE_HI * (1.0 - GUMBEL_SHARE_HI)           # 同一样本换噪声即翻面的下界
FLIP_RATE_MAX = 2.0 * GUMBEL_SHARE_LO * (1.0 - GUMBEL_SHARE_LO)           # 上界
TRAIN_DROP_MIN = 1.0 - GUMBEL_SHARE_HI                    # 训练期被硬切换丢弃 attention 的样本占比下界
TRAIN_DROP_MAX = 1.0 - GUMBEL_SHARE_LO
STE_ULP = 2.0 ** -24                                      # float32 在 [1,2) 上的 ulp：STE 在 g_hard=1 时的舍入级

# ---- 预注册判据（只允许在看到结果之前修改；规则文本是预注册原文，单独硬编码）----
AUC_TEST_MIN = 0.8521                      # 效应阈值（基线实测 0.850069）
EVAL_ATTN_SHARE_MIN = 0.999                # 判据 1：eval 期路由到 attention 的样本占比 ≥ 0.999
ROUTING_L1_MIN = 1e-6                      # 判据 2：|W − W_attn| 的 L1 < 1e-6（逐样本路由等价于纯 attention）
GRAD_NORM_MIN = 0.0                        # 判据 3：affinity_mlp 训练期梯度范数最大值 == 0
NOISE_SHARE_WINDOW_MAX = 0.32              # 保真判据：解析窗口上界 0.3019 + 余量（超出 ⇒ 复现偏离历史）
PRESIGMOID_SATURATION_ABS = 8.0            # 饱和读数阈值：|z| ≥ 8 ⇒ sigmoid' ≤ 3.4e-4
RULE_EVAL_SHARE = "eval_attn_share >= 0.999"
RULE_ROUTING_L1 = "routing_l1 < 1e-6"
RULE_GRAD_NORM = "grad_norm == 0"
RULE_NOISE_WINDOW = "noise_share_window <= 0.32"
RULE_G_LOGIT_INTERVAL = "g_logit_in_unit_interval"
RULE_EFFECT = "auc_test_education >= 0.8521"

# ---- 探针默认参数（固定 seed ⇒ 读数可复现；私有 Generator ⇒ 不动全局 RNG）----
NOISE_PROBE_DRAWS = 256
NOISE_PROBE_SEED = 20260930


def ratio(numerator: float, denominator: float):
    """衰减比：分母为 0 时返回 None（JSON 可序列化；不得写成 inf）。"""
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0.0 else None


def exact_attention_share(presigmoid: torch.Tensor) -> torch.Tensor:
    """D1 的闭式解：训练期硬切换到 attention 的概率 `1 − exp(−exp(sigmoid(z)))`。

    推导：`g_hard = 1 ⟺ g_soft > 0.5 ⟺ g_logit + G > 0 ⟺ G > −g_logit`，
    `G ~ Gumbel(0,1)`（CDF `exp(−e^{−x})`）⇒ `P = 1 − exp(−exp(g_logit))`，`g_logit = sigmoid(z)`。
    与 `torch.rand_like` 的抽样相互独立，可用来给蒙特卡洛探针做交叉验证。
    """
    g_logit = torch.sigmoid(presigmoid.detach().double())
    return 1.0 - torch.exp(-torch.exp(g_logit))


def exact_flip_rate(presigmoid: torch.Tensor) -> torch.Tensor:
    """同一样本在两个独立噪声下的硬判据翻面概率 `2p(1−p)`（p = `exact_attention_share`）。"""
    p = exact_attention_share(presigmoid)
    return 2.0 * p * (1.0 - p)


def gumbel_noise_like(like: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """历史原式 `-(-rand_like(x).clamp(1e-10).log()).clamp(1e-10).log()`（= −log(−log U)）。

    `generator=None` ⇒ 走 `torch.rand_like`（**消耗全局 RNG**，历史行为，D3）；
    给定 `generator` ⇒ 走 `torch.rand(..., generator=...)`（探针专用，不动全局 RNG）。
    """
    if generator is None:
        uniform = torch.rand_like(like)
    else:
        uniform = torch.rand(like.shape, generator=generator, dtype=like.dtype, device=like.device)
    return -(-uniform.clamp(CLAMP_EPS).log()).clamp(CLAMP_EPS).log()


def routing_pipeline(newtask, dnn_input, gen_rep, spec_reps, env_embs, w):
    """历史 `forward` 中 `W` **之后**的管线（逐行照搬；`forward` 与 D6 反事实读出共用）。

    `new_spec_rep = Σ_k W_k spec_rep_k` → `env_aware = new_env_emb ⊙ new_spec_rep`
    → `fused_rep = [env_aware, gen_rep] · gate_network(x)` → `tower(fused_rep)`。
    """
    new_env_emb = newtask.env_embedding_network(newtask.new_env_idx).squeeze(0)
    if newtask.num_source_tasks is not None:
        spec_reps = spec_reps[:newtask.num_source_tasks]
    new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), w).squeeze()
    env_aware_rep = new_spec_rep * new_env_emb
    gate_out = newtask.gate_network(dnn_input).unsqueeze(dim=2)
    all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
    fused_rep = torch.matmul(all_reps, gate_out).squeeze()
    return newtask.tower_network(fused_rep).squeeze()


class HistoricalAffinityGateNewTask(NewTask):
    """历史 `NewTask(fusion_mode="affinity_gate")` 的忠实复刻（含 D1–D5 五个历史缺陷）。

    构造顺序与历史**逐行同序**（`nn.Module.__init__` 之后手工搭模块树，而不是 `super().__init__()`）：
    历史在 `projection_network` 与 `gate_network` **之间**构造 `affinity_mlp`，两次 `nn.Linear` 的默认
    初始化会消耗全局 RNG ⇒ 之后的 `gate_network` / `tower_network` 与基线拿到不同初值（D3 的一部分；
    历史 affinity_gate-vs-prompt 的对比因此不是单变量对比）。子模块名与形状仍与 master `NewTask`
    完全一致（测试锁定 `state_dict` 键集只多 `affinity_mlp.*`）。

    `last_terms`：最近一次 `forward` 存下的**已 detach** 中间量，供训练轨迹读取（不重算 ⇒ 不额外消耗 RNG）。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 temperature=TEMPERATURE, num_source_tasks=None):
        nn.Module.__init__(self)                       # 故意不走 NewTask.__init__：见类 docstring
        self.reg_dnn = reg_dnn
        self.device = device
        self.fusion_mode = FUSION_MODE
        self.num_source_tasks = num_source_tasks
        self.temperature = temperature
        self.env_embedding_network = nn.Embedding(1, rep_dim)
        self.register_buffer("new_env_idx", torch.tensor([0]), persistent=True)
        self.projection_network = nn.Sequential(
            nn.Linear(input_size, rep_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(rep_dim // 2, rep_dim, bias=False),
            nn.LayerNorm(rep_dim),
        )
        self.tc_fusion = None                          # 历史 else 分支的赋值（不产生参数、不消耗 RNG）
        self.affinity_mlp = nn.Sequential(             # 位置即历史位置（projection 与 gate 之间）
            nn.Linear(AFFINITY_INPUT_DIM, AFFINITY_HIDDEN),
            nn.ReLU(),
            nn.Linear(AFFINITY_HIDDEN, 1),
            nn.Sigmoid(),                              # ← D1：这个 Sigmoid 是"logit 被压扁"的根源
        )
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, 2, bias=False), nn.Softmax(dim=-1)
        )
        self.tower_network = MLP(
            list(tower_dnn_hidden_units) + [1], input_size=rep_dim, output_activation="sigmoid"
        )
        self.last_terms = None                         # 最近一次 forward 的中间量（diagnostic 专用）

    # ---- 历史公式 ----
    def gate_terms(self, dnn_input, gen_rep, spec_reps, env_embs, *, noise=None, generator=None,
                   use_noise=None) -> dict:
        """历史 affinity_gate 分支的全部中间量（纯函数：不改参数、不写状态；forward 与诊断共用同一口径）。

        `noise` / `use_noise` / `generator` 是给探针用的**只读扩展**，默认值下与原式逐位一致：
          * `use_noise=None` ⇒ 跟随 `self.training`（历史行为）；
          * `noise` 给定 ⇒ 直接用给定噪声（不抽 RNG）；`generator` 给定且 `noise=None` ⇒ 用私有 generator 抽；
            两者都不给 ⇒ `torch.rand_like`（**全局 RNG**，历史行为，D3）。
        """
        exist_env_embs = torch.stack(env_embs, dim=1)
        if self.num_source_tasks is not None:
            exist_env_embs = exist_env_embs[:, :self.num_source_tasks]
            spec_reps = spec_reps[:self.num_source_tasks]
            num_tasks = self.num_source_tasks
        else:
            num_tasks = len(spec_reps)

        h_out = self.projection_network(dnn_input)
        # 布局守卫（fail-fast，不改变任何数值）：E 必须是 [rep_dim, K]（env_embs 元素为一维 [rep_dim]）。
        # 传成 [K, rep_dim] 或 [B, rep_dim] 时历史代码会静默算错或抛出难懂的 mm 形状错误。
        if exist_env_embs.dim() != 2 or exist_env_embs.shape[0] != h_out.shape[-1]:
            raise ValueError(f"env_embs 布局应为 {ENV_LAYOUT}（每个元素一维 [rep_dim]），"
                             f"实际 stack(dim=1) 后为 {tuple(exist_env_embs.shape)}，"
                             f"而 projection 输出为 {tuple(h_out.shape)}")

        # Step 1–2：历史"cosine similarity"特征（E 为 [d, K]，`normalize(dim=0)` 即逐任务列归一化 ⇒ 真余弦）
        h_p_norm = F.normalize(h_out, dim=-1)
        e_norm = F.normalize(exist_env_embs, dim=ENV_NORMALIZE_DIM)
        cos_sim = torch.mm(h_p_norm, e_norm)
        max_cos = cos_sim.max(dim=-1, keepdim=True)[0]
        min_cos = cos_sim.min(dim=-1, keepdim=True)[0]
        affinity_input = torch.cat([
            max_cos,
            min_cos,
            max_cos - min_cos,
            cos_sim.abs().mean(dim=-1, keepdim=True),
        ], dim=-1)

        # Step 3：z 是 MLP 的前 sigmoid 输出（D6 修正口径要用它）；g_logit = sigmoid(z) 才是历史"logit"
        presigmoid = self.affinity_mlp[2](self.affinity_mlp[1](self.affinity_mlp[0](affinity_input))).squeeze(-1)
        g_logit = torch.sigmoid(presigmoid)

        # Gumbel 硬切换（D1+D4：eval 恒为 1；训练期是噪声主导的硬判据 + straight-through）
        noise_on = self.training if use_noise is None else bool(use_noise)
        if noise_on:
            # 三条路径（见 `gumbel_noise_like`）：给定 `noise` 直接用（不抽 RNG）；给定 `generator` 从私有
            # generator 抽（探针专用，不动全局 RNG）；两者都不给才 `torch.rand_like`（**全局 RNG**，历史行为，
            # D3）。梯度探针的 historical 路径必须在私有 generator 上抽（experiments 文档 5.3），漏传会把
            # 全局 RNG 拖进读数（I5）。
            drawn = noise if noise is not None else gumbel_noise_like(g_logit, generator=generator)
            g_soft = torch.sigmoid((g_logit + drawn) / GUMBEL_TAU)
        else:
            drawn, g_soft = None, torch.sigmoid(g_logit / GUMBEL_TAU)
        g_hard = (g_soft > 0.5).float()
        g = g_hard.detach() + g_soft - g_soft.detach()

        # Step 4–5：两套权重 + 硬切换（W_attn 用**未归一化**的 H_out / E，与历史一致）
        w_attn = F.softmax(torch.mm(h_out, exist_env_embs) / self.temperature, dim=-1).unsqueeze(2)
        w_fw = torch.ones_like(w_attn) / num_tasks
        w = g.view(-1, 1, 1) * w_attn + (1 - g.view(-1, 1, 1)) * w_fw

        return {"h_out": h_out, "env_embs": exist_env_embs, "e_norm": e_norm, "cos_sim": cos_sim,
                "affinity_input": affinity_input, "presigmoid": presigmoid, "g_logit": g_logit,
                "gumbel_noise": drawn, "g_soft": g_soft, "g_hard": g_hard, "g": g,
                "w_attn": w_attn, "w_fw": w_fw, "w": w, "num_tasks": num_tasks}

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs, *, noise=None, generator=None,
                use_noise=None):
        terms = self.gate_terms(dnn_input, gen_rep, spec_reps, env_embs,
                                noise=noise, generator=generator, use_noise=use_noise)
        # 训练轨迹读取真实前向的中间量（detach 只脱图、不复制数据；不重算 ⇒ 不额外抽 RNG）
        self.last_terms = {key: (value.detach() if torch.is_tensor(value) else value)
                           for key, value in terms.items()}
        return routing_pipeline(self, dnn_input, gen_rep, spec_reps, env_embs, terms["w"])


class AffinityStats:
    """门控读数累加器（CPU float64；流式、显存 O(1)，口径见 experiments 文档第 5 节）。

    样本级聚合（与分批方式无关）：硬切换占比 `g_hard_mean`（= eval 期路由到 attention 的样本占比）、
    `g_soft_mean/min/max`、`g_logit_min/max`（双 sigmoid 的结构签名：精确算术下 ∈ (0,1)，float32 可饱和到 0/1）、
    `presigmoid_*`（饱和读数）、`routing_l1_mean/max`（|W − W_attn| 的逐样本 L1）、`routing_fw_l1_mean`、
    `routing_identity_frac`（W 与 W_attn **逐位**相同的样本占比）、`ste_exact_frac`（STE 值 g 与 g_hard
    **逐位**相同的样本占比——float32 下 `fl(1+g_soft) − g_soft` 会有 ±2⁻²⁴ 的舍入，见 D1 精度留痕）、
    cosine 读数（`cos_max_mean` / `cos_min_mean` / `cos_range_mean` / `cos_absmean_mean` / `cos_entry_min/max`）、
    逐源任务列读数（`cos_col_mean[k]` / `routing_col_mean[k]`）、口径自证（`env_normalize_dim` = 特征维 /
    `env_layout` = 布局串）、
    解析读数（`noise_share_exact_*` = 训练期路由概率的闭式解；`attention_drop_frac_exact`；
    `corrected_routing_share_attn` / `corrected_gate_mean` = D6 修正口径）。
    空累加器：所有**数值**字段为 0.0（`env_layout` 为字符串；`*_abs` / `abs_*` 阈值字段恒回显阈值）。
    """

    def __init__(self) -> None:
        self.n_samples = self.n_batches = 0
        self.num_tasks = None
        self._g_sum = self._g_hard_sum = self._g_soft_sum = 0.0
        self._g_min = self._g_max = self._g_soft_min = self._g_soft_max = None
        self._g_logit_min = self._g_logit_max = None
        self._z_sum = self._z_abs_sum = self._z_abs_max = 0.0
        self._sat_count = 0
        self._l1_attn_sum = self._l1_attn_max = self._l1_fw_sum = 0.0
        self._identical_count = 0
        self._ste_exact_count = 0
        self._cos_max_sum = self._cos_min_sum = self._cos_abs_sum = 0.0
        self._cos_entry_min = self._cos_entry_max = None
        self._cos_col_sum = self._routing_col_sum = None
        self._share_sum = self._share_min = self._share_max = None
        self._flip_sum = 0.0
        self._corrected_share_count = 0
        self._corrected_gate_sum = 0.0

    def _init_tasks(self, num_tasks: int) -> None:
        if self.num_tasks is None:
            self.num_tasks = int(num_tasks)
            self._cos_col_sum = torch.zeros(self.num_tasks, dtype=torch.float64)
            self._routing_col_sum = torch.zeros(self.num_tasks, dtype=torch.float64)
        elif self.num_tasks != int(num_tasks):
            raise ValueError(f"逐批 num_tasks 不一致: {self.num_tasks} vs {num_tasks}")

    def update(self, terms: dict) -> None:
        g = terms["g"].detach().double().reshape(-1).cpu()
        g_hard = terms["g_hard"].detach().double().reshape(-1).cpu()
        g_soft = terms["g_soft"].detach().double().reshape(-1).cpu()
        g_logit = terms["g_logit"].detach().double().reshape(-1).cpu()
        z = terms["presigmoid"].detach().double().reshape(-1).cpu()
        batch = int(g.numel())
        self._init_tasks(terms["num_tasks"])

        cos_sim = terms["cos_sim"].detach().double().cpu()                 # [B, K]
        w = terms["w"].detach().double().cpu()                             # [B, K, 1]
        w_attn = terms["w_attn"].detach().double().cpu()
        w_fw = terms["w_fw"].detach().double().cpu()
        l1_attn = (w - w_attn).abs().reshape(batch, -1).sum(dim=1)         # [B]
        l1_fw = (w - w_fw).abs().reshape(batch, -1).sum(dim=1)
        identical = (w == w_attn).reshape(batch, -1).all(dim=1)            # [B] 逐位相等
        ste_exact = (g == g_hard)                                          # [B] STE 舍入留痕（float32）
        share = exact_attention_share(z).reshape(-1)                       # [B] 训练期路由概率（闭式）
        corrected_share = (torch.sigmoid(z / GUMBEL_TAU) > 0.5).double()   # [B] D6 修正判据（z > 0）

        self.n_samples += batch
        self.n_batches += 1
        self._g_sum += float(g.sum())
        self._g_hard_sum += float(g_hard.sum())
        self._g_soft_sum += float(g_soft.sum())
        self._z_sum += float(z.sum())
        self._z_abs_sum += float(z.abs().sum())
        self._sat_count += int((z.abs() >= PRESIGMOID_SATURATION_ABS).sum())
        self._l1_attn_sum += float(l1_attn.sum())
        self._l1_attn_max = max(self._l1_attn_max, float(l1_attn.max()))
        self._l1_fw_sum += float(l1_fw.sum())
        self._identical_count += int(identical.sum())
        self._ste_exact_count += int(ste_exact.sum())
        self._cos_max_sum += float(cos_sim.max(dim=-1)[0].sum())
        self._cos_min_sum += float(cos_sim.min(dim=-1)[0].sum())
        self._cos_abs_sum += float(cos_sim.abs().mean(dim=-1).sum())
        self._cos_col_sum += cos_sim.sum(dim=0)
        self._routing_col_sum += w.reshape(batch, self.num_tasks).sum(dim=0)
        self._share_sum = float(share.sum()) if self._share_sum is None else self._share_sum + float(share.sum())
        if batch:
            self._share_min = float(share.min()) if self._share_min is None else min(self._share_min,
                                                                                    float(share.min()))
            self._share_max = float(share.max()) if self._share_max is None else max(self._share_max,
                                                                                    float(share.max()))
            self._cos_entry_min = (float(cos_sim.min()) if self._cos_entry_min is None
                                   else min(self._cos_entry_min, float(cos_sim.min())))
            self._cos_entry_max = (float(cos_sim.max()) if self._cos_entry_max is None
                                   else max(self._cos_entry_max, float(cos_sim.max())))
        self._flip_sum += float(exact_flip_rate(z).sum())
        self._corrected_share_count += int(corrected_share.sum())
        self._corrected_gate_sum += float(torch.sigmoid(z / GUMBEL_TAU).sum())
        if batch:
            self._g_min = float(g.min()) if self._g_min is None else min(self._g_min, float(g.min()))
            self._g_max = float(g.max()) if self._g_max is None else max(self._g_max, float(g.max()))
            self._g_soft_min = (float(g_soft.min()) if self._g_soft_min is None
                                else min(self._g_soft_min, float(g_soft.min())))
            self._g_soft_max = (float(g_soft.max()) if self._g_soft_max is None
                                else max(self._g_soft_max, float(g_soft.max())))
            self._g_logit_min = (float(g_logit.min()) if self._g_logit_min is None
                                 else min(self._g_logit_min, float(g_logit.min())))
            self._g_logit_max = (float(g_logit.max()) if self._g_logit_max is None
                                 else max(self._g_logit_max, float(g_logit.max())))
            self._z_abs_max = max(self._z_abs_max, float(z.abs().max()))

    def result(self) -> dict:
        n, batches = self.n_samples, self.n_batches
        share_mean = (self._share_sum / n) if n else 0.0
        return {
            "n_samples": n, "n_batches": batches, "num_tasks": self.num_tasks,
            # 硬切换 / 软门控读数
            "g_hard_mean": self._g_hard_sum / n if n else 0.0,
            "g_mean": self._g_sum / n if n else 0.0,
            "g_min": self._g_min if self._g_min is not None else 0.0,
            "g_max": self._g_max if self._g_max is not None else 0.0,
            "g_soft_mean": self._g_soft_sum / n if n else 0.0,
            "g_soft_min": self._g_soft_min if self._g_soft_min is not None else 0.0,
            "g_soft_max": self._g_soft_max if self._g_soft_max is not None else 0.0,
            # D1 的结构签名（`g_logit = σ(z) ∈ (0,1)`；float32 饱和时为 1.0/0.0，见 evaluate_gate 注释）
            "g_logit_min": self._g_logit_min if self._g_logit_min is not None else 0.0,
            "g_logit_max": self._g_logit_max if self._g_logit_max is not None else 0.0,
            # 路由等价性（判据 2 的读数）
            "routing_l1_mean": self._l1_attn_sum / n if n else 0.0,
            "routing_l1_max": self._l1_attn_max,
            "routing_fw_l1_mean": self._l1_fw_sum / n if n else 0.0,
            "routing_identity_frac": self._identical_count / n if n else 0.0,
            "ste_exact_frac": self._ste_exact_count / n if n else 0.0,     # D1 精度留痕（g == g_hard 的样本占比）
            # 门控输入特征（K=2 时这 4 个数由 (C_1, C_2) 完全决定）+ 逐源任务列读数
            "cos_max_mean": self._cos_max_sum / n if n else 0.0,
            "cos_min_mean": self._cos_min_sum / n if n else 0.0,
            "cos_range_mean": (self._cos_max_sum - self._cos_min_sum) / n if n else 0.0,
            "cos_absmean_mean": self._cos_abs_sum / n if n else 0.0,
            "cos_col_mean": (self._cos_col_sum / n).tolist() if (n and self.num_tasks) else [],
            "routing_col_mean": (self._routing_col_sum / n).tolist() if (n and self.num_tasks) else [],
            # D2（复核后口径）：`C` 是**真余弦**（E 为 [rep_dim, K]，dim=0 即特征维）——落盘布局与归一化维
            # 自证；"归一化维错误"的早期说法已撤回，保留的 D2 是设计层问题（同源 + 像空间 ≤ 2 维）。
            "cos_entry_min": self._cos_entry_min if self._cos_entry_min is not None else 0.0,
            "cos_entry_max": self._cos_entry_max if self._cos_entry_max is not None else 0.0,
            "env_normalize_dim": int(ENV_NORMALIZE_DIM),
            "env_layout": ENV_LAYOUT,
            # 前 sigmoid 读数
            "presigmoid_mean": self._z_sum / n if n else 0.0,
            "presigmoid_abs_mean": self._z_abs_sum / n if n else 0.0,
            "presigmoid_abs_max": self._z_abs_max,
            "presigmoid_saturation_frac": self._sat_count / n if n else 0.0,
            "presigmoid_saturation_abs": PRESIGMOID_SATURATION_ABS,
            # 解析读数（D1）：训练期路由概率的闭式解
            "noise_share_exact_mean": share_mean,
            "noise_share_exact_min": self._share_min if self._share_min is not None else 0.0,
            "noise_share_exact_max": self._share_max if self._share_max is not None else 0.0,
            "noise_share_exact_window": ((self._share_max - self._share_min)
                                         if self._share_min is not None else 0.0),
            "attention_drop_frac_exact": (1.0 - share_mean) if n else 0.0,
            "flip_rate_exact_mean": self._flip_sum / n if n else 0.0,
            # D6 修正口径（把前 sigmoid 输出当 logit：sigmoid(z/τ)，无噪声）
            "corrected_routing_share_attn": self._corrected_share_count / n if n else 0.0,
            "corrected_gate_mean": self._corrected_gate_sum / n if n else 0.0,
            "gumbel_tau": GUMBEL_TAU,
        }


def affinity_grad_norm(newtask) -> float:
    """affinity_mlp 全参数梯度范数（backward 之后、`optimizer.step()` 之前读取；全为 None ⇒ 0.0）。"""
    total = 0.0
    for param in newtask.affinity_mlp.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().double().norm(2)) ** 2
    return total ** 0.5


class TrainGateTrace:
    """训练期逐 epoch 轨迹：门控读数（`AffinityStats` 口径）+ affinity_mlp 梯度范数。

    与 CGR 的诊断不同点（D3 要求）：这里**不重算**门控 —— 训练期重算会多抽一次 Gumbel 噪声，
    破坏"忠实复现"的随机流。读数直接取 `newtask.last_terms`（真实前向的中间量）。

    runner 顺序：`start_epoch(epoch)` → 每个 batch 在 `loss.backward()` 之后、`optimizer.step()` 之前
    `record_step(newtask)` → 一个 epoch 结束 `end_epoch()`。
    """

    def __init__(self) -> None:
        self.stats = AffinityStats()
        self._per_epoch: list[dict] = []
        self._epoch = None
        self._epoch_stats: AffinityStats | None = None
        self._epoch_grads: list[float] = []
        self._grads: list[float] = []

    def start_epoch(self, epoch: int) -> None:
        if self._epoch is not None:
            raise RuntimeError(f"epoch {self._epoch} 尚未 end_epoch()，不能开始 epoch {epoch}")
        self._epoch = int(epoch)
        self._epoch_stats, self._epoch_grads = AffinityStats(), []

    def record_step(self, newtask) -> float:
        if not isinstance(newtask, HistoricalAffinityGateNewTask):
            raise ValueError(f"affinity 轨迹只适用于 {HistoricalAffinityGateNewTask.__name__}，"
                             f"收到 {type(newtask).__name__}")
        if self._epoch is None:
            raise RuntimeError("record_step 之前必须先 start_epoch()")
        terms = newtask.last_terms
        if terms is None:
            raise RuntimeError("forward 尚未调用：训练轨迹读取真实前向的 last_terms（不重算 ⇒ 不额外抽 RNG）")
        norm = affinity_grad_norm(newtask)
        self._epoch_grads.append(norm)
        self._grads.append(norm)
        self._epoch_stats.update(terms)
        self.stats.update(terms)
        return norm

    def end_epoch(self) -> dict:
        if self._epoch is None:
            raise RuntimeError("end_epoch 之前必须先 start_epoch()")
        grads = self._epoch_grads
        record = {"epoch": self._epoch, "steps": len(grads),
                  "grad_norm_mean": sum(grads) / len(grads) if grads else 0.0,
                  "grad_norm_max": max(grads) if grads else 0.0,
                  "grad_nonzero_steps": sum(1 for norm in grads if norm > 0.0)}
        record.update(self._epoch_stats.result())
        self._per_epoch.append(record)
        self._epoch, self._epoch_stats, self._epoch_grads = None, None, []
        return record

    def per_epoch(self) -> list[dict]:
        return list(self._per_epoch)

    def result(self) -> dict:
        steps = len(self._grads)
        return {**self.stats.result(),
                "epochs": len(self._per_epoch), "steps": steps,
                "grad_norm_mean": sum(self._grads) / steps if steps else 0.0,
                "grad_norm_max": max(self._grads) if steps else 0.0,
                "grad_nonzero_steps": sum(1 for norm in self._grads if norm > 0.0),
                "per_epoch": self.per_epoch()}


@torch.no_grad()
def evaluate_gate(newtask, backbone, loader, device) -> dict:
    """best_state 载入后的门控诊断（只用给定划分前向一遍：不训练、不反传、不改参数、不抽 RNG）。

    调用点固定在**选点完成、best_state 已载入之后**，只用 val；不参与 early stop / 选点 / test 评测，
    因此不可能改变 A/B 门禁或效应判定的结果。诊断在 **eval 模式**下进行 ⇒ 门控走无噪分支，
    读到的是历史 runner 实际评测到的那条路径（D1/D6 的签名读数就在这里）。

    随机流口径：门控本身一次 RNG 都不抽（eval 无噪）。但**迭代传进来的 loader** 属于 PyTorch 机制：
    `DataLoader.__iter__` 每次都抽一次 `_base_seed`（走 `loader.generator`，默认 `None` ⇒ 全局 RNG）。
    需要"调用前后全局 RNG 逐位不变"的严格保证时，请传一只自带 `generator=` 的 loader
    （夹具 `tiny_loader` 就是这么做的）；本函数不为此改动任何数值。
    """
    if not isinstance(newtask, HistoricalAffinityGateNewTask):
        raise ValueError(f"affinity 诊断只适用于 {HistoricalAffinityGateNewTask.__name__}，"
                         f"收到 {type(newtask).__name__}")
    was_training = newtask.training
    try:
        newtask.eval()
        backbone.eval()                                      # 三件套之 2：backbone 恒为 eval
        stats = AffinityStats()
        for _, _, _, features in loader:
            features = {key: value.to(device) for key, value in features.items()}
            dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3
            stats.update(newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
    finally:
        newtask.train(was_training)
    readings = stats.result()
    return {**readings,
            "mode": "eval",
            "eval_routing_share_attn": readings["g_hard_mean"],      # 判据 1 的读数（= 1.0 为 D1 签名）
            "routing_l1": readings["routing_l1_mean"],               # 判据 2 的读数（恒 < 1e-6，见 STE 精度留痕）
            "grad_norm": None,                                       # 判据 3 的读数来自训练轨迹，不在这里
            "noise_share_window": readings["noise_share_exact_window"],   # 保真判据的读数
            # D1 的**结构性**签名（不是 STOP 判据）：精确算术下 g_logit = σ(z) ∈ (0,1)。float32 下 sigmoid
            # 在 z ≳ 17 处饱和到 1.0、在 z ≲ −104 处下溢到 0.0，此时该读数可为 False——那是同一签名的
            # 更强形式（门控彻底饱和），不是复现偏离；空装载器同样为 False（无样本可判）。
            "g_logit_interval": RULE_G_LOGIT_INTERVAL,
            "g_logit_in_unit_interval": bool(readings["n_samples"]
                                             and 0.0 < readings["g_logit_min"]
                                             and readings["g_logit_max"] < 1.0),
            "temperature": float(newtask.temperature),
            "affinity_input_dim": int(AFFINITY_INPUT_DIM),
            "affinity_hidden_units": int(AFFINITY_HIDDEN),
            "eval_g_soft_lower": EVAL_G_SOFT_MIN,
            "eval_g_soft_upper": EVAL_G_SOFT_MAX}


@torch.no_grad()
def noise_signal_probe(newtask, dnn_input, gen_rep, spec_reps, env_embs, *,
                       draws: int = NOISE_PROBE_DRAWS, seed: int = NOISE_PROBE_SEED) -> dict:
    """噪声 / 信号分解（D1+D4 的实测口径；**私有 Generator ⇒ 不动全局 RNG**，不写 `.grad`）。

    * 对每样本重复抽 `draws` 次 Gumbel 噪声 → 经验路由概率 `noise_share_mc_*`（噪声轴）；
    * 与闭式解 `noise_share_exact_*` 比对（`mc_exact_max_abs_error`）；
    * `flip_rate_mc`：同一样本换一次噪声就翻面的比例（噪声把判据变成硬币）；
    * `g_soft_within_sample_std`（噪声轴离散度）vs `g_soft_between_sample_std`（样本轴离散度）→ 比值。
    """
    if not isinstance(newtask, HistoricalAffinityGateNewTask):
        raise ValueError(f"affinity 探针只适用于 {HistoricalAffinityGateNewTask.__name__}，"
                         f"收到 {type(newtask).__name__}")
    was_training = newtask.training
    try:
        newtask.eval()                                       # 基础读数：无噪、无 RNG
        base = newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        g_logit = base["g_logit"]
        generator = torch.Generator(device=g_logit.device).manual_seed(seed)
        decisions, soft = [], []
        for _ in range(int(draws)):
            terms = newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs,
                                       noise=gumbel_noise_like(g_logit, generator=generator),
                                       use_noise=True)
            decisions.append(terms["g_hard"])
            soft.append(terms["g_soft"])
    finally:
        newtask.train(was_training)

    hard = torch.stack(decisions).double().cpu()             # [M, B]
    soft_stack = torch.stack(soft).double().cpu()            # [M, B]
    share = hard.mean(dim=0)                                 # [B] 每样本经验路由概率
    exact = exact_attention_share(base["presigmoid"]).reshape(-1).cpu()
    flips = float((hard[1:] != hard[:-1]).double().mean()) if hard.shape[0] > 1 else 0.0
    within = float(soft_stack.std(dim=0, unbiased=False).mean())      # 噪声轴
    between = float(soft_stack.mean(dim=0).std(unbiased=False))       # 样本轴
    share_min, share_max = float(share.min()), float(share.max())
    return {
        "draws": int(draws), "seed": int(seed), "n_samples": int(hard.shape[1]),
        "noise_share_mc_mean": float(share.mean()),
        "noise_share_mc_min": share_min,
        "noise_share_mc_max": share_max,
        "noise_share_mc_window": share_max - share_min,
        "noise_share_exact_mean": float(exact.mean()),
        "noise_share_exact_min": float(exact.min()),
        "noise_share_exact_max": float(exact.max()),
        "noise_share_exact_window": float(exact.max() - exact.min()),
        "mc_exact_max_abs_error": float((share - exact).abs().max()),
        "flip_rate_mc": flips,
        "flip_rate_exact_mean": float(exact_flip_rate(base["presigmoid"]).mean()),
        "train_eval_disagreement_mc": 1.0 - float(share.mean()),
        "g_soft_within_sample_std": within,
        "g_soft_between_sample_std": between,
        "noise_to_signal_ratio": ratio(within, between),
        "gumbel_share_lower": GUMBEL_SHARE_LO,
        "gumbel_share_upper": GUMBEL_SHARE_HI,
        "gumbel_share_window_max_exact": GUMBEL_SHARE_WINDOW_MAX,
        "flip_rate_min_exact": FLIP_RATE_MIN,
        "flip_rate_max_exact": FLIP_RATE_MAX,
    }


def gradient_probe(newtask, dnn_input, gen_rep, spec_reps, env_embs, *,
                   seed: int = NOISE_PROBE_SEED, tau: float = GUMBEL_TAU,
                   label: float = 1.0) -> dict:
    """三条路径的 affinity_mlp 梯度范数（用 `torch.autograd.grad` 显式取 ⇒ **不写 `.grad`**、不影响训练）。

      * `grad_norm_historical`：历史路径（双 sigmoid + Gumbel 噪声，私有 generator，训练模式）；
      * `grad_norm_noiseless`：去掉噪声（eval 分支，双 sigmoid，硬判据恒 1）；
      * `grad_norm_corrected`：D6 修正路径（把前 sigmoid 输出 `z` 当 logit：`sigmoid(z/τ)`，无噪声）；
      * 三个衰减比：噪声衰减 `noiseless/historical`、双 sigmoid 衰减 `corrected/noiseless`、总衰减。

    伪标签固定为 `label`（默认全 1）：本函数只比较三条路径的**相对**大小，不构成训练信号。
    """
    if not isinstance(newtask, HistoricalAffinityGateNewTask):
        raise ValueError(f"affinity 探针只适用于 {HistoricalAffinityGateNewTask.__name__}，"
                         f"收到 {type(newtask).__name__}")
    params = list(newtask.affinity_mlp.parameters())
    y = torch.full((dnn_input.shape[0],), float(label), device=dnn_input.device)
    loss_func = nn.BCELoss()
    was_training = newtask.training

    def _norm(pred) -> float:
        loss = loss_func(pred, y)
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
        return float(sum(float(g.detach().double().norm(2)) ** 2 for g in grads if g is not None) ** 0.5)

    try:
        newtask.train()
        generator = torch.Generator(device=dnn_input.device).manual_seed(seed)
        historical = _norm(newtask(dnn_input, gen_rep, spec_reps, env_embs, generator=generator))
        newtask.eval()
        noiseless = _norm(newtask(dnn_input, gen_rep, spec_reps, env_embs))
        terms = newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)     # eval：无噪、无 RNG
        g_soft = torch.sigmoid(terms["presigmoid"] / tau)                       # D6 修正：z 当 logit
        g_hard = (g_soft > 0.5).float()
        g = g_hard.detach() + g_soft - g_soft.detach()
        w = g.view(-1, 1, 1) * terms["w_attn"] + (1 - g.view(-1, 1, 1)) * terms["w_fw"]
        corrected = _norm(routing_pipeline(newtask, dnn_input, gen_rep, spec_reps, env_embs, w))
    finally:
        newtask.train(was_training)
    return {
        "grad_norm_historical": historical,
        "grad_norm_noiseless": noiseless,
        "grad_norm_corrected": corrected,
        "attenuation_noise": ratio(noiseless, historical),
        "attenuation_double_sigmoid": ratio(corrected, noiseless),
        "attenuation_total": ratio(corrected, historical),
        "tau": float(tau), "seed": int(seed), "pseudo_label": float(label),
        "local_sensitivity_logit_max": 0.25,      # max_z σ′(z)：历史把 logit 压扁后的局部灵敏度上限
        "local_sensitivity_corrected_max": 1.0 / float(tau),   # 修正路径 max_z (1/τ)σ′(z/τ)
    }


def degeneracy_verdict(*, eval_share: float, routing_l1: float, grad_norm: float,
                       noise_window: float) -> dict:
    """主导预注册判据：机制 STOP 三条任一命中 ⇒ CONFIRMED_DEGENERATE（效应阈值不再解释）。

    输入口径：`eval_share` / `routing_l1` / `noise_window` 取 **best_state 载入后**的 val 诊断
    （`evaluate_gate` 的 `eval_routing_share_attn` / `routing_l1` / `noise_share_window`）；
    `grad_norm` 取训练轨迹的 `grad_norm_max`（"门控从未收到过任何梯度"）。
    保真判据（`noise_window > 0.32`）单独由 `fidelity_verdict` 判定，优先级更高。
    """
    checks = {
        "eval_share_ge_0.999": {"rule": RULE_EVAL_SHARE, "threshold": float(EVAL_ATTN_SHARE_MIN),
                                "observed": float(eval_share),
                                "triggered": bool(eval_share >= EVAL_ATTN_SHARE_MIN)},
        "routing_l1_lt_1e-6": {"rule": RULE_ROUTING_L1, "threshold": float(ROUTING_L1_MIN),
                               "observed": float(routing_l1),
                               "triggered": bool(routing_l1 < ROUTING_L1_MIN)},
        "grad_norm_eq_0": {"rule": RULE_GRAD_NORM, "threshold": float(GRAD_NORM_MIN),
                           "observed": float(grad_norm),
                           "triggered": bool(grad_norm == GRAD_NORM_MIN)},
    }
    triggered = [check["rule"] for check in checks.values() if check["triggered"]]
    return {"status": "CONFIRMED_DEGENERATE" if triggered else "NOT_DEGENERATE",
            "triggered_rules": triggered, "checks": checks,
            "observed": {"eval_share": float(eval_share), "routing_l1": float(routing_l1),
                         "grad_norm": float(grad_norm)}}


def fidelity_verdict(*, noise_window: float, window_max: float = NOISE_SHARE_WINDOW_MAX) -> dict:
    """保真判据：D1 的解析窗口上界 `1 − e^{−e} − (1 − e^{−1}) = 0.3019`。

    忠实复现下 `noise_window` **不可能**超过该上界（`g_logit = sigmoid(z) ∈ (0,1)` 是恒等式）；
    一旦超出，说明复现偏离历史（例如"顺手修好"D1），此时任何机制结论都不可信 ⇒ 优先 STOP。
    """
    return {"rule": RULE_NOISE_WINDOW, "threshold": float(window_max),
            "observed": float(noise_window),
            "analytic_max": float(GUMBEL_SHARE_WINDOW_MAX),
            "triggered": bool(noise_window > window_max)}


def arm_verdict(test_auc: float, *, degeneracy: dict, fidelity: dict,
                auc_min: float = AUC_TEST_MIN) -> dict:
    """臂级判定：保真 > 退化 > 效应。任何 STOP 下 `effect.evaluated = false`、`pass = null`。"""
    fidelity_break = bool(fidelity["triggered"])
    degenerate = degeneracy["status"] == "CONFIRMED_DEGENERATE"
    stopped = fidelity_break or degenerate
    effect = {"rule": RULE_EFFECT, "threshold": float(auc_min), "observed": float(test_auc),
              "evaluated": not stopped,
              "pass": None if stopped else bool(test_auc >= auc_min)}
    if fidelity_break:
        status, stop_reason = "STOP", "fidelity_break"
    elif degenerate:
        status, stop_reason = "STOP", "degenerate"
    else:
        status, stop_reason = ("EFFECT_CONFIRMED" if effect["pass"] else "EFFECT_NOT_CONFIRMED"), None
    return {"status": status, "pass": bool(status == "EFFECT_CONFIRMED"), "stop_reason": stop_reason,
            "fidelity": fidelity, "degeneracy": degeneracy, "effect": effect}


def preregistered_criteria() -> dict:
    """落盘用的预注册判据快照（与 experiments 文档第 6 节一致；改阈值必须先改文档再改这里）。"""
    return {"eval_attn_share_min": float(EVAL_ATTN_SHARE_MIN),
            "routing_l1_min": float(ROUTING_L1_MIN),
            "grad_norm_min": float(GRAD_NORM_MIN),
            "noise_share_window_max": float(NOISE_SHARE_WINDOW_MAX),
            "noise_share_window_analytic_max": float(GUMBEL_SHARE_WINDOW_MAX),
            "auc_test_min": float(AUC_TEST_MIN),
            "presigmoid_saturation_abs": float(PRESIGMOID_SATURATION_ABS),
            "gumbel_tau": float(GUMBEL_TAU),
            "gumbel_share_bounds": [float(GUMBEL_SHARE_LO), float(GUMBEL_SHARE_HI)],
            "rules": {"stop_mechanism": [RULE_EVAL_SHARE, RULE_ROUTING_L1, RULE_GRAD_NORM],
                      "stop_fidelity": RULE_NOISE_WINDOW,
                      "effect": RULE_EFFECT},
            "dominance": "stop_fidelity > stop_mechanism > effect"}


def provenance() -> dict:
    """审计溯源快照（写进 `metrics.json:affinity_arm`，便于把 run 与历史实现对起来）。"""
    return {"historical_source": "archive/exploration: run_newtask_from_ckpt.py (mode='affinity_gate') + "
                                 "multitaskrec/model.py NewTask(fusion_mode='affinity_gate')",
            "historical_claim": "archive/exploration: logs/all_experiment_results.md —— '逐样本 cosine similarity "
                                "特征能有效检测 OOD 样本并切换到 FW'；arch_notes.md —— 'OOD-aware 硬切换，"
                                "最稳定（σ=0.0014）但均值偏低，过于保守'",
            "historical_result": "archive/exploration: logs/all_experiment_results.md "
                                 "(CensusIncome affinity_gate mean test AUC 0.8601±0.0026, Δ vs prompt −0.0028; "
                                 "AliCCP 0.6714±0.0014)",
            "env_layout": ENV_LAYOUT,                         # env_embs 元素一维 [rep_dim] ⇒ stack(dim=1)
            "env_normalize_dim": int(ENV_NORMALIZE_DIM),      # = 特征维 ⇒ 逐任务列归一化 = 真余弦（非缺陷）
            "d2_audit_correction": "撤回'归一化维错误（dim=0 应为 dim=-1）'之说：get_infos 以 0 维索引取 "
                                   "env 嵌入（每个 [rep_dim]），stack(dim=1) 得 [rep_dim, K]，dim=0 即特征维；"
                                   "保留的 D2 是设计层问题（门控特征与 W_attn 同源；K=2 时第 4 维是前三维的"
                                   "函数 ⇒ 实际只有 2 个自由数）",
            "gumbel_noise_rng": "global",                     # D3：训练期每步 torch.rand_like 吃全局 RNG
            "gumbel_tau": float(GUMBEL_TAU), "temperature": float(TEMPERATURE),
            "affinity_input_dim": int(AFFINITY_INPUT_DIM), "affinity_hidden_units": int(AFFINITY_HIDDEN),
            "affinity_init": "default_torch",                 # 历史未做常量初始化（与 CGR 不同）
            "affinity_mlp_in_l2": False,                      # D5：get_l2_reg 只正则 tower（继承 master）
            "eval_hard_gate": "constant_1",                   # D1：eval 期 g_hard ≡ 1（对任意参数/输入）
            "rng_construction_order": "historical",           # affinity_mlp 建在 projection 与 gate 之间
            "spec": "docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-revalidation.md"}


def build_newtask(variant: str, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                  temperature: float = TEMPERATURE) -> NewTask:
    """变体工厂：`baseline` → 基线 `NewTask`（逐位同 master）；`affinity` → 忠实复现的处理臂。"""
    if variant == BASELINE_VARIANT:
        return NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
    if variant == VARIANT:
        return HistoricalAffinityGateNewTask(
            input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
            reg_dnn=reg_dnn, device=device, temperature=temperature)
    raise ValueError(f"未知 variant: {variant!r}（可选 {VARIANTS}）")


def stage2_config(variant: str) -> dict:
    """处理臂专属的 config.json 字段（基线臂不写，避免污染基线键集）。"""
    if variant == BASELINE_VARIANT:
        return {}
    return {"fusion_mode": FUSION_MODE, "affinity_tau": float(GUMBEL_TAU),
            "affinity_hidden_units": int(AFFINITY_HIDDEN),
            "noise_probe_draws": int(NOISE_PROBE_DRAWS), "noise_probe_seed": int(NOISE_PROBE_SEED)}
