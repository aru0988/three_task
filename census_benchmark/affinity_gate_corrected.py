"""修正版逐样本 Affinity 软门控：确定性、无 Gumbel、无 STE、train/eval 同一公式。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-corrected.md

被修正的历史缺陷（`exp/stage2-affinity-gate-repro` @ `88d787c` 的忠实复现在公平协议下单 seed 短跑得到
`status = STOP` / `CONFIRMED_DEGENERATE`，见 docs/superpowers/experiments/
2026-09-30-stage2-affinity-gate-revalidation.md 第 3、8 节）：

  D1  双 sigmoid + eval 期硬门控 `g_hard ≡ 1`（对任意参数、任意输入）⇒ 评测路径上根本没有路由，
      被评的模型就是纯 attention；
  D1(c)/D6 训练期 Gumbel 硬切换是噪声主导的硬币（`noise_to_signal_ratio = 15.68`、`flip_rate ≈ 0.126`），
      且不存活到 eval（训练/评测路由错配 ≥ 0.066）；
  D3  训练期前向每步消耗全局 RNG（`torch.rand_like`）+ 构造顺序位移（`affinity_mlp` 建在 projection 与
      gate 之间 ⇒ 同 seed 下共享参数初值与基线**不同**）；
  D4  硬 STE + 噪声污染梯度 ⇒ 门控被自身动力学推向恒 attention 角（`presigmoid_abs_mean` 1.65 → 6.49）；
  D2(b) 4 维门控特征全部由同一个 `H(x)` 派生（与 `W_attn` 同源），K=2 时第 4 维是前三维的函数 ⇒
      门控实际只看到 2 个自由数。

本模块的修正（**不主张任何新颖性**：这在 soft gating / 门控融合 / MoE 路由的已有大类之内）：

  1. **构造**：先走基线 `NewTask.__init__`，门控参数在**其后**追加 ⇒ 共享参数与它们消耗的全局 RNG
     与基线逐位一致（D3 的构造位移被消掉）；门控前向不抽任何随机数（D3 的随机流位移被消掉）。
  2. **单 sigmoid**：`gate_logit = MLP(features)` 无约束、`g = sigmoid(gate_logit)` 恰好一次（D1）；
     `MLP` 内不得有 `nn.Sigmoid`。末层 bias 恒零、权重按 `GATE_INIT_WEIGHT_SCALE` 缩小 ⇒ 初始 gate
     均值 ≈ 0.5、非饱和，且特征通路在第 1 步就有非零梯度（不做 zero-init 的"死启动"）。
  3. **同一公式 train / eval**：无噪声、无硬判据、无 STE、无 `self.training` 分支（D1(c)/D4/D6）。
  4. **blend**：`W = g·W_attn + (1−g)·W_fw`，`W_fw = 1/K` 均匀路由；`g` 逐样本（无 batch 轴归约）。
  5. **特征（3 维，可解释、非同一来源）**：
     `[注意力归一化熵, cos(gen_rep, attention 融合源表征), cos(gen_rep, 均匀融合源表征)]`。
     第 1 维是"这次路由有多自信"；第 2、3 维是"通用表征与两种候选融合有多一致"——后者依赖
     `gen_rep` / `spec_reps`，是历史 4 维特征集（只依赖 `H(x)` 与 `E`）**结构上看不到**的信息（D2(b)）。
  6. **源贡献诊断（val、训练与选点全部结束之后、零泄漏）**：留一源（L1O）边际预测效应、逐源依赖份额、
     gate 与"attention 路由相对均匀路由的逐样本效用"的秩相关。**标量 gate 没有逐源偏好** ⇒ 逐源
     "权重 vs 依赖"相关只作诊断（且带力学耦合），不设阈值、不参与判定——见
     `source_contribution_probe` 的 `per_source_correlation_note`。

预注册判据（**只允许在看到结果之前修改**；与 experiments 文档逐字同数）：

    机制（任一不满足 ⇒ STOP_MECHANISM，主判据）：
      gate_std >= 0.01   且  0.05 <= gate_mean <= 0.95   且  routing_l1_mean >= 1e-4
      且 gate_grad_norm_min_step > 0（每个训练步非零）  且 train_eval_identical  且 no_global_rng_consumed
    效应（仅机制通过后才解释）：
      auc_test_education >= 0.8521（基线 0.8500685307175756）
    对齐（第三层；不达标**不**翻转效应结论）：
      spearman(gate, utility) >= 0.05

用法（接线见 `run_census_benchmark.run_stage2`）：

    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --variant affinity_corrected --tag short
    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --affinity-corrected        --tag short
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from multitaskrec.model import NewTask

# ---- 臂与接线（沿用其它 exp/* 分支的"协议 run_id + 臂后缀"约定）----
BASELINE_VARIANT = "baseline"
VARIANT = "affinity_corrected"
VARIANTS = (BASELINE_VARIANT, VARIANT)
RUN_ID_SUFFIX = "-affcorr"
FUSION_MODE = "affinity_gate_corrected"

# ---- 门控结构（小 MLP：3 → 8 → 1，共 41 个参数）----
FEATURE_NAMES = ("attn_entropy_norm", "cos_gen_attn", "cos_gen_fw")
GATE_FEATURE_DIM = len(FEATURE_NAMES)
GATE_HIDDEN = 8
GATE_INIT_WEIGHT_SCALE = 0.1               # 末层权重相对默认初始化的缩放（中性、非饱和、非零）
ENTROPY_EPS = 1e-12                        # 熵里的 log(0) 保护（对 one-hot 的贡献 ~1e-11）
PRED_EPS = 1e-7                            # 逐样本 BCE 的概率夹逼（防 log(0) = inf）
L1O_EPS = 1e-12                            # 留一权重重新归一化的分母下限
SATURATION_ABS = 8.0                       # 饱和读数阈值：|logit| ≥ 8 ⇒ sigmoid' ≤ 3.4e-4
# env_embs 的元素是一维 [rep_dim]（`MPTRec.get_infos` 逐个 index 取 env 嵌入），stack(dim=1) 得
# [rep_dim, K]（特征在前、任务在后）。本模块**不**归一化 E（不使用历史 C 特征），所以不存在
# "归一化维"问题；若引用该事实，唯一正确表述见 `provenance()["env_normalize_note"]`。
ENV_LAYOUT = "feature_first_(rep_dim, K)"

# ---- 预注册阈值（只允许在看到结果之前修改；规则文本 = 预注册原文，单独硬编码）----
AUC_TEST_MIN = 0.8521                      # 效应阈值
BASELINE_TEST_AUC = 0.8500685307175756     # 协议基线行 20260929-1735-s20260929-m1685480945-short-904f8d0
GATE_STD_MIN = 0.01                        # 逐样本 gate 标准差下限（val、ddof=0）
GATE_MEAN_MIN, GATE_MEAN_MAX = 0.05, 0.95  # 未坍缩窗口
ROUTING_L1_MIN = 1e-4                      # mean_x ||W − W_attn||_1 下限（"确实偏离纯 attention"）
GATE_GRAD_NORM_MIN = 0.0                   # 判据为**严格** > 0（每个训练步）
ALIGNMENT_SPEARMAN_MIN = 0.05              # 保守阈值：n≈5e4 时零假设下 SE ≈ 4.5e-3 ⇒ 0.05 ≈ 11 SE

RULE_GATE_STD = "gate_std >= 0.01"
RULE_GATE_MEAN = "0.05 <= gate_mean <= 0.95"
RULE_ROUTING_L1 = "routing_l1_mean >= 1e-4"
RULE_GATE_GRAD = "gate_grad_norm_min_step > 0"
RULE_TRAIN_EVAL = "train_eval_identical"
RULE_RNG = "no_global_rng_consumed"
RULE_EFFECT = "auc_test_education >= 0.8521"
RULE_ALIGNMENT = "spearman(gate, utility) >= 0.05"


# ---- 纯函数：门控几何 / 路由 / 相关性（逐样本；只允许在源轴与特征轴上归约）----
def normalized_attention_entropy(w_attn: torch.Tensor, eps: float = ENTROPY_EPS) -> torch.Tensor:
    """注意力路由的归一化熵 ∈ [0, 1]（1 = 完全均匀 = 最不自信，0 = 完全 one-hot = 最自信）。

    除以 `log K` ⇒ 与源数无关，可跨 K 比较。`w_attn` 形状 [B, K]，返回 [B]。
    """
    p = w_attn.clamp_min(eps)
    entropy = -(w_attn * p.log()).sum(dim=-1)
    num_tasks = int(w_attn.shape[-1])
    return entropy / (math.log(num_tasks) if num_tasks > 1 else 1.0)


def uniform_routing(w_attn: torch.Tensor) -> torch.Tensor:
    """均匀路由 `W_fw = 1/K`（与 `w_attn` 同形状 [B, K]）。"""
    return torch.full_like(w_attn, 1.0 / w_attn.shape[-1])


def blend_routing(w_attn: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """`W = g·W_attn + (1−g)·W_fw`（`g` 形状 [B]，返回 [B, K, 1]：与基线融合管线的形状约定一致）。"""
    w_fw = uniform_routing(w_attn)
    weight = g.reshape(-1, 1, 1)
    return weight * w_attn.unsqueeze(2) + (1.0 - weight) * w_fw.unsqueeze(2)


def leave_one_out_weights(w: torch.Tensor, k: int, *, eps: float = L1O_EPS):
    """把第 k 个源的权重置 0 并在其余源上重新归一（K=2 时即 one-hot 到另一个源）。

    其余源权重之和 ≤ eps（全部权重都在被移除的源上）时，该样本的"留一"反事实在数学上**无定义**：
    回退为其余源上的均匀权重，并用第二个返回值（[B] 布尔）计数。K=2 时回退值与极限一致，故不影响数值。

    `w` 形状 [B, K, 1]；返回 (w_l1o [B, K, 1], undefined [B])。
    """
    keep = torch.ones_like(w)
    keep[..., int(k), :] = 0.0
    masked = w * keep
    denom = masked.sum(dim=1, keepdim=True)
    undefined = denom <= eps
    fallback = keep / keep.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.where(undefined, fallback, masked / denom.clamp_min(eps)), undefined.reshape(-1)


def routing_pipeline(newtask, dnn_input, gen_rep, spec_reps, w):
    """`W` 之后的融合管线：与基线 `NewTask.forward` **逐行同式**（含 `.squeeze()` 在批大小为 1 时的语义）。

    `new_spec_rep = Σ_k W_k·spec_rep_k` → `env_aware = new_env_emb ⊙ new_spec_rep`
    → `fused = [env_aware, gen_rep]·gate_network(x)` → `tower(fused)`。
    """
    new_env_emb = newtask.env_embedding_network(newtask.new_env_idx).squeeze(0)
    new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), w).squeeze()
    env_aware_rep = new_spec_rep * new_env_emb
    gate_out = newtask.gate_network(dnn_input).unsqueeze(dim=2)
    all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
    fused_rep = torch.matmul(all_reps, gate_out).squeeze()
    return newtask.tower_network(fused_rep).squeeze()


def gate_grad_norm(newtask) -> float:
    """门控 MLP 全参数梯度 L2 范数（`backward()` 之后、`optimizer.step()` 之前读取；全 None ⇒ 0.0）。"""
    total = 0.0
    for param in newtask.routing_gate_mlp.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().double().norm(2)) ** 2
    return total ** 0.5


def _as_float_array(values) -> np.ndarray:
    if torch.is_tensor(values):
        values = values.detach().cpu().double().numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def correlation_summary(x, y) -> dict:
    """Pearson 与 Spearman（平均秩）相关；样本数 < 2 或任一输入无方差 ⇒ 对应值为 None（不得返回 nan）。"""
    xs, ys = _as_float_array(x), _as_float_array(y)
    if xs.size != ys.size:
        raise ValueError(f"x 与 y 长度不一致: {xs.size} vs {ys.size}")
    out = {"n": int(xs.size), "pearson": None, "spearman": None}
    if xs.size >= 2 and float(xs.std()) > 0.0 and float(ys.std()) > 0.0:
        out["pearson"] = float(np.corrcoef(xs, ys)[0, 1])
        out["spearman"] = float(np.corrcoef(rankdata(xs), rankdata(ys))[0, 1])
    return out


# ---- 门控头（基线 NewTask 的**子类**；不动 multitaskrec/model.py）----
def build_gate_mlp(init_weight_scale: float = GATE_INIT_WEIGHT_SCALE) -> nn.Sequential:
    """门控 MLP：`Linear(3, 8) → ReLU → Linear(8, 1)`；末层**无激活**（输出是无约束 logit）。

    中性初始化：末层 bias 置零（无先验偏移）、权重按 `init_weight_scale` 缩小（初始 |logit| 很小 ⇒
    不饱和）。缩放用 `mul_` / `zero_`（在 `no_grad` 下）⇒ **不消耗全局 RNG**；权重**不**置零，
    以免第 1 步特征通路的梯度恒为 0（zero-init 的"死启动"）。
    """
    mlp = nn.Sequential(nn.Linear(GATE_FEATURE_DIM, GATE_HIDDEN), nn.ReLU(), nn.Linear(GATE_HIDDEN, 1))
    with torch.no_grad():
        mlp[2].weight.mul_(float(init_weight_scale))
        mlp[2].bias.zero_()
    return mlp


class CorrectedAffinityGateNewTask(NewTask):
    """修正版逐样本软门控头。

    **构造顺序**：`super().__init__`（= 基线 `NewTask` 的模块与它们的全局 RNG 消耗，逐位同 master）
    → 再追加 `routing_gate_mlp` ⇒ 同 seed 下所有共享参数与基线**逐位一致**（历史 D3 的构造位移被消掉），
    且门控构造之后不再有基线模块 ⇒ 训练随机流（dropout / batch 顺序）与基线完全同源。

    `last_terms`：最近一次 `forward` 存下的**已 detach** 中间量，供训练轨迹读取（不重算 ⇒ 不抽 RNG）。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 init_weight_scale: float = GATE_INIT_WEIGHT_SCALE):
        super().__init__(input_size=input_size, rep_dim=rep_dim,
                         tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
        self.routing_gate_mlp = build_gate_mlp(init_weight_scale)
        self.last_terms = None

    # ---- 门控公式（纯函数：不改参数、不写状态、不抽随机数）----
    def gate_terms(self, dnn_input, gen_rep, spec_reps, env_embs) -> dict:
        """门控的全部中间量；`forward` 与所有诊断/轨迹共用同一口径。

        逐样本性：只在该样本自己的特征轴上归约（`W_attn` 的 softmax、源轴平均、cosine 的特征轴），
        不做任何跨 batch 的归约；因此同一样本在任意 batch 组成下的输出一致。
        """
        exist_env_embs = torch.stack(env_embs, dim=1)          # [rep_dim, K]
        h_out = self.projection_network(dnn_input)             # [B, rep_dim]
        # 布局守卫（fail-fast，不改变任何数值）：E 必须是 [rep_dim, K]（每个 env 嵌入是一维 [rep_dim]）
        if exist_env_embs.dim() != 2 or exist_env_embs.shape[0] != h_out.shape[-1]:
            raise ValueError(
                f"env_embs 布局应为 {ENV_LAYOUT}（每个元素一维 [rep_dim]），"
                f"实际 stack(dim=1) 后为 {tuple(exist_env_embs.shape)}，"
                f"projection 输出为 {tuple(h_out.shape)}")
        num_tasks = int(exist_env_embs.shape[1])

        # 路由（与基线同一 attention：同样的 H、E 与温度）
        w_attn = F.softmax(torch.mm(h_out, exist_env_embs) / self.temperature, dim=-1)   # [B, K]
        spec_stack = torch.stack(spec_reps, dim=2)                                       # [B, d, K]
        fused_attn = torch.matmul(spec_stack, w_attn.unsqueeze(2)).squeeze(2)             # [B, d]
        fused_fw = spec_stack.mean(dim=2)                                                 # [B, d]

        # 3 维门控特征：[注意力自信度, 与 attention 融合源表征的一致性, 与均匀融合源表征的一致性]
        attn_entropy = normalized_attention_entropy(w_attn)                               # [B]
        cos_gen_attn = F.cosine_similarity(gen_rep, fused_attn, dim=-1)                   # [B]
        cos_gen_fw = F.cosine_similarity(gen_rep, fused_fw, dim=-1)                       # [B]
        gate_features = torch.stack([attn_entropy, cos_gen_attn, cos_gen_fw], dim=-1)     # [B, 3]

        # 恰好一次 sigmoid；logit 无约束（不是被压到 (0,1) 的"二次 sigmoid"输入）
        gate_logit = self.routing_gate_mlp(gate_features).squeeze(-1)                     # [B]
        g = torch.sigmoid(gate_logit)
        w_fw = uniform_routing(w_attn)
        w = blend_routing(w_attn, g)                                                      # [B, K, 1]
        return {"h_out": h_out, "env_embs": exist_env_embs, "w_attn": w_attn, "w_fw": w_fw,
                "fused_attn": fused_attn, "fused_fw": fused_fw, "attn_entropy": attn_entropy,
                "cos_gen_attn": cos_gen_attn, "cos_gen_fw": cos_gen_fw,
                "gate_features": gate_features, "gate_logit": gate_logit, "g": g, "w": w,
                "num_tasks": num_tasks, "temperature": float(self.temperature)}

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        terms = self.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        # 轨迹读取真实前向的中间量（detach 只脱图、不复制数据；不重算 ⇒ 不抽 RNG）
        self.last_terms = {key: (value.detach() if torch.is_tensor(value) else value)
                           for key, value in terms.items()}
        return routing_pipeline(self, dnn_input, gen_rep, spec_reps, terms["w"])


# ---- 读数累加器（流式、显存 O(1)；累加器恒在 CPU float64，与 metrics.py 的 GateStats 同口径）----
class RoutingGateStats:
    """门控读数（样本级聚合，与分批方式无关）：gate 分布、logit 饱和、注意力熵、逐源路由列均值、
    `routing_l1_*`（与纯 attention / 纯均匀的距离）。

    说明：**统计聚合必然跨样本求和**（读数口径，与 `metrics.py:GateStats` 一致），这类归约只影响
    **读数**、不影响任何模型输出；门控本身的逐样本性由 `gate_terms` 保证（源码守卫 + batch 组成无关测试）。
    空累加器：全部数值字段 0.0，列均值返回 `[]`。
    """

    def __init__(self) -> None:
        self.n_samples = self.n_batches = 0
        self.num_tasks = None
        self._g_sum = self._g_sq_sum = 0.0
        self._g_min = self._g_max = None
        self._logit_sum = self._logit_abs_sum = self._logit_abs_max = 0.0
        self._sat_count = 0
        self._entropy_sum = 0.0                    # 和式累加器：必须是数值 0（None 会让首次 update() 抛 TypeError）
        self._entropy_min = self._entropy_max = None   # 极值哨兵：惰性初始化（result() 把 None 读成 0.0）
        self._cos_attn_sum = self._cos_fw_sum = 0.0
        self._l1_sum = self._l1_max = self._l1_fw_sum = 0.0
        self._w_attn_sum = 0.0
        self._w_attn_col_sum = self._w_col_sum = self._feature_sum = None

    def _init_tasks(self, num_tasks: int) -> None:
        if self.num_tasks is None:
            self.num_tasks = int(num_tasks)
            self._w_attn_col_sum = torch.zeros(self.num_tasks, dtype=torch.float64)
            self._w_col_sum = torch.zeros(self.num_tasks, dtype=torch.float64)
            self._feature_sum = torch.zeros(GATE_FEATURE_DIM, dtype=torch.float64)
        elif self.num_tasks != int(num_tasks):
            raise ValueError(f"逐批 num_tasks 不一致: {self.num_tasks} vs {num_tasks}")

    def update(self, terms: dict) -> None:
        self._init_tasks(int(terms["num_tasks"]))
        g = terms["g"].detach().double().reshape(-1).cpu()
        logit = terms["gate_logit"].detach().double().reshape(-1).cpu()
        entropy = terms["attn_entropy"].detach().double().reshape(-1).cpu()
        cos_attn = terms["cos_gen_attn"].detach().double().reshape(-1).cpu()
        cos_fw = terms["cos_gen_fw"].detach().double().reshape(-1).cpu()
        features = terms["gate_features"].detach().double().reshape(-1, GATE_FEATURE_DIM).cpu()
        w = terms["w"].detach().double().reshape(-1, self.num_tasks).cpu()
        w_attn = terms["w_attn"].detach().double().cpu()
        w_fw = terms["w_fw"].detach().double().cpu()
        batch = int(g.numel())
        l1 = (w - w_attn).abs().sum(dim=1)
        l1_fw = (w - w_fw).abs().sum(dim=1)
        logit_abs = logit.abs()

        self.n_samples += batch
        self.n_batches += 1
        self._g_sum += float(g.sum())
        self._g_sq_sum += float((g * g).sum())
        self._logit_sum += float(logit.sum())
        self._logit_abs_sum += float(logit_abs.sum())
        self._logit_abs_max = max(self._logit_abs_max, float(logit_abs.max()))
        self._sat_count += int((logit_abs >= SATURATION_ABS).sum())
        self._entropy_sum += float(entropy.sum())
        self._cos_attn_sum += float(cos_attn.sum())
        self._cos_fw_sum += float(cos_fw.sum())
        self._l1_sum += float(l1.sum())
        self._l1_max = max(self._l1_max, float(l1.max()))
        self._l1_fw_sum += float(l1_fw.sum())
        self._w_attn_sum += float(w_attn.sum())
        self._w_attn_col_sum += w_attn.sum(dim=0)
        self._w_col_sum += w.sum(dim=0)
        self._feature_sum += features.sum(dim=0)
        if batch:
            self._g_min = float(g.min()) if self._g_min is None else min(self._g_min, float(g.min()))
            self._g_max = float(g.max()) if self._g_max is None else max(self._g_max, float(g.max()))
            self._entropy_min = (float(entropy.min()) if self._entropy_min is None
                                 else min(self._entropy_min, float(entropy.min())))
            self._entropy_max = (float(entropy.max()) if self._entropy_max is None
                                 else max(self._entropy_max, float(entropy.max())))

    def result(self) -> dict:
        n = self.n_samples
        mean = (self._g_sum / n) if n else 0.0
        var = ((self._g_sq_sum / n) - mean * mean) if n else 0.0
        return {
            "n_samples": n, "n_batches": self.n_batches, "num_tasks": self.num_tasks,
            "gate_mean": mean,
            "gate_std": math.sqrt(var) if var > 0.0 else 0.0,        # 总体标准差（ddof=0）
            "gate_min": self._g_min if self._g_min is not None else 0.0,
            "gate_max": self._g_max if self._g_max is not None else 0.0,
            "gate_logit_mean": (self._logit_sum / n) if n else 0.0,
            "gate_logit_abs_mean": (self._logit_abs_sum / n) if n else 0.0,
            "gate_logit_abs_max": self._logit_abs_max,
            "gate_logit_saturation_frac": (self._sat_count / n) if n else 0.0,
            "saturation_abs": SATURATION_ABS,
            "attn_entropy_mean": (self._entropy_sum / n) if n else 0.0,
            "attn_entropy_min": self._entropy_min if self._entropy_min is not None else 0.0,
            "attn_entropy_max": self._entropy_max if self._entropy_max is not None else 0.0,
            "cos_gen_attn_mean": (self._cos_attn_sum / n) if n else 0.0,
            "cos_gen_fw_mean": (self._cos_fw_sum / n) if n else 0.0,
            "cos_gen_diff_mean": ((self._cos_attn_sum - self._cos_fw_sum) / n) if n else 0.0,
            "routing_l1_mean": (self._l1_sum / n) if n else 0.0,
            "routing_l1_max": self._l1_max,
            "routing_l1_fw_mean": (self._l1_fw_sum / n) if n else 0.0,
            "w_attn_mean": (self._w_attn_sum / (n * self.num_tasks)) if (n and self.num_tasks) else 0.0,
            "w_attn_col_mean": (self._w_attn_col_sum / n).tolist() if (n and self.num_tasks) else [],
            "w_col_mean": (self._w_col_sum / n).tolist() if (n and self.num_tasks) else [],
            "gate_feature_mean": (self._feature_sum / n).tolist() if (n and self.num_tasks) else [],
            "gate_feature_names": list(FEATURE_NAMES),
        }


class TrainGateTrace:
    """训练期轨迹：逐 epoch 的门控读数 + 逐**步**的门控梯度范数（评据 4 的输入）。

    runner 顺序：`start_epoch(epoch)` → 每步在 `loss.backward()` 之后、`optimizer.step()` 之前
    `record_step(newtask)` → 一个 epoch 结束 `end_epoch()`。读数取**真实前向**的 `last_terms`
    （不重算 ⇒ 不抽 RNG、不改随机流）。
    """

    def __init__(self) -> None:
        self.stats = RoutingGateStats()
        self._per_epoch: list[dict] = []
        self._epoch = None
        self._epoch_stats = None
        self._epoch_grads: list[float] = []
        self._grads: list[float] = []

    def start_epoch(self, epoch: int) -> None:
        if self._epoch is not None:
            raise RuntimeError(f"epoch {self._epoch} 尚未 end_epoch()，不能开始 epoch {epoch}")
        self._epoch = int(epoch)
        self._epoch_stats, self._epoch_grads = RoutingGateStats(), []

    def record_step(self, newtask) -> float:
        if not isinstance(newtask, CorrectedAffinityGateNewTask):
            raise ValueError(f"轨迹只适用于 {CorrectedAffinityGateNewTask.__name__}，"
                             f"收到 {type(newtask).__name__}")
        if self._epoch is None:
            raise RuntimeError("record_step 之前必须先 start_epoch()")
        terms = newtask.last_terms
        if terms is None:
            raise RuntimeError("forward 尚未调用：轨迹读取真实前向的 last_terms（不重算）")
        norm = gate_grad_norm(newtask)
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
                  "grad_norm_min": min(grads) if grads else 0.0,
                  "grad_nonzero_steps": sum(1 for norm in grads if norm > 0.0),
                  "grad_zero_steps": sum(1 for norm in grads if norm == 0.0)}
        record.update(self._epoch_stats.result())
        self._per_epoch.append(record)
        self._epoch, self._epoch_stats, self._epoch_grads = None, None, []
        return record

    def per_epoch(self) -> list[dict]:
        return list(self._per_epoch)

    def result(self) -> dict:
        grads = self._grads
        return {**self.stats.result(),
                "epochs": len(self._per_epoch), "steps": len(grads),
                "grad_norm_mean": sum(grads) / len(grads) if grads else 0.0,
                "grad_norm_max": max(grads) if grads else 0.0,
                "grad_norm_min": min(grads) if grads else 0.0,     # 判据 4：空轨迹按 0（⇒ 判据命中）
                "grad_nonzero_steps": sum(1 for norm in grads if norm > 0.0),
                "grad_zero_steps": sum(1 for norm in grads if norm == 0.0),
                "per_epoch": self.per_epoch()}


# ---- 诊断（val、best_state 载入之后；不训练、不反传、不改参数、不抽随机数）----
def _require_corrected_head(newtask, who: str) -> None:
    if not isinstance(newtask, CorrectedAffinityGateNewTask):
        raise ValueError(f"{who} 只适用于 {CorrectedAffinityGateNewTask.__name__}，"
                         f"收到 {type(newtask).__name__}")


def _runtime_invariants(newtask, dnn_input, gen_rep, spec_reps, env_embs) -> dict:
    """机制判据 5、6 的运行时读数：同输入下 train / eval 前向**逐位一致**、且不消耗全局 RNG。"""
    cuda_checked = bool(dnn_input.device.type == "cuda" and torch.cuda.is_available())
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state() if cuda_checked else None
    was_training = newtask.training
    newtask.eval()
    out_eval = newtask(dnn_input, gen_rep, spec_reps, env_embs)
    newtask.train()
    out_train = newtask(dnn_input, gen_rep, spec_reps, env_embs)
    newtask.train(was_training)
    cpu_after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state() if cuda_checked else None
    return {"train_eval_identical": bool(torch.equal(out_eval, out_train)),
            "no_global_rng_consumed": bool(torch.equal(cpu_before, cpu_after)
                                           and (cuda_before is None or torch.equal(cuda_before, cuda_after))),
            "cuda_rng_checked": cuda_checked,
            "checked_batch_size": int(out_eval.reshape(-1).numel())}


@torch.no_grad()
def evaluate_routing_gate(newtask, backbone, loader, device) -> dict:
    """门控诊断：给定划分（协议路径上只用 val）前向一遍，累加读数并检查运行时不变式。

    调用点固定在**选点完成、best_state 已载入之后**；不参与 early stop / 选点 / test 评测，
    因此不可能改变 A/B 门禁或效应判定的结果。空 loader ⇒ `invariants = {}`（无样本可检）。
    """
    _require_corrected_head(newtask, "门控诊断")
    was_training = newtask.training
    stats = RoutingGateStats()
    invariants: dict = {}
    try:
        newtask.eval()
        backbone.eval()                                        # 三件套之 2：backbone 恒为 eval
        for _, _, _, features in loader:
            features = {key: value.to(device) for key, value in features.items()}
            dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3
            if not invariants:
                invariants = _runtime_invariants(newtask, dnn_input, gen_rep, spec_reps, env_embs)
            stats.update(newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
    finally:
        newtask.train(was_training)
    return {**stats.result(),
            "mode": "eval",
            "invariants": invariants,
            "temperature": float(newtask.temperature),
            "gate_feature_dim": int(GATE_FEATURE_DIM),
            "gate_hidden_units": int(GATE_HIDDEN),
            "gate_init_weight_scale": float(GATE_INIT_WEIGHT_SCALE),
            "env_layout": ENV_LAYOUT,
            "reading_scope": "best_state 载入后、只用给定划分前向一遍；不训练、不反传、不改参数、不抽随机数"}


@torch.no_grad()
def source_contribution_probe(newtask, backbone, loader, device, *, max_samples=None) -> dict:
    """源贡献诊断（只用 val；训练与选点**全部结束**之后运行，零泄漏）。

    给出四类读数：

    1. **留一源（L1O）边际预测效应**：`Δ_k(x) = p_W(x) − p_{W^{(−k)}}(x)`（把源 k 的权重置 0 并在其余
       源上重新归一）⇒ `source_reliance_mean_abs_delta[k]` / `source_reliance_mean_delta[k]`；
    2. **逐源依赖份额**：`share_k(x) = |Δ_k| / Σ_j |Δ_j|`（尺度无关，`Σ_k share_k = 1`）；
    3. **gate 实际移动了多少预测**：`pred_delta_vs_attn_mean_abs` / `pred_delta_vs_fw_mean_abs`；
    4. **对齐读数**：逐样本效用 `u(x) = BCE(p_fw, y) − BCE(p_attn, y)`（> 0 ⇒ 该样本上 attention 路由
       更优）与 `g(x)` 的 Spearman/Pearson（`gate_utility_*`）+ AUC（ties 排除在外）。

    **逐源相关性只作诊断**：标量 `g` 决定的是"attention 路由 vs 均匀路由"的整体混合比例，**没有逐源
    偏好** ⇒ "gate 偏好源 k" 不可识别；这里报告的 `routing_weight_vs_reliance_*`（路由权重 vs 依赖份额）
    还带力学耦合（权重本身进入融合、直接放大该源的边际效应），因此不设阈值、不参与判定。

    效用读 val 标签，但只在训练与选点结束之后计算，不进入任何梯度 / 选点 / test 路径。
    """
    _require_corrected_head(newtask, "源贡献诊断")
    was_training = newtask.training
    g_parts, u_parts, w_parts, share_parts = [], [], [], []
    abs_delta_sum = delta_sum = share_sum = None
    abs_attn_sum = abs_fw_sum = 0.0
    n_samples = n_undefined = n_utility_nonzero = n_utility_zero = 0
    num_tasks = 0
    try:
        newtask.eval()
        backbone.eval()
        for _, _, y, features in loader:
            if max_samples is not None and n_samples >= int(max_samples):
                break
            features = {key: value.to(device) for key, value in features.items()}
            dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
            terms = newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
            num_tasks = int(terms["num_tasks"])
            if abs_delta_sum is None:
                abs_delta_sum = [0.0] * num_tasks
                delta_sum = [0.0] * num_tasks
                share_sum = [0.0] * num_tasks
            w = terms["w"]                                               # [B, K, 1]
            pred_full = routing_pipeline(newtask, dnn_input, gen_rep, spec_reps, w)
            pred_attn = routing_pipeline(newtask, dnn_input, gen_rep, spec_reps,
                                         terms["w_attn"].unsqueeze(2))
            pred_fw = routing_pipeline(newtask, dnn_input, gen_rep, spec_reps,
                                       terms["w_fw"].unsqueeze(2))
            batch = int(pred_full.reshape(-1).numel())
            keep = batch if max_samples is None else min(batch, int(max_samples) - n_samples)
            if keep <= 0:
                break
            view = slice(0, keep)

            abs_deltas = []
            for k in range(num_tasks):
                w_k, undefined = leave_one_out_weights(w, k)
                delta_k = (pred_full - routing_pipeline(newtask, dnn_input, gen_rep, spec_reps,
                                                        w_k))[view]
                abs_deltas.append(delta_k.abs())
                abs_delta_sum[k] += float(delta_k.abs().double().sum())
                delta_sum[k] += float(delta_k.double().sum())
                n_undefined += int(undefined[view].sum())
            total_abs = sum(abs_deltas).clamp_min(L1O_EPS)       # 对**源轴**求和（不跨 batch）
            shares = [abs_deltas[k] / total_abs for k in range(num_tasks)]
            for k in range(num_tasks):
                share_sum[k] += float(shares[k].double().sum())

            labels = y.float().to(device)[view]
            bce_fw = F.binary_cross_entropy(pred_fw[view].clamp(PRED_EPS, 1.0 - PRED_EPS), labels,
                                            reduction="none")
            bce_attn = F.binary_cross_entropy(pred_attn[view].clamp(PRED_EPS, 1.0 - PRED_EPS), labels,
                                              reduction="none")
            utility = bce_fw - bce_attn                                  # > 0 ⇒ attention 路由更优
            g_parts.append(terms["g"][view].double().cpu().numpy())
            u_parts.append(utility.double().cpu().numpy())
            w_parts.append(w.reshape(batch, num_tasks)[view].double().cpu().numpy())
            share_parts.append(torch.stack(shares, dim=1).double().cpu().numpy())
            abs_attn_sum += float((pred_full - pred_attn)[view].abs().double().sum())
            abs_fw_sum += float((pred_full - pred_fw)[view].abs().double().sum())
            n_utility_nonzero += int((utility != 0).sum())
            n_utility_zero += int((utility == 0).sum())
            n_samples += keep
    finally:
        newtask.train(was_training)

    g_all = np.concatenate(g_parts) if g_parts else np.zeros(0)
    u_all = np.concatenate(u_parts) if u_parts else np.zeros(0)
    w_all = np.concatenate(w_parts).reshape(-1) if w_parts else np.zeros(0)
    share_all = np.concatenate(share_parts).reshape(-1) if share_parts else np.zeros(0)
    utility_corr = correlation_summary(g_all, u_all)
    source_corr = correlation_summary(w_all, share_all)
    alignment_auc = None
    if 0 < n_utility_nonzero < int(u_all.size):
        nonzero = u_all != 0.0
        binary = (u_all[nonzero] > 0.0).astype(int)
        if binary.min() != binary.max():
            alignment_auc = float(roc_auc_score(binary, g_all[nonzero]))

    n = max(n_samples, 1)
    zeros = [0.0] * max(num_tasks, 0)
    return {
        "n_samples": n_samples, "num_tasks": num_tasks,
        "max_samples": (None if max_samples is None else int(max_samples)),
        "source_reliance_mean_abs_delta": [v / n for v in (abs_delta_sum or zeros)],
        "source_reliance_mean_delta": [v / n for v in (delta_sum or zeros)],
        "source_reliance_share_mean": [v / n for v in (share_sum or zeros)],
        "l1o_undefined_count": n_undefined,
        "l1o_undefined_frac": (n_undefined / n) if n_samples else 0.0,
        "pred_delta_vs_attn_mean_abs": (abs_attn_sum / n) if n_samples else 0.0,
        "pred_delta_vs_fw_mean_abs": (abs_fw_sum / n) if n_samples else 0.0,
        "gate_utility_spearman": utility_corr["spearman"],
        "gate_utility_pearson": utility_corr["pearson"],
        "gate_utility_auc": alignment_auc,
        "n_utility_nonzero": n_utility_nonzero, "n_utility_zero": n_utility_zero,
        "utility_definition": "u(x) = BCE(p_fw, y) − BCE(p_attn, y)（逐样本；> 0 ⇒ 该样本上 attention "
                              "路由更优；概率夹逼到 [1e-7, 1−1e-7] 防 log(0)）",
        "routing_weight_vs_reliance_pearson": source_corr["pearson"],
        "routing_weight_vs_reliance_spearman": source_corr["spearman"],
        "per_source_correlation_status": "diagnostic_only",
        "per_source_correlation_note": (
            "标量 gate 只决定 attention 路由与均匀路由的整体混合比例，没有逐源偏好 ⇒ "
            "【gate 偏好源 k】不可识别；此处报告的是**路由权重** W_k 与**依赖份额** share_k 的相关性，"
            "且带力学耦合（W_k 本身进入融合、直接放大该源的边际效应），"
            "故只作诊断、不设阈值、不参与判定；与 gate 的**可识别**对齐读数是 gate_utility_*（见上）。"),
        "leakage_note": "本探针只在训练与选点全部结束之后运行；val 标签只用于效用读数，不进入任何梯度路径",
    }


# ---- 预注册判据与判定（机制 > 效应 > 对齐）----
def mechanism_verdict(*, gate_std: float, gate_mean: float, routing_l1_mean: float,
                      gate_grad_norm_min_step: float, train_eval_identical: bool,
                      rng_unchanged: bool) -> dict:
    """机制 6 条（任一不满足 ⇒ `MECHANISM_FAIL`；机制是主判据，不满足就不解释效应阈值）。"""
    checks = {
        "gate_std_ge_min": {"rule": RULE_GATE_STD, "threshold": float(GATE_STD_MIN),
                            "observed": float(gate_std), "passed": bool(float(gate_std) >= GATE_STD_MIN)},
        "gate_mean_in_window": {"rule": RULE_GATE_MEAN,
                                "threshold": [float(GATE_MEAN_MIN), float(GATE_MEAN_MAX)],
                                "observed": float(gate_mean),
                                "passed": bool(GATE_MEAN_MIN <= float(gate_mean) <= GATE_MEAN_MAX)},
        "routing_l1_ge_min": {"rule": RULE_ROUTING_L1, "threshold": float(ROUTING_L1_MIN),
                              "observed": float(routing_l1_mean),
                              "passed": bool(float(routing_l1_mean) >= ROUTING_L1_MIN)},
        "gate_grad_nonzero_every_step": {"rule": RULE_GATE_GRAD, "threshold": float(GATE_GRAD_NORM_MIN),
                                         "observed": float(gate_grad_norm_min_step),
                                         "passed": bool(float(gate_grad_norm_min_step) > GATE_GRAD_NORM_MIN)},
        "train_eval_identical": {"rule": RULE_TRAIN_EVAL, "threshold": True,
                                 "observed": bool(train_eval_identical),
                                 "passed": bool(train_eval_identical)},
        "no_global_rng_consumed": {"rule": RULE_RNG, "threshold": True,
                                   "observed": bool(rng_unchanged), "passed": bool(rng_unchanged)},
    }
    failed = [check["rule"] for check in checks.values() if not check["passed"]]
    return {"status": "MECHANISM_OK" if not failed else "MECHANISM_FAIL",
            "failed_rules": failed, "checks": checks,
            "observed": {"gate_std": float(gate_std), "gate_mean": float(gate_mean),
                         "routing_l1_mean": float(routing_l1_mean),
                         "gate_grad_norm_min_step": float(gate_grad_norm_min_step),
                         "train_eval_identical": bool(train_eval_identical),
                         "no_global_rng_consumed": bool(rng_unchanged)}}


def alignment_verdict(*, spearman, auc=None, n: int = 0,
                      threshold: float = ALIGNMENT_SPEARMAN_MIN) -> dict:
    """对齐判据（第三层）：gate 与逐样本效用 `u(x)` 的秩相关。`None`（无方差）按保守处理 ⇒ 不算通过。"""
    base = {"rule": RULE_ALIGNMENT, "threshold": float(threshold), "n": int(n),
            "auc": (None if auc is None else float(auc)),
            "threshold_rationale": "n≈5e4 时零假设下 Spearman 的标准误 1/sqrt(n−1) ≈ 4.5e-3 ⇒ "
                                   "0.05 ≈ 11 个标准误（保守）"}
    if spearman is None:
        return {**base, "observed": None, "pass": False, "status": "ALIGNMENT_UNDEFINED",
                "reason": "gate 或 utility 在 val 上无方差（常函数）⇒ Spearman 无定义，按保守处理"}
    passed = bool(float(spearman) >= threshold)
    return {**base, "observed": float(spearman), "pass": passed,
            "status": "ALIGNMENT_OK" if passed else "ALIGNMENT_FAIL",
            "reason": None if passed else "低于预注册保守阈值"}


def arm_verdict(test_auc: float, *, mechanism: dict, alignment: dict,
                auc_min: float = AUC_TEST_MIN) -> dict:
    """臂级判定：`机制 > 效应 > 对齐`。机制不过 ⇒ 不得谈效应（`pass = null`）。"""
    mechanism_ok = mechanism["status"] == "MECHANISM_OK"
    effect_pass = bool(float(test_auc) >= auc_min)
    effect = {"rule": RULE_EFFECT, "threshold": float(auc_min), "observed": float(test_auc),
              "evaluated": mechanism_ok, "pass": (effect_pass if mechanism_ok else None)}
    if not mechanism_ok:
        status, stop_reason = "STOP_MECHANISM", "mechanism_fail"
    elif not effect_pass:
        status, stop_reason = "STOP_EFFECT", "effect_below_threshold"
    elif alignment["status"] != "ALIGNMENT_OK":
        status, stop_reason = "PASS_ALIGNMENT_FAIL", "alignment_below_threshold"
    else:
        status, stop_reason = "PASS", None
    return {"status": status, "pass": bool(status == "PASS"), "stop_reason": stop_reason,
            "dominance": "mechanism > effect > alignment",
            "mechanism": mechanism, "effect": effect, "alignment": alignment}


def preregistered_criteria() -> dict:
    """落盘用的预注册判据快照（与 experiments 文档第 6 节逐字同数；改阈值必须先改文档再改这里）。"""
    return {"auc_test_min": float(AUC_TEST_MIN),
            "baseline_test_auc": float(BASELINE_TEST_AUC),
            "gate_std_min": float(GATE_STD_MIN),
            "gate_mean_min": float(GATE_MEAN_MIN), "gate_mean_max": float(GATE_MEAN_MAX),
            "routing_l1_min": float(ROUTING_L1_MIN),
            "gate_grad_norm_min": float(GATE_GRAD_NORM_MIN),
            "alignment_spearman_min": float(ALIGNMENT_SPEARMAN_MIN),
            "gate_init_weight_scale": float(GATE_INIT_WEIGHT_SCALE),
            "gate_hidden_units": int(GATE_HIDDEN),
            "gate_feature_names": list(FEATURE_NAMES),
            "rules": {"stop_mechanism": [RULE_GATE_STD, RULE_GATE_MEAN, RULE_ROUTING_L1,
                                          RULE_GATE_GRAD, RULE_TRAIN_EVAL, RULE_RNG],
                      "effect": RULE_EFFECT,
                      "alignment": RULE_ALIGNMENT},
            "dominance": "mechanism > effect > alignment"}


def provenance() -> dict:
    """审计溯源快照（写进 `metrics.json:affinity_corrected_arm`，便于与历史复现件逐条对照）。"""
    return {
        "historical_source": "archive/exploration: run_newtask_from_ckpt.py (mode='affinity_gate') + "
                             "multitaskrec/model.py NewTask(fusion_mode='affinity_gate')",
        "historical_repro": "docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-revalidation.md"
                            "（exp/stage2-affinity-gate-repro @ 88d787c：公平协议下单 seed 短跑 "
                            "status = STOP、CONFIRMED_DEGENERATE，AUC-Test-Education 0.8465147339710887 仅留痕）",
        "fixed_defects": {
            "D1": "双 sigmoid + eval 期硬门控恒选 attention（g_hard ≡ 1）⇒ 本模块单 sigmoid、无硬判据："
                  "g = sigmoid(logit) 就是**输出**，不再被当作 logit 二次 sigmoid",
            "D2": "门控特征与 W_attn 同源、K=2 时 4 维特征实际只有 2 个自由数 ⇒ 本模块改用 3 维"
                  "[注意力熵, cos(gen, attention 融合), cos(gen, 均匀融合)]：后两维依赖 gen_rep/spec_reps，"
                  "与路由权重不是同一来源，且满列秩（无精确线性恒等式）",
            "D3": "训练期每步 Gumbel 抽样（torch.rand_like）消耗全局 RNG + 构造顺序位移 ⇒ 本模块构造在"
                  "基线模块**之后**（共享参数与其 RNG 消耗与基线逐位一致），前向不抽任何随机数"
                  "（也不消耗全局 RNG）",
            "D4": "硬 STE + 噪声污染的梯度 ⇒ 本模块无 Gumbel、无 STE：梯度直接来自 sigmoid 的软输出",
            "D6": "eval 规则是参数的常函数、历史读数不可识别 ⇒ 本模块不需要反事实读出：训练与评测"
                  "走同一条公式（机制判据 5 在运行时逐位验证）",
        },
        "preserved_defect": {"D5": "门控不进 L2（历史行为，本模块**保留**）：get_l2_reg 继承基线只正则 "
                                   "tower；门控 MLP 仅 41 个参数，L2 影响可忽略"},
        "gate_in_l2": False,
        "gumbel_used": False,
        "ste_used": False,
        "hard_gate_used": False,
        "blend": "W = g·W_attn + (1−g)·W_fw（g 逐样本；W_fw = 1/K 均匀路由）",
        "gate_features": list(FEATURE_NAMES),
        "env_layout": ENV_LAYOUT,
        "env_normalize_note": ("本模块不归一化 E（不使用历史 C 特征），因此不涉及归一化维问题。"
                               "唯一正确表述：env_embs 的元素是一维 [rep_dim]，"
                               "torch.stack(env_embs, dim=1) 得 [rep_dim, K]（特征在前、任务在后），"
                               "归一化维 dim=0 就是特征维（逐任务列归一化）——**不是缺陷**；"
                               "历史复现件已**撤回**早前那条错误结论，本文档不得复述它。"),
        "non_novelty_note": ("非新颖性：与 soft gating / 门控融合 / MoE 路由（mixture-of-experts routing）"
                             "同族，是已有大类；本分支**不主张任何新颖性**，只回答【修正后的确定性软门控"
                             "能否在公平协议下同时通过机制与效应判据】这一可证伪问题。"),
        "spec": "docs/superpowers/experiments/2026-09-30-stage2-affinity-gate-corrected.md",
    }


# ---- 变体工厂与接线（基线臂逐位不受影响）----
def build_newtask(variant: str, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None) -> NewTask:
    """`baseline` → 基线 `NewTask`（逐位同 master）；`affinity_corrected` → 修正版软门控头。"""
    if variant == BASELINE_VARIANT:
        return NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
    if variant == VARIANT:
        return CorrectedAffinityGateNewTask(
            input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
            reg_dnn=reg_dnn, device=device)
    raise ValueError(f"未知 variant: {variant!r}（可选 {VARIANTS}）")


def stage2_config(variant: str) -> dict:
    """处理臂专属的 config.json 字段（基线臂不写，避免污染基线键集）。"""
    if variant == BASELINE_VARIANT:
        return {}
    return {"fusion_mode": FUSION_MODE,
            "gate_feature_dim": int(GATE_FEATURE_DIM),
            "gate_feature_names": list(FEATURE_NAMES),
            "gate_hidden_units": int(GATE_HIDDEN),
            "gate_init_weight_scale": float(GATE_INIT_WEIGHT_SCALE)}
