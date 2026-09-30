"""逐样本 CGR（per-sample confidence-gated routing）——修正后的工程消融实现。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-cgr-instance.md
上游协议：docs/superpowers/specs/2026-09-29-census-stage2-benchmark-design.md（划分 / 种子 / 真冻结 /
A-B 门禁 / SUMMARY 一律沿用；`protocol.py` · `metrics.py` · `multitaskrec/model.py` 零改动）

被修正的对象是 `exp/stage2-cgr-repro` 的忠实复现（该臂判定 `CONFIRMED_DEGENERATE`，
三条退化判据全部命中、`grad_norm ≡ 0`）。本模块把审计出的缺陷逐条修好，重新跑**一次**单 seed 短跑：

  * **D1 批级门控 → 逐样本门控**：门控输入只含逐样本量，不含任何批级归约；
    同一样本的 g 与批大小、批组成无关（测试 `test_same_sample_keeps_the_same_g_*` 锁定）。
  * **D2 RNG 构造顺序 → 共享参数先行**：`super().__init__()` 先建全部共享模块
    （env_embedding / projection / gate / tower，消耗与基线逐位相同的 RNG），门控参数在其**之后**追建。
  * **D3 常量初始化 + 对称不破缺 → 标准初始化 + 零初始化末层**：隐层用 `nn.Linear` 默认初始化
    （32 行互不相同），末层权重与偏置全零 ⇒ 预激活 z ≡ 0 ⇒ g ≡ 0.5（精确），sigmoid′(0) = 0.25，不饱和。
  * **D4 warmup 时序怪癖 → 无 warmup**：不存在 alpha 插值；g 就是 σ(z)。
    （理由：warmup 的历史动机是"门控起点不可信"；零初始化已使起点精确落在中性点 g = 0.5，
    再加 warmup 只会把 g 再乘一次 (1−alpha) 拉向 0.5，属于无依据的额外自由度。）
  * **D6 单样本批 NaN → 逐样本量 + `squeeze(-1)`**：不存在批内自由度 ≤ 0 的统计量；
    forward 用显式维度压缩，B=1 时不 NaN、不形状报错。
  * **D5 梯度信号恒 0 → 机制可学**：门控输出层在初始步即收到非零梯度；一步之后门控全部参数梯度非零。

门控（设 x = `dnn_input`，H = `projection_network`，E = stack(env_embs)，K = 源任务数，T = 150）：

    H_out(x)   ∈ R^D                   逐样本 attention 投影（末端 LayerNorm）
    W_attn(x)  = softmax(H_out(x) E / T)             [B, K]      既有 attention 权重
    p_max      = max_k W_attn[k]                     [B]         注意力置信度
    H_norm     = −Σ_k W_attn[k] log W_attn[k] / log K [B]        归一化熵（0 = 尖峰，1 = 均匀）
    spec_mix   = Σ_k W_attn[k] · spec_rep_k          [B, D]      attention 融合后的源表征
    agree      = cos(spec_mix, gen_rep)              [B]         与通用表征的一致性
    gate_in    = [H_out, p_max, H_norm, agree]       [B, D+3]    逐样本，无批级量
    z          = MLP(gate_in)   (D+3 → 32 → 1, ReLU) [B, 1]
    g          = σ(z) ∈ (0,1)                        [B, 1]
    W          = g·W_attn + (1−g)·W_fw               [B, K, 1]   W_fw = 1/K 均匀权重

W 之后的管线与基线 `NewTask.forward` 逐行相同；`get_l2_reg` 不重写（只正则 tower，与基线/历史同口径）。
参数量：门控 (D+3)·32 + 32 + 32 + 1（CensusIncome D=128 ⇒ 4257），远小于历史门控的 20545。

**非新颖性**：置信门控 / 条件化路由 / 门控融合（MoE 门控、FiLM、条件化归一化、prompt 调制）是已有大类。
本臂不是新方法，是一次**修正后的工程消融**；历史 CGR 在 CensusIncome 上也从未超越 prompt。

预注册判据（**只允许在看到结果之前修改**；规则原文单独硬编码，见 `RULE_*`）：

    机制门（任一不满足 ⇒ MECHANISM_FAILED ⇒ STOP，与 AUC 无关）：
      g_within_batch_std_mean >= 0.01   且   grad_norm > 0
      0.05 <= g_mean <= 0.95            且   weight_l1_mean >= 1e-4
    效应门（仅机制通过时才解释）：auc_test_education >= 0.8521（基线 0.8500685307175756）

用法（接线见 run_census_benchmark.run_stage2）：

    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --variant cgr-instance --tag short
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitaskrec.model import NewTask

# ---- 臂与接线 ----
BASELINE_VARIANT = "baseline"
VARIANT = "cgr-instance"
VARIANTS = (BASELINE_VARIANT, VARIANT)
RUN_ID_SUFFIX = "-cgrinst"                 # 沿用各 exp 分支"协议 run_id + 臂后缀"的约定

# ---- 门控设计（本模块固定；常量进 config.json / prereg 快照，便于复现）----
GATE_HIDDEN = 32                           # 隐层宽度（与历史门控同宽，便于对照）
GATE_SCALAR_FEATURES = ("p_max", "entropy_norm", "cos_agreement")
ATTENTION_TEMPERATURE = 150                # 基线 NewTask.temperature，未调参
WARMUP = False                             # 无 warmup（理由见模块 docstring）

# ---- 预注册判据（只允许在看到结果之前修改；规则文本是预注册原文，单独硬编码）----
AUC_TEST_MIN = 0.8521                      # 效应阈值（基线实测 0.8500685307175756）
BASELINE_TEST_AUC = 0.8500685307175756     # 协议基线行 20260929-1735-...-904f8d0 的 AUC-Test-Education
GATE_STD_MIN = 0.01                        # 机制 1：批内 g 的样本标准差（批间均值）下限
GRAD_NORM_MIN = 0.0                        # 机制 2：门控梯度范数下界（严格大于）
G_MEAN_MIN, G_MEAN_MAX = 0.05, 0.95        # 机制 3：门控均值区间（未坍缩到单一分支）
WEIGHT_L1_MIN = 1e-4                       # 机制 4：|W − W_attn| 的 L1 下限
PRESIGMOID_SATURATION_ABS = 8.0            # 饱和读数阈值：|z| ≥ 8 ⇒ σ′ ≤ 3.4e-4
RULE_GATE_STD = "g_within_batch_std_mean >= 0.01"
RULE_GRAD_NORM = "grad_norm > 0"
RULE_G_MEAN = "0.05 <= g_mean <= 0.95"
RULE_WEIGHT_L1 = "weight_l1_mean >= 1e-4"
RULE_EFFECT = "auc_test_education >= 0.8521"


def sigmoid_slope(presigmoid: float) -> float:
    """σ′(z) = σ(z)(1 − σ(z))，数值安全写法（大 |z| 时避免 exp 溢出）。"""
    if presigmoid >= 0:
        exp_neg = math.exp(-presigmoid)
        value = 1.0 / (1.0 + exp_neg)
    else:
        exp_pos = math.exp(presigmoid)
        value = exp_pos / (1.0 + exp_pos)
    return value * (1.0 - value)


class CGRInstanceNewTask(NewTask):
    """基线 `NewTask` 的**修正版逐样本置信门控**头（子类扩展，不改 `model.py`）。

    构造顺序（D2 修复的关键）：先 `super().__init__()` 建完全部共享模块——这一步消耗的全局 RNG
    与直接构造基线 `NewTask` **逐位相同**，因此同一 model seed 下 `projection_network` /
    `gate_network` / `tower_network` / `env_embedding_network` 与基线**逐位相同**；
    门控参数在其后追建，只多消耗之后的 RNG，不位移任何共享初值。

    门控输出层零初始化是**有意的、可检验的**设计：z ≡ 0 ⇒ g ≡ 0.5（精确）⇒
    W = 0.5·W_attn + 0.5·W_fw，起点正好落在"attention / 均匀"的中性混合，且 sigmoid 斜率最大（0.25），
    不饱和。代价是隐层在**第一步**拿不到梯度（∂z/∂hidden ∝ W_out = 0），一步之后即恢复。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 gate_hidden=GATE_HIDDEN):
        super(CGRInstanceNewTask, self).__init__(input_size=input_size, rep_dim=rep_dim,
                                                 tower_dnn_hidden_units=tower_dnn_hidden_units,
                                                 reg_dnn=reg_dnn, device=device)
        self.rep_dim = int(rep_dim)
        self.gate_hidden = int(gate_hidden)
        # 逐样本门控：隐层标准初始化（行互不相同，对称可破缺），末层零初始化（g 起点 = 0.5，不饱和）
        self.confidence_mlp = nn.Sequential(
            nn.Linear(self.gate_input_dim, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, 1),
        )
        nn.init.zeros_(self.confidence_mlp[2].weight)
        nn.init.zeros_(self.confidence_mlp[2].bias)

    @property
    def gate_input_dim(self) -> int:
        return self.rep_dim + len(GATE_SCALAR_FEATURES)

    def gate_features(self, dnn_input, gen_rep, spec_reps, env_embs) -> dict:
        """逐样本门控特征（纯函数：不改参数、不写状态；forward 与诊断共用同一口径）。

        返回（B = 批大小，D = rep_dim，K = 源任务数）：
          `exist_env_embs` [D, K]；`h_out` [B, D]；`w_attn` [B, K]；`w_fw` [K]；
          `p_max` [B]；`entropy_norm` [B]；`spec_mix` [B, D]；`cos_agreement` [B]；
          `gate_input` [B, D+3]。**全部逐样本**——没有任何跨样本（批级）归约。
        """
        exist_env_embs = torch.stack(env_embs, dim=1)                     # [D, K]
        num_tasks = len(spec_reps)
        h_out = self.projection_network(dnn_input)                        # [B, D]
        w_attn = F.softmax(torch.mm(h_out, exist_env_embs) / self.temperature, dim=-1)   # [B, K]
        w_fw = torch.full((num_tasks,), 1.0 / num_tasks, dtype=h_out.dtype, device=h_out.device)

        p_max = w_attn.max(dim=-1).values                                 # [B]
        entropy = -torch.special.xlogy(w_attn, w_attn).sum(dim=-1)        # [B]（0·log0 = 0，无 NaN）
        entropy_norm = entropy / (math.log(num_tasks) if num_tasks > 1 else 1.0)
        spec_mix = torch.matmul(torch.stack(spec_reps, dim=2), w_attn.unsqueeze(2)).squeeze(-1)
        cos_agreement = F.cosine_similarity(spec_mix, gen_rep, dim=-1, eps=1e-8)         # [B]

        gate_input = torch.cat([h_out, p_max.unsqueeze(-1), entropy_norm.unsqueeze(-1),
                                cos_agreement.unsqueeze(-1)], dim=-1)                     # [B, D+3]
        return {"exist_env_embs": exist_env_embs, "h_out": h_out, "w_attn": w_attn, "w_fw": w_fw,
                "p_max": p_max, "entropy_norm": entropy_norm, "spec_mix": spec_mix,
                "cos_agreement": cos_agreement, "gate_input": gate_input}

    def gate_terms(self, dnn_input, gen_rep, spec_reps, env_embs) -> dict:
        """门控全部中间量 = `gate_features` + `presigmoid` [B,1] + `g` [B,1] + `w` [B,K,1]。

        `w` 是 W_attn 与均匀权重 W_fw 的凸混合：`w = g·W_attn + (1−g)·W_fw`，
        因此 g ≡ 1 时 `w` 与基线 attention 权重**逐位相同**，g ≡ 0 时退化为均匀权重。
        """
        features = self.gate_features(dnn_input, gen_rep, spec_reps, env_embs)
        presigmoid = self.confidence_mlp(features["gate_input"])          # [B, 1]
        g = torch.sigmoid(presigmoid)                                     # [B, 1]
        g3 = g.unsqueeze(-1)                                              # [B, 1, 1]
        w = g3 * features["w_attn"].unsqueeze(-1) + (1.0 - g3) * features["w_fw"].view(1, -1, 1)
        return {**features, "presigmoid": presigmoid, "g": g, "w": w}

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        """与基线 `NewTask.forward` 逐行同管线，只有两处不同：用的是混合权重 `w`；
        维度压缩用 `squeeze(-1)` 而非 `squeeze()`（B=1 安全，D6）。"""
        terms = self.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        new_env_emb = self.env_embedding_network(self.new_env_idx).squeeze(0)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), terms["w"]).squeeze(-1)
        env_aware_rep = new_spec_rep * new_env_emb
        gate_out = self.gate_network(dnn_input).unsqueeze(dim=2)
        all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
        fused_rep = torch.matmul(all_reps, gate_out).squeeze(-1)
        return self.tower_network(fused_rep).squeeze(-1)

    # get_l2_reg 不重写：与基线一致，只正则 tower_network（门控不进 L2，历史同口径）


@torch.no_grad()
def hidden_row_symmetry(newtask) -> dict:
    """隐层行对称性读数（D3 的签名量）。

    `hidden_row_symmetry_deviation` = mean|W1[i,j] − mean_i W1[i,j]|（= 0 ⟺ 各行逐位相同）；
    `hidden_rows_identical` = 偏差是否为 0。历史常量初始化下恒为 0（对称永不破缺）；
    标准初始化下必须 > 0。
    """
    weight = newtask.confidence_mlp[0].weight.detach().double()
    row_mean = weight.mean(dim=0, keepdim=True)
    deviation = float((weight - row_mean).abs().mean())
    return {"hidden_row_symmetry_deviation": deviation,
            "hidden_row_symmetry_deviation_max": float((weight - row_mean).abs().sum(dim=1).max()),
            "hidden_rows_identical": bool(deviation == 0.0)}


class CGRInstanceStats:
    """门控读数累加器（CPU float64；流式、显存 O(1)）。

    样本级聚合（与分批方式无关）：`g_mean/g_min/g_max`、`presigmoid_abs_mean/max`、
    `presigmoid_saturation_frac`（|z| ≥ 阈值 的样本占比）、三个门控特征的均值、`n_samples`。
    批级聚合（**按定义依赖分批**，取批间均值/极值）：`g_within_batch_std_*`（批内样本维总体 std，
    ddof=0）、`g_within_batch_unique_frac_*`（批内唯一 g 值数 / 批大小）、`weight_l1_*`
    （批内逐样本 ‖W − W_attn‖₁ 的均值），以及批数 `n_batches`。

    字段名与 exp/stage2-cgr-repro 的 `CGRStats` **逐名对齐**，便于两臂直接对照；
    无 warmup，故本臂的 `g` 即历史读数的 `g_raw`（历史另有 alpha 插值后的 `g`）。
    空累加器：所有数值字段为 0.0（`presigmoid_saturation_abs` 恒回显阈值）。
    """

    def __init__(self) -> None:
        self.n_samples = self.n_batches = 0
        self._g_sum = self._p_max_sum = self._entropy_sum = self._cos_sum = 0.0
        self._presig_abs_sum = 0.0
        self._sat_count = 0
        self._g_min = self._g_max = None
        self._presig_abs_max = 0.0
        self._std_sum = self._std_min = self._std_max = None
        self._unique_frac_sum = self._unique_frac_max = 0.0
        self._weight_l1_sum = self._weight_l1_max = 0.0

    def update(self, terms: dict) -> None:
        g = terms["g"].detach().double().reshape(-1).cpu()
        presig_abs = terms["presigmoid"].detach().double().reshape(-1).abs().cpu()
        w = terms["w"].detach().double().cpu()                                  # [B, K, 1]
        w_attn = terms["w_attn"].detach().double().cpu()                        # [B, K]
        weight_l1 = (w.squeeze(-1) - w_attn).abs().sum(dim=1)                   # [B]
        batch = int(g.numel())
        std = float(g.std(unbiased=False)) if batch else 0.0
        unique_frac = (float(torch.unique(g).numel()) / batch) if batch else 0.0

        self.n_samples += batch
        self.n_batches += 1
        self._g_sum += float(g.sum())
        self._presig_abs_sum += float(presig_abs.sum())
        self._sat_count += int((presig_abs >= PRESIGMOID_SATURATION_ABS).sum())
        self._p_max_sum += float(terms["p_max"].detach().double().sum())
        self._entropy_sum += float(terms["entropy_norm"].detach().double().sum())
        self._cos_sum += float(terms["cos_agreement"].detach().double().sum())
        if batch:
            self._g_min = float(g.min()) if self._g_min is None else min(self._g_min, float(g.min()))
            self._g_max = float(g.max()) if self._g_max is None else max(self._g_max, float(g.max()))
            self._presig_abs_max = max(self._presig_abs_max, float(presig_abs.max()))
        self._std_sum = std if self._std_sum is None else self._std_sum + std
        self._std_min = std if self._std_min is None else min(self._std_min, std)
        self._std_max = std if self._std_max is None else max(self._std_max, std)
        self._unique_frac_sum += unique_frac
        self._unique_frac_max = max(self._unique_frac_max, unique_frac)
        self._weight_l1_sum += float(weight_l1.mean())
        self._weight_l1_max = max(self._weight_l1_max, float(weight_l1.max()))

    def result(self) -> dict:
        n, batches = self.n_samples, self.n_batches
        return {
            "n_samples": n, "n_batches": batches,
            "g_mean": self._g_sum / n if n else 0.0,
            "g_min": self._g_min if self._g_min is not None else 0.0,
            "g_max": self._g_max if self._g_max is not None else 0.0,
            "g_within_batch_std_mean": self._std_sum / batches if batches else 0.0,
            "g_within_batch_std_min": self._std_min if self._std_min is not None else 0.0,
            "g_within_batch_std_max": self._std_max if self._std_max is not None else 0.0,
            "g_within_batch_unique_frac_mean": self._unique_frac_sum / batches if batches else 0.0,
            "g_within_batch_unique_frac_max": self._unique_frac_max,
            "weight_l1_mean": self._weight_l1_sum / batches if batches else 0.0,
            "weight_l1_max": self._weight_l1_max,
            "presigmoid_abs_mean": self._presig_abs_sum / n if n else 0.0,
            "presigmoid_abs_max": self._presig_abs_max,
            "presigmoid_saturation_frac": self._sat_count / n if n else 0.0,
            "presigmoid_saturation_abs": PRESIGMOID_SATURATION_ABS,
            "p_max_mean": self._p_max_sum / n if n else 0.0,
            "entropy_norm_mean": self._entropy_sum / n if n else 0.0,
            "cos_agreement_mean": self._cos_sum / n if n else 0.0,
        }


def confidence_grad_norm(newtask) -> float:
    """门控全参数梯度范数（backward 之后、optimizer.step() 之前读取；全为 None ⇒ 0.0）。"""
    total = 0.0
    for param in newtask.confidence_mlp.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().double().norm(2)) ** 2
    return total ** 0.5


class TrainGateTrace:
    """训练期逐 epoch 轨迹：门控读数（`CGRInstanceStats` 口径）+ 门控梯度范数。

    runner 顺序：`start_epoch(epoch)` → 每个 batch 在 `loss.backward()` 之后、`optimizer.step()`
    之前 `record_step(...)` → 一个 epoch 结束 `end_epoch()`。无 warmup ⇒ 不需要写任何 alpha。
    """

    def __init__(self) -> None:
        self.stats = CGRInstanceStats()
        self._per_epoch: list[dict] = []
        self._epoch = None
        self._epoch_stats: CGRInstanceStats | None = None
        self._epoch_grads: list[float] = []
        self._grads: list[float] = []

    def start_epoch(self, epoch: int) -> None:
        if self._epoch is not None:
            raise RuntimeError(f"epoch {self._epoch} 尚未 end_epoch()，不能开始 epoch {epoch}")
        self._epoch = int(epoch)
        self._epoch_stats, self._epoch_grads = CGRInstanceStats(), []

    def record_step(self, newtask, dnn_input, gen_rep, spec_reps, env_embs) -> float:
        if not isinstance(newtask, CGRInstanceNewTask):
            raise ValueError(f"门控轨迹只适用于 {CGRInstanceNewTask.__name__}，收到 {type(newtask).__name__}")
        if self._epoch is None:
            raise RuntimeError("record_step 之前必须先 start_epoch()")
        norm = confidence_grad_norm(newtask)
        with torch.no_grad():                                   # 复算同一步的门控读数（不改参数、不建图）
            terms = newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
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
    """best_state 载入后的门控诊断（只用给定划分前向一遍：不训练、不反传、不改参数）。

    调用点固定在**选点完成、best_state 已载入之后**，只用 val；不参与 early stop / 选点 / test 评测，
    因此不可能改变 A/B 门禁或效应判定的结果。
    """
    if not isinstance(newtask, CGRInstanceNewTask):
        raise ValueError(f"门控诊断只适用于 {CGRInstanceNewTask.__name__}，收到 {type(newtask).__name__}")
    newtask.eval()
    backbone.eval()                                          # 三件套之 2：backbone 恒为 eval
    stats = CGRInstanceStats()
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽表征
        stats.update(newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
    return {**stats.result(), **hidden_row_symmetry(newtask),
            "gate_hidden": int(newtask.gate_hidden), "gate_input_dim": int(newtask.gate_input_dim),
            "gate_output_weight_absmax": float(newtask.confidence_mlp[2].weight.detach().abs().max()),
            "gate_output_bias": float(newtask.confidence_mlp[2].bias.detach().reshape(-1)[0])}


def mechanism_verdict(*, g_std: float, grad_norm: float, g_mean: float, weight_l1: float) -> dict:
    """主导预注册判据：四条机制门**全部**满足才 MECHANISM_OK，任一不满足 ⇒ MECHANISM_FAILED ⇒ STOP。

    输入口径：`g_std` / `g_mean` / `weight_l1` 取 **best_state 载入后**的 val 诊断
    （`evaluate_gate` 的 `g_within_batch_std_mean` / `g_mean` / `weight_l1_mean`）；
    `grad_norm` 取训练轨迹的 `grad_norm_max`（"门控是否收到过梯度"）。
    """
    checks = {
        "gate_std_ge_min": {"rule": RULE_GATE_STD, "threshold": float(GATE_STD_MIN),
                            "observed": float(g_std), "pass": bool(g_std >= GATE_STD_MIN)},
        "grad_norm_gt_zero": {"rule": RULE_GRAD_NORM, "threshold": float(GRAD_NORM_MIN),
                              "observed": float(grad_norm), "pass": bool(grad_norm > GRAD_NORM_MIN)},
        "g_mean_in_range": {"rule": RULE_G_MEAN, "threshold": [float(G_MEAN_MIN), float(G_MEAN_MAX)],
                            "observed": float(g_mean),
                            "pass": bool(G_MEAN_MIN <= g_mean <= G_MEAN_MAX)},
        "weight_l1_ge_min": {"rule": RULE_WEIGHT_L1, "threshold": float(WEIGHT_L1_MIN),
                             "observed": float(weight_l1), "pass": bool(weight_l1 >= WEIGHT_L1_MIN)},
    }
    failed = [check["rule"] for check in checks.values() if not check["pass"]]
    return {"status": "MECHANISM_OK" if not failed else "MECHANISM_FAILED",
            "pass": not failed, "failed_rules": failed, "checks": checks,
            "observed": {"g_std": float(g_std), "grad_norm": float(grad_norm),
                         "g_mean": float(g_mean), "weight_l1": float(weight_l1)}}


def arm_verdict(test_auc: float, *, mechanism: dict, auc_min: float = AUC_TEST_MIN) -> dict:
    """臂级判定：机制失败 ⇒ STOP（效应阈值不解释）；机制通过才比较 `AUC-Test-Education >= 0.8521`。"""
    failed = mechanism["status"] != "MECHANISM_OK"
    effect = {"rule": RULE_EFFECT, "threshold": float(auc_min), "observed": float(test_auc),
              "evaluated": not failed, "pass": None if failed else bool(test_auc >= auc_min)}
    return {"status": "STOP" if failed else ("EFFECT_CONFIRMED" if effect["pass"] else "EFFECT_NOT_CONFIRMED"),
            "pass": bool(not failed and effect["pass"]), "mechanism": mechanism, "effect": effect}


def preregistered_criteria() -> dict:
    """落盘用的预注册判据快照（与 experiments 文档第 5 节一致；改阈值必须先改文档再改这里）。"""
    return {"auc_test_min": float(AUC_TEST_MIN), "baseline_test_auc": float(BASELINE_TEST_AUC),
            "gate_std_min": float(GATE_STD_MIN), "grad_norm_min": float(GRAD_NORM_MIN),
            "g_mean_min": float(G_MEAN_MIN), "g_mean_max": float(G_MEAN_MAX),
            "weight_l1_min": float(WEIGHT_L1_MIN),
            "presigmoid_saturation_abs": float(PRESIGMOID_SATURATION_ABS),
            "gate_hidden": int(GATE_HIDDEN), "gate_scalar_features": list(GATE_SCALAR_FEATURES),
            "warmup": bool(WARMUP),
            "rules": {"mechanism": [RULE_GATE_STD, RULE_GRAD_NORM, RULE_G_MEAN, RULE_WEIGHT_L1],
                      "effect": RULE_EFFECT}}


def provenance() -> dict:
    """修正溯源快照（写进 `metrics.json:cgr_instance_arm`，便于与历史实现、被修正的复现臂对齐）。"""
    return {"corrective_source": "exp/stage2-cgr-repro（忠实复现判定 CONFIRMED_DEGENERATE："
                                 "unique_frac 0.00391 <= 0.01、weight_l1 0.0 < 1e-6、grad_norm 0.0 == 0）",
            "historical_source": "archive/exploration: run_newtask_from_ckpt.py (mode='cgr') + "
                                 "multitaskrec/model.py NewTask(fusion_mode='cgr')",
            "fixes": {
                "D1": "门控输入改为逐样本（H_out + attention 置信度 + agreement），无任何批级归约",
                "D2": "先 super().__init__() 建共享模块、后追建门控 ⇒ 共享参数与基线逐位同 seed 同初值",
                "D3": "隐层标准初始化（行对称可破缺）+ 门控输出层零初始化 ⇒ z ≡ 0、g ≡ 0.5、不饱和",
                "D4": "移除 warmup（零初始化已使起点落在中性点 g = 0.5，不再需要 alpha 插值）",
                "D6": "逐样本量 + forward 用 squeeze(-1) ⇒ 单样本批不 NaN、不形状报错",
            },
            "gate_input": "per_sample", "warmup": "removed", "construction_order": "baseline_first",
            "gate_scalar_features": list(GATE_SCALAR_FEATURES), "gate_hidden": int(GATE_HIDDEN),
            "attention_temperature": int(ATTENTION_TEMPERATURE),
            "spec": "docs/superpowers/experiments/2026-09-30-stage2-cgr-instance.md"}


def build_newtask(variant: str, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                  gate_hidden: int = GATE_HIDDEN) -> NewTask:
    """变体工厂：`baseline` → 基线 `NewTask`（逐位同 master，且不多消耗一次 RNG）；
    `cgr-instance` → 修正的逐样本身份门控头。"""
    if variant == BASELINE_VARIANT:
        return NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
    if variant == VARIANT:
        return CGRInstanceNewTask(input_size=input_size, rep_dim=rep_dim,
                                  tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn,
                                  device=device, gate_hidden=gate_hidden)
    raise ValueError(f"未知 variant: {variant!r}（可选 {VARIANTS}）")


def stage2_config(variant: str, *, rep_dim: int | None = None) -> dict:
    """处理臂专属的 config.json 字段（基线臂返回 `{}` ⇒ 基线 payload 键集逐键不变）。"""
    if variant == BASELINE_VARIANT:
        return {}
    return {"variant": VARIANT, "gate_hidden": int(GATE_HIDDEN),
            "gate_input_dim": (int(rep_dim) + len(GATE_SCALAR_FEATURES)) if rep_dim is not None else None,
            "gate_scalar_features": list(GATE_SCALAR_FEATURES), "warmup": bool(WARMUP),
            "attention_temperature": int(ATTENTION_TEMPERATURE)}
