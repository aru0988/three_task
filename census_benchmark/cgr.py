"""历史 CGR（Confidence-Gated Routing）的**忠实复现与门控诊断**（不是新方法）。

唯一事实来源：docs/superpowers/experiments/2026-09-30-stage2-cgr-revalidation.md

审计对象（`archive/exploration` 分支的历史实现，逐项对齐、不做"顺手修好"）：

    run_newtask_from_ckpt.py        mode="cgr"；warmup_epochs=5；alpha=min(1, epoch/5)
    multitaskrec/model.py           NewTask(fusion_mode="cgr") 的整条 cgr 分支
    analysis/cgr_diagnosis.py       g 的逐 epoch 读数（g_mean / g_std）
    analysis/cgr_gradient_diagnosis.py  projection_network 的梯度范数读数

历史公式（$x$ = `dnn_input`，$H(x)$ = `projection_network(x)`，$E$ = stack(env_embs, 1)，$T=150$）：

    W_attn(x) = softmax(H(x) E / T)                      # 既有 attention 权重
    W_fw      = 1/K                                      # 均匀权重
    z(x)      = MLP([e_new, mean_k E_k, var_k E_k, mean_b gen_rep, var_b gen_rep])   # 5·rep_dim 输入
    g_raw     = sigmoid(z)
    g         = alpha·g_raw + (1−alpha)·0.5              # alpha 由 runner 逐 epoch 写入
    W(x)      = g·W_attn(x) + (1−g)·W_fw
    new_spec_rep = Σ_k W_k spec_rep_k ,  env_aware = new_env_emb ⊙ new_spec_rep
    fused_rep = [env_aware, gen_rep] · gate_network(x) ,  pred = tower(fused_rep)

以下历史缺陷**故意**复现、不修（编号与 experiments 文档第 3 节一致；D4 warmup 时序、D5 梯度信号弱
由读数留痕，不改实现）：

  * **D1 批级门控**：`z` 的输入全部是**批级/环境级统计量**（`gen_rep.mean(dim=0)`、`gen_rep.var(dim=0)`、
    `E.mean(dim=1)`、`E.var(dim=1)`），没有逐样本信息 ⇒ g 在**批内逐位相同**（批级常量），
    同一样本换个批上下文就换一个 g。历史根因分析（commit 6b023d7）称之为"全局标量门控"。
  * **D2 RNG 构造顺序**：历史把 `confidence_mlp` 建在 `projection_network` 与 `gate_network` 之间，
    因此在同一 model seed 下 `gate_network` / `tower_network` 的初始化与基线**不同**——
    历史 CGR-vs-Prompt 的对比并不是单变量对比（审计缺陷）。
  * **D3 饱和 + 行对称**：`confidence_mlp` 常量初始化（0.01 / 0.01、0.1 / 2.0）⇒（a）门控起点偏向
    g≈0.9（"信任 attention"）；（b）32 个隐层单元的行完全相同，且第二层也是常量 ⇒ 对称永不破缺，
    MLP 退化成"输入的标量和"的一个标量函数。
  * **D6 单样本批 NaN**（本轮审计新发现）：门控输入里的 `gen_rep.var(dim=0)` 用 ddof=1，
    批大小 = 1 时自由度 ≤ 0 ⇒ `g = NaN` ⇒ 该步整批预测与损失全毁。CensusIncome 恰好躲开
    （训练集 199523 = 779×256 + 99、val 49881 = 194×256 + 217，末批都不是 1），
    但这是历史设计自带的爆点：任何 batch_size 或样本数变化都可能踩中。

硬约束：

  * **不改 `multitaskrec/model.py`**：本模块以 `NewTask` 的**子类**扩展，`gate_terms` / `forward`
    只复刻历史 cgr 分支的表达式。
  * **不影响基线臂**：`build_newtask(BASELINE_VARIANT, ...)` 返回的就是基线 `NewTask` 本身
    （逐位同 master，构造不多消耗一次 RNG）。
  * **protocol.py / metrics.py 零改动**：A/B 门禁语义不变；臂级判定只写在 `metrics.json:cgr_arm`。
  * **不主张新颖性**：置信门控 / 条件化路由是已有大类；历史 CGR 在 CensusIncome 上也从未超越
    prompt（见 experiments 文档第 2 节）。本方向只回答"忠实复现后，历史结论是否还站得住"。

预注册判据（**只允许在看到结果之前修改**）：

    unique_frac <= 0.01  或  weight_l1 < 1e-6  或  grad_norm == 0   ⇒ CONFIRMED_DEGENERATE ⇒ STOP
    （退化是主导判据；效应阈值 AUC-Test-Education >= 0.8521 仅在**非退化**时才有意义）

用法（接线见 run_census_benchmark.run_stage2）：

    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --variant cgr --tag short
    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --cgr        --tag short
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from census_benchmark import protocol as P
from multitaskrec.model import MLP, NewTask

# ---- 臂与接线 ----
BASELINE_VARIANT = "baseline"
VARIANT = "cgr"
VARIANTS = (BASELINE_VARIANT, VARIANT)
RUN_ID_SUFFIX = "-cgr"                     # 沿用 null-expert / attn / csb 分支的"协议 run_id + 臂后缀"约定

# ---- 历史常量（archive/exploration 原值，不得改）----
WARMUP_EPOCHS = 5                          # run_newtask_from_ckpt.py: warmup_epochs = 5
TEMPERATURE = 150                          # NewTask.temperature
HIDDEN_UNITS = 32                          # confidence_mlp 隐层宽度
GATE_INPUT_BLOCKS = 5                      # [e_new, source_mean, source_var, gen_mean, gen_var]

# ---- 预注册判据（只允许在看到结果之前修改；规则文本是预注册原文，单独硬编码）----
AUC_TEST_MIN = 0.8521                      # 效应阈值（基线实测 0.850069）
UNIQUE_FRAC_MAX = 0.01                     # 判据 1：批内 g 唯一值占比 ≤ 1%
WEIGHT_L1_MIN = 1e-6                       # 判据 2：|W − W_attn| 的 L1 < 1e-6
GRAD_NORM_MIN = 0.0                        # 判据 3：置信 MLP 训练期梯度范数 == 0
PRESIGMOID_SATURATION_ABS = 8.0            # 饱和读数阈值：|z| ≥ 8 ⇒ sigmoid' ≤ 3.4e-4
RULE_UNIQUE_FRAC = "unique_frac <= 0.01"
RULE_WEIGHT_L1 = "weight_l1 < 1e-6"
RULE_GRAD_NORM = "grad_norm == 0"
RULE_EFFECT = "auc_test_education >= 0.8521"


class CGRNewTask(NewTask):
    """历史 `NewTask(fusion_mode="cgr")` 的忠实复刻（含 D1–D3 三个历史缺陷）。

    构造顺序与历史**逐行同序**（`nn.Module.__init__` 之后手工搭模块树，而不是 `super().__init__()`）：
    历史代码在 `projection_network` 与 `gate_network` 之间构造了 `confidence_mlp`，
    两次 `nn.Linear` 的默认初始化会消耗全局 RNG ⇒ 之后的 `gate_network` / `tower_network`
    与基线不同（D2）。这是被审计出来的对比缺陷，必须原样保留才能称"忠实复现"；
    子模块名与形状仍与 master `NewTask` 完全一致（测试锁定 `state_dict` 键集只多 confidence_mlp）。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 temperature=TEMPERATURE):
        nn.Module.__init__(self)                       # 故意不走 NewTask.__init__：见类 docstring（D2）
        self.reg_dnn = reg_dnn
        self.device = device
        self.temperature = temperature
        self.env_embedding_network = nn.Embedding(1, rep_dim)
        self.register_buffer("new_env_idx", torch.tensor([0]), persistent=True)
        self.projection_network = nn.Sequential(
            nn.Linear(input_size, rep_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(rep_dim // 2, rep_dim, bias=False),
            nn.LayerNorm(rep_dim),
        )
        self.confidence_mlp = nn.Sequential(           # 位置即历史位置（projection 与 gate 之间）
            nn.Linear(GATE_INPUT_BLOCKS * rep_dim, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 1),
            nn.Sigmoid(),
        )
        # 历史常量初始化：g 起点偏向 1（"信任 attention"），且 32 个隐层行完全相同（D3）
        nn.init.constant_(self.confidence_mlp[0].weight, 0.01)
        nn.init.constant_(self.confidence_mlp[0].bias, 0.01)
        nn.init.constant_(self.confidence_mlp[2].weight, 0.1)
        nn.init.constant_(self.confidence_mlp[2].bias, 2.0)
        self.gate_warmup_alpha = 1.0                   # 历史默认值（runner 每 epoch 覆写）
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, 2, bias=False), nn.Softmax(dim=-1)
        )
        self.tower_network = MLP(
            list(tower_dnn_hidden_units) + [1], input_size=rep_dim, output_activation="sigmoid"
        )

    def gate_terms(self, dnn_input, gen_rep, spec_reps, env_embs) -> dict:
        """历史 cgr 分支的全部中间量（纯函数：不改参数、不写状态；forward 与诊断共用同一口径）。

        返回（均为 [B, …] 张量，可带梯度）：
          `w_attn` 既有 attention 权重；`w_fw` 均匀权重；`presigmoid` = z；`g_raw` = sigmoid(z)；
          `g` warmup 插值后的置信标量（**批级常量**，D1）；`w` = g·W_attn + (1−g)·W_fw。
        """
        exist_env_embs = torch.stack(env_embs, dim=1)
        new_env_emb = self.env_embedding_network(self.new_env_idx).squeeze(0)
        num_tasks = len(spec_reps)

        h_out = self.projection_network(dnn_input)
        w_attn = F.softmax(torch.mm(h_out, exist_env_embs) / self.temperature, dim=-1).unsqueeze(2)
        w_fw = (torch.ones(dnn_input.shape[0], num_tasks, device=dnn_input.device) / num_tasks).unsqueeze(2)

        # D1：门控输入只含批级 / 环境级统计量 ⇒ 批内每个样本拿到同一个 g
        gate_input = torch.cat([new_env_emb, exist_env_embs.mean(dim=1), exist_env_embs.var(dim=1),
                                gen_rep.mean(dim=0), gen_rep.var(dim=0)], dim=-1)
        gate_input = gate_input.unsqueeze(0).expand(dnn_input.shape[0], -1)
        presigmoid = self.confidence_mlp[2](self.confidence_mlp[1](self.confidence_mlp[0](gate_input)))
        g_raw = torch.sigmoid(presigmoid)
        g = self.gate_warmup_alpha * g_raw + (1 - self.gate_warmup_alpha) * 0.5
        w = g.unsqueeze(-1) * w_attn + (1 - g.unsqueeze(-1)) * w_fw
        return {"w_attn": w_attn, "w_fw": w_fw, "presigmoid": presigmoid, "g_raw": g_raw, "g": g, "w": w}

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        terms = self.gate_terms(dnn_input, gen_rep, spec_reps, env_embs)
        new_env_emb = self.env_embedding_network(self.new_env_idx).squeeze(0)
        new_spec_rep = torch.matmul(torch.stack(spec_reps, dim=2), terms["w"]).squeeze()
        env_aware_rep = new_spec_rep * new_env_emb
        gate_out = self.gate_network(dnn_input).unsqueeze(dim=2)
        all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
        fused_rep = torch.matmul(all_reps, gate_out).squeeze()
        return self.tower_network(fused_rep).squeeze()          # 历史不变量：get_l2_reg 只正则 tower


# ---- 忠实 warmup（历史式：由 runner 逐 epoch 写入，alpha 不是 buffer、不进 state_dict）----
def warmup_alpha(epoch: int, warmup_epochs: int = WARMUP_EPOCHS) -> float:
    """`min(1.0, epoch / warmup_epochs)`——run_newtask_from_ckpt.py 原式。"""
    return min(1.0, epoch / warmup_epochs)


def set_warmup_alpha(newtask, epoch: int, warmup_epochs: int = WARMUP_EPOCHS) -> float:
    """每个 epoch 开头写入（历史如此）；提前停止后 alpha 停在最后一轮的取值（历史怪癖，留痕）。"""
    alpha = warmup_alpha(epoch, warmup_epochs)
    newtask.gate_warmup_alpha = alpha
    return alpha


@torch.no_grad()
def hidden_row_symmetry(newtask) -> dict:
    """`confidence_mlp[0].weight` 的隐层行对称性读数（D3）。

    `hidden_row_symmetry_deviation` = mean|W1[i,j] − mean_i W1[i,j]|（= 0 ⟺ 32 行逐位相同）；
    `hidden_row_symmetry_deviation_max` = max_i ‖W1[i] − mean_row‖₁；
    `hidden_rows_identical` = 偏差是否为 0。常量初始化 + 常量第二层 ⇒ 对称永不破缺（测试锁定）。
    """
    weight = newtask.confidence_mlp[0].weight.detach().double()
    row_mean = weight.mean(dim=0, keepdim=True)
    deviation = float((weight - row_mean).abs().mean())
    return {"hidden_row_symmetry_deviation": deviation,
            "hidden_row_symmetry_deviation_max": float((weight - row_mean).abs().sum(dim=1).max()),
            "hidden_rows_identical": bool(deviation == 0.0)}


class CGRStats:
    """门控读数累加器（CPU float64；流式、显存 O(1)，口径见 experiments 文档第 5 节）。

    样本级聚合（与分批方式无关）：`g_mean/g_min/g_max`、`g_raw_*`、`presigmoid_abs_mean/max`、
    `presigmoid_saturation_frac`（|z| ≥ 8 的样本占比）、`n_samples`。
    批级聚合（**按定义依赖分批**，取批间均值/最大值）：`g_within_batch_std_*`（批内样本维总体 std，
    ddof=0）、`g_within_batch_unique_frac_*`（批内唯一 g 值数 / 批大小）、`weight_l1_*`
    （批内逐样本 ‖W − W_attn‖₁ 的均值）、以及批数 `n_batches`。
    空累加器：所有数值字段为 0.0（`presigmoid_saturation_abs` 恒回显阈值）。
    """

    def __init__(self) -> None:
        self.n_samples = self.n_batches = 0
        self._g_sum = self._g_raw_sum = self._presig_abs_sum = 0.0
        self._sat_count = 0
        self._g_min = self._g_max = self._g_raw_min = self._g_raw_max = None
        self._presig_abs_max = 0.0
        self._std_sum = self._unique_frac_sum = self._weight_l1_sum = 0.0
        self._std_max = self._unique_frac_max = self._weight_l1_max = 0.0

    def update(self, terms: dict) -> None:
        g = terms["g"].detach().double().reshape(-1).cpu()
        g_raw = terms["g_raw"].detach().double().reshape(-1).cpu()
        presig_abs = terms["presigmoid"].detach().double().reshape(-1).abs().cpu()
        weight_l1 = float((terms["w"].detach().double().cpu()
                           - terms["w_attn"].detach().double().cpu()).abs().sum(dim=1).mean())
        batch = int(g.numel())
        std = float(g.std(unbiased=False)) if batch else 0.0
        unique_frac = (float(torch.unique(g).numel()) / batch) if batch else 0.0

        self.n_samples += batch
        self.n_batches += 1
        self._g_sum += float(g.sum())
        self._g_raw_sum += float(g_raw.sum())
        self._presig_abs_sum += float(presig_abs.sum())
        self._sat_count += int((presig_abs >= PRESIGMOID_SATURATION_ABS).sum())
        if batch:
            self._g_min = float(g.min()) if self._g_min is None else min(self._g_min, float(g.min()))
            self._g_max = float(g.max()) if self._g_max is None else max(self._g_max, float(g.max()))
            self._g_raw_min = (float(g_raw.min()) if self._g_raw_min is None
                               else min(self._g_raw_min, float(g_raw.min())))
            self._g_raw_max = (float(g_raw.max()) if self._g_raw_max is None
                               else max(self._g_raw_max, float(g_raw.max())))
            self._presig_abs_max = max(self._presig_abs_max, float(presig_abs.max()))
        self._std_sum += std
        self._std_max = max(self._std_max, std)
        self._unique_frac_sum += unique_frac
        self._unique_frac_max = max(self._unique_frac_max, unique_frac)
        self._weight_l1_sum += weight_l1
        self._weight_l1_max = max(self._weight_l1_max, weight_l1)

    def result(self) -> dict:
        n, batches = self.n_samples, self.n_batches
        return {
            "n_samples": n, "n_batches": batches,
            "g_mean": self._g_sum / n if n else 0.0,
            "g_min": self._g_min if self._g_min is not None else 0.0,
            "g_max": self._g_max if self._g_max is not None else 0.0,
            "g_raw_mean": self._g_raw_sum / n if n else 0.0,
            "g_raw_min": self._g_raw_min if self._g_raw_min is not None else 0.0,
            "g_raw_max": self._g_raw_max if self._g_raw_max is not None else 0.0,
            "g_within_batch_std_mean": self._std_sum / batches if batches else 0.0,
            "g_within_batch_std_max": self._std_max,
            "g_within_batch_unique_frac_mean": self._unique_frac_sum / batches if batches else 0.0,
            "g_within_batch_unique_frac_max": self._unique_frac_max,
            "weight_l1_mean": self._weight_l1_sum / batches if batches else 0.0,
            "weight_l1_max": self._weight_l1_max,
            "presigmoid_abs_mean": self._presig_abs_sum / n if n else 0.0,
            "presigmoid_abs_max": self._presig_abs_max,
            "presigmoid_saturation_frac": self._sat_count / n if n else 0.0,
            "presigmoid_saturation_abs": PRESIGMOID_SATURATION_ABS,
        }


def confidence_grad_norm(newtask) -> float:
    """置信 MLP 全参数梯度范数（backward 之后、optimizer.step() 之前读取；全为 None ⇒ 0.0）。"""
    total = 0.0
    for param in newtask.confidence_mlp.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().double().norm(2)) ** 2
    return total ** 0.5


class TrainGateTrace:
    """训练期逐 epoch 轨迹：门控读数（CGRStats 口径）+ 置信 MLP 梯度范数。

    runner 顺序：`start_epoch(epoch, alpha)` → 每个 batch 在 `loss.backward()` 之后、
    `optimizer.step()` 之前 `record_step(...)` → 一个 epoch 结束 `end_epoch()`。
    """

    def __init__(self) -> None:
        self.stats = CGRStats()
        self._per_epoch: list[dict] = []
        self._epoch = self._alpha = None
        self._epoch_stats: CGRStats | None = None
        self._epoch_grads: list[float] = []
        self._grads: list[float] = []

    def start_epoch(self, epoch: int, alpha: float) -> None:
        if self._epoch is not None:
            raise RuntimeError(f"epoch {self._epoch} 尚未 end_epoch()，不能开始 epoch {epoch}")
        self._epoch, self._alpha = int(epoch), float(alpha)
        self._epoch_stats, self._epoch_grads = CGRStats(), []

    def record_step(self, newtask, dnn_input, gen_rep, spec_reps, env_embs) -> float:
        if not isinstance(newtask, CGRNewTask):
            raise ValueError(f"CGR 轨迹只适用于 {CGRNewTask.__name__}，收到 {type(newtask).__name__}")
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
        record = {"epoch": self._epoch, "alpha": self._alpha, "steps": len(grads),
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
    if not isinstance(newtask, CGRNewTask):
        raise ValueError(f"CGR 诊断只适用于 {CGRNewTask.__name__}，收到 {type(newtask).__name__}")
    newtask.eval()
    backbone.eval()                                          # 三件套之 2：backbone 恒为 eval
    stats = CGRStats()
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽表征
        stats.update(newtask.gate_terms(dnn_input, gen_rep, spec_reps, env_embs))
    return {**stats.result(), "warmup_alpha_at_eval": float(newtask.gate_warmup_alpha),
            "temperature": float(newtask.temperature), **hidden_row_symmetry(newtask)}


def degeneracy_verdict(*, unique_frac: float, weight_l1: float, grad_norm: float) -> dict:
    """主导预注册判据：三条任一命中 ⇒ CONFIRMED_DEGENERATE（效果阈值不再解释）。

    输入口径：`unique_frac` / `weight_l1` 取 **best_state 载入后**的 val 诊断
    （`evaluate_gate` 的 `g_within_batch_unique_frac_mean` / `weight_l1_mean`）；
    `grad_norm` 取训练轨迹的 `grad_norm_max`（"门控从未收到过任何梯度"）。
    """
    checks = {
        "unique_frac_le_0.01": {"rule": RULE_UNIQUE_FRAC, "threshold": float(UNIQUE_FRAC_MAX),
                                "observed": float(unique_frac),
                                "triggered": bool(unique_frac <= UNIQUE_FRAC_MAX)},
        "weight_l1_lt_1e-6": {"rule": RULE_WEIGHT_L1, "threshold": float(WEIGHT_L1_MIN),
                              "observed": float(weight_l1),
                              "triggered": bool(weight_l1 < WEIGHT_L1_MIN)},
        "grad_norm_eq_0": {"rule": RULE_GRAD_NORM, "threshold": float(GRAD_NORM_MIN),
                           "observed": float(grad_norm),
                           "triggered": bool(grad_norm == GRAD_NORM_MIN)},
    }
    triggered = [check["rule"] for check in checks.values() if check["triggered"]]
    return {"status": "CONFIRMED_DEGENERATE" if triggered else "NOT_DEGENERATE",
            "triggered_rules": triggered, "checks": checks,
            "observed": {"unique_frac": float(unique_frac), "weight_l1": float(weight_l1),
                         "grad_norm": float(grad_norm)}}


def arm_verdict(test_auc: float, *, degeneracy: dict, auc_min: float = AUC_TEST_MIN) -> dict:
    """臂级判定：退化 ⇒ STOP（效应阈值不解释）；非退化才比较 `AUC-Test-Education >= 0.8521`。"""
    degenerate = degeneracy["status"] == "CONFIRMED_DEGENERATE"
    effect = {"rule": RULE_EFFECT, "threshold": float(auc_min), "observed": float(test_auc),
              "evaluated": not degenerate,
              "pass": None if degenerate else bool(test_auc >= auc_min)}
    status = "STOP" if degenerate else ("EFFECT_CONFIRMED" if effect["pass"] else "EFFECT_NOT_CONFIRMED")
    return {"status": status, "pass": bool(not degenerate and effect["pass"]),
            "degeneracy": degeneracy, "effect": effect}


def preregistered_criteria() -> dict:
    """落盘用的预注册判据快照（与 experiments 文档第 4 节一致；改阈值必须先改文档再改这里）。"""
    return {"unique_frac_max": float(UNIQUE_FRAC_MAX), "weight_l1_min": float(WEIGHT_L1_MIN),
            "grad_norm_min": float(GRAD_NORM_MIN), "auc_test_min": float(AUC_TEST_MIN),
            "presigmoid_saturation_abs": float(PRESIGMOID_SATURATION_ABS),
            "warmup_epochs": int(WARMUP_EPOCHS),
            "rules": {"degenerate": [RULE_UNIQUE_FRAC, RULE_WEIGHT_L1, RULE_GRAD_NORM],
                      "effect": RULE_EFFECT}}


def provenance() -> dict:
    """审计溯源快照（写进 `metrics.json:cgr_arm`，便于把 run 与历史实现对起来）。"""
    return {"historical_source": "archive/exploration: run_newtask_from_ckpt.py (mode='cgr', "
                                 "warmup_epochs=5) + multitaskrec/model.py NewTask(fusion_mode='cgr')",
            "historical_diagnostics": "archive/exploration: analysis/cgr_diagnosis.py, "
                                      "analysis/cgr_gradient_diagnosis.py",
            "historical_result": "archive/exploration: logs/all_experiment_results.md "
                                 "(CensusIncome cgr mean test AUC 0.8618±0.0048, Δ vs prompt −0.0011)",
            "rng_construction_order": "historical",          # D2：confidence_mlp 建在 projection 与 gate 之间
            "gate_input": "batch_global_statistics",         # D1：门控输入 5·rep_dim，无逐样本信息
            "hidden_init": "constant_0.01_0.01_0.1_2.0",     # D3：常量初始化 + 行对称
            "temperature": float(TEMPERATURE), "hidden_units": int(HIDDEN_UNITS),
            "spec": "docs/superpowers/experiments/2026-09-30-stage2-cgr-revalidation.md"}


def build_newtask(variant: str, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                  temperature: float = TEMPERATURE) -> NewTask:
    """变体工厂：`baseline` → 基线 `NewTask`（逐位同 master）；`cgr` → 忠实复现的处理臂。"""
    if variant == BASELINE_VARIANT:
        return NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
    if variant == VARIANT:
        return CGRNewTask(input_size=input_size, rep_dim=rep_dim,
                          tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn,
                          device=device, temperature=temperature)
    raise ValueError(f"未知 variant: {variant!r}（可选 {VARIANTS}）")


def stage2_config(variant: str) -> dict:
    """处理臂专属的 config.json 字段（基线臂不写，避免污染基线键集）。"""
    if variant == BASELINE_VARIANT:
        return {}
    return {"warmup_epochs": int(WARMUP_EPOCHS), "temperature": float(TEMPERATURE)}
