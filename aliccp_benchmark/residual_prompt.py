"""AliCCP 阶段 2 范数受控残差 Prompt（可学习门控）——canonical seed 2 复现（seed 1688723740）。

唯一事实来源：docs/superpowers/specs/2026-10-04-aliccp-stage2-residual-prompt-seed-replication-design.md
移植自 `79ddefa:aliccp_benchmark/residual_prompt.py`（git blob
`5dc7158ca999c9e7e6217a999c37869231411549`）；相对钉死 blob 恰 2 处 run-reference 适配（模块
docstring + 3 个基线常量；预注册 §1.4-A1′），其余逐字节一致（守卫测试重建等式钉死）。机制本体
（公式、初始化、门控、注入点）与 seed1 实现及 Census 祖本（`99b9510`，blob
`107221b26382da7fd44990e167d9a61da2680d92`）逐字一致（测试 `TestMechanismPin` 守卫）。

公式（逐样本；$x$ = `dnn_input`，$d$ = `rep_dim`，三路 $h_s$ = `gen_rep` 与两个 `spec_rep`）：

    P(x)        = tanh(W2·ReLU(W1·x + b1) + b2)            # 有界方向，来自上下文
    g_eff(x)    = |alpha| · ||P(x)|| / sqrt(d)             # 逐样本有效门控
    delta_s(x)  = alpha · (||h_s|| / sqrt(d)) · P(x)       # 残差 = 门控 × 单位方向 × 基向量范数
    h_s'        = h_s + delta_s

结构性事实（测试锁定）：
  * S1 范数受控：ratio_s = ||delta_s|| / ||h_s|| = g_eff ≤ |alpha|（||P|| ≤ sqrt(d)）；
  * S2 跨流一致：同一 x 下三路相对扰动相同，与各流自身范数无关；
  * S3 保守初值：alpha = 0 ⇒ delta ≡ 0 ⇒ 前向与基线逐位一致（共享权重相同时）。

硬约束：不改 `multitaskrec/model.py`（子类扩展）；新增模块构造在 `isolated_cpu_rng()` 内，
共享参数与全局 RNG 端点与基线逐位一致；不主张新颖性（prompt / 适配器 / 条件化调制均为既有方法族）。

用法（接线见 aliccp_benchmark.bench.run_stage2）：

    python run_aliccp_benchmark.py stage2 --stage1-id <sid> --variant residual-prompt --tag short \\
        --model-seed 1688723740 --prompt-reference-newtask <baseline_run>/newtask.pt
"""
from __future__ import annotations

import contextlib
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from aliccp_benchmark import metrics
from multitaskrec.model import NewTask

# ---- 臂与接线（预注册 2.3）----
BASELINE_VARIANT = "baseline"
VARIANT = "residual-prompt"
VARIANTS = (BASELINE_VARIANT, VARIANT)
RUN_ID_SUFFIX = "-rpg"                     # 协议 run_id + 臂后缀（与 Census 同约定）

# ---- 预注册判据（预注册 3/4；只允许在看到结果之前修改）----
PROMPT_HIDDEN = 16                         # 生成器隐层（Census 超参，固定，不搜索）
AUC_TEST_DELTA_MIN = 0.0055                # U1：Δtest 下限（用户指定 provisional，严于协议 10.1 的 +0.005）
AUC_VAL_DIRECTION_MIN = 0.0                # U2：Δval 严格 > 0
BASELINE_AUC_TEST = 0.5974422649550507     # 基线 run 20261003-0624-…-79b5e07 实测（raw seed2 配对基线；2026-10-04 复核逐位一致）
BASELINE_AUC_VAL = 0.5809347091990792      # 同上（AUC-Val-BSI best）
PRED_STD_MIN_RATIO = 0.5                   # G7：pred_std ≥ 0.5 × ref_pred_std（BSI 预测天然集中，禁用绝对下限）
REFERENCE_PRED_STD = 0.005217193225189258  # 预注册前实测的参照头 val pred_std（M0/G7；seed2 参照）
REFERENCE_IDENTITY_TOL = 1e-9              # M0：参照复算与预注册记录逐位一致（float64 容差）
EXPECTED_NEW_PARAMS_TOTAL = 2385           # G8：16×80+16 + 64×16+64 + 1
EXPECTED_HEAD_PARAMS = 8129                # G8：AliCCP NewTask 头（input 80 / rep 64 / tower 32,32）
RATIO_BAND = (0.005, 0.5)                  # G5：逐流平均 ratio 活性带（含端点）
BOUND_TOL = 1e-6                           # G4：结构性界的 fp32 容差
EXTRA_PARAM_NAMES = ("prompt_gate", "prompt_generator.0.bias", "prompt_generator.0.weight",
                     "prompt_generator.2.bias", "prompt_generator.2.weight")


@contextlib.contextmanager
def isolated_cpu_rng():
    """保存—恢复全局 CPU RNG：新增模块的构造不得改变与基线共享的随机流端点（spec 2.2）。"""
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


class ResidualPromptNewTask(NewTask):
    """`NewTask` 的子类：上下文生成的、有界的、范数受控的残差 prompt + 零初始化标量门控。

    只重写 `forward` 的"注入"部分，融合管线原样委托 `super().forward`（零复制、零漂移）。
    `prompt_gate` 零初始化且生成器构造在隔离 RNG 内（不消耗共享随机流、不改变 RNG 端点）。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 hidden: int = PROMPT_HIDDEN):
        super().__init__(input_size=input_size, rep_dim=rep_dim,
                         tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
        with isolated_cpu_rng():   # 生成器初始化不触碰全局 RNG（构造后端点与基线一致）
            self.prompt_generator = nn.Sequential(
                nn.Linear(input_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, rep_dim),
            )
        self.prompt_gate = nn.Parameter(torch.zeros(()))     # 标量门控，精确 0，不消耗 RNG

    def prompt_deltas(self, dnn_input: torch.Tensor, reps: list[torch.Tensor]):
        """纯函数（无副作用）：返回 (逐路 delta, 有界方向 m)。forward 与诊断共用同一实现。"""
        m = torch.tanh(self.prompt_generator(dnn_input))     # 逐维 ∈ (-1, 1) ⇒ ||m|| < sqrt(d)
        deltas = []
        for h in reps:
            if h.shape[-1] != m.shape[-1]:
                raise ValueError(f"表征维度 {h.shape[-1]} 与 prompt 方向 {m.shape[-1]} 不一致，拒绝出数")
            scale = h.detach().norm(dim=-1, keepdim=True) / math.sqrt(m.shape[-1])
            deltas.append(self.prompt_gate * scale * m)
        return deltas, m

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        deltas, _ = self.prompt_deltas(dnn_input, [gen_rep, *spec_reps])
        gen_rep_p = gen_rep + deltas[0]
        spec_reps_p = [spec + delta for spec, delta in zip(spec_reps, deltas[1:])]
        return super().forward(dnn_input, gen_rep_p, spec_reps_p, env_embs)


def effective_gate(m: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """逐样本有效门控 g_eff = |alpha| · ||m|| / sqrt(d)（fp64）——相对扰动的实测值，上界为 |alpha|。"""
    alpha_abs = alpha.detach().double().abs()
    return alpha_abs * m.detach().double().norm(dim=-1) / math.sqrt(m.shape[-1])


def build_newtask(variant, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device):
    """按 variant 构造新任务头；未知 variant 直接拒绝（不允许静默回退到基线）。"""
    if variant == BASELINE_VARIANT:
        return NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
    if variant == VARIANT:
        return ResidualPromptNewTask(input_size=input_size, rep_dim=rep_dim,
                                     tower_dnn_hidden_units=tower_dnn_hidden_units,
                                     reg_dnn=reg_dnn, device=device)
    raise ValueError(f"未知 variant: {variant!r}（可选 {VARIANTS}）")


class PromptAudit:
    """M0/G1/G2/G3 运行期审计（处理臂专用；对真实构造出来的那个实例做检查）。

    - G1：从构造前 RNG 现场重建基线头，逐位比较全部共享参数/buffer、RNG 端点与新增键集；
    - G2：真实第一个训练 batch 上，α=0 的处理头前向与"载入同一共享权重"的基线头前向 `torch.equal`；
    - G3：逐 epoch 首个 batch 反传后的 α / 生成器梯度范数与 α 取值。
    """

    def __init__(self, newtask, *, rng_before, rng_after, input_size, rep_dim,
                 tower_dnn_hidden_units, reg_dnn, device=None):
        self.newtask = newtask
        saved = torch.get_rng_state()
        try:
            torch.set_rng_state(rng_before)
            reference = NewTask(input_size=input_size, rep_dim=rep_dim,
                                tower_dnn_hidden_units=list(tower_dnn_hidden_units),
                                reg_dnn=reg_dnn, device=device)
            endpoint = torch.get_rng_state()
        finally:
            torch.set_rng_state(saved)
        ref_sd, var_sd = reference.state_dict(), newtask.state_dict()
        shared_identical = all(torch.equal(ref_sd[key].cpu(), var_sd[key].cpu()) for key in ref_sd)
        extra_keys = sorted(set(var_sd) - set(ref_sd))
        self.construction = {
            "shared_params_bit_identical": bool(shared_identical),
            "global_rng_endpoint_identical": bool(torch.equal(endpoint, rng_after)),
            "extra_keys": extra_keys,
            "expected_extra_keys": sorted(EXTRA_PARAM_NAMES),
            "alpha_at_construction": float(newtask.prompt_gate.detach()),
        }
        # 共享权重载入参照：G1 已验证逐位一致；载入后 G2 只差"prompt 注入通路"这一个变量
        reference.load_state_dict({key: value for key, value in var_sd.items() if key in ref_sd})
        self._reference = reference
        self.init_forward: dict | None = None
        self.grad_records: list[dict] = []

    @torch.no_grad()
    def init_forward_check(self, dnn_input, gen_rep, spec_reps, env_embs) -> None:
        """只在真实第一个训练 batch 上执行一次（幂等）。"""
        if self.init_forward is not None:
            return
        reference = self._reference.to(dnn_input.device).eval()
        was_training = self.newtask.training
        self.newtask.eval()
        try:
            out_variant = self.newtask(dnn_input, gen_rep, spec_reps, env_embs)
            out_reference = reference(dnn_input, gen_rep, spec_reps, env_embs)
        finally:
            self.newtask.train(was_training)
        self.init_forward = {
            "bit_identical": bool(torch.equal(out_variant, out_reference)),
            "max_abs_diff": float((out_variant - out_reference).abs().max()),
            "n_samples": int(out_variant.numel()),
        }

    def after_backward(self, epoch: int, step: int) -> None:
        """逐 epoch 首个 batch：记录 α 与 α/生成器梯度范数（诊断 + G3 判据）。"""
        if step != 0:
            return
        alpha = self.newtask.prompt_gate
        generator_sq = sum(float(param.grad.detach().pow(2).sum())
                           for param in self.newtask.prompt_generator.parameters()
                           if param.grad is not None)
        self.grad_records.append({
            "epoch": int(epoch),
            "alpha": float(alpha.detach()),
            "alpha_grad_norm": float(alpha.grad.detach().norm()) if alpha.grad is not None else None,
            "generator_grad_norm": math.sqrt(generator_sq),
        })

    def result(self, alpha_final: float) -> dict:
        return {"construction": self.construction, "init_forward": self.init_forward,
                "grad_probe": self.grad_records, "alpha_final": float(alpha_final)}


class PromptStats:
    """逐流 ratio / cos 与逐样本有效门控的流式统计（fp64、CPU 累加、显存 O(1)）。"""

    stream_names = ("gen", "spec_0", "spec_1")

    def __init__(self) -> None:
        streams = len(self.stream_names)
        self.ratio_sum = torch.zeros(streams, dtype=torch.float64)
        self.ratio_sq_sum = torch.zeros(streams, dtype=torch.float64)
        self.ratio_max = torch.zeros(streams, dtype=torch.float64)
        self.delta_norm_sum = torch.zeros(streams, dtype=torch.float64)
        self.h_norm_sum = torch.zeros(streams, dtype=torch.float64)
        self.cos_sum = torch.zeros(streams, dtype=torch.float64)
        self.n_zero_rep = torch.zeros(streams, dtype=torch.long)
        self.geff_sum = self.geff_sq_sum = 0.0
        self.geff_min, self.geff_max = float("inf"), 0.0
        self.count = 0

    def update(self, deltas: list[torch.Tensor], reps: list[torch.Tensor],
               geff: torch.Tensor) -> None:
        if len(deltas) != len(self.stream_names) or len(reps) != len(deltas):
            raise ValueError(f"流数不匹配：deltas={len(deltas)} reps={len(reps)}")
        batch = torch.atleast_2d(reps[0]).shape[0]
        for i, (delta, h) in enumerate(zip(deltas, reps)):
            d = torch.atleast_2d(delta.detach().double().cpu())
            hh = torch.atleast_2d(h.detach().double().cpu())
            h_norm, d_norm = hh.norm(dim=1), d.norm(dim=1)
            ratio = d_norm / h_norm.clamp_min(1e-12)          # ||h||=0 ⇒ δ=0 ⇒ ratio 记 0
            self.ratio_sum[i] += ratio.sum()
            self.ratio_sq_sum[i] += (ratio * ratio).sum()
            self.ratio_max[i] = torch.maximum(self.ratio_max[i], ratio.max())
            self.delta_norm_sum[i] += d_norm.sum()
            self.h_norm_sum[i] += h_norm.sum()
            self.cos_sum[i] += F.cosine_similarity(d, hh, dim=1).sum()
            self.n_zero_rep[i] += int((h_norm == 0).sum())
        gate = torch.atleast_1d(geff.detach().double().cpu())
        self.geff_sum += float(gate.sum())
        self.geff_sq_sum += float((gate * gate).sum())
        self.geff_min = min(self.geff_min, float(gate.min()))
        self.geff_max = max(self.geff_max, float(gate.max()))
        self.count += int(batch)

    def result(self) -> dict:
        n = max(self.count, 1)
        ratio_mean = self.ratio_sum / n
        ratio_std = (self.ratio_sq_sum / n - ratio_mean * ratio_mean).clamp_min(0).sqrt()
        geff_mean = self.geff_sum / n
        geff_std = max(self.geff_sq_sum / n - geff_mean * geff_mean, 0.0) ** 0.5
        return {"streams": {"ratio_mean": ratio_mean.tolist(), "ratio_std": ratio_std.tolist(),
                            "ratio_max": self.ratio_max.tolist(),
                            "delta_norm_mean": (self.delta_norm_sum / n).tolist(),
                            "h_norm_mean": (self.h_norm_sum / n).tolist(),
                            "cos_mean": (self.cos_sum / n).tolist(),
                            "n_zero_rep": [int(v) for v in self.n_zero_rep]},
                "gate": {"geff_mean": geff_mean, "geff_std": geff_std,
                         "geff_min": self.geff_min if self.count else 0.0, "geff_max": self.geff_max},
                "count": self.count}


@torch.no_grad()
def evaluate_prompt_diagnostics(newtask, backbone, loader, device) -> dict:
    """选点完成、`best_state` 已载入后，只用 val 前向一遍（不训练、不参与选点）。"""
    newtask.eval()
    backbone.eval()
    stats = PromptStats()
    preds = []
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
        reps = [gen_rep, *spec_reps]
        deltas, m = newtask.prompt_deltas(dnn_input, reps)
        stats.update(deltas, reps, effective_gate(m, newtask.prompt_gate.detach()))
        preds.append(newtask(dnn_input, gen_rep, spec_reps, env_embs).detach().float().cpu())
    result = stats.result()
    pred = torch.cat(preds).double()
    result["dispersion"] = {
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std(unbiased=False)),
        "pred_min": float(pred.min()), "pred_max": float(pred.max()),
        "pred_q05": float(torch.quantile(pred, 0.05)), "pred_q50": float(torch.quantile(pred, 0.50)),
        "pred_q95": float(torch.quantile(pred, 0.95)), "n_samples": int(pred.numel())}
    result["alpha_final"] = float(newtask.prompt_gate.detach())
    return result


@torch.no_grad()
def reference_head_stats(newtask_path, backbone, loader, device, *, input_size, rep_dim,
                         tower_dnn_hidden_units, reg_dnn) -> dict:
    """只读载入一个参照 NewTask checkpoint（零训练），在 val 上重算预测离散度与 AUC 对账。

    缺失 → FileNotFoundError；非 state_dict / 键或形状不符 → ValueError。参照不参与任何判定。
    """
    path = Path(newtask_path)
    if not path.is_file():
        raise FileNotFoundError(f"参照 newtask checkpoint 不存在: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"参照 checkpoint 不是 state_dict: {type(state)}")
    with isolated_cpu_rng():
        head = NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=list(tower_dnn_hidden_units), reg_dnn=reg_dnn, device=device)
    try:
        head.load_state_dict(state)
    except RuntimeError as exc:
        raise ValueError(f"参照 checkpoint 与 NewTask 结构不符: {exc}") from exc
    head.to(device).eval()
    backbone.eval()
    ys, preds = [], []
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
        preds.append(head(dnn_input, gen_rep, spec_reps, env_embs).detach().float().cpu())
        ys.append(y)
    pred = torch.cat(preds).double()
    return {"newtask_checkpoint": str(path), "n_samples": int(pred.numel()),
            "val_auc": metrics.auc_score(torch.cat(ys), pred),
            "pred_dispersion": {"pred_mean": float(pred.mean()), "pred_std": float(pred.std(unbiased=False)),
                                "pred_min": float(pred.min()), "pred_max": float(pred.max()),
                                "pred_q05": float(torch.quantile(pred, 0.05)),
                                "pred_q50": float(torch.quantile(pred, 0.50)),
                                "pred_q95": float(torch.quantile(pred, 0.95))}}


def param_report(newtask) -> dict:
    """新增参数清单与预算核算（spec 4："可训练参数清单"）。"""
    items = [{"name": name, "shape": list(param.shape), "numel": int(param.numel())}
             for name, param in newtask.named_parameters() if name.startswith("prompt_")]
    new_total = sum(item["numel"] for item in items)
    total = sum(int(param.numel()) for param in newtask.parameters())
    head_total = total - new_total
    return {"new_param_list": items, "new_params_total": new_total, "head_params": head_total,
            "total_params": total, "new_ratio_of_head": new_total / head_total}


def arm_verdict(*, auc_test: float, auc_val: float, probe: dict, val_stats: dict, params: dict,
                reference: dict | None, protocol_ok: bool = True,
                baseline_auc_test: float = BASELINE_AUC_TEST,
                baseline_auc_val: float = BASELINE_AUC_VAL,
                expected_reference_auc: float = BASELINE_AUC_VAL,
                expected_reference_pred_std: float = REFERENCE_PRED_STD) -> dict:
    """M0 + G1–G8 + U1/U2 判定与三标签结局分类（预注册第 4/5 节；判据先于结果写定，不得事后修改）。"""
    construction = probe.get("construction") or {}
    init_forward = probe.get("init_forward") or {}
    records = probe.get("grad_probe") or []
    alpha_final = float(probe.get("alpha_final") or 0.0)
    gate = val_stats.get("gate") or {}
    dispersion = val_stats.get("dispersion") or {}

    reference = reference or {}
    reference_dispersion = reference.get("pred_dispersion") or {}
    ref_auc = reference.get("val_auc")
    ref_std = reference_dispersion.get("pred_std")
    m0_pass = bool(ref_auc is not None and ref_std is not None
                   and abs(float(ref_auc) - expected_reference_auc) <= REFERENCE_IDENTITY_TOL
                   and abs(float(ref_std) - expected_reference_pred_std) <= REFERENCE_IDENTITY_TOL)

    g1_pass = bool(construction.get("shared_params_bit_identical")
                   and construction.get("global_rng_endpoint_identical")
                   and construction.get("extra_keys") == sorted(EXTRA_PARAM_NAMES))
    g2_pass = bool(construction.get("alpha_at_construction") == 0.0
                   and init_forward.get("bit_identical"))
    if records:
        first, last = records[0], records[-1]
        g3_pass = bool(float(first.get("alpha_grad_norm") or 0.0) != 0.0
                       and float(last.get("alpha") or 0.0) != 0.0
                       and float(last.get("generator_grad_norm") or 0.0) > 0.0
                       and alpha_final != 0.0)
    else:
        g3_pass = False
    ratio_max = [float(r) for r in val_stats["streams"]["ratio_max"]]
    g4_pass = all(r <= abs(alpha_final) + BOUND_TOL for r in ratio_max)
    ratio_mean = [float(r) for r in val_stats["streams"]["ratio_mean"]]
    band_low = any(r < RATIO_BAND[0] for r in ratio_mean)
    band_high = any(r > RATIO_BAND[1] for r in ratio_mean)
    g5_pass = not band_low and not band_high
    g6_pass = bool(float(gate.get("geff_std") or 0.0) > 0.0
                   and float(gate.get("geff_max") or 0.0) > 0.0
                   and float(gate.get("geff_min") or 0.0) >= 0.0)
    pred_std = float(dispersion.get("pred_std") or 0.0)
    g7_pass = bool(pred_std > 0.0 and ref_std is not None and float(ref_std) > 0.0
                   and pred_std >= PRED_STD_MIN_RATIO * float(ref_std))
    names = sorted(str(item["name"]) for item in (params.get("new_param_list") or []))
    g8_pass = bool(names == sorted(EXTRA_PARAM_NAMES)
                   and int(params.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                   and int(params.get("head_params", -1)) == EXPECTED_HEAD_PARAMS)

    delta_test = float(auc_test) - baseline_auc_test
    delta_val = float(auc_val) - baseline_auc_val
    u1_pass = bool(delta_test >= AUC_TEST_DELTA_MIN)
    u2_pass = bool(delta_val > AUC_VAL_DIRECTION_MIN)
    u_pass = bool(u1_pass and u2_pass)

    if not m0_pass:
        classification, subreason = "MECHANISM_FAIL", "REFERENCE_IDENTITY"
    elif not (g1_pass and g2_pass and g4_pass and g8_pass):
        classification, subreason = "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"
    elif not g3_pass:
        classification, subreason = "MECHANISM_FAIL", "MECHANISM_INACTIVE"
    elif band_low:
        classification, subreason = "MECHANISM_FAIL", "MECHANISM_SILENT"
    elif band_high:
        classification, subreason = "MECHANISM_FAIL", "MECHANISM_OVER_PERTURB"
    elif not g6_pass:
        classification, subreason = "MECHANISM_FAIL", "GATE_DEGENERATE"
    elif not g7_pass:
        classification, subreason = "MECHANISM_FAIL", "PREDICTION_COLLAPSE"
    elif not protocol_ok:
        classification, subreason = "MECHANISM_FAIL", "PROTOCOL_INVALID"
    elif u_pass:
        classification, subreason = "VALID_POSITIVE", None
    else:
        classification, subreason = "VALID_NEGATIVE", None
    return {
        "M0": {"pass": m0_pass,
               "rule": f"|ref_val_auc - {expected_reference_auc}| <= {REFERENCE_IDENTITY_TOL} AND "
                       f"|ref_pred_std - {expected_reference_pred_std}| <= {REFERENCE_IDENTITY_TOL}",
               "observed": {"ref_val_auc": ref_auc, "ref_pred_std": ref_std}},
        "G1": {"pass": g1_pass, "rule": "shared bit-identical AND rng endpoint identical AND extra keys exact",
               "observed": {"shared_params_bit_identical": construction.get("shared_params_bit_identical"),
                            "global_rng_endpoint_identical": construction.get("global_rng_endpoint_identical"),
                            "extra_keys": construction.get("extra_keys")}},
        "G2": {"pass": g2_pass, "rule": "alpha_at_construction == 0 AND init forward bit-identical",
               "observed": {"alpha_at_construction": construction.get("alpha_at_construction"),
                            "init_bit_identical": init_forward.get("bit_identical")}},
        "G3": {"pass": g3_pass,
               "rule": "first alpha grad != 0 AND last epoch alpha != 0 AND generator grad > 0 AND best alpha != 0",
               "observed": {"first": records[0] if records else None, "last": records[-1] if records else None,
                            "alpha_final": alpha_final}},
        "G4": {"pass": g4_pass, "rule": f"every stream ratio_max <= |alpha_final| + {BOUND_TOL}",
               "observed": {"ratio_max": ratio_max, "alpha_abs": abs(alpha_final)}},
        "G5": {"pass": g5_pass, "rule": f"every stream ratio_mean in [{RATIO_BAND[0]}, {RATIO_BAND[1]}]",
               "observed": {"ratio_mean": ratio_mean, "band_low": band_low, "band_high": band_high}},
        "G6": {"pass": g6_pass, "rule": "geff_std > 0 AND geff_max > 0 AND geff_min >= 0",
               "observed": {"geff_std": gate.get("geff_std"), "geff_min": gate.get("geff_min"),
                            "geff_max": gate.get("geff_max")}},
        "G7": {"pass": g7_pass, "rule": f"pred_std > 0 AND pred_std >= {PRED_STD_MIN_RATIO} * ref_pred_std",
               "observed": {"pred_std": pred_std, "ref_pred_std": ref_std}},
        "G8": {"pass": g8_pass,
               "rule": f"5 exact prompt_ keys AND new_params_total == {EXPECTED_NEW_PARAMS_TOTAL} "
                       f"AND head_params == {EXPECTED_HEAD_PARAMS}",
               "observed": {"names": names, "new_params_total": params.get("new_params_total"),
                            "head_params": params.get("head_params")}},
        "U": {"pass": u_pass,
              "U1": {"pass": u1_pass, "rule": f"delta_test >= +{AUC_TEST_DELTA_MIN}",
                     "observed": {"auc_test": float(auc_test), "delta_test": delta_test,
                                  "baseline_auc_test": baseline_auc_test}},
              "U2": {"pass": u2_pass, "rule": "delta_val > 0",
                     "observed": {"auc_val": float(auc_val), "delta_val": delta_val,
                                  "baseline_auc_val": baseline_auc_val}}},
        "protocol_10_1": {"delta_test_ge_0.005": bool(delta_test >= 0.005), "val_positive": u2_pass},
        "classification": classification,
        "subreason": subreason,
        "pass": classification == "VALID_POSITIVE",
    }
