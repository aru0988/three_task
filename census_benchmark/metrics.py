"""AUC、机制指标 M1/M3/M4、A/B 门禁判定（spec 7、8）。不 import fvcore，不做 FLOPs。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from census_benchmark import protocol as P


def auc(y_true: torch.Tensor, y_hat: torch.Tensor) -> float:
    return float(roc_auc_score(y_true.int().cpu(), y_hat.detach().cpu()))


def env_accuracy(env_pred_logprob: torch.Tensor, env_ids: torch.Tensor) -> float:
    """M1：env_pred 与当前 env_ids 的一致率（梯度反转生效时应趋近 1/NUM_TASKS = 0.5）。"""
    return float((env_pred_logprob.argmax(dim=1).cpu() == env_ids.cpu()).float().mean())


class GateStats:
    """M3：各 gate 在样本维度的平均权重（流式累加，显存 O(1)）。

    gate_networks[i] 输出 2 维（specific 分支 / general 分支）且过 softmax → 两维互补，
    故 result() 报告**每个任务 specific 分支**的样本均值（= 1 − general 分支均值），长度 = num_tasks；
    这正是 B3「未坍缩到单一分支」判据所需的量（0.05 / 0.95 即两端的坍缩边界）。
    """

    def __init__(self, num_tasks: int = P.NUM_TASKS):
        self.total = torch.zeros(num_tasks, 2, dtype=torch.float64)
        self.count = 0

    def update(self, gate_outs: list[torch.Tensor]) -> None:
        for i, gate_out in enumerate(gate_outs):
            # 累加器恒在 CPU：先在原设备归约，再显式 .cpu()，否则 CPU 累加器接 CUDA 张量会 device mismatch
            self.total[i] += gate_out.detach().double().sum(dim=0).cpu()
        self.count += gate_outs[0].shape[0]

    def result(self) -> list[float]:
        return (self.total[:, 0] / max(self.count, 1)).tolist()


class RepStats:
    """M4：cos(gen_rep, spec_rep_i) 均值 + gen_rep 各维标准差均值（防常量表征）。

    用累加和 / 平方和代替"按固定 seed 抽样后保存中间张量"：全量、显存 O(1)、不受抽样影响
    （spec 7.2 实现约束以显存峰值为目的，流式累加达成同一目的且更强）。
    """

    def __init__(self, num_tasks: int = P.NUM_TASKS):
        self.cos_sum = torch.zeros(num_tasks, dtype=torch.float64)
        self.gen_sum = self.gen_sq_sum = None
        self.count = 0

    def update(self, gen_rep: torch.Tensor, spec_reps: list[torch.Tensor]) -> None:
        gen = gen_rep.detach().double()
        # 同 GateStats：归约留在原设备，结果显式 .cpu()，三个累加器恒为 CPU（result() 的返回类型不变）
        gen_sum = gen.sum(dim=0).cpu()
        gen_sq_sum = (gen * gen).sum(dim=0).cpu()
        self.gen_sum = gen_sum if self.gen_sum is None else self.gen_sum + gen_sum
        self.gen_sq_sum = gen_sq_sum if self.gen_sq_sum is None else self.gen_sq_sum + gen_sq_sum
        for i, spec in enumerate(spec_reps):
            self.cos_sum[i] += F.cosine_similarity(gen, spec.detach().double(), dim=1).sum().cpu()
        self.count += gen.shape[0]

    def result(self) -> dict:
        n = max(self.count, 1)
        mean = self.gen_sum / n
        std = (self.gen_sq_sum / n - mean * mean).clamp_min(0).sqrt()
        return {"cos_gen_spec": (self.cos_sum / n).tolist(), "gen_std": float(std.mean())}


@torch.no_grad()
def evaluate_newtask(newtask, backbone, loader, device, *, mechanism: bool = False) -> dict:
    """评估新任务头。mechanism=True 时额外算 M3/M4（只在 val 上开一次）。"""
    newtask.eval(); backbone.eval()                 # 三件套之 2：backbone 恒为 eval
    ys, preds = [], []
    gate = GateStats() if mechanism else None
    rep = RepStats() if mechanism else None
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽取
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        ys.append(y); preds.append(pred.detach())
        if mechanism:
            gate.update([backbone.gate_networks[i](dnn_input) for i in range(P.NUM_TASKS)])
            rep.update(gen_rep, spec_reps)
    out = {"auc": auc(torch.cat(ys), torch.cat(preds))}
    if mechanism:
        out["gate_mean"] = gate.result(); out.update(rep.result())
    return out


def judge(*, backbone_sha_equal: bool, grads_all_none: bool, split_ok: bool, split_stats_ok: bool,
          env_ids_ok: bool, auc_val_income: float, auc_val_marital: float, auc_test_education: float,
          auc_val_education_best: float, gate_mean: list[float], env_shares: list[float]) -> dict:
    """A1/A2/A4/A5 + B1–B4（spec 8）。A3 按需触发，由调用方另行写入，不进入 overall_pass。"""

    def verdict(passed: bool, **detail) -> dict:
        return {"pass": bool(passed), "detail": detail}

    report = {
        "A1": verdict(backbone_sha_equal and grads_all_none,
                      backbone_sha_equal=backbone_sha_equal, grads_all_none=grads_all_none),
        "A2": verdict(split_ok, split_fingerprint_consistent=split_ok),
        "A4": verdict(split_stats_ok, disjoint_and_complete=split_stats_ok),
        "A5": verdict(env_ids_ok, env_ids_sha256_matches_stage1=env_ids_ok),
        "B1": verdict(min(auc_val_income, auc_val_marital, auc_test_education) >= P.AUC_FLOOR,
                      auc_val_income=auc_val_income, auc_val_marital=auc_val_marital,
                      auc_test_education=auc_test_education, floor=P.AUC_FLOOR),
        "B2": verdict(abs(auc_val_education_best - auc_test_education) <= P.VAL_TEST_GAP,
                      gap=abs(auc_val_education_best - auc_test_education), limit=P.VAL_TEST_GAP),
        "B3": verdict(all(P.GATE_MIN <= w <= P.GATE_MAX for w in gate_mean), gate_mean=gate_mean),
        "B4": verdict(all(share >= P.ENV_SHARE_MIN for share in env_shares), env_shares=env_shares),
    }
    report["overall_pass"] = all(item["pass"] for item in report.values())
    report["failures"] = [key for key, item in report.items() if isinstance(item, dict) and not item["pass"]]
    return report


# ---- 跨数据集迁移臂（docs/superpowers/specs/2026-10-03-census-stage2-attenuation-transfer-design.md §1、§5）----
ALICCP_NULL_RUN_ID = "20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx"
ALICCP_NULL_MEAN = 0.3027766269669533              # 系数唯一来源：AliCCP null 臂 val 聚合 null 权重
SPEC_ATTENUATION_COEF = 1.0 - ALICCP_NULL_MEAN     # = 0.6972233730330467（文档字面量；冻结，禁止调参）
SPEC_ATTENUATION_COEF_F32 = 0.6972233653068542     # float32 铸造有效因子（记录备查，判定不依赖）
ALICCP_CONTROL_DELTA_TEST = 0.012706144575588052  # AliCCP 对照臂 Δtest（descriptive 参照）
ALICCP_REPRO_RATIO = 0.917640706647481             # AliCCP 对照臂重现率（descriptive 参照）

TRANSFER_BASELINE_RUN_ID = "20260929-1735-s20260929-m1685480945-short-904f8d0"
TRANSFER_BASELINE_TEST_AUC = 0.8500685307175756
TRANSFER_BASELINE_VAL_BEST = 0.8527881905614896
TRANSFER_TEST_THRESHOLD = 0.8521                   # 逐字沿用 exp/stage2-null-expert 预注册 §5；本分支不新设
TRANSFER_NULL_RUN_ID = "20260929-1802-s20260929-m1685480945-short-1eadd05-nullx"
TRANSFER_NULL_TEST_AUC = 0.8513648272440768
TRANSFER_NULL_VAL_BEST = 0.853020431042571
TRANSFER_NULL_DELTA_TEST = TRANSFER_NULL_TEST_AUC - TRANSFER_BASELINE_TEST_AUC   # = 0.001296296526501206


def transfer_arm_verdict(*, test_auc: float, val_best: float, construction_ok: bool, a_gates_ok: bool) -> dict:
    """跨数据集迁移臂机械判定（预注册 §5）：四条同时满足 → TRANSFER_SUPPORTED，否则 TRANSFER_NOT_SUPPORTED。

    test 阈值为闭边界（≥ 0.8521，逐字沿用 null-expert 预注册，不新设）；val 方向为严格大于基线 best。
    Δ/r 为 descriptive 记录，不参与分类。
    """
    criteria = {
        "test_auc_ge_threshold": bool(test_auc >= TRANSFER_TEST_THRESHOLD),
        "val_direction_improves": bool(val_best > TRANSFER_BASELINE_VAL_BEST),
        "construction_audit_ok": bool(construction_ok),
        "a_gates_ok": bool(a_gates_ok),
    }
    delta_test = float(test_auc) - TRANSFER_BASELINE_TEST_AUC
    return {
        "verdict": "TRANSFER_SUPPORTED" if all(criteria.values()) else "TRANSFER_NOT_SUPPORTED",
        "criteria": criteria,
        "evidence": {
            "test_auc": float(test_auc), "val_best": float(val_best),
            "threshold": TRANSFER_TEST_THRESHOLD,
            "baseline_test_auc": TRANSFER_BASELINE_TEST_AUC, "baseline_val_best": TRANSFER_BASELINE_VAL_BEST,
            "delta_test_vs_baseline": delta_test,
            "delta_val_vs_baseline": float(val_best) - TRANSFER_BASELINE_VAL_BEST,
            "delta_test_vs_null": float(test_auc) - TRANSFER_NULL_TEST_AUC,
            "reproduction_ratio_vs_null": delta_test / TRANSFER_NULL_DELTA_TEST,
            "aliccp_reference": {"delta_test": ALICCP_CONTROL_DELTA_TEST, "reproduction_ratio": ALICCP_REPRO_RATIO},
        },
    }


def _as_float64(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1)


def pred_dispersion(values) -> dict:
    """预测离散度（float64 终算；std/var 用 unbiased=False）。"""
    t = _as_float64(values)
    quant = torch.quantile(t, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=torch.float64))
    return {"mean": float(t.mean()), "std": float(t.std(unbiased=False)), "var": float(t.var(unbiased=False)),
            "q10": float(quant[0]), "q25": float(quant[1]), "q50": float(quant[2]),
            "q75": float(quant[3]), "q90": float(quant[4]),
            "min": float(t.min()), "max": float(t.max()), "n": int(t.numel())}


def pearson_corr(a, b) -> float:
    """Pearson 相关（float64 终算；退化输入返回 nan）。"""
    x, y = _as_float64(a), _as_float64(b)
    if x.numel() != y.numel():
        raise ValueError(f"长度不一致: {x.numel()} vs {y.numel()}")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(xc.norm() * yc.norm())
    return float(xc @ yc) / denom if denom > 0 else float("nan")


def _average_ranks(t: torch.Tensor) -> torch.Tensor:
    """平均秩（并列取均值），float64。"""
    n = t.numel()
    order = torch.argsort(t, stable=True)
    sorted_t = t[order]
    new_group = torch.ones(n, dtype=torch.bool)
    if n > 1:
        new_group[1:] = sorted_t[1:] != sorted_t[:-1]
    group_id = torch.cumsum(new_group.to(torch.int64), dim=0) - 1
    counts = torch.bincount(group_id, minlength=int(group_id[-1]) + 1).to(torch.float64)
    rank_sums = torch.zeros_like(counts).scatter_add_(0, group_id, torch.arange(1, n + 1, dtype=torch.float64))
    out = torch.empty(n, dtype=torch.float64)
    out[order] = (rank_sums / counts)[group_id]
    return out


def spearman_corr(a, b) -> float:
    """Spearman 秩相关（平均秩；等价于对秩求 Pearson）。"""
    return pearson_corr(_average_ranks(_as_float64(a)), _average_ranks(_as_float64(b)))


def logit_clip(preds, eps: float = 1e-6) -> torch.Tensor:
    """logit(clamp(p, eps, 1-eps))：相关分析的数值稳定口径。"""
    t = torch.as_tensor(preds, dtype=torch.float64).clamp(eps, 1.0 - eps)
    return torch.log(t / (1.0 - t))


def paired_delta_stats(a, b) -> dict:
    """逐样本配对差 a − b 的统计量（float64 终算）。"""
    d = _as_float64(a) - _as_float64(b)
    quant = torch.quantile(d, torch.tensor([0.10, 0.50, 0.90], dtype=torch.float64))
    return {"mean": float(d.mean()), "std": float(d.std(unbiased=False)),
            "min": float(d.min()), "max": float(d.max()),
            "q10": float(quant[0]), "q50": float(quant[1]), "q90": float(quant[2]),
            "frac_pos": float((d > 0).to(torch.float64).mean()), "n": int(d.numel())}


@torch.no_grad()
def newtask_probe(head, backbone, loader, device, *, num_source_tasks: int = 2) -> dict:
    """头级 val 探针（预注册 §6.1）：一次遍历、no_grad、流式 float64。

    与 `NewTask.forward` 同算子同顺序重算 specific 混合（衰减前/后范数）、融合路径范数与 gate 均值；
    对 null 头（use_null_expert=True）候选集含零专家，source gate 仅统计前 K 个源任务权重。
    """
    head.eval(); backbone.eval()
    attenuation = float(getattr(head, "spec_attenuation", 1.0))
    use_null = bool(getattr(head, "use_null_expert", False))
    preds = []
    head_gate_sum = source_gate_sum = null_share_sum = None
    norm_sum = {"spec_mix_pre": 0.0, "spec_mix_post": 0.0, "env_aware": 0.0, "fused": 0.0, "gen": 0.0}
    n = 0
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
        exist_env_embs = torch.stack(env_embs, dim=1)
        new_env_emb = head.env_embedding_network(head.new_env_idx).squeeze(0)
        projection = head.projection_network(dnn_input)
        if use_null:
            keys = torch.cat([exist_env_embs, head.null_key.detach().unsqueeze(1)], dim=1)
            values = torch.cat([torch.stack(spec_reps, dim=2), torch.zeros_like(spec_reps[0]).unsqueeze(2)], dim=2)
        else:
            keys, values = exist_env_embs, torch.stack(spec_reps, dim=2)
        W = F.softmax(torch.mm(projection, keys) / head.temperature, dim=-1)
        gate_out = head.gate_network(dnn_input)
        spec_mix = torch.matmul(values, W.unsqueeze(2)).squeeze()
        spec_post = spec_mix * attenuation
        env_aware = spec_post * new_env_emb
        fused = torch.matmul(torch.stack([env_aware, gen_rep], dim=2), gate_out.unsqueeze(2)).squeeze()
        pred = head(dnn_input, gen_rep, spec_reps, env_embs)
        preds.append(pred.detach().float().cpu())
        head_gate = gate_out.detach().double().sum(dim=0).cpu()
        source_gate = W[:, :num_source_tasks].detach().double().sum(dim=0).cpu()
        head_gate_sum = head_gate if head_gate_sum is None else head_gate_sum + head_gate
        source_gate_sum = source_gate if source_gate_sum is None else source_gate_sum + source_gate
        if use_null:
            null_share = W[:, num_source_tasks:].detach().double().sum(dim=0).cpu()
            null_share_sum = null_share if null_share_sum is None else null_share_sum + null_share
        norm_sum["spec_mix_pre"] += float(spec_mix.detach().double().norm(dim=1).sum())
        norm_sum["spec_mix_post"] += float(spec_post.detach().double().norm(dim=1).sum())
        norm_sum["env_aware"] += float(env_aware.detach().double().norm(dim=1).sum())
        norm_sum["fused"] += float(fused.detach().double().norm(dim=1).sum())
        norm_sum["gen"] += float(gen_rep.detach().double().norm(dim=1).sum())
        n += dnn_input.shape[0]
    out = {"pred": pred_dispersion(torch.cat(preds)),
           "head_gate_mean": (head_gate_sum / n).tolist(),
           "source_gate_mean": (source_gate_sum / n).tolist(),
           "norm_mean": {key: value / n for key, value in norm_sum.items()},
           "spec_attenuation": attenuation, "n": n}
    if use_null:
        out["null_share_mean"] = float(null_share_sum.sum() / n)
    return out
