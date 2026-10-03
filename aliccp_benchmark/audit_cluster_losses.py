"""AliCCP Stage-1 `cluster_2` 机制审计（只读诊断工具；判定与预注册见
docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md）。

目的：在固定公平基线的 Stage-1 产物上量化 `cluster_2`（multitaskrec/train.py：逐样本
per-task BCE 直接 argmin）的原始损失尺度结构，判定 B4 退化（env_0 仅 ~0.03%）是否由
CTR/CVR 逐样本损失尺度不可比驱动；并给出候选尺度校正（z-score / 稳健 z / rank-quantile）
在聚类时刻损失向量上的分配对照与分位数统计。

两种模式：
  1) 仅分析（默认）：用产物中保存的 best 权重复算损失向量。注意 best_epoch 可能 != 聚类
     epoch（产物只保存 best 权重），此时损失为近似；env_ids 交叉表与 env_pred 探针仍是
     精确记录值。
  2) `--reproduce`：跑一次与产物记录**逐位一致**的 stage1 复现（同 seed / 预算 / 配置 /
     顺序；训练全程无 shuffle、无额外 RNG 消耗），捕获聚类时刻（epoch%2==0 的 epoch 末）
     的模型权重与逐样本损失向量，并断言逐位复现（cluster 事件、逐 epoch val AUC、逐 epoch
     损失、best epoch、backbone sha、test AUC、env_ids sha 全部相等）。只有复现全过，
     捕获量才被当作"聚类时刻真值"，用于预注册预测（新 stage1_id 的聚类分配）。

本脚本不修改 multitaskrec/*、不改任何既有产物（产物目录只读打开）；全部输出写入 --out-dir
（artifacts/aliccp_bench/audit/，gitignore）。CPU/GPU 均可运行。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec

from . import bench, protocol

# 分位数口径（记录用；不参与任何判定）
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
ENV_PROBE_BATCHES = 200  # 与 bench.run_stage1 的 env_accuracy_probe 默认一致（训练集前 400k 行）


# ---------------------------------------------------------------- 候选尺度校正（参考实现）
def ref_zscore(losses: torch.Tensor) -> torch.Tensor:
    """候选 1：逐任务 z-score（population std）。零方差列 → 恒 0（确定性兜底）。float64。"""
    l = losses.double()
    mu = l.mean(dim=0, keepdim=True)
    sd = l.std(dim=0, unbiased=False, keepdim=True)
    z = torch.zeros_like(l)
    ok = (sd > 0).expand_as(l)
    z[ok] = ((l - mu) / sd)[ok]
    return z


def ref_robust_zscore(losses: torch.Tensor) -> torch.Tensor:
    """候选 2：逐任务稳健标准化（median / IQR）。零 IQR 列 → 恒 0（确定性兜底）。float64。"""
    l = losses.double()
    med = l.median(dim=0, keepdim=True).values
    q1 = torch.quantile(l, 0.25, dim=0, keepdim=True)
    q3 = torch.quantile(l, 0.75, dim=0, keepdim=True)
    iqr = q3 - q1
    z = torch.zeros_like(l)
    ok = (iqr > 0).expand_as(l)
    z[ok] = ((l - med) / iqr)[ok]
    return z


def ref_rank01(losses: torch.Tensor) -> torch.Tensor:
    """候选 3（最终选择）：逐任务 rank/quantile 变换，average-rank 归一化到 [0, 1]。

    语义（钉死；实现见 aliccp_benchmark/normalized_clustering.py，等价性由单测钉住）：
    - 1-based average rank（并列取平均秩）→ q = (rank - 1) / (N - 1)；
    - N == 1 → 恒 0；全并列 → 恒 0.5（确定性，无 NaN）；
    - 输入按 float64 计算（float32 损失上采后排序，输出 float64）。
    """
    l = losses.double()
    n = l.shape[0]
    cols = []
    for k in range(l.shape[1]):
        uniq, inv, counts = torch.unique(l[:, k], return_inverse=True, return_counts=True)
        cum = torch.cumsum(counts, dim=0)
        avg_rank = (cum - counts).double() + (counts.double() + 1.0) / 2.0  # 1-based 平均秩
        ranks = avg_rank[inv]
        cols.append((ranks - 1.0) / max(n - 1, 1))
    return torch.stack(cols, dim=1)


CANDIDATES = {
    "raw": lambda losses: losses.double(),
    "zscore": ref_zscore,
    "robust_zscore": ref_robust_zscore,
    "rank01": ref_rank01,
}


# ---------------------------------------------------------------- 复算损失向量（与 cluster_2 同 ops）
@torch.no_grad()
def loss_vectors(model, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """逐样本 per-task BCE（(N, 2) float32 CPU）+ 标签 ((N, 2) float32 CPU)。

    逐行操作与 multitaskrec/train.py `MPTRecTrainManager.cluster_2` 完全一致：
    `pred = model.cluster_predict(features)`；`BCELoss(reduction='none')(pred[k].cpu(), y_k.float())`。
    不改变任何训练状态（eval 模式、no_grad、不消耗 RNG）。
    """
    model.eval()
    loss_func = torch.nn.BCELoss(reduction="none")
    loss_chunks, y_chunks = [], []
    for y_0, y_1, _, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        pred = model.cluster_predict(features)
        loss_0 = loss_func(pred[0].cpu(), y_0.float())
        loss_1 = loss_func(pred[1].cpu(), y_1.float())
        loss_chunks.append(torch.stack([loss_0, loss_1], dim=1))
        y_chunks.append(torch.stack([y_0.float(), y_1.float()], dim=1))
    return torch.cat(loss_chunks, dim=0), torch.cat(y_chunks, dim=0)


class _CapturingManager(bench.RecordingMPTRecTrainManager):
    """复现运行专用 manager：完整走父类 `cluster_2`（含事件记录），额外**只读记录**
    聚类时刻权重、损失向量与标签。不改变任何计算行为、不消耗 RNG。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_capture: dict | None = None

    def cluster_2(self):
        result = super().cluster_2()
        losses, ys = loss_vectors(self.model, self.train_loader, self.device)
        mismatch = int((torch.argmin(losses, dim=1) != result).sum())
        self.cluster_capture = {
            "epoch": len(self.val_epoch_aucs) + 1,
            "state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "losses": losses,
            "ys": ys,
            "assignment": result.detach().cpu(),
            "raw_argmin_mismatch": mismatch,  # 复算损失向量的 argmin 与父类分配必须逐位一致 → 期望 0
            "env_ids_sha256": protocol.sha256_tensor(result.detach().cpu()),
        }
        return result


# ---------------------------------------------------------------- 统计工具
def _quantile_stats(values: torch.Tensor) -> dict:
    v = values.double()
    stats = {
        "n": int(v.numel()),
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
    }
    for q in QUANTILES:
        stats[f"q{q:g}"] = float(torch.quantile(v, q))
    return stats


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """两向量的 Spearman 秩相关（average-rank；由 float64 秩的 Pearson 计算）。"""
    ra = ref_rank01(a.double().unsqueeze(1)).squeeze(1)
    rb = ref_rank01(b.double().unsqueeze(1)).squeeze(1)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(torch.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum()) / denom if denom > 0 else float("nan")


def _env_crosstab(env_ids: torch.Tensor, ys: torch.Tensor) -> dict:
    """env 分配与 (click, purchase) 标签的交叉表 + 集合等式判定。"""
    click, purchase = ys[:, 0] > 0.5, ys[:, 1] > 0.5
    out = {"n": int(env_ids.numel()), "env_0": int((env_ids == 0).sum()), "env_1": int((env_ids == 1).sum())}
    for env in (0, 1):
        m = env_ids == env
        out[f"env_{env}_purchase1"] = int((m & purchase).sum())
        out[f"env_{env}_click1"] = int((m & click).sum())
        out[f"env_{env}_share"] = float(m.double().mean())
    out["purchase1_total"] = int(purchase.sum())
    out["click1_total"] = int(click.sum())
    out["env0_equals_purchase1_set"] = bool(torch.equal(env_ids == 0, purchase))
    out["env0_equals_click1_set"] = bool(torch.equal(env_ids == 0, click))
    out["env0_subset_of_purchase1"] = bool(((env_ids == 0) & ~purchase).sum() == 0)
    return out


def _assignment_summary(assignment: torch.Tensor, ys: torch.Tensor) -> dict:
    summary = _env_crosstab(assignment, ys)
    return summary


@torch.no_grad()
def env_pred_probe(model, loader, env_ids, batch_size, device, max_batches=ENV_PROBE_BATCHES) -> dict:
    """env_pred 与最终 env_ids 的一致率（复刻 bench.env_accuracy_probe 的批切语义）+ 平衡准确率
    （macro recall，替代在退化分配下误导性的大多数类准确率）。"""
    model.eval()
    preds, ids_list = [], []
    for step, (_, _, _, features) in enumerate(loader):
        if step >= max_batches:
            break
        for key in features:
            features[key] = features[key].to(device)
        env_pred = model(features)["env_pred"]
        ids = env_ids[batch_size * step : batch_size * (step + 1)]
        n = len(ids)
        preds.append(env_pred.argmax(dim=1)[:n].cpu())
        ids_list.append(ids.cpu())
    pred = torch.cat(preds)
    ids = torch.cat(ids_list)
    acc = float((pred == ids).float().mean())
    recalls = {}
    for env in (0, 1):
        m = ids == env
        recalls[env] = float((pred[m] == env).float().mean()) if bool(m.any()) else float("nan")
    bal = (recalls[0] + recalls[1]) / 2.0
    return {"n": int(ids.numel()), "acc": acc, "balanced_acc": bal,
            "recall_env_0": recalls[0], "recall_env_1": recalls[1]}


# ---------------------------------------------------------------- 复现运行
def _rebuild_from_meta(meta: dict, data_files: dict, device) -> tuple[dict, dict, MPTRec]:
    """按产物 meta 记录的配置重建 dataset/loader/model（保真复现；顺序与 bench.run_stage1 一致）。"""
    budgets, cfg = meta["budgets"], meta["config"]
    datasets = {s: AliCCPDataset(data_files[s], budgets[s]) for s in ("train", "val", "test")}
    loaders = {
        s: DataLoader(datasets[s], batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
        for s in ("train", "val", "test")
    }
    model = MPTRec(
        num_tasks=cfg["num_tasks"],
        feature_vocabulary={str(k): int(v) for k, v in cfg["vocab"].items()},
        embedding_size=cfg["embedding_size"],
        input_size=cfg["input_size"],
        expert_dnn_hidden_units=list(cfg["expert_hidden"]),
        tower_dnn_hidden_units=list(cfg["tower_hidden"]),
        dropout=list(cfg["dropout"]),
        reg_embedding=cfg["reg_embedding"],
        reg_dnn=cfg["reg_dnn"],
        device=device,
    ).to(device)
    return datasets, loaders, model


def reproduce(meta: dict, data_files: dict, device, log=print) -> dict:
    """跑一次与产物记录逐位一致的 stage1，返回 (model, loaders, manager, checks)。"""
    cfg = meta["config"]
    protocol.seed_model(meta["model_seed"])
    datasets, loaders, model = _rebuild_from_meta(meta, data_files, device)
    log(f"[audit] 复现运行：train={len(datasets['train'])} val={len(datasets['val'])} test={len(datasets['test'])}")
    env_ids = protocol.make_env_ids(len(datasets["train"]), meta["env_seed"])
    manager = _CapturingManager(
        model=model, train_loader=loaders["train"], val_loader=loaders["val"], env_ids=env_ids,
        task_name=["CTR", "CVR"], lr=cfg["lr"], batch_size=cfg["batch_size"],
        uni_coe=cfg["uni_coe"], env_coe=cfg["env_coe"], epochs=meta["epochs"], patience=meta["patience"],
    )
    manager.train_two_task()

    checks: dict = {}
    checks["cluster_events_equal"] = manager.cluster_events == meta["cluster_events"]
    per_epoch = meta["per_epoch"]
    checks["per_epoch_len_equal"] = len(manager.val_epoch_aucs) == len(per_epoch)
    same = True
    for i, rec in enumerate(per_epoch):
        ok = (
            manager.val_epoch_aucs[i][0] == rec["auc_val_ctr"]
            and manager.val_epoch_aucs[i][1] == rec["auc_val_cvr"]
            and float(manager.uni_loss_0_list[i]) == rec["uni_loss_0"]
            and float(manager.uni_loss_1_list[i]) == rec["uni_loss_1"]
            and float(manager.fused_loss_0_list[i]) == rec["fuse_loss_0"]
            and float(manager.fused_loss_1_list[i]) == rec["fuse_loss_1"]
            and float(manager.env_loss_list[i]) == rec["env_loss"]
        )
        same = same and ok
    checks["per_epoch_values_bit_equal"] = bool(same)
    checks["best_epoch_equal"] = manager.best_epoch() == meta["best_epoch"]

    model.load_state_dict(manager.best_weight)
    checks["best_backbone_sha_equal"] = protocol.backbone_sha256(model) == meta["backbone_sha256"]
    test_aucs = manager.evaluation_two_task(loaders["test"])
    checks["test_auc_ctr_equal"] = float(test_aucs[0]) == meta["test_auc_ctr"]
    checks["test_auc_cvr_equal"] = float(test_aucs[1]) == meta["test_auc_cvr"]
    checks["env_ids_sha_equal"] = protocol.sha256_tensor(manager.env_ids.detach().cpu()) == meta["env_ids_sha256"]
    capture = manager.cluster_capture or {}
    checks["capture_argmin_matches_parent_assignment"] = capture.get("raw_argmin_mismatch") == 0
    checks["capture_assignment_sha_equal_env_ids_meta"] = capture.get("env_ids_sha256") == meta["env_ids_sha256"]
    return {"model": model, "loaders": loaders, "datasets": datasets, "manager": manager,
            "checks": checks, "capture": capture}


# ---------------------------------------------------------------- 主流程
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_audit(stage1_dir: Path, out_dir: Path, data_files: dict, device, reproduce_epoch: bool,
              log=print) -> dict:
    t0 = time.time()
    stage1_dir = Path(stage1_dir)
    meta = json.loads((stage1_dir / "meta.json").read_text(encoding="utf-8"))
    env_ids = torch.load(stage1_dir / "env_ids.pt", map_location="cpu")
    backbone_state = torch.load(stage1_dir / "backbone.pt", map_location="cpu")
    env_ids_sha_ok = protocol.sha256_tensor(env_ids) == meta["env_ids_sha256"]
    log(f"[audit] 产物 {meta['stage1_id']}：env_ids sha 与 meta 一致 = {env_ids_sha_ok}；"
        f"cluster_events={meta['cluster_events']}")
    if not env_ids_sha_ok:
        raise AssertionError("env_ids.pt 与 meta 记录不一致（产物被改动？）")

    report: dict = {
        "audit": {
            "stage1_dir": str(stage1_dir), "mode": "reproduce" if reproduce_epoch else "artifact_only",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "commit": protocol.code_commit(), "device": str(device),
        },
        "artifact": {
            "stage1_id": meta["stage1_id"], "model_seed": meta["model_seed"], "env_seed": meta["env_seed"],
            "epochs": meta["epochs"], "patience": meta["patience"], "best_epoch": meta["best_epoch"],
            "fingerprint_sha256": meta["fingerprint_sha256"], "backbone_sha256": meta["backbone_sha256"],
            "env_ids_sha256": meta["env_ids_sha256"], "cluster_events": meta["cluster_events"],
            "env_acc_recorded": meta.get("env_acc"),
            "test_auc_ctr": meta["test_auc_ctr"], "test_auc_cvr": meta["test_auc_cvr"],
            "best_val_auc_ctr": meta["best_val_auc_ctr"], "best_val_auc_cvr": meta["best_val_auc_cvr"],
        },
    }

    if reproduce_epoch:
        rep = reproduce(meta, data_files, device, log=log)
        checks = rep["checks"]
        report["reproduction"] = {"checks": checks, "all_pass": bool(all(checks.values()))}
        if not report["reproduction"]["all_pass"]:
            log("[audit] 警告：复现未全部逐位一致——捕获量按近似量对待，预测不可用。")
        capture = rep["capture"]
        report["reproduction"]["capture"] = {
            "epoch": capture["epoch"],
            "raw_argmin_mismatch": capture["raw_argmin_mismatch"],
            "env_ids_sha256_from_capture": capture["env_ids_sha256"],
        }
        losses, ys = capture["losses"], capture["ys"]
        assignment = capture["assignment"]
        losses_source = f"captured_epoch{capture['epoch']}"
        model = rep["model"]
        loaders = rep["loaders"]
        batch_size = meta["config"]["batch_size"]
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(capture["state"], out_dir / f"epoch{capture['epoch']}_state.pt")
        torch.save(losses, out_dir / "cluster_losses.pt")
    else:
        _, loaders, model = _rebuild_from_meta(meta, data_files, device)  # loader 语义 (batch/顺序) 与产物一致
        model.load_state_dict(backbone_state)
        batch_size = meta["config"]["batch_size"]
        losses, ys = loss_vectors(model, loaders["train"], device)
        assignment = env_ids
        losses_source = "artifact_best_backbone（近似聚类时刻；best_epoch 可能 != 聚类 epoch）"

    report["loss_vectors"] = {
        "source": losses_source,
        "finite": bool(torch.isfinite(losses).all()),
        "argmin_agreement_with_final_env_ids": {
            "matches": int((torch.argmin(losses, dim=1) == env_ids).sum()),
            "fraction": float((torch.argmin(losses, dim=1) == env_ids).double().mean()),
        },
    }
    report["env_crosstab_final_env_ids"] = _env_crosstab(env_ids, ys)

    # ---- 原始损失尺度结构 ----
    raw_stats = {}
    for k, task in enumerate(("ctr_task0", "cvr_task1")):
        col, y = losses[:, k].double(), ys[:, k] > 0.5
        raw_stats[task] = {
            "all": _quantile_stats(col),
            "positive": _quantile_stats(col[y]) if bool(y.any()) else None,
            "negative": _quantile_stats(col[~y]),
        }
    entropy = {}
    for k, task in enumerate(("ctr_task0", "cvr_task1")):
        q = float((ys[:, k] > 0.5).double().mean())
        entropy[task] = {
            "base_rate": q,
            "binary_entropy": -(q * torch.log(torch.tensor(q, dtype=torch.float64))
                                + (1 - q) * torch.log(torch.tensor(1 - q, dtype=torch.float64))).item()
            if 0 < q < 1 else 0.0,
            "empirical_mean_loss": float(losses[:, k].double().mean()),
        }
    report["loss_stats"] = {
        "raw": raw_stats, "entropy_vs_mean_loss": entropy,
        "mean_scale_ratio_task0_over_task1": entropy["ctr_task0"]["empirical_mean_loss"] / max(
            entropy["cvr_task1"]["empirical_mean_loss"], 1e-300),
    }

    # ---- 候选尺度校正分配对照（同一损失向量） ----
    candidates = {}
    for name, fn in CANDIDATES.items():
        q = fn(losses)
        assign = torch.argmin(q, dim=1)
        candidates[name] = {
            "env_crosstab": _assignment_summary(assign, ys),
            "normalized_stats": {
                task: _quantile_stats(q[:, k]) for k, task in enumerate(("ctr_task0", "cvr_task1"))
            },
        }
    report["candidates"] = candidates
    report["rank_correlation"] = {
        "spearman_task0_task1_all": _spearman(losses[:, 0], losses[:, 1]),
        "spearman_task0_task1_purchase0": _spearman(losses[ys[:, 1] <= 0.5, 0], losses[ys[:, 1] <= 0.5, 1]),
        "spearman_task0_task1_click0": _spearman(losses[ys[:, 0] <= 0.5, 0], losses[ys[:, 0] <= 0.5, 1]),
    }

    # ---- env_pred 探针（best 权重） ----
    probe = env_pred_probe(model, loaders["train"], env_ids, batch_size, device)
    probe["env_acc_recorded"] = meta.get("env_acc")
    probe["env_acc_matches_recorded"] = (
        meta.get("env_acc") is not None
        and abs(probe["acc"] - float(meta["env_acc"])) <= 1e-12
    )
    report["env_pred_probe"] = probe
    report["audit"]["wall_seconds"] = round(time.time() - t0, 1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report, log)
    return report


def _print_summary(report: dict, log=print) -> None:
    art = report["artifact"]
    log("=" * 100)
    log(f"[audit] {art['stage1_id']}  mode={report['audit']['mode']}  损失来源: {report['loss_vectors']['source']}")
    if "reproduction" in report:
        checks = report["reproduction"]["checks"]
        log(f"[audit] 复现校验: all_pass={report['reproduction']['all_pass']} "
            f"({sum(bool(v) for v in checks.values())}/{len(checks)})")
        for key, val in checks.items():
            if not val:
                log(f"[audit]   !! {key} = {val}")
    ct = report["env_crosstab_final_env_ids"]
    log(f"[audit] 最终 env_ids 交叉表: env_0={ct['env_0']} ({ct['env_0_share']:.5%}), env_1={ct['env_1']}; "
        f"purchase1={ct['purchase1_total']}; env_0==purchase1 集合: {ct['env0_equals_purchase1_set']}; "
        f"env_0⊆purchase1: {ct['env0_subset_of_purchase1']}")
    ls = report["loss_stats"]
    for task in ("ctr_task0", "cvr_task1"):
        st = ls["raw"][task]["all"]
        log(f"[audit] raw {task}: mean={st['mean']:.6f} std={st['std']:.6f} "
            f"q0={st['q0']:.3g} q50={st['q0.5']:.6g} q99={st['q0.99']:.4g} q100={st['q1']:.4g}")
    log(f"[audit] 熵/尺度: " + "; ".join(
        f"{t}: rate={v['base_rate']:.6g} H={v['binary_entropy']:.6f} mean_loss={v['empirical_mean_loss']:.6f}"
        for t, v in ls["entropy_vs_mean_loss"].items()))
    log(f"[audit] 平均尺度比 task0/task1 = {ls['mean_scale_ratio_task0_over_task1']:.1f}×")
    for name, c in report["candidates"].items():
        e = c["env_crosstab"]
        log(f"[audit] 候选 {name:14s}: env_0={e['env_0']:>9d} ({e['env_0_share']:.5%}) "
            f"env_1={e['env_1']:>9d} | env_0∩purchase1={e['env_0_purchase1']} | "
            f"env0==purchase1集合: {e['env0_equals_purchase1_set']}")
    rc = report["rank_correlation"]
    log(f"[audit] Spearman(loss0, loss1): all={rc['spearman_task0_task1_all']:.4f} "
        f"purchase0={rc['spearman_task0_task1_purchase0']:.4f} click0={rc['spearman_task0_task1_click0']:.4f}")
    probe = report["env_pred_probe"]
    log(f"[audit] env_pred: acc={probe['acc']:.6f}（记录值 {probe['env_acc_recorded']}，一致={probe['env_acc_matches_recorded']}）"
        f" balanced_acc={probe['balanced_acc']:.4f} recall_env0={probe['recall_env_0']:.4f} "
        f"recall_env1={probe['recall_env_1']:.4f}")
    log("=" * 100)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AliCCP Stage-1 cluster_2 机制审计（只读；见设计文档）")
    parser.add_argument("--stage1-dir", type=str, required=True, help="Stage-1 产物目录（只读打开）")
    parser.add_argument("--out-dir", type=str, required=True, help="审计输出目录（gitignore）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true", help="强制 CPU（默认 cuda:<gpu>）")
    parser.add_argument("--reproduce", action="store_true",
                        help="跑一次与产物逐位一致的 stage1 复现并捕获聚类时刻权重/损失")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.init()
    run_audit(
        stage1_dir=Path(args.stage1_dir), out_dir=Path(args.out_dir),
        data_files=protocol.DATA_FILES, device=device, reproduce_epoch=args.reproduce,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
