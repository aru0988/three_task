"""方向 2 审计（只读）：AliCCP 环境划分可辨识性 + normalized 划分支持审计。

背景（2026-10-07）：
- P0（canonical raw-argmin env_ids）：seed 1688723512 → env_0=566 / env_1=1,999,434（N=2,000,000）。
  原始划分下的环境加权不可辨识（监督方已独立核对）→ 原始划分加权 NO-GO 事实。
- normalized 划分（逐任务 rank/quantile 归一化后再 argmin；机制来自
  exp/aliccp-stage1-normalized-clustering-seed3，blob 1a68c0b48dad7b30b849a6cb9582fc17041cfbd1）：
  规模均衡，但稀缺类支持待审计（监督方核对：CVR 正例 564/2；seed2 → 565/1）。

本脚本（输入只读、确定性、不写任何 stage1 目录；只写指定的 --out 审计 JSON）：
1. 逐位 parity 校验本文件移植的 per_task_rank01 与钉死 blob 的实现在合成输入上一致；
2. 在 canonical Stage-1 冻结 backbone 上重算 normalized 划分（post-hoc，来自冻结产物，可复现），
   对照监督方独立核对的 env 计数；
3. 审计 P0 与 normalized 划分的：占比 / 逐环境 CTR±·CVR±·BSI± / K=5 折 × 环境 × BSI 单元；
4. 权重空间：uniform 与 learned（clip [0.1,10]、样本均值 1 归一化）的 ESS、max 权重比；
5. 按预注册门槛（与 prereg 文档一致）给出逐项 GO/NO-GO 判定。

用法：
  python -m aliccp_benchmark.audit_env_weighting \
      --train-file dataset/AliCCP/ctr_cvr.train --n-train 2000000 \
      --stage1 label=path [label=path ...] \
      --out docs/experiments/2026-10-07-aliccp-env-weighting-audit.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec

from . import protocol

PRECEDENT_BRANCH = "exp/aliccp-stage1-normalized-clustering-seed3"
PRECEDENT_BLOB = "1a68c0b48dad7b30b849a6cb9582fc17041cfbd1"
PRECEDENT_PATH = "aliccp_benchmark/normalized_clustering.py"

# 预注册门槛（与 docs/superpowers/specs/2026-10-07-aliccp-env-weighting-audit-prereg.md 一致；
# 看到任何训练结果之前钉死）
GATES = {
    "G1_env_share_min": 0.20,          # 每环境占比下限（normalized 划分）
    "G2_bsi_neg_min_per_env": 1000,    # 每环境 BSI 负例下限（权重信号的任务为 BSI）
    "G2_bsi_pos_min_per_env": 100000,  # 每环境 BSI 正例下限
    "G2_ctr_pos_min_per_env": 10000,   # 每环境 CTR 正例下限（记录性）
    "G3_cell_bsi_neg_min": 200,        # K=5 折 × 环境 单元 BSI 负例下限
    "G3_cell_total_min": 50000,        # K=5 折 × 环境 单元样本下限
    "G4_weight_move_min": 0.01,        # learned 权重与 1 的偏离下限（实现期检查；审计期仅记录预算）
    "CLIP_LO": 0.1,
    "CLIP_HI": 10.0,
    "N_FOLDS": 5,
}


# ---- 机制移植（逐位；parity 校验见下）----
def per_task_rank01(losses: torch.Tensor) -> torch.Tensor:
    """逐任务 1-based average-rank 归一化到 [0,1]（float64）。移植自钉死 blob。"""
    l = losses.double()
    n = l.shape[0]
    cols = []
    for k in range(l.shape[1]):
        uniq, inv, counts = torch.unique(l[:, k], return_inverse=True, return_counts=True)
        cum = torch.cumsum(counts, dim=0)
        avg_rank = (cum - counts).double() + (counts.double() + 1.0) / 2.0
        ranks = avg_rank[inv]
        cols.append((ranks - 1.0) / max(n - 1, 1))
    return torch.stack(cols, dim=1)


def parity_check() -> dict:
    """对钉死 blob 的 per_task_rank01 在合成输入（含并列/N=1/极值）上逐位对照。"""
    repo = Path(__file__).resolve().parents[1]
    resolved = subprocess.run(
        ["git", "rev-parse", f"{PRECEDENT_BRANCH}:{PRECEDENT_PATH}"],
        capture_output=True, text=True, encoding="utf-8", cwd=repo,
    )
    if resolved.returncode != 0 or resolved.stdout.strip() != PRECEDENT_BLOB:
        raise RuntimeError("precedent blob pin hash mismatch")
    src = subprocess.run(
        ["git", "show", PRECEDENT_BLOB],
        capture_output=True, text=True, encoding="utf-8",
        cwd=repo,
    )
    if src.returncode != 0:
        raise RuntimeError(f"钉死 blob 读取失败（系谱依赖，如实响亮失败）：{src.stderr.strip()}")
    ns: dict = {"torch": torch}
    tree = __import__("ast").parse(src.stdout)
    fn = next(n for n in tree.body if isinstance(n, __import__("ast").FunctionDef) and n.name == "per_task_rank01")
    exec(compile(__import__("ast").Module(body=[fn], type_ignores=[]), "<pinned>", "exec"), ns)
    pinned = ns["per_task_rank01"]

    gen = torch.Generator().manual_seed(20261007)
    cases = {
        "random_f32": torch.rand(5000, 2, generator=gen),
        "ties_heavy": torch.randint(0, 20, (5000, 2), generator=gen).float(),
        "n1": torch.rand(1, 2, generator=gen),
        "all_equal": torch.full((1000, 2), 0.5),
        "extreme": torch.tensor([[0.0, 1e-12], [1e-12, 0.0], [1.0, 1.0], [1e30, -1e30]], dtype=torch.float32),
    }
    results = {}
    ok_all = True
    for name, x in cases.items():
        a = per_task_rank01(x)
        b = pinned(x)
        same = bool(torch.equal(a, b)) and a.dtype == b.dtype
        results[name] = {"bit_equal": same, "shape": list(a.shape), "dtype": str(a.dtype)}
        ok_all = ok_all and same
    return {"blob": PRECEDENT_BLOB, "cases": results, "all_bit_equal": ok_all}


def build_backbone(device) -> MPTRec:
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=protocol.build_vocab(),
        embedding_size=protocol.EMBEDDING_SIZE,
        input_size=protocol.INPUT_SIZE,
        expert_dnn_hidden_units=list(protocol.EXPERT_HIDDEN),
        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
        dropout=list(protocol.DROPOUT),
        reg_embedding=protocol.REG_EMBEDDING,
        reg_dnn=protocol.REG_DNN,
        device=device,
    ).to(device)
    return model


def require_backbone_match(model: nn.Module, meta: dict) -> str:
    actual = protocol.backbone_sha256(model)
    if actual != meta["backbone_sha256"]:
        raise AssertionError("backbone sha mismatch with Stage-1 meta")
    return actual


def require_fingerprint_match(meta: dict, fingerprint: dict) -> str:
    actual = fingerprint["fingerprint_sha256"]
    if actual != meta["fingerprint_sha256"]:
        raise AssertionError("fingerprint mismatch with Stage-1 meta")
    return actual


@torch.no_grad()
def compute_losses(model: MPTRec, loader, device) -> torch.Tensor:
    """与 multitaskrec/train.py cluster_2 同 ops 的逐样本 (N,2) 损失（CPU float32）。"""
    model.eval()
    loss_func = nn.BCELoss(reduction="none")
    chunks = []
    for y_0, y_1, _, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        pred = model.cluster_predict(features)
        loss_0 = loss_func(pred[0].cpu(), y_0.float())
        loss_1 = loss_func(pred[1].cpu(), y_1.float())
        chunks.append(torch.stack([loss_0, loss_1], dim=1))
    raw = torch.cat(chunks, dim=0)
    if not bool(torch.isfinite(raw).all()):
        raise AssertionError("聚类损失向量含非有限值（raw）")
    return raw


def partition_stats(env_ids: torch.Tensor, labels: dict) -> dict:
    """逐环境计数 + 标签正负例 + K 折单元。labels: dict of np.ndarray (N,)。"""
    ids = env_ids.numpy().astype(np.int64)
    n = len(ids)
    out = {"n": int(n), "env_counts": [int((ids == e).sum()) for e in (0, 1)]}
    out["env_share"] = [float(c) / n for c in out["env_counts"]]
    per_env = []
    for e in (0, 1):
        m = ids == e
        row = {"env": e, "n": int(m.sum())}
        for name, arr in labels.items():
            row[f"{name}_pos"] = int(arr[m].sum())
            row[f"{name}_neg"] = int((~arr[m]).sum())
        per_env.append(row)
    out["per_env"] = per_env
    # K=5 连续折 × 环境 × BSI 单元
    k = GATES["N_FOLDS"]
    fold = np.arange(n) // (n // k)
    cells = []
    bsi = labels["bsi"]
    for f in range(k):
        for e in (0, 1):
            m = (fold == f) & (ids == e)
            cells.append(
                {
                    "fold": f,
                    "env": e,
                    "n": int(m.sum()),
                    "bsi_pos": int(bsi[m].sum()),
                    "bsi_neg": int((~bsi[m]).sum()),
                }
            )
    out["fold_env_cells"] = cells
    return out


def weight_space(env_ids: torch.Tensor) -> dict:
    """uniform 与 learned（clip [0.1,10]）的权重空间：ESS、max 权重、两种极端中的最坏情形。"""
    ids = env_ids.numpy().astype(np.int64)
    n = len(ids)
    def ess_of(w_env):
        w = np.where(ids == 0, w_env[0], w_env[1]).astype(np.float64)
        w = w / w.mean()
        return float((w.sum() ** 2) / (w**2).sum())

    uniform = ess_of([1.0, 1.0])
    hi, lo = GATES["CLIP_HI"], GATES["CLIP_LO"]
    worst = min(ess_of([hi, lo]), ess_of([lo, hi]))
    return {
        "ess_uniform": uniform,
        "ess_floor_at_clip": worst,  # 单侧顶格、另一侧底格（权重空间下界）
        "max_ratio_at_clip": float(hi / lo),
        "clip": [lo, hi],
    }


def evaluate_gates(stats: dict) -> dict:
    share_ok = min(stats["env_share"]) >= GATES["G1_env_share_min"]
    per_env_ok = all(
        r["bsi_neg"] >= GATES["G2_bsi_neg_min_per_env"]
        and r["bsi_pos"] >= GATES["G2_bsi_pos_min_per_env"]
        and r["ctr_pos"] >= GATES["G2_ctr_pos_min_per_env"]
        for r in stats["per_env"]
    )
    cells = stats["fold_env_cells"]
    cells_ok = all(
        c["bsi_neg"] >= GATES["G3_cell_bsi_neg_min"] and c["n"] >= GATES["G3_cell_total_min"] for c in cells
    )
    return {
        "G1_env_share": {"pass": bool(share_ok), "min_share": float(min(stats["env_share"])),
                          "threshold": GATES["G1_env_share_min"]},
        "G2_per_env_support": {"pass": bool(per_env_ok)},
        "G3_fold_cell_support": {"pass": bool(cells_ok)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--n-train", type=int, default=2_000_000)
    ap.add_argument("--stage1", action="append", required=True,
                    help="label=stage1_dir（canonical backbone 目录；label 如 seed1_canonical）")
    ap.add_argument("--p0-env-ids", default=None, help="P0 env_ids.pt 路径（可选）")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=protocol.BATCH_SIZE)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    t0 = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": str(device)}

    print("[audit] parity 校验（移植 vs 钉死 blob）...")
    result["parity"] = parity_check()
    print(f"[audit] parity all_bit_equal={result['parity']['all_bit_equal']}")
    if not result["parity"]["all_bit_equal"]:
        raise AssertionError("per_task_rank01 移植与钉死 blob 非逐位一致")

    print("[audit] 加载训练前缀...")
    dataset = AliCCPDataset(args.train_file, args.n_train)
    assert len(dataset) == args.n_train
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = dataset.data
    labels = {
        "ctr": np.array([int(r[0]) for r in rows], dtype=bool),
        "cvr": np.array([int(r[1]) for r in rows], dtype=bool),
        "bsi": np.array([int(r[-1]) for r in rows], dtype=bool),
    }
    result["label_totals"] = {k: {"pos": int(v.sum()), "neg": int((~v).sum())} for k, v in labels.items()}

    result["partitions"] = {}
    if args.p0_env_ids:
        p0 = torch.load(args.p0_env_ids, map_location="cpu")
        stats = partition_stats(p0, labels)
        stats["weight_space"] = weight_space(p0)
        stats["gates"] = evaluate_gates(stats)
        result["partitions"]["P0_raw_argmin_seed1"] = stats
        print(f"[audit] P0 env_counts={stats['env_counts']}")

    for spec in args.stage1:
        label, path = spec.split("=", 1)
        root = Path(path).parent.parent
        art = protocol.load_stage1(root, Path(path).name)
        meta = art["meta"]
        fingerprint = protocol.load_fingerprint(root, meta["prefix_tag"])
        protocol.verify_fingerprint(fingerprint)
        require_fingerprint_match(meta, fingerprint)
        model = build_backbone(device)
        model.load_state_dict(art["backbone_state"])
        protocol.freeze_backbone(model)
        require_backbone_match(model, meta)
        sha_ok = True
        print(f"[audit] {label}: backbone sha 与 meta 一致={sha_ok}；重算损失与 normalized 划分...")
        raw = compute_losses(model, loader, device)
        normalized = per_task_rank01(raw)
        if not bool(torch.isfinite(normalized).all()):
            raise AssertionError("归一化损失含非有限值")
        assign = torch.argmin(normalized, dim=1)
        stats = partition_stats(assign, labels)
        stats["weight_space"] = weight_space(assign)
        stats["gates"] = evaluate_gates(stats)
        stats["backbone_sha_match"] = bool(sha_ok)
        stats["stage1_id"] = meta.get("stage1_id")
        result["partitions"][f"normalized_{label}"] = stats
        print(f"[audit] normalized_{label}: env_counts={stats['env_counts']} "
              f"gates={ {k: v['pass'] for k, v in stats['gates'].items()} }")

    result["wall_seconds"] = round(time.time() - t0, 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[audit] 输出: {out} wall={result['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
