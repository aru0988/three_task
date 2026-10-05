"""seed1/seed2 归一化聚类证据的独立机制审计（只读；预注册见
docs/superpowers/specs/2026-10-05-aliccp-stage1-normalized-clustering-seed3-design.md）。

目的：在 seed3 实验之前，从**原始产物**（stage1 meta/backbone/env_ids + stage2 run 记录）机械
复核两个既有分支的结论：
  - `exp/aliccp-stage1-normalized-env-clustering`（seed1 = 1688723512，tip `2058de8`）
  - `exp/aliccp-stage1-normalized-clustering-seed-replication`（seed2 = 1688723740，tip `7ab445c`）
即：归一化聚类在两 seed 上均修复环境均衡/退化（B4 由 FAIL 转 PASS），而 Stage-2 效用 seed1
为正、seed2 近零且 val 反向。本脚本把"已记录值"变为"可从产物复算并逐项断言"的证据。

做法（全部只读、无训练、不写任何既有产物）：
  1. 按钉死的 stage1_id / run_id 在 `--artifacts-root`（可多个，按序查找）定位产物；
  2. 复算 env_ids / backbone 张量 sha256 == meta 记录；重建 MPTRec 载入 backbone 并复算
     `protocol.backbone_sha256`；
  3. 扫描训练集前缀标签（click/purchase），复算 env 分配 × 标签交叉表（精确，逐位对齐）；
  4. 重建模型 + AliCCPDataset，复算 env_pred 探针（整数计数口径；acc / balanced_acc / 各环境
     recall）——与 `bench.env_accuracy_and_balance_probe` 同语义；
  5. 读取 stage2 run 的 metrics/gate_report，逐项断言；
  6. 可选 `--seed1-capture`：载入 seed1 认证捕获的聚类时刻损失向量，复算 rank01 候选分配，
     断言 == seed1 归一化臂的事件三元组与 env_ids sha256（bit 级跨 worktree 复核）。

任何 pin 不符都会出现在 JSON 的 `checks` 与 `all_pins_pass=false` 中；本脚本不修改、不解释，
仅如实报告。判定与结论见审计文档。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec

from . import bench, protocol

# ---- 钉死常量（= 两既有分支的已提交/已记录值；来自 SUMMARY 行、预注册文档与产物）----
SEED1, SEED2, SEED3 = 1688723512, 1688723740, 1688738016
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"

ARMS = {
    ("seed1", "raw"): {
        "stage1_id": "s1-5c060b9c-m1688723512-e3-3a30e2c0",
        "cfg_hash": "3a30e2c0b1e8a2b4e9fecaa4d76893b775f6ee9e597922dce7303ac3b77b08c4",
        "model_seed": SEED1, "commit": "b2e17f9",
        "event": {"epoch": 2, "diff_num": 1000466, "env_0": 566, "env_1": 1999434},
        "test_ctr": 0.5481837665250074, "test_cvr": 0.5280080347106456,
        "env_acc": 0.99977, "env_bal_acc": None, "clustering": None,
        "env_ids_sha256": "b839e51df9cb46a6b46fa5ea5e757e006c8288e00a2716fa0d4a89967a4f2a15",
        "backbone_sha256": "5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee",
        "crosstab": {"env_0": 566, "env_1": 1999434, "env_0_purchase1": 566,
                     "env0_equals_purchase1_set": True, "env0_subset_of_purchase1": True},
        "probe": {"acc": 0.9997699856758118, "balanced_acc": 0.5, "recall_env_0": 0.0, "recall_env_1": 1.0},
        "run": {
            "run_id": "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9", "commit": "b2e17f9",
            "best_val_auc_bsi": 0.5781533414372665, "test_auc_bsi": 0.5988392178311113,
            "gate_mean": [0.851149, 0.1488510625], "hard_pass": False,
            "per_epoch_val": [0.49373140290270867, 0.5037099476993491, 0.5177881501359481,
                              0.5433982610490566, 0.5781533414372665],
            "gates": {"A1": "PASS", "A2": "PASS", "A3": "SKIP", "A4": "PASS", "A5": "PASS",
                      "A6": "PASS", "B1": "FAIL", "B2": "PASS", "B3": "PASS", "B4": "FAIL"},
        },
    },
    ("seed1", "norm"): {
        "stage1_id": "s1-5c060b9c-m1688723512-e3-ad3b353f",
        "cfg_hash": "ad3b353f6d436ac95c26703e504a318e3441933a4e00e3f3d6c5e22a05bfe96e",
        "model_seed": SEED1, "commit": "66a8fa9",
        "event": {"epoch": 2, "diff_num": 999966, "env_0": 959244, "env_1": 1040756},
        "test_ctr": 0.546661154445625, "test_cvr": 0.5238168287730445,
        "env_acc": 0.18375, "env_bal_acc": 0.1844824230498549, "clustering": "rank_normalized",
        "env_ids_sha256": "4b983fc9f485d7cba853b8d5f0846b292e85344fc5b6f3725845dbba5ea0d52b",
        "backbone_sha256": "d86ce16971804036b67923cf651349d191f011d64fe34369be5aa8077fdf5b91",
        "crosstab": {"env_0": 959244, "env_1": 1040756, "env_0_purchase1": 564,
                     "env0_equals_purchase1_set": False, "env0_subset_of_purchase1": False},
        "probe": {"acc": 0.18375, "balanced_acc": 0.1844824230498549,
                  "recall_env_0": 0.20399289321114644, "recall_env_1": 0.16497195288856334},
        "run": {
            "run_id": "20261003-0838-p2M-v500k-t1M-m1688723512-norm-529090e", "commit": "529090e",
            "best_val_auc_bsi": 0.5882126133134417, "test_auc_bsi": 0.6070820576951792,
            "gate_mean": [0.5515200625, 0.44847984375], "hard_pass": False,
            "per_epoch_val": [0.4990801851401279, 0.5085663599101901, 0.5288994639272553,
                              0.5583654899846621, 0.5882126133134417],
            "gates": {"A1": "PASS", "A2": "PASS", "A3": "SKIP", "A4": "PASS", "A5": "PASS",
                      "A6": "PASS", "B1": "FAIL", "B2": "PASS", "B3": "PASS", "B4": "PASS"},
        },
    },
    ("seed2", "raw"): {
        "stage1_id": "s1-5c060b9c-m1688723740-e3-4e1b5c6f",
        "cfg_hash": "4e1b5c6ffe9b6ff49cda34691de27390669396337e64e867f0f83b6ec4da7981",
        "model_seed": SEED2, "commit": "79b5e07",
        "event": {"epoch": 2, "diff_num": 1000418, "env_0": 1532, "env_1": 1998468},
        "test_ctr": 0.5534062000766764, "test_cvr": 0.5420522697385088,
        "env_acc": 0.9992525, "env_bal_acc": None, "clustering": None,
        "env_ids_sha256": "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0",
        "backbone_sha256": "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c",
        "crosstab": {"env_0": 1532, "env_1": 1998468, "env_0_purchase1": 564,
                     "env0_equals_purchase1_set": False, "env0_subset_of_purchase1": False},
        "probe": {"acc": 0.9992525, "balanced_acc": 0.5, "recall_env_0": 0.0, "recall_env_1": 1.0},
        "run": {
            "run_id": "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07", "commit": "79b5e07",
            "best_val_auc_bsi": 0.5809347091990792, "test_auc_bsi": 0.5974422649550507,
            "gate_mean": [0.789178, 0.2108221875], "hard_pass": False,
            "per_epoch_val": [0.4645711559431739, 0.48564481224085004, 0.5142344439088465,
                              0.5508809596059387, 0.5809347091990792],
            "gates": {"A1": "PASS", "A2": "PASS", "A3": "SKIP", "A4": "PASS", "A5": "PASS",
                      "A6": "PASS", "B1": "PASS", "B2": "PASS", "B3": "PASS", "B4": "FAIL"},
        },
    },
    ("seed2", "norm"): {
        "stage1_id": "s1-5c060b9c-m1688723740-e3-820697f6",
        "cfg_hash": "820697f6b29dec50af3335a6b8fe35d5418e822401699441cec1388d50cc266b",
        "model_seed": SEED2, "commit": "a39c170",
        "event": {"epoch": 2, "diff_num": 998878, "env_0": 1106286, "env_1": 893714},
        "test_ctr": 0.5536026554267786, "test_cvr": 0.5486753127564796,
        "env_acc": 0.668745, "env_bal_acc": 0.6474166148852035, "clustering": "rank_normalized",
        "env_ids_sha256": "285365696d98a22c50c064439921d76c455815b1d25cb4e0431a86040e728a9e",
        "backbone_sha256": "037e084b68b8cb206eeb218cbb51ac70d79f87b540e922cd3131dba69187a1e8",
        "crosstab": {"env_0": 1106286, "env_1": 893714, "env_0_purchase1": 565,
                     "env0_equals_purchase1_set": False, "env0_subset_of_purchase1": False},
        "probe": {"acc": 0.668745, "balanced_acc": 0.6474166148852035,
                  "recall_env_0": None, "recall_env_1": None},  # 记录值（-reanalysis 无盘上文件）
        "run": {
            "run_id": "20261003-0932-p2M-v500k-t1M-m1688723740-norm-7b2c26a", "commit": "7b2c26a",
            "best_val_auc_bsi": 0.5794515447344156, "test_auc_bsi": 0.5979010844832302,
            "gate_mean": [0.80868025, 0.19131965625], "hard_pass": True,
            "per_epoch_val": [0.46673476960220306, 0.48695931686719585, 0.5141675968824724,
                              0.5492497027405067, 0.5794515447344156],
            "gates": {"A1": "PASS", "A2": "PASS", "A3": "SKIP", "A4": "PASS", "A5": "PASS",
                      "A6": "PASS", "B1": "PASS", "B2": "PASS", "B3": "PASS", "B4": "PASS"},
        },
    },
}

# 记录值里 acc 为 float32 均值口径（env_acc 由 bench.env_accuracy_probe 旧口径记录）；探针复算为
# 整数计数口径（ebefd19 修正）。允许 1e-6 容差的字段（其余一律精确相等）。
ACC_TOL = 1e-6


def _jload(path: Path):
    return json.load(io.open(path, encoding="utf-8"))


def _find_dir(roots, kind: str, name: str) -> Path:
    for root in roots:
        candidate = Path(root) / kind / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"未在任何 --artifacts-root 下找到 {kind}/{name}")


def _scan_labels(train_file: Path, n: int) -> torch.Tensor:
    """训练集前 n 行的 (click, purchase) 标签，float32，行序 = 文件序（与 env_ids 对齐）。"""
    clicks, purchases = [], []
    with io.open(train_file, encoding="utf-8") as handle:
        handle.readline()
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"训练文件行数不足 {n}（第 {i + 1} 行截断）")
            parts = line.split(",", 2)
            clicks.append(int(parts[0])); purchases.append(int(parts[1]))
    return torch.tensor([clicks, purchases], dtype=torch.float32).t()


def _crosstab(env_ids: torch.Tensor, ys: torch.Tensor) -> dict:
    click, purchase = ys[:, 0] > 0.5, ys[:, 1] > 0.5
    m0 = env_ids == 0
    return {
        "env_0": int(m0.sum()), "env_1": int((env_ids == 1).sum()),
        "env_0_share": float(m0.double().mean()),
        "env_0_purchase1": int((m0 & purchase).sum()),
        "env_0_click1": int((m0 & click).sum()),
        "purchase1_total": int(purchase.sum()),
        "env0_equals_purchase1_set": bool(torch.equal(m0, purchase)),
        "env0_subset_of_purchase1": bool((m0 & ~purchase).sum() == 0),
    }


def _build_model(meta: dict, device) -> MPTRec:
    cfg = meta["config"]
    return MPTRec(
        num_tasks=cfg["num_tasks"],
        feature_vocabulary={str(k): int(v) for k, v in cfg["vocab"].items()},
        embedding_size=cfg["embedding_size"], input_size=cfg["input_size"],
        expert_dnn_hidden_units=list(cfg["expert_hidden"]),
        tower_dnn_hidden_units=list(cfg["tower_hidden"]),
        dropout=list(cfg["dropout"]), reg_embedding=cfg["reg_embedding"], reg_dnn=cfg["reg_dnn"],
        device=device,
    ).to(device)


@torch.no_grad()
def _probe(model, dataset, env_ids, batch_size, device, max_batches=200) -> dict:
    """env_pred 探针（整数计数口径，语义同 bench.env_accuracy_and_balance_probe）。"""
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, ids_list = [], []
    for step, (_, _, _, features) in enumerate(loader):
        if step >= max_batches:
            break
        for key in features:
            features[key] = features[key].to(device)
        env_pred = model(features)["env_pred"]
        ids = env_ids[batch_size * step: batch_size * (step + 1)].to(device)
        n = len(ids)
        preds.append(env_pred.argmax(dim=1)[:n].cpu()); ids_list.append(ids.cpu())
    pred = torch.cat(preds); ids = torch.cat(ids_list)
    acc = int((pred == ids).sum()) / max(1, int(ids.numel()))
    recalls = {e: int((pred[ids == e] == e).sum()) / max(1, int((ids == e).sum())) for e in (0, 1)}
    return {"n": int(ids.numel()), "acc": acc, "balanced_acc": (recalls[0] + recalls[1]) / 2.0,
            "recall_env_0": recalls[0], "recall_env_1": recalls[1]}


def _close(a, b, tol=0.0) -> bool:
    if a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= tol


def audit_arm(roots, train_file: Path, key, spec: dict, device, log=print) -> dict:
    seed_tag, arm_tag = key
    log(f"[audit] === {seed_tag}/{arm_tag}  stage1_id={spec['stage1_id']} ===")
    stage1_dir = _find_dir(roots, "stage1", spec["stage1_id"])
    run_dir = _find_dir(roots, "runs", spec["run"]["run_id"])
    meta = _jload(stage1_dir / "meta.json")
    metrics = _jload(run_dir / "metrics.json")
    gate_report = _jload(run_dir / "gate_report.json")

    env_ids = torch.load(stage1_dir / "env_ids.pt", map_location="cpu")
    backbone_state = torch.load(stage1_dir / "backbone.pt", map_location="cpu")
    model = _build_model(meta, device)
    model.load_state_dict(backbone_state)

    checks: dict = {}
    checks["stage1_id"] = meta["stage1_id"] == spec["stage1_id"]
    checks["cfg_hash"] = meta["config_hash"] == spec["cfg_hash"]
    checks["model_seed"] = int(meta["model_seed"]) == spec["model_seed"]
    checks["fingerprint_sha256"] = meta["fingerprint_sha256"] == FINGERPRINT_SHA
    checks["commit"] = meta.get("commit") == spec["commit"]
    checks["dirty_false"] = meta.get("git", {}).get("dirty") is False
    checks["cluster_events"] = meta["cluster_events"] == [spec["event"]]
    checks["env_ids_sha256_recomputed"] = protocol.sha256_tensor(env_ids) == spec["env_ids_sha256"]
    checks["env_ids_sha256_matches_meta"] = meta["env_ids_sha256"] == spec["env_ids_sha256"]
    checks["backbone_sha256_recomputed"] = protocol.backbone_sha256(model) == spec["backbone_sha256"]
    checks["backbone_sha256_matches_meta"] = meta["backbone_sha256"] == spec["backbone_sha256"]
    checks["test_ctr"] = meta["test_auc_ctr"] == spec["test_ctr"]
    checks["test_cvr"] = meta["test_auc_cvr"] == spec["test_cvr"]
    checks["env_acc_recorded"] = _close(meta.get("env_acc"), spec["env_acc"], ACC_TOL)
    checks["clustering_marker"] = meta.get("clustering") == spec["clustering"]
    if spec["env_bal_acc"] is not None:
        checks["env_bal_acc_recorded"] = _close(meta.get("env_bal_acc"), spec["env_bal_acc"], 1e-12)

    # 复算交叉表（精确，env_ids × 原始标签，行序对齐）
    ys = _scan_labels(train_file, len(env_ids))
    cross = _crosstab(env_ids, ys)
    checks["crosstab_env_counts"] = (
        cross["env_0"] == spec["crosstab"]["env_0"] and cross["env_1"] == spec["crosstab"]["env_1"]
    )
    checks["crosstab_env0_purchase1"] = cross["env_0_purchase1"] == spec["crosstab"]["env_0_purchase1"]
    checks["crosstab_env0_equals_set"] = cross["env0_equals_purchase1_set"] == spec["crosstab"]["env0_equals_purchase1_set"]
    checks["crosstab_env0_subset"] = cross["env0_subset_of_purchase1"] == spec["crosstab"]["env0_subset_of_purchase1"]

    # 复算 env_pred 探针（best 权重，前 200 batch）
    dataset = AliCCPDataset(str(train_file), len(env_ids))
    probe = _probe(model, dataset, env_ids, meta["config"]["batch_size"], device)
    del dataset
    checks["probe_acc"] = _close(probe["acc"], spec["probe"]["acc"], ACC_TOL)
    checks["probe_balanced_acc"] = _close(probe["balanced_acc"], spec["probe"]["balanced_acc"], ACC_TOL)
    if spec["probe"]["recall_env_0"] is not None:
        checks["probe_recall_env_0"] = _close(probe["recall_env_0"], spec["probe"]["recall_env_0"], ACC_TOL)
        checks["probe_recall_env_1"] = _close(probe["recall_env_1"], spec["probe"]["recall_env_1"], ACC_TOL)

    # stage2 run 记录
    checks["run_stage1_id"] = metrics["stage1_id"] == spec["stage1_id"]
    checks["run_commit"] = metrics.get("commit") == spec["run"]["commit"]
    checks["run_dirty_false"] = metrics.get("git", {}).get("dirty") is False
    checks["run_best_val"] = metrics["best_val_auc_bsi"] == spec["run"]["best_val_auc_bsi"]
    checks["run_test"] = metrics["test_auc_bsi"] == spec["run"]["test_auc_bsi"]
    checks["run_gate_mean"] = [round(float(v), 10) for v in metrics["gate_mean"]] == [
        round(float(v), 10) for v in spec["run"]["gate_mean"]
    ]
    checks["run_hard_pass"] = bool(metrics["hard_pass"]) == spec["run"]["hard_pass"]
    checks["run_per_epoch_val"] = all(
        abs(rec["val_auc_bsi"] - pin) < 5e-7
        for rec, pin in zip(metrics["per_epoch"], spec["run"]["per_epoch_val"])
    ) and len(metrics["per_epoch"]) == len(spec["run"]["per_epoch_val"])
    checks["run_gates"] = {k: v["verdict"] for k, v in gate_report["gates"].items()} == spec["run"]["gates"]

    failed = sorted(k for k, v in checks.items() if v is not True)
    log(f"[audit]   pins {'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")
    log(f"[audit]   env 占比 {cross['env_0_share']:.5%}/{1 - cross['env_0_share']:.5%}；"
        f"probe acc={probe['acc']:.6f} bal={probe['balanced_acc']:.4f} "
        f"(r0={probe['recall_env_0']:.4f}, r1={probe['recall_env_1']:.4f})")
    return {"checks": checks, "all_pass": not failed,
            "rederived": {"crosstab": cross, "probe": probe},
            "recorded": {"meta_env_acc": meta.get("env_acc"), "meta_env_bal_acc": meta.get("env_bal_acc")}}


def seed1_capture_check(capture_path: Path, log=print) -> dict:
    """seed1 认证捕获（聚类时刻损失向量）→ rank01 分配 == seed1 归一化臂事件/env_ids sha（bit 级）。"""
    from .audit_cluster_losses import ref_rank01
    losses = torch.load(capture_path, map_location="cpu")
    assign = torch.argmin(ref_rank01(losses), dim=1)
    counts = torch.bincount(assign, minlength=2)
    sha = protocol.sha256_tensor(assign)
    norm = ARMS[("seed1", "norm")]
    checks = {
        "capture_env_0": int(counts[0]) == norm["event"]["env_0"],
        "capture_env_1": int(counts[1]) == norm["event"]["env_1"],
        "capture_env_ids_sha256": sha == norm["env_ids_sha256"],
    }
    log(f"[audit] seed1 捕获复算：env_0={int(counts[0])} env_1={int(counts[1])} sha={sha[:16]}… "
        f"→ {'ALL PASS' if all(checks.values()) else 'FAILED: ' + ', '.join(k for k, v in checks.items() if not v)}")
    return checks


def pattern_report(results: dict, log=print) -> dict:
    """从复算值机械导出跨 seed 模式（判定问题：修复两 seed 稳定 + 效用 seed1 正 / seed2 近零负 val）。"""
    def ev(seed, arm):
        return ARMS[(seed, arm)]["event"]
    def run(seed, arm):
        return ARMS[(seed, arm)]["run"]

    pattern = {}
    for seed in ("seed1", "seed2"):
        n = ev(seed, "norm")["env_0"] + ev(seed, "norm")["env_1"]
        raw_min = min(ev(seed, "raw")["env_0"], ev(seed, "raw")["env_1"]) / n
        norm_min = min(ev(seed, "norm")["env_0"], ev(seed, "norm")["env_1"]) / n
        pattern[f"{seed}_B4"] = {
            "raw_min_share": raw_min, "norm_min_share": norm_min,
            "raw_B4_FAIL": raw_min < protocol.ENV_SHARE_MIN,
            "norm_B4_PASS": norm_min >= protocol.ENV_SHARE_MIN,
        }
        d_test = run(seed, "norm")["test_auc_bsi"] - run(seed, "raw")["test_auc_bsi"]
        d_val = run(seed, "norm")["best_val_auc_bsi"] - run(seed, "raw")["best_val_auc_bsi"]
        pattern[f"{seed}_utility"] = {"delta_test": d_test, "delta_val": d_val}
    for seed in ("seed1", "seed2"):
        p = pattern[f"{seed}_B4"]
        p["repair_confirmed"] = bool(p["raw_B4_FAIL"] and p["norm_B4_PASS"])
    p1, p2 = pattern["seed1_utility"], pattern["seed2_utility"]
    pattern["utility_shape"] = {
        "seed1_positive": bool(p1["delta_test"] > 0 and p1["delta_val"] > 0),
        "seed2_near_zero_test_below_0p001": bool(0 < p2["delta_test"] < 0.001),
        "seed2_negative_val": bool(p2["delta_val"] < 0),
    }
    log("[audit] 模式：seed1 修复 + Δtest/Δval 均为正；seed2 修复 + Δtest 近零、Δval 为负（如实报告）")
    return pattern


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="seed1/seed2 归一化聚类证据独立审计（只读）")
    parser.add_argument("--artifacts-root", action="append", required=True,
                        help="包含 stage1/ 与 runs/ 的产物根（可多次；按序查找）")
    parser.add_argument("--train-file", type=str, default=protocol.DATA_FILES["train"])
    parser.add_argument("--out", type=str, required=True, help="审计 JSON 输出路径（gitignore 目录）")
    parser.add_argument("--seed1-capture", type=str, default=None,
                        help="可选：seed1 认证捕获 cluster_losses.pt 路径（bit 级复核）")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    roots = [Path(r) for r in args.artifacts_root]
    train_file = Path(args.train_file)
    t0 = time.time()
    report = {
        "audit": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                  "commit": protocol.code_commit(), "device": str(device),
                  "roots": [str(r) for r in roots], "train_file": str(train_file)},
        "arms": {},
    }
    for key, spec in ARMS.items():
        report["arms"][f"{key[0]}_{key[1]}"] = audit_arm(roots, train_file, key, spec, device)
    if args.seed1_capture:
        report["seed1_capture_check"] = seed1_capture_check(Path(args.seed1_capture))
    report["pattern"] = pattern_report(report["arms"])
    report["all_pins_pass"] = all(v["all_pass"] for v in report["arms"].values()) and all(
        v for k, v in report.items() if k == "seed1_capture_check"
    ) if "seed1_capture_check" in report else all(v["all_pass"] for v in report["arms"].values())
    report["audit"]["wall_seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[audit] all_pins_pass={report['all_pins_pass']}；输出：{args.out}（wall={report['audit']['wall_seconds']}s）")
    return 0 if report["all_pins_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
