"""seed4/5 判定与五 seed 汇总：跑后只读验证与机械分类（预注册 §3.2/§4/§5；只读，不训练、不改产物）。

三个子命令（`--model-seed` 参数化；机制与方法学复用 seed3 验证脚本 `verify_seed3_results` 的未改函数）：

1) `freeze-p4`（每 seed：R1 认证捕获后、R2 运行前；登记 P4）
   在 R1 `--reproduce` 捕获的聚类时刻损失向量上，用**两个独立实现**
   （`normalized_clustering.per_task_rank01` 与 `audit_cluster_losses.ref_rank01`，
   由单测钉逐位等价）计算 argmin 分配，必须逐位一致；导出该 seed 的 R2 预测事件三元组、
   `env_ids_sha256`、环境占比、`env_0∩purchase1`，写 JSON（含 model_seed）。

2) `verify`（每 seed：R1–R4 完成后；机械判定）
   读取该 seed 的四个产物，执行预注册 M/U 门禁、§5.2 三分法分类、§5.3 material contradiction、
   §5.7 headroom 与 §5.4 逐 seed 结论标签，输出 JSON + 摘要。全部阈值来自预注册文档字面值，
   本脚本只做机械执行。

3) `aggregate`（seed4/5 均完成后；终局汇总）
   五 canonical seed 的配对 Δtest/Δval 汇总统计（均值、样本标准差 ddof=1、正提升计数、
   符号正计数、最差 seed、配对 95% CI，t 冻结常数 §5.6）、`reclassification` 字段（不覆盖历史标签）
   与耐久性标签 `UTILITY_DURABLE` / `MECHANISM_REPAIR_ONLY_UNSTABLE_UTILITY`。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from . import protocol
from . import audit_prior_seeds as aps
from .verify_seed3_results import _jload, _run_tests, _smoke_identity

# ---- 预注册钉死值（§3.1/§5；不得随实现修改）----
SEEDS = {"seed4": 1688749593, "seed5": 1688762746}
CANONICAL_SEEDS = [1688723512, 1688723740, 1688738016, 1688749593, 1688762746]
P1 = {
    1688749593: {
        "raw_cfg_hash": "bb2b68de2dac257c20d7b50544f83e672a12eb2e04754ec476a66f90788a8cb3",
        "raw_stage1_id": "s1-5c060b9c-m1688749593-e3-bb2b68de",
        "norm_cfg_hash": "cf810ec493f52555dc7e69bcb971537c52b9f6b14dca6234759447ed1e9ac3fe",
        "norm_stage1_id": "s1-5c060b9c-m1688749593-e3-cf810ec4",
    },
    1688762746: {
        "raw_cfg_hash": "6343490f5e16d99e31213d5bb4a91459a173b782616eda2184d9c4c3d94c75d2",
        "raw_stage1_id": "s1-5c060b9c-m1688762746-e3-6343490f",
        "norm_cfg_hash": "a8238feaa483aada38f0280971f6c5b6fd5fbfdb01ab67eda68dac99a501f0ae",
        "norm_stage1_id": "s1-5c060b9c-m1688762746-e3-a8238fea",
    },
}
FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
# §5.2 分类阈值（用户预注册）
POSITIVE_MIN = 0.001
CLEAR_DEGRADATION_MAX = -0.02
# §4.2 容差
U1_TOLERANCE = -0.005
# §5.3 material contradiction 阈值
DELTA_VAL_CONTRADICTION = -0.01
# §5.6 汇总冻结常数（t 分布 df=4 的 0.975 分位；scipy 复算一致）
T975_DF4 = 2.7764451051977987
# seed3 提交记录字面值（f8ff4cf；用于 H3 与汇总；`--seed3-verification` 可交叉核对）
SEED3_DELTA_TEST = 0.02347892658289208
SEED3_DELTA_VAL = 0.017557536803150087
# §5.5 历史标签冻结表（不可变、不覆盖；原样引用各分支记录）
HISTORICAL = {
    1688723512: {
        "branch": "exp/aliccp-stage1-normalized-env-clustering", "tip": "2058de8",
        "labels": ["REPAIRED", "NO_MATERIAL_DEGRADATION"],
    },
    1688723740: {
        "branch": "exp/aliccp-stage1-normalized-clustering-seed-replication", "tip": "7ab445c",
        "labels": ["NOT_SUPPORTED"],
    },
    1688738016: {
        "branch": "exp/aliccp-stage1-normalized-clustering-seed3", "tip": "f8ff4cf",
        "labels": ["MECHANISM_REPAIRED_SEED3", "SEED3_CONDITION_PASSED", "POSITIVE_IMPROVEMENT"],
    },
}


# ---------------------------------------------------------------- 机械函数
def classify(delta_test: float) -> str:
    """§5.2 三分法（机械执行；阈值先于 run 冻结）。"""
    if delta_test >= POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_test <= CLEAR_DEGRADATION_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def aggregate_stats(deltas) -> dict:
    """§5.6 汇总统计：mean / 样本标准差（ddof=1）/ 配对 95% CI（t 冻结常数）/ 计数 / 最差 seed。"""
    values = [float(d) for d in deltas]
    n = len(values)
    if n < 2:
        raise ValueError("汇总统计至少需要 2 个 seed")
    mean = sum(values) / n
    var = sum((d - mean) ** 2 for d in values) / (n - 1)
    sd = math.sqrt(var)
    half = T975_DF4 * sd / math.sqrt(n)
    worst_index = min(range(n), key=lambda i: values[i])
    return {
        "n": n,
        "mean": mean,
        "sample_std": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "positive_count": sum(1 for d in values if d >= POSITIVE_MIN),
        "sign_positive_count": sum(1 for d in values if d > 0),
        "worst_index": worst_index,
        "worst_value": values[worst_index],
        "values": values,
    }


def prior_delta_literals() -> dict:
    """seed1/2（aps.ARMS 精确记录，M0 已复算）+ seed3（f8ff4cf 提交记录）的配对 Δ。"""
    return {
        1688723512: {
            "delta_test": aps.ARMS[("seed1", "norm")]["run"]["test_auc_bsi"]
            - aps.ARMS[("seed1", "raw")]["run"]["test_auc_bsi"],
            "delta_val": aps.ARMS[("seed1", "norm")]["run"]["best_val_auc_bsi"]
            - aps.ARMS[("seed1", "raw")]["run"]["best_val_auc_bsi"],
        },
        1688723740: {
            "delta_test": aps.ARMS[("seed2", "norm")]["run"]["test_auc_bsi"]
            - aps.ARMS[("seed2", "raw")]["run"]["test_auc_bsi"],
            "delta_val": aps.ARMS[("seed2", "norm")]["run"]["best_val_auc_bsi"]
            - aps.ARMS[("seed2", "raw")]["run"]["best_val_auc_bsi"],
        },
        1688738016: {"delta_test": SEED3_DELTA_TEST, "delta_val": SEED3_DELTA_VAL},
    }


def headroom_flags(*, mechanism_ok: bool, delta_test: float, delta_val: float, prior_deltas,
                   best_epoch_raw: int, best_epoch_norm: int, epochs: int) -> dict:
    """§5.7：H1 机制活性 / H2 验证方向 / H3 跨 seed（种子 1..k；count ≥ ⌈k/2⌉ ∧ mean > 0）/ H4 预算轨迹。"""
    priors = [float(d) for d in prior_deltas]
    k = len(priors) + 1
    need = math.ceil(k / 2)
    count_pos = sum(1 for d in priors if d > 0) + (1 if delta_test > 0 else 0)
    mean_all = (sum(priors) + delta_test) / k
    h3 = bool(delta_test > 0 and count_pos >= need and mean_all > 0)
    h2 = "favors" if delta_val >= 0 else ("contradicts" if delta_val <= DELTA_VAL_CONTRADICTION else "ambiguous")
    h4 = bool(best_epoch_raw == epochs and best_epoch_norm == epochs)
    verdict = "HEADROOM_PLAUSIBLE" if (mechanism_ok and h2 != "contradicts" and h3) else "HEADROOM_NOT_EVIDENT"
    return {
        "H1_mechanism": "favors" if mechanism_ok else "contradicts",
        "H2_validation_direction": h2,
        "H3_cross_seed": "favors" if h3 else "not",
        "H4_budget_trajectory": "favors_longer_budget" if h4 else "not",
        "H3_detail": {"k": k, "count_positive": count_pos, "need": need, "mean": mean_all},
        "verdict": verdict,
    }


# ---------------------------------------------------------------- freeze-p4
def freeze_p4(capture: Path, train_file: Path, env_seed: int, out_path: Path, model_seed: int, log=print) -> dict:
    from .audit_cluster_losses import ref_rank01
    from .normalized_clustering import per_task_rank01

    losses = torch.load(capture, map_location="cpu")
    q_impl = per_task_rank01(losses)
    q_ref = ref_rank01(losses)
    agree = bool(torch.equal(q_impl, q_ref))
    if not agree:
        raise AssertionError("两独立实现的 rank01 输出不逐位相等——P4 不得登记（按预注册 §4.3 处置）")
    assign = torch.argmin(q_impl, dim=1)
    counts = torch.bincount(assign, minlength=2)
    initial = protocol.make_env_ids(losses.shape[0], env_seed)
    ys = aps._scan_labels(train_file, losses.shape[0])
    cross = aps._crosstab(assign, ys)
    p4 = {
        "derived_from": str(capture),
        "derived_by": ["normalized_clustering.per_task_rank01", "audit_cluster_losses.ref_rank01"],
        "bitwise_agree": agree,
        "model_seed": int(model_seed),
        "n": int(losses.shape[0]),
        "event": {"epoch": 2, "diff_num": int((initial != assign).sum()),
                  "env_0": int(counts[0]), "env_1": int(counts[1])},
        "env_ids_sha256": protocol.sha256_tensor(assign),
        "env_0_share": float(counts[0]) / int(losses.shape[0]),
        "env_0_purchase1": cross["env_0_purchase1"],
        "crosstab": cross,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "commit": protocol.code_commit(),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(p4, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[p4] seed={model_seed} 预测登记：event={p4['event']} share={p4['env_0_share']:.5%} "
        f"env_ids_sha={p4['env_ids_sha256'][:16]}… env_0∩purchase1={p4['env_0_purchase1']} → {out_path}")
    return p4


# ---------------------------------------------------------------- verify
def _find_run(root: Path, stage1_id: str, tag: str, model_seed: int) -> Path:
    hits = []
    for run_dir in sorted((root / "runs").glob("*")):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            m = _jload(metrics_path)
        except Exception:  # noqa: BLE001
            continue
        if m.get("stage1_id") == stage1_id and m.get("tag") == tag and int(m.get("model_seed", -1)) == model_seed:
            hits.append(run_dir)
    if len(hits) != 1:
        raise FileNotFoundError(f"run 定位不唯一（{tag}/{stage1_id}/m{model_seed}）：{hits}")
    return hits[0]


def _epoch12(meta: dict):
    return meta["per_epoch"][:2]


def verify(root: Path, out_path: Path, *, model_seed: int, raw_sid=None, norm_sid=None,
           p4_path: Path | None = None, repro_audit: Path | None = None, train_file: Path,
           smoke_reference_meta: Path | None = None, seed3_verification: Path | None = None,
           prior_verifications=(), run_tests: bool = True, device, log=print) -> dict:
    t0 = time.time()
    p1 = P1[model_seed]
    raw_sid = raw_sid or p1["raw_stage1_id"]
    norm_sid = norm_sid or p1["norm_stage1_id"]
    seed_name = next(n for n, s in SEEDS.items() if s == model_seed)
    raw_dir = root / "stage1" / raw_sid
    norm_dir = root / "stage1" / norm_sid
    raw_meta = _jload(raw_dir / "meta.json")
    norm_meta = _jload(norm_dir / "meta.json")
    raw_run_dir = _find_run(root, raw_sid, "short", model_seed)
    norm_run_dir = _find_run(root, norm_sid, "norm", model_seed)
    raw_run = _jload(raw_run_dir / "metrics.json")
    norm_run = _jload(norm_run_dir / "metrics.json")
    raw_gates = _jload(raw_run_dir / "gate_report.json")
    norm_gates = _jload(norm_run_dir / "gate_report.json")

    raw_env = torch.load(raw_dir / "env_ids.pt", map_location="cpu")
    norm_env = torch.load(norm_dir / "env_ids.pt", map_location="cpu")
    n = len(raw_env)
    ys = aps._scan_labels(train_file, n)
    raw_cross = aps._crosstab(raw_env, ys)
    norm_cross = aps._crosstab(norm_env, ys)

    checks: dict = {}

    # ---- M1: B4 修复 ----
    events = norm_meta["cluster_events"]
    m1 = (len(events) == 1 and events[0]["env_0"] >= protocol.ENV_SHARE_MIN * n
          and events[0]["env_1"] >= protocol.ENV_SHARE_MIN * n)
    m1 = bool(m1 and norm_gates["gates"]["B4"]["verdict"] == "PASS")
    checks["M1_B4_repair"] = m1

    # ---- M2a: 身份 ----
    checks["M2a_raw_identity"] = (raw_meta["stage1_id"] == raw_sid
                                  and raw_meta["config_hash"] == p1["raw_cfg_hash"])
    checks["M2a_norm_identity"] = (norm_meta["stage1_id"] == norm_sid
                                   and norm_meta["config_hash"] == p1["norm_cfg_hash"])

    # ---- M2b: epoch 1-2 逐位相等 ----
    checks["M2b_epoch12_bit_equal"] = _epoch12(raw_meta) == _epoch12(norm_meta)

    # ---- M2c: P4 ----
    p4 = _jload(p4_path) if p4_path and Path(p4_path).is_file() else None
    if p4 is not None:
        if int(p4.get("model_seed", -1)) != model_seed:
            raise AssertionError(f"P4 的 model_seed 不符：{p4.get('model_seed')} != {model_seed}")
    if p4 is None:
        checks["M2c_event_matches_P4"] = None  # 未判定（§4.3）
        checks["M2c_env_ids_sha_matches_P4"] = None
    else:
        checks["M2c_event_matches_P4"] = events == [p4["event"]]
        checks["M2c_env_ids_sha_matches_P4"] = norm_meta["env_ids_sha256"] == p4["env_ids_sha256"]

    # ---- M3: 有限性 ----
    diags = norm_meta.get("cluster_diagnostics", [])
    finite = bool(diags) and all(
        torch.isfinite(torch.tensor(list(v.values()))).all()
        for diag in diags for v in diag.get("normalized_loss_quantiles", {}).values()
    ) and all(
        torch.isfinite(torch.tensor(list(v.values()))).all()
        for diag in diags for v in diag.get("raw_loss_quantiles", {}).values()
    )
    checks["M3_finite_diagnostics"] = bool(finite)

    # ---- M4: 非标签恢复 ----
    m4 = (norm_cross["env_0"] > 0
          and norm_cross["env_0_purchase1"] <= 0.05 * norm_cross["env_0"]
          and not norm_cross["env0_equals_purchase1_set"]
          and not norm_cross["env0_subset_of_purchase1"])
    checks["M4_non_label_recovery"] = bool(m4)

    # ---- M4b: 退化签名消除（探针 recall 双正） ----
    model = aps._build_model(norm_meta, device)
    model.load_state_dict(torch.load(norm_dir / "backbone.pt", map_location="cpu"))
    from multitaskrec.dataset import AliCCPDataset
    ds = AliCCPDataset(str(train_file), n)
    probe = aps._probe(model, ds, norm_env, norm_meta["config"]["batch_size"], device)
    raw_model = aps._build_model(raw_meta, device)
    raw_model.load_state_dict(torch.load(raw_dir / "backbone.pt", map_location="cpu"))
    del ds
    ds_raw = AliCCPDataset(str(train_file), n)
    raw_probe = aps._probe(raw_model, ds_raw, raw_env, raw_meta["config"]["batch_size"], device)
    del ds_raw
    checks["M4b_recalls_positive"] = bool(probe["recall_env_0"] > 0 and probe["recall_env_1"] > 0)

    # ---- M5: 有效变化 ----
    checks["M5_effective_change"] = bool(
        events and events[0]["diff_num"] >= 0.05 * n
        and int((raw_env != norm_env).sum()) >= 0.05 * n
    )

    # ---- M6: 环境逐位可复现（reproduce） ----
    repro = _jload(repro_audit) if repro_audit and Path(repro_audit).is_file() else None
    if repro is None:
        checks["M6_reproduce_10of10"] = None
    else:
        rep = repro.get("reproduction", {})
        checks["M6_reproduce_10of10"] = bool(rep.get("all_pass") and rep.get("checks")
                                             and all(rep["checks"].values()))

    # ---- M7: A 类完整性 ----
    def a_ok(gate_doc):
        g = gate_doc["gates"]
        return all(g[k]["verdict"] == "PASS" for k in ("A1", "A2", "A4", "A5", "A6"))
    checks["M7_A_class_both_runs"] = bool(a_ok(raw_gates) and a_ok(norm_gates))

    # ---- M8a: 测试套件 ----
    checks["M8a_tests"] = _run_tests(log)["ok"] if run_tests else None
    # ---- M8b: smoke 恒等 ----
    if smoke_reference_meta and Path(smoke_reference_meta).is_file():
        checks["M8b_smoke_identity"] = _smoke_identity(root, Path(smoke_reference_meta), device, log)["pass"]
    else:
        checks["M8b_smoke_identity"] = None

    # ---- U 组 ----
    delta_test_ctr = norm_meta["test_auc_ctr"] - raw_meta["test_auc_ctr"]
    delta_test = norm_run["test_auc_bsi"] - raw_run["test_auc_bsi"]
    delta_val = norm_run["best_val_auc_bsi"] - raw_run["best_val_auc_bsi"]
    checks["U1_ctr_no_regression"] = bool(delta_test_ctr >= U1_TOLERANCE)

    classification = classify(delta_test)

    m_gates = {k: v for k, v in checks.items() if k.startswith("M")}
    mechanism_repaired = all(v is True for v in m_gates.values())  # None（未判定）不算通过
    contradiction = {
        "(a)_delta_val_le_-0.01": bool(delta_val <= DELTA_VAL_CONTRADICTION),
        "(b)_mechanism_gate_fail": not mechanism_repaired,
        "(c)_U1_fail": not checks["U1_ctr_no_regression"],
        "(d)_A_class_fail": not checks["M7_A_class_both_runs"],
    }
    has_contradiction = any(contradiction.values())
    expand_condition = bool(delta_test >= POSITIVE_MIN and not has_contradiction)

    # ---- §5.7 headroom（种子 1..k 的可用集合；不引入未来 seed） ----
    priors = prior_delta_literals()
    if seed3_verification and Path(seed3_verification).is_file():
        v3 = _jload(seed3_verification)
        if v3["stage2"]["delta_test_bsi"] != SEED3_DELTA_TEST or v3["stage2"]["delta_val_bsi"] != SEED3_DELTA_VAL:
            raise AssertionError("seed3 verification JSON 与冻结字面值不符")
    expected_prior_seeds = CANONICAL_SEEDS[: CANONICAL_SEEDS.index(model_seed)]
    have = set(priors)
    for path in prior_verifications:
        doc = _jload(Path(path))
        ps = int(doc["identities"]["model_seed"])
        if ps == model_seed:
            raise AssertionError(f"--prior-verification 不得包含当前 seed 自身：{ps}")
        priors[ps] = {"delta_test": doc["stage2"]["delta_test_bsi"], "delta_val": doc["stage2"]["delta_val_bsi"]}
        have.add(ps)
    missing = [s for s in expected_prior_seeds if s not in have]
    if missing:
        raise AssertionError(f"H3 需要前置 seed 的 verification（缺 {missing}）")
    prior_deltas = [priors[s]["delta_test"] for s in expected_prior_seeds]
    headroom = headroom_flags(
        mechanism_ok=mechanism_repaired, delta_test=delta_test, delta_val=delta_val, prior_deltas=prior_deltas,
        best_epoch_raw=raw_run["best_epoch"], best_epoch_norm=norm_run["best_epoch"], epochs=raw_run["epochs"],
    )

    label = "SEED4" if seed_name == "seed4" else "SEED5"
    if expand_condition:
        nxt = ("seed5 same 1+1+1+1 protocol per prereg §5.4" if label == "SEED4"
               else "aggregate five seeds per prereg §5.6")
        verdict = f"{label}_CONDITION_PASSED__next: {nxt}"
    elif mechanism_repaired:
        verdict = "mechanism repair effective but utility unstable"
    else:
        verdict = f"MECHANISM_NOT_REPAIRED_ON_{label}"

    report = {
        "verify": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "commit": protocol.code_commit(), "root": str(root), "device": str(device)},
        "identities": {"seed": seed_name, "model_seed": model_seed,
                       "raw_stage1_id": raw_sid, "norm_stage1_id": norm_sid,
                       "fingerprint_sha": FINGERPRINT_SHA,
                       "raw_run_id": raw_run["run_id"], "norm_run_id": norm_run["run_id"]},
        "provenance": {
            "raw_stage1": {"commit": raw_meta.get("commit"), "dirty": raw_meta.get("git", {}).get("dirty")},
            "norm_stage1": {"commit": norm_meta.get("commit"), "dirty": norm_meta.get("git", {}).get("dirty")},
            "raw_run": {"commit": raw_run.get("commit"), "dirty": raw_run.get("git", {}).get("dirty")},
            "norm_run": {"commit": norm_run.get("commit"), "dirty": norm_run.get("git", {}).get("dirty")},
        },
        "mechanism": {
            "raw_event": raw_meta["cluster_events"], "norm_event": norm_meta["cluster_events"],
            "raw_shares": {"env_0": raw_cross["env_0"] / n, "env_1": raw_cross["env_1"] / n},
            "norm_shares": {"env_0": norm_cross["env_0"] / n, "env_1": norm_cross["env_1"] / n},
            "raw_crosstab": raw_cross, "norm_crosstab": norm_cross,
            "norm_probe": probe, "raw_probe": raw_probe,
            "norm_diagnostics": diags,
            "env_ids_sha256": {"raw": raw_meta["env_ids_sha256"], "norm": norm_meta["env_ids_sha256"],
                               "norm_equals_p4": (p4 or {}).get("env_ids_sha256")},
        },
        "stage2": {
            "raw": {"run_id": raw_run["run_id"], "best_val_auc_bsi": raw_run["best_val_auc_bsi"],
                    "test_auc_bsi": raw_run["test_auc_bsi"], "gate_mean": raw_run["gate_mean"],
                    "per_epoch": raw_run["per_epoch"], "best_epoch": raw_run["best_epoch"],
                    "hard_pass": raw_run["hard_pass"],
                    "gates": {k: v["verdict"] for k, v in raw_gates["gates"].items()}},
            "norm": {"run_id": norm_run["run_id"], "best_val_auc_bsi": norm_run["best_val_auc_bsi"],
                     "test_auc_bsi": norm_run["test_auc_bsi"], "gate_mean": norm_run["gate_mean"],
                     "per_epoch": norm_run["per_epoch"], "best_epoch": norm_run["best_epoch"],
                     "hard_pass": norm_run["hard_pass"],
                     "gates": {k: v["verdict"] for k, v in norm_gates["gates"].items()}},
            "delta_test_bsi": delta_test, "delta_val_bsi": delta_val, "delta_test_ctr": delta_test_ctr,
        },
        "checks": checks,
        "mechanism_repaired": bool(mechanism_repaired),
        "classification": classification,
        "material_contradiction": {"flags": contradiction, "present": bool(has_contradiction)},
        "expand_condition": expand_condition,
        "headroom": headroom,
        "cross_seed_delta_test": {str(s): priors[s]["delta_test"] for s in expected_prior_seeds} | {str(model_seed): delta_test},
        "verdict": verdict,
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("=" * 96)
    log(f"[verify] seed={seed_name} ({model_seed}) Δtest={delta_test:+.6f} Δval={delta_val:+.6f} Δtest_ctr={delta_test_ctr:+.6f}")
    log(f"[verify] 机制 M 组：" + ", ".join(f"{k}={'PASS' if v is True else v}" for k, v in m_gates.items()))
    log(f"[verify] 分类={classification}；material contradiction={has_contradiction}；"
        f"headroom={headroom['verdict']}")
    log(f"[verify] 结论：{verdict}")
    log(f"[verify] 输出：{out_path}（wall={report['wall_seconds']}s）")
    log("=" * 96)
    return report


# ---------------------------------------------------------------- aggregate
def aggregate(out_path: Path, *, seed4_verification: Path, seed5_verification: Path,
              seed3_verification: Path | None = None, log=print) -> dict:
    t0 = time.time()
    v4 = _jload(Path(seed4_verification))
    v5 = _jload(Path(seed5_verification))
    if int(v4["identities"]["model_seed"]) != SEEDS["seed4"]:
        raise AssertionError(f"seed4 verification 的 model_seed 不符：{v4['identities']['model_seed']}")
    if int(v5["identities"]["model_seed"]) != SEEDS["seed5"]:
        raise AssertionError(f"seed5 verification 的 model_seed 不符：{v5['identities']['model_seed']}")
    if seed3_verification and Path(seed3_verification).is_file():
        v3 = _jload(seed3_verification)
        if v3["stage2"]["delta_test_bsi"] != SEED3_DELTA_TEST or v3["stage2"]["delta_val_bsi"] != SEED3_DELTA_VAL:
            raise AssertionError("seed3 verification JSON 与冻结字面值不符")

    priors = prior_delta_literals()
    by_seed = {
        1688723512: {"delta_test": priors[1688723512]["delta_test"], "delta_val": priors[1688723512]["delta_val"]},
        1688723740: {"delta_test": priors[1688723740]["delta_test"], "delta_val": priors[1688723740]["delta_val"]},
        1688738016: {"delta_test": priors[1688738016]["delta_test"], "delta_val": priors[1688738016]["delta_val"]},
        1688749593: {"delta_test": v4["stage2"]["delta_test_bsi"], "delta_val": v4["stage2"]["delta_val_bsi"]},
        1688762746: {"delta_test": v5["stage2"]["delta_test_bsi"], "delta_val": v5["stage2"]["delta_val_bsi"]},
    }
    mechanism_flags = {
        1688723512: True,  # M0 复算（B4 修复 confirmed）
        1688723740: True,  # M0 复算（B4 修复 confirmed）
        1688738016: bool(_jload(Path(seed3_verification))["mechanism_repaired"]) if seed3_verification else True,
        1688749593: bool(v4["mechanism_repaired"]),
        1688762746: bool(v5["mechanism_repaired"]),
    }
    verification_docs = {1688749593: v4, 1688762746: v5}

    rows = []
    for i, seed in enumerate(CANONICAL_SEEDS):
        entry = by_seed[seed]
        if seed in HISTORICAL:
            historical = {
                "branch": HISTORICAL[seed]["branch"], "tip": HISTORICAL[seed]["tip"],
                "labels": list(HISTORICAL[seed]["labels"]),
            }
            source = f"recorded ({HISTORICAL[seed]['tip']})"
        else:
            doc = verification_docs[seed]
            historical = {"branch": "this (exp/aliccp-stage1-normalized-clustering-seeds45)",
                          "labels": [doc["classification"], doc["verdict"]]}
            source = doc["identities"]["raw_run_id"] + " / " + doc["identities"]["norm_run_id"]
        rows.append({
            "canonical_index": i + 1,
            "seed": seed,
            "delta_test": entry["delta_test"],
            "delta_val": entry["delta_val"],
            "reclassification": classify(entry["delta_test"]),
            "historical_labels": historical,
            "mechanism_repaired": mechanism_flags[seed],
            "source": source,
        })

    stats_test = aggregate_stats([r["delta_test"] for r in rows])
    stats_val = aggregate_stats([r["delta_val"] for r in rows])
    durable = bool(stats_test["ci95_low"] > 0 and stats_test["positive_count"] >= 3)
    verdict = "UTILITY_DURABLE" if durable else "MECHANISM_REPAIR_ONLY_UNSTABLE_UTILITY"
    headroom_by_seed = {str(s): verification_docs[s]["headroom"] for s in verification_docs}

    report = {
        "aggregate": {"timestamp": datetime.now().isoformat(timespec="seconds"),
                      "commit": protocol.code_commit(),
                      "seed4_verification": str(seed4_verification),
                      "seed5_verification": str(seed5_verification),
                      "seed3_verification": str(seed3_verification) if seed3_verification else None},
        "canonical_seeds": CANONICAL_SEEDS,
        "rows": rows,
        "stats": {"delta_test": stats_test, "delta_val": stats_val},
        "mechanism_consistency": {
            "repaired_count": sum(1 for s in CANONICAL_SEEDS if mechanism_flags[s]),
            "n": len(CANONICAL_SEEDS),
        },
        "headroom_by_seed": headroom_by_seed,
        "worst_seed": {"seed": rows[stats_test["worst_index"]]["seed"],
                       "delta_test": stats_test["worst_value"]},
        "durability_verdict": verdict,
        "wall_seconds": round(time.time() - t0, 1),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("=" * 96)
    for r in rows:
        log(f"[aggregate] {r['canonical_index']}. seed={r['seed']} Δtest={r['delta_test']:+.8f} "
            f"Δval={r['delta_val']:+.8f} → {r['reclassification']}（历史：{r['historical_labels']['labels']}）")
    log(f"[aggregate] Δtest 均值={stats_test['mean']:+.8f} 样本std={stats_test['sample_std']:.8f} "
        f"95%CI=[{stats_test['ci95_low']:+.8f}, {stats_test['ci95_high']:+.8f}] "
        f"正提升={stats_test['positive_count']}/5 符号正={stats_test['sign_positive_count']}/5 "
        f"最差={report['worst_seed']['seed']}({stats_test['worst_value']:+.8f})")
    log(f"[aggregate] 机制修复 {report['mechanism_consistency']['repaired_count']}/5；终局标签：{verdict}")
    log(f"[aggregate] 输出：{out_path}（wall={report['wall_seconds']}s）")
    log("=" * 96)
    return report


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description="seed4/5 判定与五 seed 汇总（只读；预注册 §3.2/§4/§5）")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze-p4", help="在 R1 捕获上登记该 seed 的 P4 预测")
    f.add_argument("--model-seed", type=int, required=True, choices=sorted(SEEDS.values()))
    f.add_argument("--capture", type=str, required=True)
    f.add_argument("--train-file", type=str, default=protocol.DATA_FILES["train"])
    f.add_argument("--env-seed", type=int, default=protocol.ENV_SEED)
    f.add_argument("--out", type=str, required=True)

    v = sub.add_parser("verify", help="执行 M/U 门禁与分类")
    v.add_argument("--model-seed", type=int, required=True, choices=sorted(SEEDS.values()))
    v.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    v.add_argument("--raw-stage1-id", type=str, default=None)
    v.add_argument("--norm-stage1-id", type=str, default=None)
    v.add_argument("--p4", type=str, default=None)
    v.add_argument("--repro-audit", type=str, default=None)
    v.add_argument("--train-file", type=str, default=protocol.DATA_FILES["train"])
    v.add_argument("--smoke-reference-meta", type=str, default=None)
    v.add_argument("--seed3-verification", type=str, default=None)
    v.add_argument("--prior-verification", action="append", default=[],
                   help="前置 canonical seed 的 verification JSON（seed5 需要 seed4 的）")
    v.add_argument("--skip-tests", action="store_true")
    v.add_argument("--out", type=str, required=True)
    v.add_argument("--cpu", action="store_true")
    v.add_argument("--gpu", type=int, default=0)

    a = sub.add_parser("aggregate", help="五 seed 终局汇总（§5.6）")
    a.add_argument("--seed4-verification", type=str, required=True)
    a.add_argument("--seed5-verification", type=str, required=True)
    a.add_argument("--seed3-verification", type=str, default=None)
    a.add_argument("--out", type=str, required=True)

    args = parser.parse_args(argv)
    if args.command == "freeze-p4":
        freeze_p4(Path(args.capture), Path(args.train_file), args.env_seed, Path(args.out), args.model_seed)
        return 0
    if args.command == "aggregate":
        aggregate(Path(args.out), seed4_verification=Path(args.seed4_verification),
                  seed5_verification=Path(args.seed5_verification),
                  seed3_verification=Path(args.seed3_verification) if args.seed3_verification else None)
        return 0
    device = torch.device("cpu") if args.cpu else torch.device(f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.init()
    report = verify(
        Path(args.root), Path(args.out), model_seed=args.model_seed,
        raw_sid=args.raw_stage1_id, norm_sid=args.norm_stage1_id,
        p4_path=Path(args.p4) if args.p4 else None,
        repro_audit=Path(args.repro_audit) if args.repro_audit else None,
        train_file=Path(args.train_file),
        smoke_reference_meta=Path(args.smoke_reference_meta) if args.smoke_reference_meta else None,
        seed3_verification=Path(args.seed3_verification) if args.seed3_verification else None,
        prior_verifications=[Path(p) for p in args.prior_verification],
        run_tests=not args.skip_tests, device=device,
    )
    return 0 if report["mechanism_repaired"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
