"""三方对照（预注册 §6.2）：同一 Stage-1 产物 + 基线/nullx/处理三头，同一 test/val 全序重评测。

纯分析：无训练、不改任何 run 产物；完整性检查先于解读（任何一项失败 → 诊断解读作废）。
null 头类从 git blob 动态载入（sha256 钉死），前臂代码不移植进本分支。
见 docs/superpowers/specs/2026-10-03-census-stage2-attenuation-transfer-design.md。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from census_benchmark import metrics
from census_benchmark import protocol as P
from multitaskrec.model import NewTask

REPO = Path(__file__).resolve().parents[1]
NULL_HEAD_REV = "1eadd05"
NULL_HEAD_PATH = "multitaskrec/model.py"
NULL_HEAD_SHA256 = "836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c"
DEFAULT_BASELINE_RUN = metrics.TRANSFER_BASELINE_RUN_ID
DEFAULT_NULL_RUN = metrics.TRANSFER_NULL_RUN_ID
AUC_TOLERANCE = 1e-9
PROBE_TOLERANCE = 1e-12
PAIRS = (("control", "baseline"), ("control", "null"), ("null", "baseline"))


def load_null_newtask_class(repo_root: Path = REPO, *, rev: str = NULL_HEAD_REV):
    """从 git blob 动态载入 null 臂的 NewTask 类；内容 sha256 与预注册钉死值不符即拒绝。"""
    blob = subprocess.check_output(["git", "show", f"{rev}:{NULL_HEAD_PATH}"], cwd=repo_root)
    sha = hashlib.sha256(blob).hexdigest()
    if sha != NULL_HEAD_SHA256:
        raise RuntimeError(f"null 头 blob sha256 不符（拒绝载入）: {sha} != {NULL_HEAD_SHA256}")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "census_null_head_model.py"
        path.write_bytes(blob)
        spec = importlib.util.spec_from_file_location("census_null_head_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.NewTask


def _infer_head_sizes(state: dict) -> tuple[int, int]:
    """从头权重形状反推 (input_size, rep_dim)：projection_network.0 = Linear(input_size, rep_dim // 2)。"""
    weight = state["projection_network.0.weight"]
    return int(weight.shape[1]), int(weight.shape[0]) * 2


def _infer_tower_hidden(state: dict) -> list[int]:
    """从 tower_network.mlp.linear{i}.weight 形状反推 hidden 列表（末层输出 1 不属 hidden）。"""
    hidden = []
    index = 0
    while f"tower_network.mlp.linear{index}.weight" in state:
        hidden.append(int(state[f"tower_network.mlp.linear{index}.weight"].shape[0]))
        index += 1
    return hidden[:-1]


@torch.no_grad()
def predict_arm(head, backbone, loader, device):
    """按 loader 全序逐样本预测（与 metrics.evaluate_newtask 同序同算子）。"""
    head.eval(); backbone.eval()
    preds, ys = [], []
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
        preds.append(head(dnn_input, gen_rep, spec_reps, env_embs).detach().float().cpu())
        ys.append(y.cpu())
    return torch.cat(preds), torch.cat(ys)


def _load_run(root: Path, run_id: str) -> dict:
    run_dir = Path(root) / "runs" / run_id
    return {"run_id": run_id, "dir": run_dir,
            "config": json.loads((run_dir / "config.json").read_text(encoding="utf-8")),
            "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))}


def _build_head(arm: str, state: dict, device):
    input_size, rep_dim = _infer_head_sizes(state)
    common = dict(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=_infer_tower_hidden(state),
                  reg_dnn=P.REG_DNN, device=device)
    if arm == "null":
        head = load_null_newtask_class()(use_null_expert=True, **common)
    elif arm == "control":
        head = NewTask(spec_attenuation=metrics.SPEC_ATTENUATION_COEF, **common)
    else:
        head = NewTask(**common)
    head.load_state_dict(state, strict=True)
    return head.to(device).eval()


def _max_abs_diff(a, b) -> float:
    """嵌套结构（dict/list/数值）的最大绝对差；结构不一致或非数值不等 → inf。"""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max((_max_abs_diff(a[key], b[key]) for key in a), default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def run_compare(root=P.ARTIFACT_ROOT, *, stage1_dir, baseline_run=DEFAULT_BASELINE_RUN,
                null_run=DEFAULT_NULL_RUN, control_run, device=None, loaders=None, stats=None,
                indices=None, model=None) -> dict:
    """三方对照主流程；产物 = <control-run>/three_way_compare.json（+ .log）。"""
    root = Path(root)
    device = device or torch.device("cuda:0")
    sid = Path(stage1_dir).name
    checkpoint = P.load_stage1(root, sid)                       # spec 4.3：唯一 backbone 来源
    meta = checkpoint["meta"]
    if loaders is None:
        from run_census_benchmark import build_census_loaders
        loaders, stats, indices = build_census_loaders(meta["split_seed"])
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=meta["split_seed"], stats=stats, val_idx=val_idx, test_idx=test_idx)

    if model is None:
        from run_census_benchmark import build_mptrec
        model = build_mptrec(device)
    backbone = model.to(device)
    backbone.load_state_dict(checkpoint["backbone_state"])
    P.freeze_backbone(backbone)

    runs = {"baseline": _load_run(root, baseline_run), "null": _load_run(root, null_run),
            "control": _load_run(root, control_run)}
    heads = {arm: _build_head(arm, torch.load(run["dir"] / "newtask.pt", map_location="cpu"), device)
             for arm, run in runs.items()}

    # ---- 完整性检查（先于解读）----
    checks: dict[str, dict] = {}
    for arm, run in runs.items():
        checks[f"{arm}_stage1_id_matches"] = {
            "pass": bool(run["config"].get("stage1_id") == sid and run["metrics"].get("stage1_id") == sid),
            "run": run["run_id"], "sid": sid}
    checks["split_fingerprint_matches_stage1"] = {
        "pass": bool(fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"]),
        "recomputed": fp["fingerprint_sha256"], "stage1": meta["split_fingerprint_sha256"]}
    for arm, run in runs.items():
        recorded = run["metrics"].get("split_sha256", {}).get("fingerprint")
        checks[f"{arm}_split_fingerprint_matches"] = {
            "pass": bool(recorded == fp["fingerprint_sha256"]), "run": recorded, "recomputed": fp["fingerprint_sha256"]}
    backbone_sha = P.backbone_sha256(backbone)
    checks["backbone_sha256_matches_stage1"] = {
        "pass": bool(backbone_sha == meta["backbone_sha256"]), "recomputed": backbone_sha, "stage1": meta["backbone_sha256"]}
    checks["null_head_blob_pinned"] = {"pass": True, "rev": NULL_HEAD_REV, "sha256": NULL_HEAD_SHA256}

    test_preds, test_ys = {}, {}
    for arm, run in runs.items():
        preds, ys = predict_arm(heads[arm], backbone, loaders["test"], device)
        test_preds[arm], test_ys[arm] = preds, ys
        recomputed = metrics.auc(ys, preds)
        recorded = run["metrics"]["stage2"]["test_auc"]
        checks[f"{arm}_test_auc_recomputed"] = {
            "pass": bool(abs(recomputed - recorded) <= AUC_TOLERANCE),
            "recomputed": recomputed, "recorded": recorded, "abs_diff": abs(recomputed - recorded)}
    y_shas = {arm: P.sha256_tensor(ys) for arm, ys in test_ys.items()}
    checks["y_true_identical_across_arms"] = {"pass": len(set(y_shas.values())) == 1, "sha256": y_shas}

    param_counts = {arm: sum(p.numel() for p in head.parameters()) for arm, head in heads.items()}
    checks["control_zero_extra_params_vs_baseline"] = {
        "pass": bool(param_counts["control"] == param_counts["baseline"]), "counts": param_counts}

    val_probe = {arm: metrics.newtask_probe(head, backbone, loaders["val"], device) for arm, head in heads.items()}
    if "probe" in runs["control"]["metrics"]:
        diff = _max_abs_diff(runs["control"]["metrics"]["probe"], val_probe["control"])
        checks["control_probe_matches_in_run"] = {"pass": bool(diff <= PROBE_TOLERANCE), "max_abs_diff": diff}

    integrity = {"checks": checks, "all_pass": all(item["pass"] for item in checks.values())}

    # ---- 诊断量（test 同序）----
    corr = {"_".join(pair): {"pearson": metrics.pearson_corr(test_preds[pair[0]], test_preds[pair[1]]),
                             "pearson_logit": metrics.pearson_corr(metrics.logit_clip(test_preds[pair[0]]),
                                                                   metrics.logit_clip(test_preds[pair[1]])),
                             "spearman": metrics.spearman_corr(test_preds[pair[0]], test_preds[pair[1]])}
            for pair in PAIRS}
    paired = {f"{a}_minus_{b}": metrics.paired_delta_stats(test_preds[a], test_preds[b]) for a, b in PAIRS}
    result = {
        "stage1_id": sid, "runs": {arm: run["run_id"] for arm, run in runs.items()},
        "heads": {arm: {"trainable_params": int(param_counts[arm]), "state_sha256": P.backbone_sha256(head)}
                  for arm, head in heads.items()},
        "integrity": integrity,
        "test": {"dispersion": {arm: metrics.pred_dispersion(test_preds[arm]) for arm in runs},
                 "corr": corr, "paired_delta": paired},
        "val": {"probe": val_probe},
        "gates": {"head_gate_mean": {arm: val_probe[arm]["head_gate_mean"] for arm in runs},
                  "source_gate_mean": {arm: val_probe[arm]["source_gate_mean"] for arm in runs},
                  "norm_mean": {arm: val_probe[arm]["norm_mean"] for arm in runs}},
    }

    out_dir = runs["control"]["dir"]
    (out_dir / "three_way_compare.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"three-way compare: stage1={sid}",
             f"runs: baseline={baseline_run} null={null_run} control={control_run}",
             f"integrity all_pass={integrity['all_pass']}"]
    for name, item in checks.items():
        lines.append(f"  check {name}: {'PASS' if item['pass'] else 'FAIL'}")
    for pair, values in corr.items():
        lines.append(f"  corr {pair}: pearson={values['pearson']:.6f} logit={values['pearson_logit']:.6f} "
                     f"spearman={values['spearman']:.6f}")
    for pair, values in paired.items():
        lines.append(f"  paired {pair}: mean={values['mean']:.6g} frac_pos={values['frac_pos']:.4f}")
    (out_dir / "three_way_compare.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for line in lines[:4]:
        print(f"[compare] {line}")
    return result
