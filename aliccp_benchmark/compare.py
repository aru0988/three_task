"""三方对照分析（spec §6.2）：基线 / null 处理臂（前臂）/ 特定混合衰减对照臂。

在同一 Stage-1 产物（真冻结、no_grad）上，把三个 head `newtask.pt` 各自 strict 载入，
在**同一** test/val 序上重评测：先过完整性检查（重算 AUC/gate 与各 run 记录值比对），
再输出离散度、相关（Pearson-logit / Spearman）与 paired delta。

纯分析：不做任何训练、不改任何 run 产物、不改任何分支文件；本分支**不移植**前臂代码——
null 头类从 git blob `6224c0f:multitaskrec/model.py` 动态载入（sha256 钉死）。

对照说明：docs/superpowers/specs/2026-10-03-aliccp-stage2-specific-attenuation-control-design.md
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import types
from pathlib import Path

import torch

from multitaskrec.model import MPTRec, NewTask

from . import bench, metrics, protocol

REPO = Path(__file__).resolve().parents[1]
NULL_HEAD_COMMIT = "6224c0f"          # 前臂 model.py（含 use_null_expert）所在 commit
NULL_HEAD_MODEL_SHA256 = "836db5327b550cdcd07dee004c1c30e75010df43b38d5f950a41f1977539786c"
HEAD_SOURCE_RELPATH = "multitaskrec/model.py"
ARMS = ("baseline", "null", "control")


def load_null_head_class(commit: str = NULL_HEAD_COMMIT):
    """从 git blob 动态载入前臂 NewTask 类（与测试同一机制，不落地任何文件）。返回 (cls, sha256)。"""
    src_bytes = subprocess.run(
        ["git", "show", f"{commit}:{HEAD_SOURCE_RELPATH}"], cwd=REPO, check=True, capture_output=True
    ).stdout
    sha = hashlib.sha256(src_bytes).hexdigest()
    module = types.ModuleType("model_null_head")
    exec(compile(src_bytes.decode("utf-8"), f"{commit}:{HEAD_SOURCE_RELPATH}", "exec"), module.__dict__)
    return module.NewTask, sha


def build_frozen_backbone(
    root, stage1_id, device, *,
    vocab=None, expert_hidden=protocol.EXPERT_HIDDEN, tower_hidden=protocol.TOWER_HIDDEN,
    embedding_size=protocol.EMBEDDING_SIZE, input_size=protocol.INPUT_SIZE,
    reg_dnn=protocol.REG_DNN, reg_embedding=protocol.REG_EMBEDDING,
):
    """加载固定 Stage-1 产物并真冻结（与 bench.run_stage2 的构造/冻结口径一致）。返回 (model, meta)。"""
    art = protocol.load_stage1(root, stage1_id)
    vocab = dict(vocab) if vocab is not None else protocol.build_vocab()
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=embedding_size,
        input_size=input_size,
        expert_dnn_hidden_units=list(expert_hidden),
        tower_dnn_hidden_units=list(tower_hidden),
        dropout=list(protocol.DROPOUT),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device,
    ).to(device)
    model.load_state_dict(art["backbone_state"])
    protocol.freeze_backbone(model)
    return model, art["meta"]


@torch.no_grad()
def predict_newtask(newtask, model, loader, device):
    """与 bench.evaluate_newtask 同一遍历口径，但返回 (y_true, preds) 逐样本张量（cat 后同序）。"""
    newtask.eval()
    ys, preds = [], []
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        preds.append(newtask(dnn_input, gen_rep, spec_reps, env_embs).detach())
        ys.append(y)
    return torch.cat(ys), torch.cat(preds)


@torch.no_grad()
def source_gate_probe(model, loader, device) -> list:
    """backbone 源任务 gate 均值（val 同序；与 null 臂 SourceGateStats 同口径）。"""
    model.eval()
    gate_stats = metrics.SourceGateStats(protocol.NUM_TASKS)
    for _, _, _, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, _, _, _ = model.get_infos(features)
        gate_stats.update([model.gate_networks[i](dnn_input) for i in range(protocol.NUM_TASKS)])
    return gate_stats.result()


def _read_metrics(root, run_id: str) -> dict:
    path = protocol.run_dir(root, run_id) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"run metrics 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_head(arm: str, *, input_size, rep_dim, tower_hidden, reg_dnn, device):
    """按臂构造头类（control = 预注册系数衰减；null = git blob 前臂类；baseline = 基点行为）。"""
    common = dict(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(tower_hidden),
                  reg_dnn=reg_dnn, device=device)
    if arm == "null":
        null_cls, sha = load_null_head_class()
        return null_cls(use_null_expert=True, **common), sha
    if arm == "control":
        return NewTask(spec_attenuation=metrics.SPEC_ATTENUATION_COEF, **common), None
    return NewTask(**common), None


def _load_head(root, run_id: str, arm: str, **kwargs):
    head, source_sha = _build_head(arm, **kwargs)
    state = torch.load(protocol.run_dir(root, run_id) / "newtask.pt", map_location="cpu")
    head.load_state_dict(state, strict=True)
    return head.to(kwargs["device"]), source_sha


def run_comparison(
    *,
    root,
    stage1_id,
    run_ids,
    data_files,
    budgets,
    prefix_tag,
    device,
    batch_size=protocol.BATCH_SIZE,
    vocab=None,
    expert_hidden=protocol.EXPERT_HIDDEN,
    tower_hidden=protocol.TOWER_HIDDEN,
    embedding_size=protocol.EMBEDDING_SIZE,
    input_size=protocol.INPUT_SIZE,
    reg_dnn=protocol.REG_DNN,
    log=print,
    output_path=None,
) -> dict:
    """三方对照（spec §6.2）：完整性检查先行；任一失败 → integrity.pass=False（照常落盘，供审计）。"""
    root = Path(root)
    run_ids = {arm: str(run_ids[arm]) for arm in ARMS}
    recorded = {arm: _read_metrics(root, run_ids[arm]) for arm in ARMS}
    log(f"[compare] stage1_id={stage1_id} arms="
        + ", ".join(f"{arm}={run_ids[arm]}" for arm in ARMS))

    # ---- 前缀身份（A2 口径）：同一 test/val 序的锚 ----
    fp = protocol.load_fingerprint(root, prefix_tag)
    protocol.verify_fingerprint(fp)

    model, art_meta = build_frozen_backbone(
        root, stage1_id, device, vocab=vocab, expert_hidden=expert_hidden, tower_hidden=tower_hidden,
        embedding_size=embedding_size, input_size=input_size, reg_dnn=reg_dnn,
    )
    backbone_sha = protocol.backbone_sha256(model)

    datasets, loaders = bench._loaders(data_files, budgets, batch_size)
    for split in ("train", "val", "test"):
        if len(datasets[split]) != budgets[split]:
            raise AssertionError(f"样本数与预算不符（{split}）: {len(datasets[split])} != {budgets[split]}")

    integrity = {"checks": {}, "pass": True}

    def _check(name: str, ok: bool, detail: str) -> None:
        integrity["checks"][name] = {"pass": bool(ok), "detail": detail}
        integrity["pass"] = integrity["pass"] and bool(ok)

    _check("fingerprint_verified", True, f"prefix={prefix_tag} sha={fp['fingerprint_sha256'][:16]}")
    _check("backbone_sha_matches_stage1", backbone_sha == art_meta.get("backbone_sha256"),
           f"loaded={backbone_sha[:16]} stage1={str(art_meta.get('backbone_sha256'))[:16]}")

    rep_dim = int(list(expert_hidden)[-1])
    head_kwargs = dict(input_size=input_size, rep_dim=rep_dim, tower_hidden=tower_hidden,
                       reg_dnn=reg_dnn, device=device)

    arms_out, preds_test, y_shas = {}, {}, {}
    for arm in ARMS:
        head, source_sha = _load_head(root, run_ids[arm], arm, **head_kwargs)
        y_val, p_val = predict_newtask(head, model, loaders["val"], device)
        y_test, p_test = predict_newtask(head, model, loaders["test"], device)
        val_auc = metrics.auc_score(y_val, p_val)
        test_auc = metrics.auc_score(y_test, p_test)
        gate_mean = bench.newtask_gate_mean(head, model, loaders["val"], device)
        rec = recorded[arm]

        _check(f"{arm}.test_auc_matches_recorded", abs(test_auc - rec["test_auc_bsi"]) <= 1e-9,
               f"recomputed={test_auc!r} recorded={rec['test_auc_bsi']!r} diff={test_auc - rec['test_auc_bsi']!r}")
        _check(f"{arm}.val_auc_matches_recorded", abs(val_auc - rec["best_val_auc_bsi"]) <= 1e-9,
               f"recomputed={val_auc!r} recorded={rec['best_val_auc_bsi']!r} diff={val_auc - rec['best_val_auc_bsi']!r}")
        _check(f"{arm}.gate_mean_matches_recorded",
               all(abs(a - b) <= 1e-12 for a, b in zip(gate_mean, rec["gate_mean"])),
               f"recomputed={gate_mean!r} recorded={rec['gate_mean']!r}")
        if arm == "null":
            _check("null_head_source_sha256", source_sha == NULL_HEAD_MODEL_SHA256,
                   f"sha={source_sha}")

        arms_out[arm] = {
            "run_id": run_ids[arm],
            "commit": rec.get("commit"),
            "test_auc_recomputed": test_auc,
            "test_auc_recorded": rec["test_auc_bsi"],
            "val_auc_recomputed": val_auc,
            "val_auc_recorded": rec["best_val_auc_bsi"],
            "gate_mean_recomputed": gate_mean,
            "gate_mean_recorded": rec["gate_mean"],
            "val_dispersion": metrics.pred_dispersion(p_val),
            "test_dispersion": metrics.pred_dispersion(p_test),
        }
        preds_test[arm] = p_test
        y_shas[arm] = {"val": protocol.sha256_tensor(y_val), "test": protocol.sha256_tensor(y_test)}

    _check("y_true_identical_across_arms",
           len({s["val"] for s in y_shas.values()}) == 1 and len({s["test"] for s in y_shas.values()}) == 1,
           json.dumps({arm: y_shas[arm]["test"][:16] for arm in ARMS}))

    # ---- 源 gate 均值（backbone 级，三臂同值）：与 null 臂记录值、本臂 in-run 探针交叉核对 ----
    source_gate = source_gate_probe(model, loaders["val"], device)
    null_recorded_sg = recorded["null"].get("mechanism", {}).get("source_gate_mean")
    if null_recorded_sg is not None:
        _check("source_gate_mean_matches_null_run",
               all(abs(a - b) <= 1e-12 for a, b in zip(source_gate, null_recorded_sg)),
               f"recomputed={source_gate!r} null_run={null_recorded_sg!r}")
    control_probe = recorded["control"].get("probe", {})
    if "source_gate_mean" in control_probe:
        _check("source_gate_mean_matches_control_probe",
               all(abs(a - b) <= 1e-12 for a, b in zip(source_gate, control_probe["source_gate_mean"])),
               f"recomputed={source_gate!r} probe={control_probe['source_gate_mean']!r}")

    # ---- 离散度交叉核对：control 与 in-run 探针逐位一致；null 与记录值（同公式）一致 ----
    control_probe_std = control_probe.get("pred_std")
    if control_probe_std is not None:
        recomputed_std = arms_out["control"]["val_dispersion"]["pred_std"]
        _check("control.pred_std_matches_probe", recomputed_std == control_probe_std,
               f"recomputed={recomputed_std!r} probe={control_probe_std!r}")
    null_recorded_std = recorded["null"].get("mechanism", {}).get("pred_std")
    if null_recorded_std is not None:
        recomputed_std = arms_out["null"]["val_dispersion"]["pred_std"]
        _check("null.pred_std_matches_recorded", abs(recomputed_std - null_recorded_std) <= 1e-12,
               f"recomputed={recomputed_std!r} recorded={null_recorded_std!r}")

    # ---- 两两对照（test 同序）----
    comparisons = {}
    for a, b in (("control", "null"), ("control", "baseline"), ("null", "baseline")):
        comparisons[f"{a}_vs_{b}"] = metrics.cross_arm_stats(preds_test[a], preds_test[b])

    corr_cn = comparisons["control_vs_null"]
    corr_cb = comparisons["control_vs_baseline"]
    expectation = {
        "corr_control_null_gt_control_baseline": {
            "pearson_logit": (corr_cn["pearson_logit"] is not None and corr_cb["pearson_logit"] is not None
                              and corr_cn["pearson_logit"] > corr_cb["pearson_logit"]),
            "spearman_pred": (corr_cn["spearman_pred"] is not None and corr_cb["spearman_pred"] is not None
                              and corr_cn["spearman_pred"] > corr_cb["spearman_pred"]),
        }
    }

    result = {
        "meta": {
            "stage1_id": stage1_id,
            "run_ids": run_ids,
            "prefix_tag": prefix_tag,
            "budgets": dict(budgets),
            "spec_attenuation_coef": metrics.SPEC_ATTENUATION_COEF,
            "null_run_null_mean": metrics.NULL_RUN_NULL_MEAN,
            "null_head_commit": NULL_HEAD_COMMIT,
            "null_head_model_sha256": NULL_HEAD_MODEL_SHA256,
            "backbone_sha256": backbone_sha,
            "fingerprint_sha256": fp["fingerprint_sha256"],
            "commit": protocol.code_commit(),
            "git": protocol.git_state(),
            "device": str(device),
        },
        "integrity": integrity,
        "arms": arms_out,
        "source_gate_mean": source_gate,
        "comparisons": comparisons,
        "prereg_expectation": expectation,
    }
    if output_path is None:
        output_path = protocol.run_dir(root, run_ids["control"]) / "three_way_compare.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_path"] = str(output_path)
    log(f"[compare] integrity_pass={integrity['pass']} output={output_path}")
    for pair, stats in comparisons.items():
        log(f"[compare] {pair}: pearson_logit={stats['pearson_logit']} spearman_pred={stats['spearman_pred']} "
            f"delta_mean={stats['delta_mean']:+.3e} frac_pos={stats['frac_pos']:.4f}")
    for arm in ARMS:
        log(f"[compare] {arm}: val pred_std={arms_out[arm]['val_dispersion']['pred_std']:.10f} "
            f"gate_mean={arms_out[arm]['gate_mean_recomputed']}")
    return result
