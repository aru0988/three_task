"""第二 seed 稳定性复现对照（预注册：docs/superpowers/specs/2026-10-03-aliccp-stage2-attenuation-seed-replication-design.md §6）。

在同一 Stage-1 产物（真冻结、no_grad）上，把基线臂与处理臂（固定衰减 c）两个 head `newtask.pt`
各自 strict 载入，在**同一** test/val 序上重评测：先过完整性检查（重算 AUC/gate/离散度与各 run
记录值比对、配对构造校验），再输出配对统计、§5.1 判定、§5.2 pooled 与联合分类，并落盘 JSON。

纯分析：不做任何训练、不改任何 run 产物、不改任何分支文件；本分支不移植 seed-1 的 null 臂与
git blob 头类重构（无 null 臂）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from multitaskrec.model import MPTRec, NewTask

from . import bench, metrics, protocol

ARMS = ("baseline", "arm")


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
    """backbone 源任务 gate 均值（val 同序；与 seed-1 臂 SourceGateStats 同口径）。"""
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


def _read_gate_verdicts(root, run_id: str) -> dict:
    path = protocol.run_dir(root, run_id) / "gate_report.json"
    if not path.exists():
        raise FileNotFoundError(f"run gate_report 不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["gates"]


def _build_head(arm: str, *, input_size, rep_dim, tower_hidden, reg_dnn, device):
    """按臂构造头类（arm = 预注册系数衰减；baseline = 基点行为）。"""
    common = dict(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(tower_hidden),
                  reg_dnn=reg_dnn, device=device)
    if arm == "arm":
        return NewTask(spec_attenuation=metrics.SPEC_ATTENUATION_COEF, **common)
    return NewTask(**common)


def _load_head(root, run_id: str, arm: str, **kwargs):
    head = _build_head(arm, **kwargs)
    state = torch.load(protocol.run_dir(root, run_id) / "newtask.pt", map_location="cpu")
    head.load_state_dict(state, strict=True)
    return head.to(kwargs["device"])


def run_replication(
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
    """双臂复现对照（§6）：完整性检查先行；任一失败 → integrity.pass=False（照常落盘，供审计，不抛异常）。

    判定（§5.1）与 pooled（§5.2）对**记录值**机械计算；完整性检查已确认重算值与记录值一致。
    """
    root = Path(root)
    run_ids = {arm: str(run_ids[arm]) for arm in ARMS}
    recorded = {arm: _read_metrics(root, run_ids[arm]) for arm in ARMS}
    gate_verdicts = {arm: _read_gate_verdicts(root, run_ids[arm]) for arm in ARMS}
    log(f"[replicate] stage1_id={stage1_id} arms=" + ", ".join(f"{arm}={run_ids[arm]}" for arm in ARMS))

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
        head = _load_head(root, run_ids[arm], arm, **head_kwargs)
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
        # 配对构造校验（§3/§6）：两臂必须同 stage1、同 seed；衰减标记与臂身份一致
        _check(f"{arm}.stage1_id_matches", rec.get("stage1_id") == stage1_id,
               f"recorded={rec.get('stage1_id')} expected={stage1_id}")
        _check(f"{arm}.model_seed_is_replication_seed",
               rec.get("model_seed") == metrics.REPLICATION_MODEL_SEED,
               f"recorded={rec.get('model_seed')} expected={metrics.REPLICATION_MODEL_SEED}")
        expected_coef = metrics.SPEC_ATTENUATION_COEF if arm == "arm" else 1.0
        _check(f"{arm}.spec_attenuation_is_{'frozen_coef' if arm == 'arm' else 'default'}",
               rec.get("spec_attenuation") == expected_coef,
               f"recorded={rec.get('spec_attenuation')!r} expected={expected_coef!r}")

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

    _check("model_seed_identical_across_arms",
           recorded["baseline"].get("model_seed") == recorded["arm"].get("model_seed"),
           f"baseline={recorded['baseline'].get('model_seed')} arm={recorded['arm'].get('model_seed')}")
    _check("y_true_identical_across_arms",
           len({s["val"] for s in y_shas.values()}) == 1 and len({s["test"] for s in y_shas.values()}) == 1,
           json.dumps({arm: y_shas[arm]["test"][:16] for arm in ARMS}))

    # ---- 源 gate 均值（backbone 级，两臂同值）：与本臂 in-run 探针交叉核对 ----
    source_gate = source_gate_probe(model, loaders["val"], device)
    arm_probe = recorded["arm"].get("probe", {})
    if "source_gate_mean" in arm_probe:
        _check("source_gate_mean_matches_arm_probe",
               all(abs(a - b) <= 1e-12 for a, b in zip(source_gate, arm_probe["source_gate_mean"])),
               f"recomputed={source_gate!r} probe={arm_probe['source_gate_mean']!r}")
    arm_probe_std = arm_probe.get("pred_std")
    if arm_probe_std is not None:
        recomputed_std = arms_out["arm"]["val_dispersion"]["pred_std"]
        _check("arm.pred_std_matches_probe", recomputed_std == arm_probe_std,
               f"recomputed={recomputed_std!r} probe={arm_probe_std!r}")

    # ---- A 类门禁（构造/冻结）：两臂必须全 PASS（A3 SKIP 视为通过）----
    def _a_class_ok(arm: str) -> bool:
        return all(gate_verdicts[arm][g]["verdict"] in ("PASS", "SKIP") for g in ("A1", "A2", "A3", "A4", "A5", "A6"))

    a_class_pass = all(_a_class_ok(arm) for arm in ARMS)
    _check("a_class_gates_pass_both_arms", a_class_pass,
           json.dumps({arm: {g: gate_verdicts[arm][g]["verdict"] for g in ("A1", "A2", "A3", "A4", "A5", "A6")}
                       for arm in ARMS}))

    # ---- §5.1 判定 + §5.2 pooled/联合（对记录值机械计算；完整性通过时 == 重算值）----
    verdict = metrics.stability_arm_verdict(
        recorded["baseline"]["test_auc_bsi"], recorded["baseline"]["best_val_auc_bsi"],
        recorded["arm"]["test_auc_bsi"], recorded["arm"]["best_val_auc_bsi"],
    )
    verdict["a_class_pass"] = bool(a_class_pass)
    verdict["criterion_pass"] = bool(verdict["classification"] == "STABILITY_SUPPORTED" and a_class_pass)
    verdict["classification_with_gates"] = (
        "STABILITY_SUPPORTED" if verdict["criterion_pass"] else "STABILITY_NOT_SUPPORTED"
    )
    pooled = metrics.pooled_seed_stability(
        metrics.SEED1_DELTA_TEST_AUC, metrics.SEED1_DELTA_VAL_AUC,
        verdict["delta_test_auc"], verdict["delta_val_auc"],
    )
    joint = metrics.joint_stability_classification(
        {"classification": verdict["classification_with_gates"]}, pooled
    )

    comparisons = {"arm_vs_baseline": metrics.cross_arm_stats(preds_test["arm"], preds_test["baseline"])}

    result = {
        "meta": {
            "stage1_id": stage1_id,
            "run_ids": run_ids,
            "prefix_tag": prefix_tag,
            "budgets": dict(budgets),
            "model_seed": metrics.REPLICATION_MODEL_SEED,
            "protocol_model_seed": metrics.PROTOCOL_MODEL_SEED,
            "spec_attenuation_coef": metrics.SPEC_ATTENUATION_COEF,
            "null_run_null_mean": metrics.NULL_RUN_NULL_MEAN,
            "seed1_delta_test_auc": metrics.SEED1_DELTA_TEST_AUC,
            "seed1_delta_val_auc": metrics.SEED1_DELTA_VAL_AUC,
            "stability_delta_test_min": metrics.STABILITY_DELTA_TEST_MIN,
            "backbone_sha256": backbone_sha,
            "fingerprint_sha256": fp["fingerprint_sha256"],
            "commit": protocol.code_commit(),
            "git": protocol.git_state(),
            "device": str(device),
        },
        "integrity": integrity,
        "gates": {**{arm: gate_verdicts[arm] for arm in ARMS}, "a_class_pass": bool(a_class_pass)},
        "arms": arms_out,
        "source_gate_mean": source_gate,
        "comparisons": comparisons,
        "verdict": verdict,
        "pooled": pooled,
        "joint": joint,
    }
    if output_path is None:
        output_path = protocol.run_dir(root, run_ids["arm"]) / "replication_compare.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_path"] = str(output_path)
    log(f"[replicate] integrity_pass={integrity['pass']} output={output_path}")
    log(f"[replicate] verdict={verdict['classification_with_gates']} "
        f"delta_test={verdict['delta_test_auc']:+.7f} delta_val={verdict['delta_val_auc']:+.7f} "
        f"joint={joint['classification']} mean_delta_test={pooled['mean_delta_test_auc']:+.7f}")
    stats = comparisons["arm_vs_baseline"]
    log(f"[replicate] arm_vs_baseline: pearson_logit={stats['pearson_logit']} "
        f"spearman_pred={stats['spearman_pred']} delta_mean={stats['delta_mean']:+.3e} frac_pos={stats['frac_pos']:.4f}")
    for arm in ARMS:
        log(f"[replicate] {arm}: val pred_std={arms_out[arm]['val_dispersion']['pred_std']:.10f} "
            f"gate_mean={arms_out[arm]['gate_mean_recomputed']}")
    return result
