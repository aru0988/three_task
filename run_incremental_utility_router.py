"""AliCCP 两路径增量效用路由入口（预注册 C1=40e1b9d：
docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md）。

复用冻结协议 machinery（前缀指纹 / Stage-1 产物 / 真冻结 / A 类门禁 / SUMMARY 只追加），
不修改 bench.py / residual_prompt.py / protocol.py / metrics.py / multitaskrec/*。

用法（cwd = 本 worktree 根；解释器 = 主树 venv）：
  python run_incremental_utility_router.py --tag short --model-seed 1688723512 \
      --stage1-id s1-5c060b9c-m1688723512-e3-3a30e2c0
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from aliccp_benchmark import incremental_utility_router as IUR
from aliccp_benchmark import metrics, protocol
from aliccp_benchmark import residual_prompt as RP
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec


class _Tee:
    """stdout 同时写控制台与日志文件（与 run_aliccp_benchmark.py 同式）。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}


def _fmt(value, signed: bool = False) -> str:
    """None（UNDEFINED）显式格式化，杜绝任何 NaN 参与输出/比较（统筹第四轮）。"""
    if value is None:
        return "UNDEFINED"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _fmt6(value) -> str:
    """SUMMARY 单元格用固定 6 位小数；None ⇒ 'UNDEFINED'。"""
    return "UNDEFINED" if value is None else f"{value:.6f}"


def _reset_peak_vram(device) -> None:
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def _collect(model, head_b, head_p, dataset, device, batch_size: int) -> dict:
    """单次前向收集 y / p_B / p_P / 8 维零标签特征（两头均 eval；特征构造不含标签）。"""
    head_b.eval()
    head_p.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    ys, pbs, pps, feats = [], [], [], []
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        pb = head_b(dnn_input, gen_rep, spec_reps, env_embs)
        pp = head_p(dnn_input, gen_rep, spec_reps, env_embs)
        # env_embs 为逐 env 常量嵌入（(1, rep_dim)×2）：拼接范数广播为常量列（C1c 澄清）
        env_norm = float(torch.cat([e.detach().reshape(-1) for e in env_embs]).norm())
        n_rows = int(y.shape[0])
        feats.append(
            IUR.router_features(
                pb.detach().cpu().numpy(), pp.detach().cpu().numpy(),
                dnn_input.detach().cpu().numpy(), gen_rep.detach().cpu().numpy(),
                spec_reps[0].detach().cpu().numpy(), spec_reps[1].detach().cpu().numpy(),
                np.full(n_rows, env_norm, dtype=np.float64),
            )
        )
        ys.append(y.numpy())
        pbs.append(pb.detach().cpu().numpy())
        pps.append(pp.detach().cpu().numpy())
    return {
        "y": np.concatenate(ys).astype(np.int64),
        "p_b": np.concatenate(pbs).astype(np.float64),
        "p_p": np.concatenate(pps).astype(np.float64),
        "X": np.concatenate(feats).astype(np.float64),
    }


@torch.no_grad()
def _eval_head(head, model, loader, device) -> float:
    """协议评测口径（同 benchmark.evaluate_newtask）：val BSI AUC。"""
    head.eval()
    y_true, y_hat = [], []
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        y_true.append(y)
        y_hat.append(head(dnn_input, gen_rep, spec_reps, env_embs))
    return metrics.auc_score(torch.cat(y_true), torch.cat(y_hat))


@torch.no_grad()
def _gate_mean(head_b, model, loader, device) -> list:
    """M3：val 上基线头 gate_network 平均输出（B3 判定用，同 bench.newtask_gate_mean 口径）。"""
    head_b.eval()
    total, count = None, 0
    for _, _, _, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, _, _, _ = model.get_infos(features)
        gate = head_b.gate_network(dnn_input)
        total = gate.sum(dim=0) if total is None else total + gate.sum(dim=0)
        count += gate.shape[0]
    return [float(v) / max(1, count) for v in total]


def _train_arm(head, model, train_subset, val_loader, device, *, epochs, patience, batch_size, lr) -> dict:
    """协议 stage2 同规则（Adam 1e-4 / BCE+l2 / val 选点 patience）；训练数据仅 A 子集。"""
    optimizer = torch.optim.Adam(params=head.parameters(), lr=lr)
    loss_func = torch.nn.BCELoss()
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    best_auc, best_epoch, best_state, earlystop_count = 0.0, None, None, 0
    records = []
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum, steps = 0.0, 0
        for _, _, y, features in loader:
            for key in features:
                features[key] = features[key].to(device)
            with torch.no_grad():
                infos = model.get_infos(features)
            pred = head(*infos)
            loss = loss_func(pred.cpu(), y.float()) + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            steps += 1
        val_auc = _eval_head(head, model, val_loader, device)
        records.append({"epoch": epoch, "train_loss": loss_sum / max(1, steps), "val_auc_bsi": val_auc})
        if val_auc > best_auc:
            best_auc, best_epoch, earlystop_count = val_auc, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            earlystop_count += 1
            if earlystop_count == patience:
                break
    head.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_auc_bsi": float(best_auc), "per_epoch": records}


def run_router(
    *,
    root,
    stage1_id,
    data_files,
    budgets,
    prefix_tag,
    model_seed,
    epochs,
    patience,
    tag,
    device,
    log=print,
    batch_size=protocol.BATCH_SIZE,
    lr=protocol.LR,
    enforce_b=True,
) -> dict:
    """两臂（仅 A）→ val/B/C/test 评分 → verifier(B) → 校准(C) → 六配置判定（预注册 §2–§6）。"""
    t0 = time.time()
    _reset_peak_vram(device)
    run_id = protocol.make_run_id(
        datetime.now(), prefix_tag=prefix_tag, model_seed=model_seed, tag=tag, commit=protocol.code_commit()
    ) + IUR.RUN_ID_SUFFIX
    run_path = protocol.run_dir(root, run_id)
    if run_path.exists():
        raise SystemExit(f"run 目录已存在，禁止覆盖：{run_path}")
    run_path.mkdir(parents=True)
    (run_path / "status.json").write_text(
        json.dumps({"run_id": run_id, "state": "running",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "stage1_id": stage1_id, "tag": tag, "model_seed": int(model_seed)},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"STARTED run_id={run_id} stage1_id={stage1_id} tag={tag} model_seed={model_seed} "
        f"enforce_b={enforce_b} commit={protocol.code_commit()}")
    protocol.seed_model(model_seed)

    # ---- 数据与 A2/A4 ----
    datasets = {s: AliCCPDataset(data_files[s], budgets[s]) for s in ("train", "val", "test")}
    files_budgets = {s: (Path(data_files[s]), budgets[s]) for s in ("train", "val", "test")}
    fp, created = protocol.ensure_fingerprint(root, prefix_tag, files_budgets)
    log(f"[iuv] 前缀指纹 {'构建并落盘' if created else '读取校验'}：{fp['fingerprint_sha256'][:16]}")
    prefix_sha_ok = True
    try:
        protocol.verify_fingerprint(fp)
    except AssertionError as exc:  # noqa: BLE001
        prefix_sha_ok = False
        log(f"[iuv] A2 前缀指纹校验失败：{exc}")
    len_ok = all(len(datasets[s]) == budgets[s] for s in ("train", "val", "test"))
    counts_ok = True
    for split in ("train", "val", "test"):
        try:
            protocol.verify_label_counts(datasets[split], budgets[split], fp["label_counts"][split])
        except (AssertionError, KeyError) as exc:  # noqa: BLE001
            counts_ok = False
            log(f"[iuv] A4 标签计数失败（{split}）：{exc}")

    # ---- Stage-1 产物（唯一 backbone 来源）与真冻结 ----
    art = protocol.load_stage1(root, stage1_id)
    art_meta = art["meta"]
    env_ids = art["env_ids"]
    fingerprint_sha_match = art_meta.get("fingerprint_sha256") == fp.get("fingerprint_sha256")
    env_ids_sha_match = protocol.sha256_tensor(env_ids) == art_meta.get("env_ids_sha256")

    vocab = protocol.build_vocab()
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=protocol.EMBEDDING_SIZE,
        input_size=protocol.INPUT_SIZE,
        expert_dnn_hidden_units=list(protocol.EXPERT_HIDDEN),
        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
        dropout=list(protocol.DROPOUT),
        reg_embedding=protocol.REG_EMBEDDING,
        reg_dnn=protocol.REG_DNN,
        device=device,
    ).to(device)
    model.load_state_dict(art["backbone_state"])
    backbone_sha_loaded = protocol.backbone_sha256(model)
    backbone_sha_matches_stage1 = backbone_sha_loaded == art_meta.get("backbone_sha256")
    stage1_id_recorded = art_meta.get("stage1_id") == stage1_id
    protocol.freeze_backbone(model)
    backbone_sha_before = protocol.backbone_sha256(model)
    log(f"[iuv] backbone 已加载并冻结：sha={backbone_sha_before[:16]}（A6 匹配={backbone_sha_matches_stage1}）")

    # ---- 两臂构造（固定顺序 B→P；RNG 端点记录）----
    rep_dim = int(list(protocol.EXPERT_HIDDEN)[-1])
    rng_before = torch.get_rng_state()
    head_b = RP.build_newtask(
        RP.BASELINE_VARIANT, input_size=protocol.INPUT_SIZE, rep_dim=rep_dim,
        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN), reg_dnn=protocol.REG_DNN, device=device,
    ).to(device)
    head_p = RP.build_newtask(
        RP.VARIANT, input_size=protocol.INPUT_SIZE, rep_dim=rep_dim,
        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN), reg_dnn=protocol.REG_DNN, device=device,
    ).to(device)
    rng_after = torch.get_rng_state()

    # ---- 训练内部三分割（A/B/C）----
    n_train = len(datasets["train"])
    bounds = IUR.split_bounds(n_train)
    subset_A = Subset(datasets["train"], range(*bounds["A"]))
    subset_B = Subset(datasets["train"], range(*bounds["B"]))
    subset_C = Subset(datasets["train"], range(*bounds["C"]))
    subset_index_sha = {k: IUR.range_sha(*v) for k, v in bounds.items()}
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- 两臂训练（仅 A；训练前各自重播种 ⇒ 同规则同随机流口径）----
    protocol.seed_model(model_seed)
    arm_b = _train_arm(head_b, model, subset_A, val_loader, device,
                       epochs=epochs, patience=patience, batch_size=batch_size, lr=lr)
    log(f"[iuv] 臂 B（baseline）best_epoch={arm_b['best_epoch']} best_val={arm_b['best_val_auc_bsi']:.4f}")
    protocol.seed_model(model_seed)
    arm_p = _train_arm(head_p, model, subset_A, val_loader, device,
                       epochs=epochs, patience=patience, batch_size=batch_size, lr=lr)
    log(f"[iuv] 臂 P（residual-prompt）best_epoch={arm_p['best_epoch']} best_val={arm_p['best_val_auc_bsi']:.4f}")

    arm_budget = {"A": list(bounds["A"]), "epochs": int(epochs), "patience": int(patience),
                  "lr": float(lr), "batch_size": int(batch_size)}
    arm_cfg_equal = True  # 两臂由同一代码路径、同一 arm_budget 训练（结构性；R4 记录）

    # ---- val/B/C/test 评分（两头 eval；val 仅评估，不参与学习/校准）----
    coll_V = _collect(model, head_b, head_p, datasets["val"], device, batch_size)
    coll_B = _collect(model, head_b, head_p, subset_B, device, batch_size)
    coll_C = _collect(model, head_b, head_p, subset_C, device, batch_size)
    coll_T = _collect(model, head_b, head_p, datasets["test"], device, batch_size)
    log(f"[iuv] 评分完成：val={len(coll_V['y'])} B={len(coll_B['y'])} C={len(coll_C['y'])} "
        f"test={len(coll_T['y'])}")

    # ---- verifier（仅 B）与校准（仅 C）----
    for name, arr in (("pB_B", coll_B["p_b"]), ("pP_B", coll_B["p_p"]), ("X_B", coll_B["X"]),
                      ("pB_C", coll_C["p_b"]), ("pP_C", coll_C["p_p"]),
                      ("pB_V", coll_V["p_b"]), ("pP_V", coll_V["p_p"]),
                      ("pB_T", coll_T["p_b"]), ("pP_T", coll_T["p_p"])):
        if not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
            raise AssertionError(f"{name} 含非有限值（NaN/Inf），拒绝出数")
    u_B = IUR.utility(coll_B["p_b"], coll_B["p_p"], coll_B["y"])
    u_C = IUR.utility(coll_C["p_b"], coll_C["p_p"], coll_C["y"])
    if not (np.isfinite(u_B).all() and np.isfinite(u_C).all()):
        raise AssertionError("效用 u 含非有限值（NaN/Inf），拒绝出数")
    ridge = IUR.fit_ridge(coll_B["X"], u_B)
    s_B = IUR.ridge_score(ridge, coll_B["X"])
    s_C = IUR.ridge_score(ridge, coll_C["X"])
    s_T = IUR.ridge_score(ridge, coll_T["X"])
    s_V = IUR.ridge_score(ridge, coll_V["X"])
    if not (np.isfinite(s_B).all() and np.isfinite(s_C).all() and np.isfinite(s_T).all() and np.isfinite(s_V).all()):
        raise AssertionError("router 分数含非有限值（NaN/Inf），拒绝出数")
    thr_info = IUR.select_threshold(s_C, coll_C["p_b"], coll_C["p_p"], coll_C["y"])
    mix_info = IUR.select_mix_alpha(coll_C["p_b"], coll_C["p_p"], coll_C["y"])
    diag_auroc_C = IUR.auc_or_none(IUR.beneficial(u_C), s_C)          # 单类 ⇒ None（UNDEFINED/SKIP）
    diag_spearman_C = IUR.spearman_or_none(s_C, u_C)
    log(f"[iuv] verifier：AUROC(C, vs 1[u>0])={_fmt(diag_auroc_C)} spearman={_fmt(diag_spearman_C)} "
        f"thr={thr_info['thr']:.6f}({thr_info['calibration_status']}) pi_hat={thr_info['pi_hat']:.4f} "
        f"alpha_hat={mix_info['alpha_hat']:.2f}({mix_info['calibration_status']})")

    # ---- val/test 六配置（离线组配；test 端不再训练/校准）----
    y_T, pT_b, pT_p = coll_T["y"], coll_T["p_b"], coll_T["p_p"]
    y_V, pV_b, pV_p = coll_V["y"], coll_V["p_b"], coll_V["p_p"]
    a_hat = mix_info["alpha_hat"]
    rand_seed = int(model_seed) ^ IUR.ROUTER_RANDOM_SALT
    preds = {
        "always_baseline": pT_b,
        "always_prompt": pT_p,
        "fixed_mix": a_hat * pT_p + (1.0 - a_hat) * pT_b,
        "routed": IUR.route_predict(s_T, pT_b, pT_p, thr_info["thr"]),
        "random_router": np.where(IUR.random_route_mask(len(y_T), thr_info["pi_hat"], rand_seed), pT_p, pT_b),
        "label_assisted_bce_oracle_diag": IUR.oracle_route(pT_b, pT_p, IUR.utility(pT_b, pT_p, y_T)),
    }
    preds_V = {
        "always_baseline": pV_b,
        "always_prompt": pV_p,
        "fixed_mix": a_hat * pV_p + (1.0 - a_hat) * pV_b,
        "routed": IUR.route_predict(s_V, pV_b, pV_p, thr_info["thr"]),
        "random_router": np.where(IUR.random_route_mask(len(y_V), thr_info["pi_hat"], rand_seed), pV_p, pV_b),
        "label_assisted_bce_oracle_diag": IUR.oracle_route(pV_b, pV_p, IUR.utility(pV_b, pV_p, y_V)),
    }
    aucs_T = {name: IUR.auc_or_none(y_T, pred) for name, pred in preds.items()}
    aucs_V = {name: IUR.auc_or_none(y_V, pred) for name, pred in preds_V.items()}
    aucs_C = {
        "always_baseline": IUR.auc_or_none(coll_C["y"], coll_C["p_b"]),
        "always_prompt": IUR.auc_or_none(coll_C["y"], coll_C["p_p"]),
        "fixed_mix": mix_info["auc_C_mix"],
        "routed": thr_info["auc_C_routed"],
    }

    def _delta(a, b):
        return None if (a is None or b is None) else float(a) - float(b)

    deltas = {
        "base": _delta(aucs_T["routed"], aucs_T["always_baseline"]),
        "prompt": _delta(aucs_T["routed"], aucs_T["always_prompt"]),
        "mix": _delta(aucs_T["routed"], aucs_T["fixed_mix"]),
    }
    verdict = IUR.router_value_verdict(deltas["base"], deltas["prompt"], deltas["mix"])
    log(f"[iuv] val  AUC：base={_fmt(aucs_V['always_baseline'])} prompt={_fmt(aucs_V['always_prompt'])} "
        f"mix={_fmt(aucs_V['fixed_mix'])} routed={_fmt(aucs_V['routed'])} "
        f"rand={_fmt(aucs_V['random_router'])} oracle_diag={_fmt(aucs_V['label_assisted_bce_oracle_diag'])}")
    log(f"[iuv] test AUC：base={_fmt(aucs_T['always_baseline'])} prompt={_fmt(aucs_T['always_prompt'])} "
        f"mix={_fmt(aucs_T['fixed_mix'])} routed={_fmt(aucs_T['routed'])} "
        f"rand={_fmt(aucs_T['random_router'])} oracle_diag={_fmt(aucs_T['label_assisted_bce_oracle_diag'])}")
    log(f"[iuv] Δbase={_fmt(deltas['base'], signed=True)} Δprompt={_fmt(deltas['prompt'], signed=True)} "
        f"Δmix={_fmt(deltas['mix'], signed=True)} "
        f"分类={verdict['classification_by_delta_base']} router_value={verdict['router_value']}")

    # ---- A1：冻结完整性 ----
    backbone_sha_after = protocol.backbone_sha256(model)
    try:
        protocol.assert_no_grads(model)
        backbone_grads_none = True
    except AssertionError as exc:  # noqa: BLE001
        backbone_grads_none = False
        log(f"[iuv] A1 失败：{exc}")

    # ---- 门禁（A 类复用 metrics；R1–R5 本线机械判据）----
    a_facts = {
        "backbone_sha_before": backbone_sha_before,
        "backbone_sha_after": backbone_sha_after,
        "backbone_grads_none": backbone_grads_none,
        "prefix_sha_ok": prefix_sha_ok,
        "fingerprint_sha_match": fingerprint_sha_match,
        "len_ok": len_ok,
        "counts_ok": counts_ok,
        "env_ids_sha_match": env_ids_sha_match,
        "backbone_sha_matches_stage1": backbone_sha_matches_stage1,
        "stage1_id_recorded": stage1_id_recorded,
    }
    a_gates = metrics.evaluate_a_gates(a_facts)
    if aucs_V["routed"] is None or aucs_T["routed"] is None:
        b_gates = {g: {"verdict": "N/A", "detail": "AUC UNDEFINED（单类标签）——不判定"} for g in ("B1", "B2", "B3", "B4")}
    else:
        b_gates = metrics.evaluate_b_gates(
            {
                "auc_val_ctr": art_meta.get("best_val_auc_ctr", 0.0),
                "auc_val_cvr": art_meta.get("best_val_auc_cvr", 0.0),
                "auc_val_bsi_best": aucs_V["routed"],   # 官方 validation 上的 routed AUC（v3.1；独立复算）
                "auc_test_bsi": aucs_T["routed"],
                "gate_mean": _gate_mean(head_b, model, val_loader, device),
                "cluster_events": art_meta.get("cluster_events", []),
                "train_size": budgets["train"],
            }
        )
    feature_params = list(inspect.signature(IUR.router_features).parameters)
    label_free_ok = not any(p == "y" or "label" in p.lower() for p in feature_params)
    r_gates = {
        "R1": {"verdict": "PASS" if label_free_ok else "FAIL", "detail": f"signature={feature_params}"},
        "R2": {"verdict": "PASS" if (0 < bounds["A"][1] < bounds["B"][1] < n_train) else "FAIL",
               "detail": f"bounds={bounds} n_train={n_train} index_sha={subset_index_sha}"},
        "R3": {"verdict": "PASS", "detail": "fit_target_from=B; threshold_from=C; mix_from=C; random_rate_from=C"},
        "R4": {"verdict": "PASS" if arm_cfg_equal else "FAIL", "detail": f"arm_budget={arm_budget}（两臂同式）"},
        "R5": {"verdict": "PASS" if (len(coll_B["y"]) == bounds["B"][1] - bounds["B"][0]
                                     and len(coll_C["y"]) == bounds["C"][1] - bounds["C"][0]
                                     and len(coll_T["y"]) == budgets["test"]) else "FAIL",
               "detail": f"B={len(coll_B['y'])} C={len(coll_C['y'])} T={len(coll_T['y'])}"},
    }
    gates = {**a_gates, **b_gates, **r_gates}
    hard_pass = metrics.hard_pass(a_gates, enforce_b=enforce_b, b_gates=b_gates)
    all_gates_ok = hard_pass and all(g["verdict"] in ("PASS", "SKIP", "N/A") for g in r_gates.values())
    git_dirty = protocol.git_state()["dirty"]   # 于 SUMMARY 追加前采样（台账追加会改变 tracked 文件状态）
    expand_ok = IUR.expand_eligible(verdict["router_value"], all_gates_ok, git_dirty)

    # ---- 产物（run 目录与 status.json 已于开始时创建；此处只写产物）----
    np.savez(
        run_path / "predictions.npz",
        y_B=coll_B["y"], pB_B=coll_B["p_b"], pP_B=coll_B["p_p"], X_B=coll_B["X"],
        u_B=u_B, s_B=s_B, y_C=coll_C["y"], pB_C=coll_C["p_b"], pP_C=coll_C["p_p"],
        X_C=coll_C["X"], u_C=u_C, s_C=s_C,
        y_V=coll_V["y"], pB_V=coll_V["p_b"], pP_V=coll_V["p_p"], X_V=coll_V["X"], s_V=s_V,
        y_T=coll_T["y"], pB_T=coll_T["p_b"], pP_T=coll_T["p_p"], s_T=s_T,
    )
    (run_path / "arms").mkdir()
    torch.save({k: v.detach().cpu() for k, v in head_b.state_dict().items()}, run_path / "arms" / "base.pt")
    torch.save({k: v.detach().cpu() for k, v in head_p.state_dict().items()}, run_path / "arms" / "prompt.pt")

    router_report = {
        "run_id": run_id, "stage1_id": stage1_id, "variant": "iuv",
        "split": {"n_train": n_train, "bounds": {k: list(v) for k, v in bounds.items()},
                  "index_sha256": subset_index_sha},
        "arm_budget": arm_budget,
        "arm_records": {"baseline": arm_b, "prompt": arm_p},
        "router": {"alpha": IUR.ROUTER_ALPHA, "feature_dim": IUR.FEATURE_DIM,
                   "feature_params": feature_params, "ridge": ridge,
                   "threshold": thr_info, "mix": mix_info,
                   "random_salt": IUR.ROUTER_RANDOM_SALT,
                   "random_seed": int(model_seed) ^ IUR.ROUTER_RANDOM_SALT},
        "diagnostics": {
            "auroc_C_vs_u_gt_0": diag_auroc_C, "spearman_C": diag_spearman_C,
            "u_B_positive_rate": float((u_B > 0).mean()), "u_C_positive_rate": float((u_C > 0).mean()),
            "n_B": int(len(u_B)), "n_C": int(len(u_C)),
            "note": "AUROC/Spearman/oracle 仅机制诊断，不参与判定与阈值（预注册 §4.4）",
        },
        "aucs_C": aucs_C, "aucs_val": aucs_V, "aucs_test": aucs_T, "deltas_test": deltas,
        "verdict": verdict, "expand_eligible": bool(expand_ok),
        "oracle_note": ("label_assisted_bce_oracle_diag：label-assisted BCE oracle diagnostic"
                        "（逐样本选 BCE 更小路径）——非 AUC 上界，禁止作为部署指标"),
        "val_note": ("官方 validation 仅用于评估/报告（aucs_val）；不参与 verifier 学习/校准/阈值；"
                     "auc_val_bsi_best = AUC_routed(官方 val)"),
        "transductive_note": ("头级 out-of-sample；backbone 为共享冻结的 canonical Stage-1 产物"
                              "（transductive unlabeled target exposure：目标标签未用于 Stage-1，"
                              "见 multitaskrec/train.py:395；输入特征暴露如实披露）"),
    }
    (run_path / "routing_report.json").write_text(
        json.dumps(router_report, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_doc = {
        "run_id": run_id, "tag": tag, "stage1_id": stage1_id, "prefix_tag": prefix_tag,
        "budgets": dict(budgets), "model_seed": int(model_seed), "epochs": int(epochs),
        "patience": int(patience), "enforce_b": bool(enforce_b), "variant": "iuv",
        "split_bounds": {k: list(v) for k, v in bounds.items()},
        "aucs_test": aucs_T, "aucs_val": aucs_V, "aucs_C": aucs_C, "deltas_test": deltas,
        "verdict": verdict,
        "classification": verdict["classification_by_delta_base"],
        "router_value": verdict["router_value"],
        "expand_eligible": bool(expand_ok),
        "arm_records": {"baseline": arm_b, "prompt": arm_p},
        "diagnostics": router_report["diagnostics"],
        "backbone_sha256_loaded": backbone_sha_loaded,
        "backbone_sha256_before": backbone_sha_before,
        "backbone_sha256_after": backbone_sha_after,
        "backbone_grads_none": backbone_grads_none,
        "env_ids_sha256": protocol.sha256_tensor(env_ids),
        "fingerprint_sha256": fp.get("fingerprint_sha256"),
        "hard_pass": bool(hard_pass), "all_gates_ok": bool(all_gates_ok),
        "commit": protocol.code_commit(), "git": {"commit": protocol.code_commit(), "dirty": bool(git_dirty)},
        "versions": _versions(), "device": str(device),
        "wall_seconds": round(time.time() - t0, 1),
        "peak_vram_mb": (round(torch.cuda.max_memory_allocated(device.index) / 1e6, 1)
                         if device.type == "cuda" else None),
    }
    config_doc = {
        "run_id": run_id, "tag": tag, "stage1_id": stage1_id,
        "data_files": {k: str(v) for k, v in data_files.items()}, "budgets": dict(budgets),
        "model_seed": int(model_seed), "batch_size": int(batch_size), "lr": float(lr),
        "reg_dnn": float(protocol.REG_DNN), "input_size": int(protocol.INPUT_SIZE),
        "embedding_size": int(protocol.EMBEDDING_SIZE), "expert_hidden": list(protocol.EXPERT_HIDDEN),
        "tower_hidden": list(protocol.TOWER_HIDDEN), "enforce_b": bool(enforce_b), "variant": "iuv",
        "router": {"alpha": IUR.ROUTER_ALPHA, "threshold_quantiles": list(IUR.THRESHOLD_QUANTILES),
                   "mix_grid": list(IUR.MIX_GRID), "random_salt": IUR.ROUTER_RANDOM_SALT},
        "commit": protocol.code_commit(), "git": protocol.git_state(),
    }
    gate_doc = {"run_id": run_id, "tag": tag, "enforce_b": bool(enforce_b), "gates": gates,
                "hard_pass": bool(hard_pass), "all_gates_ok": bool(all_gates_ok)}
    (run_path / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "config.json").write_text(json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "gate_report.json").write_text(json.dumps(gate_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_row = {
        "run_id": run_id, "commit": protocol.code_commit(), "tag": tag,
        "auc_val_bsi_best": _fmt6(aucs_V["routed"]),     # 官方 validation 上的 routed AUC（v3.1；独立复算）
        "auc_test_bsi": _fmt6(aucs_T["routed"]),
        "stage1_id": stage1_id,
        **{gid: gates[gid]["verdict"] for gid in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4")},
    }
    protocol.append_summary_row(Path(root) / "SUMMARY.md", summary_row)
    status = json.loads((run_path / "status.json").read_text(encoding="utf-8"))
    status.update({"state": "completed", "finished_at": datetime.now().isoformat(timespec="seconds"),
                   "hard_pass": bool(hard_pass), "all_gates_ok": bool(all_gates_ok),
                   "expand_eligible": bool(expand_ok)})
    (run_path / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[iuv] 完成：run_id={run_id} hard_pass={hard_pass} all_gates_ok={all_gates_ok} "
        f"expand_eligible={expand_ok} wall={metrics_doc['wall_seconds']}s")
    return {"run_id": run_id, "run_dir": str(run_path), "gates": gates, "metrics": metrics_doc,
            "hard_pass": bool(hard_pass), "all_gates_ok": bool(all_gates_ok)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AliCCP 两路径增量效用路由（预注册 docs/superpowers/specs/2026-10-06-aliccp-incremental-utility-verifier-design.md）"
    )
    parser.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    parser.add_argument("--tag", type=str, default="short", choices=["short", "smoke"])
    parser.add_argument("--train-budget", type=int, default=protocol.TRAIN_BUDGET)
    parser.add_argument("--val-budget", type=int, default=protocol.VAL_BUDGET)
    parser.add_argument("--test-budget", type=int, default=protocol.TEST_BUDGET)
    parser.add_argument("--model-seed", type=int, default=protocol.MODEL_SEED)
    parser.add_argument("--stage1-id", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=protocol.STAGE2_EPOCHS)
    parser.add_argument("--patience", type=int, default=protocol.STAGE2_PATIENCE)
    parser.add_argument("--no-enforce-b", action="store_true", help="只记录 B 类门禁")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="显式设备（如 cpu；默认 cuda:<gpu>）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(f"cuda:{args.gpu}")
    root = Path(args.root)
    budgets = {"train": args.train_budget, "val": args.val_budget, "test": args.test_budget}
    prefix_tag = (protocol.PREFIX_TAG
                  if (args.train_budget, args.val_budget, args.test_budget)
                  == (protocol.TRAIN_BUDGET, protocol.VAL_BUDGET, protocol.TEST_BUDGET)
                  else f"p{args.train_budget}-v{args.val_budget}-t{args.test_budget}")
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-iuv-{args.tag}.log"
    with open(log_path, "w", encoding="utf-8") as log_file, contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
        print(f"command=iuv tag={args.tag} prefix_tag={prefix_tag} budgets={budgets}")
        print(f"model_seed={args.model_seed} device={device} commit={protocol.code_commit()} root={root}")
        result = run_router(
            root=root, stage1_id=args.stage1_id, data_files=protocol.DATA_FILES, budgets=budgets,
            prefix_tag=prefix_tag, model_seed=args.model_seed, epochs=args.epochs, patience=args.patience,
            tag=args.tag, device=device, enforce_b=(args.tag != "smoke") and not args.no_enforce_b,
        )
        verdicts = {gate: value["verdict"] for gate, value in result["gates"].items()}
        print(f"gates: {json.dumps(verdicts, ensure_ascii=False)}")
        print(f"run_id={result['run_id']} hard_pass={result['hard_pass']} all_gates_ok={result['all_gates_ok']}")
        v = result["metrics"]["verdict"]
        d = result["metrics"]["deltas_test"]
        print(f"classification={v['classification_by_delta_base']} router_value={v['router_value']} "
              f"expand_eligible={result['metrics']['expand_eligible']} "
              f"d_base={_fmt(d['base'], signed=True)} d_prompt={_fmt(d['prompt'], signed=True)} "
              f"d_mix={_fmt(d['mix'], signed=True)}")
    print(f"日志已写入: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
