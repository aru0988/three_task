"""AliCCP 基准编排（spec 7/8）：stage1 训练与产物、stage2 加载 + 真冻结 + 门禁。

约束（spec 13）：不改动 multitaskrec/*；阶段 1 训练循环直接复用仓库 MPTRecTrainManager，
仅以子类方式记录 cluster 事件与逐 epoch val AUC（不改变任何计算行为）。
"""
from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

import torch
from torch.utils.data import DataLoader

from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager

from . import metrics, protocol


class ClusteringArm(NamedTuple):
    """Stage-1 聚类赋值规则的可插拔描述（预注册 §6）。

    默认路径（`run_stage1(clustering_arm=None)`）不引用任何 arm：产物、config 哈希与
    meta schema 与基线逐位一致；arm 由 CLI 显式选择时才注入（含 cfg 标识 → 新 stage1_id）。
    """

    name: str
    manager_factory: Callable[..., MPTRecTrainManager]


class RecordingMPTRecTrainManager(MPTRecTrainManager):
    """记录层：cluster 事件 + 逐 epoch val AUC + best epoch。不改变训练行为（spec 8.1）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_epoch_aucs = []
        self.cluster_events = []

    def evaluation_two_task(self, data_loader):
        aucs = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.val_epoch_aucs.append([float(a) for a in aucs])
        return aucs

    def cluster_2(self):
        old_env_ids = self.env_ids.clone()
        result = super().cluster_2()
        counts = torch.bincount(result, minlength=protocol.NUM_ENVS)
        self.cluster_events.append(
            {
                "epoch": len(self.val_epoch_aucs) + 1,  # cluster 在该 epoch 的 val 评估之前调用
                "diff_num": int((old_env_ids != result).sum()),
                "env_0": int(counts[0]),
                "env_1": int(counts[1]),
            }
        )
        return result

    def best_epoch(self) -> int:
        """与仓库选点规则一致：sum(auc_val) 首次取到最大值者（spec 7.5）。"""
        sums = [a + b for a, b in self.val_epoch_aucs]
        return int(max(range(len(sums)), key=lambda i: sums[i])) + 1


def _versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}


def _reset_peak_vram(device) -> None:
    """torch 2.6 在 CUDA 未初始化时，_cuda_resetPeakMemoryStats 对任何实参都报 Invalid device argument；
    必须先 torch.cuda.init()（实测确认，2026-10-03 smoke）。"""
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)


def _stage1_cfg(
    *, prefix_tag, budgets, model_seed, env_seed, epochs, patience, batch_size, lr,
    uni_coe, env_coe, reg_embedding, reg_dnn, embedding_size, input_size,
    expert_hidden, tower_hidden, dropout, vocab,
) -> dict:
    return {
        "prefix_tag": prefix_tag,
        "budgets": dict(budgets),
        "model_seed": int(model_seed),
        "env_seed": int(env_seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "uni_coe": float(uni_coe),
        "env_coe": float(env_coe),
        "reg_embedding": float(reg_embedding),
        "reg_dnn": float(reg_dnn),
        "embedding_size": int(embedding_size),
        "input_size": int(input_size),
        "expert_hidden": list(expert_hidden),
        "tower_hidden": list(tower_hidden),
        "dropout": [float(d) for d in dropout],
        "num_tasks": protocol.NUM_TASKS,
        "vocab": {str(k): int(v) for k, v in sorted(vocab.items())},
    }


def stage1_id_for(meta: dict) -> str:
    return protocol.make_stage1_id(
        meta["fingerprint_sha256"], meta["model_seed"], meta["epochs"], meta["config_hash"]
    )


def _loaders(data_files, budgets, batch_size):
    datasets = {split: AliCCPDataset(data_files[split], budgets[split]) for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(datasets[split], batch_size=batch_size, shuffle=False, num_workers=0)
        for split in ("train", "val", "test")
    }
    return datasets, loaders


def _budget_of(datasets, budgets) -> bool:
    return all(len(datasets[split]) == budgets[split] for split in ("train", "val", "test"))


@torch.no_grad()
def env_accuracy_probe(model, loader, env_ids, batch_size, device, max_batches=200) -> float:
    """M1：训练集前 400k 行（前 200 个 batch，确定性）上 env_pred 与最终 env_ids 的一致率。"""
    model.eval()
    correct = total = 0
    for step, (_, _, _, features) in enumerate(loader):
        if step >= max_batches:
            break
        for key in features:
            features[key] = features[key].to(device)
        env_pred = model(features)["env_pred"]
        ids = env_ids[batch_size * step : batch_size * (step + 1)].to(device)
        n = len(ids)
        correct += int((env_pred.argmax(dim=1)[:n] == ids).sum())
        total += n
    return correct / max(1, total)


@torch.no_grad()
def env_accuracy_and_balance_probe(model, loader, env_ids, batch_size, device, max_batches=200):
    """单次遍历同时返回 (acc, balanced_acc)：acc 口径与 `env_accuracy_probe` 完全一致（整数计数）；
    balanced_acc = macro recall，用于披露退化分配下多数类准确率的误导性（预注册 §6，仅 arm 路径使用）。"""
    model.eval()
    preds, ids_list = [], []
    for step, (_, _, _, features) in enumerate(loader):
        if step >= max_batches:
            break
        for key in features:
            features[key] = features[key].to(device)
        env_pred = model(features)["env_pred"]
        ids = env_ids[batch_size * step : batch_size * (step + 1)].to(device)
        n = len(ids)
        preds.append(env_pred.argmax(dim=1)[:n].cpu())
        ids_list.append(ids.cpu())
    pred = torch.cat(preds)
    ids = torch.cat(ids_list)
    acc = int((pred == ids).sum()) / max(1, int(ids.numel()))
    return acc, metrics.env_balanced_accuracy(pred, ids)


def run_stage1(
    *,
    root,
    data_files,
    budgets,
    prefix_tag,
    model_seed,
    env_seed,
    epochs,
    patience,
    device,
    log=print,
    vocab=None,
    expert_hidden=protocol.EXPERT_HIDDEN,
    tower_hidden=protocol.TOWER_HIDDEN,
    embedding_size=protocol.EMBEDDING_SIZE,
    input_size=protocol.INPUT_SIZE,
    batch_size=protocol.BATCH_SIZE,
    lr=protocol.LR,
    uni_coe=protocol.UNI_COE,
    env_coe=protocol.ENV_COE,
    reg_embedding=protocol.REG_EMBEDDING,
    reg_dnn=protocol.REG_DNN,
    dropout=protocol.DROPOUT,
    log_batches=200,
    clustering_arm: ClusteringArm | None = None,
) -> dict:
    """阶段 1：训练 → 选点 → 单次 test 评估 → 保存内容寻址 Stage-1 产物（spec 7.1/8.1）。

    `clustering_arm=None`（默认）= 仓库原聚类规则，行为与产物逐位不变；
    非 None 时仅替换聚类赋值规则并在 cfg/meta 中记录（预注册 §6）。"""
    t0 = time.time()
    _reset_peak_vram(device)
    log(f"[stage1] 开始：prefix={prefix_tag} budgets={budgets} model_seed={model_seed} env_seed={env_seed}")
    protocol.seed_model(model_seed)

    datasets, loaders = _loaders(data_files, budgets, batch_size)
    files_budgets = {split: (Path(data_files[split]), budgets[split]) for split in ("train", "val", "test")}
    fp, created = protocol.ensure_fingerprint(root, prefix_tag, files_budgets)
    log(f"[stage1] 前缀指纹 {'构建并落盘' if created else '读取校验'}：{fp['fingerprint_sha256'][:16]}")
    if not _budget_of(datasets, budgets):
        raise AssertionError("A4: 样本数与预算不符")
    for split in ("train", "val", "test"):
        protocol.verify_label_counts(datasets[split], budgets[split], fp["label_counts"][split])
    log("[stage1] A4 通过：三切分样本数与标签计数与指纹逐项相等")

    vocab = dict(vocab) if vocab is not None else protocol.build_vocab()
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=embedding_size,
        input_size=input_size,
        expert_dnn_hidden_units=list(expert_hidden),
        tower_dnn_hidden_units=list(tower_hidden),
        dropout=list(dropout),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device,
    ).to(device)

    env_ids = protocol.make_env_ids(len(datasets["train"]), env_seed)
    manager_cls = RecordingMPTRecTrainManager if clustering_arm is None else clustering_arm.manager_factory
    manager = manager_cls(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        env_ids=env_ids,
        task_name=["CTR", "CVR"],
        lr=lr,
        batch_size=batch_size,
        uni_coe=uni_coe,
        env_coe=env_coe,
        epochs=epochs,
        patience=patience,
    )
    manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    test_aucs = manager.evaluation_two_task(loaders["test"])  # test 只评一次，不参与选点（spec 7.5）
    if clustering_arm is None:
        env_acc = env_accuracy_probe(model, loaders["train"], manager.env_ids, batch_size, device, log_batches)
        env_bal_acc = None
    else:
        env_acc, env_bal_acc = env_accuracy_and_balance_probe(
            model, loaders["train"], manager.env_ids, batch_size, device, log_batches
        )

    cfg = _stage1_cfg(
        prefix_tag=prefix_tag, budgets=budgets, model_seed=model_seed, env_seed=env_seed,
        epochs=epochs, patience=patience, batch_size=batch_size, lr=lr, uni_coe=uni_coe,
        env_coe=env_coe, reg_embedding=reg_embedding, reg_dnn=reg_dnn, embedding_size=embedding_size,
        input_size=input_size, expert_hidden=expert_hidden, tower_hidden=tower_hidden,
        dropout=dropout, vocab=vocab,
    )
    if clustering_arm is not None:
        cfg["clustering"] = clustering_arm.name  # 身份的一部分：进 config_hash → 新 stage1_id（预注册 §3）
    cfg_sha = protocol.config_hash(cfg)
    sid = protocol.make_stage1_id(fp["fingerprint_sha256"], model_seed, epochs, cfg_sha)
    backbone_sha = protocol.backbone_sha256(model)
    env_ids_final = manager.env_ids.detach().cpu()

    per_epoch = []
    for i in range(len(manager.val_epoch_aucs)):
        per_epoch.append(
            {
                "epoch": i + 1,
                "auc_val_ctr": manager.val_epoch_aucs[i][0],
                "auc_val_cvr": manager.val_epoch_aucs[i][1],
                "uni_loss_0": float(manager.uni_loss_0_list[i]),
                "uni_loss_1": float(manager.uni_loss_1_list[i]),
                "fuse_loss_0": float(manager.fused_loss_0_list[i]),
                "fuse_loss_1": float(manager.fused_loss_1_list[i]),
                "env_loss": float(manager.env_loss_list[i]),
            }
        )
    best_epoch = manager.best_epoch()
    peak_vram = (
        round(torch.cuda.max_memory_allocated(device.index) / 1e6, 1) if device.type == "cuda" else None
    )
    meta = {
        "stage1_id": sid,
        "prefix_tag": prefix_tag,
        "budgets": dict(budgets),
        "model_seed": int(model_seed),
        "env_seed": int(env_seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "config_hash": cfg_sha,
        "config": cfg,
        "fingerprint_sha256": fp["fingerprint_sha256"],
        "per_epoch": per_epoch,
        "best_epoch": best_epoch,
        "best_val_auc_ctr": manager.val_epoch_aucs[best_epoch - 1][0],
        "best_val_auc_cvr": manager.val_epoch_aucs[best_epoch - 1][1],
        "test_auc_ctr": float(test_aucs[0]),
        "test_auc_cvr": float(test_aucs[1]),
        "env_acc": float(env_acc),
        "cluster_events": manager.cluster_events,
        "backbone_sha256": backbone_sha,
        "env_ids_sha256": protocol.sha256_tensor(env_ids_final),
        "commit": protocol.code_commit(),
        "git": protocol.git_state(),
        "versions": _versions(),
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 1),
        "peak_vram_mb": peak_vram,
    }
    if clustering_arm is not None:
        meta["clustering"] = clustering_arm.name
        meta["cluster_diagnostics"] = list(getattr(manager, "cluster_diagnostics", []))
        meta["env_bal_acc"] = env_bal_acc
    protocol.save_stage1(
        root, sid,
        backbone_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        env_ids=env_ids_final,
        meta=meta,
    )
    log(f"[stage1] 完成：stage1_id={sid} best_epoch={best_epoch} "
        f"test_auc_ctr={meta['test_auc_ctr']:.4f} test_auc_cvr={meta['test_auc_cvr']:.4f} "
        f"env_acc={env_acc:.4f} wall={meta['wall_seconds']}s")
    if clustering_arm is not None:
        log(f"[stage1] clustering={clustering_arm.name} env_bal_acc={env_bal_acc} "
            f"cluster_events={meta['cluster_events']}")
    return meta


@torch.no_grad()
def evaluate_newtask(newtask, model, loader, device) -> float:
    """阶段 2 评测口径与仓库 AliCCP_NewTask.py 的 evaluation() 一致（spec 7.5）。"""
    newtask.eval()
    y_true, y_hat = [], []
    for _, _, y, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        y_true.append(y)
        y_hat.append(pred)
    return metrics.auc_score(torch.cat(y_true), torch.cat(y_hat))


@torch.no_grad()
def newtask_gate_mean(newtask, model, loader, device) -> list:
    """M3：val 上 NewTask.gate_network 的平均输出（2 维），no_grad 累积（spec 9.2）。"""
    newtask.eval()
    total = None
    count = 0
    for _, _, _, features in loader:
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, _, _, _ = model.get_infos(features)
        gate = newtask.gate_network(dnn_input)
        total = gate.sum(dim=0) if total is None else total + gate.sum(dim=0)
        count += gate.shape[0]
    return [float(v) / max(1, count) for v in total]


def run_stage2(
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
    enforce_b=True,
    log=print,
    vocab=None,
    expert_hidden=protocol.EXPERT_HIDDEN,
    tower_hidden=protocol.TOWER_HIDDEN,
    embedding_size=protocol.EMBEDDING_SIZE,
    input_size=protocol.INPUT_SIZE,
    batch_size=protocol.BATCH_SIZE,
    lr=protocol.LR,
    reg_dnn=protocol.REG_DNN,
    newtask_rep_dim=None,
    run_id=None,
) -> dict:
    """阶段 2：只从 --stage1-dir 加载 backbone，真冻结三件套，训练 NewTask 头，判定门禁（spec 7/8/10）。"""
    t0 = time.time()
    _reset_peak_vram(device)
    log(f"[stage2] 开始：stage1_id={stage1_id} tag={tag} model_seed={model_seed} enforce_b={enforce_b}")
    protocol.seed_model(model_seed)  # 独立进程重播种：头初始化与阶段 1 轨迹解耦（spec 6.1）

    datasets, loaders = _loaders(data_files, budgets, batch_size)

    # ---- A2 / A4：指纹与标签对齐 ----
    prefix_sha_ok = True
    fp = None
    try:
        fp = protocol.load_fingerprint(root, prefix_tag)
        protocol.verify_fingerprint(fp)
    except (AssertionError, FileNotFoundError) as exc:  # noqa: BLE001
        prefix_sha_ok = False
        log(f"[stage2] A2 前缀指纹校验失败：{exc}")
        if fp is None:
            fp = {"fingerprint_sha256": None, "label_counts": {}}
    len_ok = _budget_of(datasets, budgets)
    counts_ok = True
    for split in ("train", "val", "test"):
        try:
            protocol.verify_label_counts(datasets[split], budgets[split], fp["label_counts"][split])
        except (AssertionError, KeyError) as exc:  # noqa: BLE001
            counts_ok = False
            log(f"[stage2] A4 标签计数失败（{split}）：{exc}")

    # ---- 加载固定 Stage-1 产物（唯一 backbone 来源）----
    art = protocol.load_stage1(root, stage1_id)
    art_meta = art["meta"]
    env_ids = art["env_ids"]
    fingerprint_sha_match = art_meta.get("fingerprint_sha256") == fp.get("fingerprint_sha256")
    env_ids_sha_match = protocol.sha256_tensor(env_ids) == art_meta.get("env_ids_sha256")

    vocab = dict(vocab) if vocab is not None else protocol.build_vocab()
    model = MPTRec(
        num_tasks=protocol.NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=embedding_size,
        input_size=input_size,
        expert_dnn_hidden_units=list(expert_hidden),
        tower_dnn_hidden_units=list(tower_hidden),
        dropout=list(protocol.DROPOUT),
        reg_embedding=protocol.REG_EMBEDDING,
        reg_dnn=reg_dnn,
        device=device,
    ).to(device)
    model.load_state_dict(art["backbone_state"])
    backbone_sha_loaded = protocol.backbone_sha256(model)
    backbone_sha_matches_stage1 = backbone_sha_loaded == art_meta.get("backbone_sha256")
    stage1_id_recorded = art_meta.get("stage1_id") == stage1_id

    # ---- 真冻结三件套（spec 7.3）----
    protocol.freeze_backbone(model)
    backbone_sha_before = protocol.backbone_sha256(model)
    log(f"[stage2] backbone 已加载并冻结：sha={backbone_sha_before[:16]}（A6 匹配={backbone_sha_matches_stage1}）")

    rep_dim = int(newtask_rep_dim) if newtask_rep_dim is not None else int(list(expert_hidden)[-1])
    if rep_dim != int(list(expert_hidden)[-1]):
        raise AssertionError("NewTask rep_dim 必须等于 expert_dnn_hidden_units[-1]（env_embs 维度约束）")
    newtask = NewTask(
        input_size=input_size,
        rep_dim=rep_dim,
        tower_dnn_hidden_units=list(tower_hidden),
        reg_dnn=reg_dnn,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=lr)
    loss_func = torch.nn.BCELoss()

    best_auc, best_epoch, best_weight, earlystop_count = 0.0, None, None, 0
    epoch_records = []
    for epoch in range(1, epochs + 1):
        newtask.train()
        loss_sum, steps = 0.0, 0
        for _, _, y, features in loaders["train"]:
            for key in features:
                features[key] = features[key].to(device)
            with torch.no_grad():  # 计算图级冻结（spec 7.3 第 3 条）；backbone 恒为 eval
                dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
            pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            steps += 1
        val_auc = evaluate_newtask(newtask, model, loaders["val"], device)
        epoch_records.append({"epoch": epoch, "train_loss": loss_sum / max(1, steps), "val_auc_bsi": val_auc})
        log(f"[stage2] Epoch:{epoch} train_loss={loss_sum / max(1, steps):.4f} AUC-Val-BSI:{val_auc:.4f}")
        if val_auc > best_auc:
            best_auc, best_epoch, earlystop_count = val_auc, epoch, 0
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            log(f"[stage2] EarlyStopping count {earlystop_count}")
            if earlystop_count == patience:
                log(f"[stage2] EarlyStopping at epoch {epoch}")
                break
    newtask.load_state_dict(best_weight)
    test_auc = evaluate_newtask(newtask, model, loaders["test"], device)  # test 只评一次（spec 7.5）
    gate_mean = newtask_gate_mean(newtask, model, loaders["val"], device)

    # ---- A1：冻结完整性 ----
    backbone_sha_after = protocol.backbone_sha256(model)
    try:
        protocol.assert_no_grads(model)
        backbone_grads_none = True
    except AssertionError as exc:  # noqa: BLE001
        backbone_grads_none = False
        log(f"[stage2] A1 失败：{exc}")

    # ---- 门禁判定（spec 10）----
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
    b_gates = metrics.evaluate_b_gates(
        {
            "auc_val_ctr": art_meta.get("best_val_auc_ctr", 0.0),
            "auc_val_cvr": art_meta.get("best_val_auc_cvr", 0.0),
            "auc_val_bsi_best": best_auc,
            "auc_test_bsi": test_auc,
            "gate_mean": gate_mean,
            "cluster_events": art_meta.get("cluster_events", []),
            "train_size": budgets["train"],
        }
    )
    gates = {**a_gates, **b_gates}
    passed = metrics.hard_pass(a_gates, enforce_b=enforce_b, b_gates=b_gates)

    # ---- 产物与 SUMMARY（spec 11）----
    if run_id is None:
        run_id = protocol.make_run_id(
            datetime.now(), prefix_tag=prefix_tag, model_seed=model_seed, tag=tag, commit=protocol.code_commit()
        )
    run_path = protocol.run_dir(root, run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in newtask.state_dict().items()}, run_path / "newtask.pt")

    metrics_doc = {
        "run_id": run_id,
        "tag": tag,
        "stage1_id": stage1_id,
        "prefix_tag": prefix_tag,
        "budgets": dict(budgets),
        "model_seed": int(model_seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "enforce_b": bool(enforce_b),
        "best_epoch": best_epoch,
        "best_val_auc_bsi": float(best_auc),
        "test_auc_bsi": float(test_auc),
        "gate_mean": gate_mean,
        "per_epoch": epoch_records,
        "backbone_sha256_loaded": backbone_sha_loaded,
        "backbone_sha256_before": backbone_sha_before,
        "backbone_sha256_after": backbone_sha_after,
        "backbone_grads_none": backbone_grads_none,
        "env_ids_sha256": protocol.sha256_tensor(env_ids),
        "fingerprint_sha256": fp.get("fingerprint_sha256"),
        "hard_pass": bool(passed),
        "commit": protocol.code_commit(),
        "git": protocol.git_state(),
        "versions": _versions(),
        "device": str(device),
        "wall_seconds": round(time.time() - t0, 1),
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated(device.index) / 1e6, 1) if device.type == "cuda" else None
        ),
    }
    config_doc = {
        "run_id": run_id,
        "tag": tag,
        "stage1_id": stage1_id,
        "data_files": {k: str(v) for k, v in data_files.items()},
        "budgets": dict(budgets),
        "model_seed": int(model_seed),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "reg_dnn": float(reg_dnn),
        "newtask_rep_dim": rep_dim,
        "expert_hidden": list(expert_hidden),
        "tower_hidden": list(tower_hidden),
        "input_size": int(input_size),
        "embedding_size": int(embedding_size),
        "enforce_b": bool(enforce_b),
        "commit": protocol.code_commit(),
        "git": protocol.git_state(),
    }
    gate_doc = {"run_id": run_id, "tag": tag, "enforce_b": bool(enforce_b), "gates": gates, "hard_pass": bool(passed)}
    (run_path / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "config.json").write_text(json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "gate_report.json").write_text(json.dumps(gate_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_row = {
        "run_id": run_id,
        "commit": protocol.code_commit(),
        "tag": tag,
        "auc_val_bsi_best": f"{best_auc:.6f}",
        "auc_test_bsi": f"{test_auc:.6f}",
        "stage1_id": stage1_id,
        **{gate_id: gates[gate_id]["verdict"] for gate_id in ("A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4")},
    }
    protocol.append_summary_row(Path(root) / "SUMMARY.md", summary_row)
    log(f"[stage2] 完成：run_id={run_id} AUC-Val-BSI(best)={best_auc:.4f} AUC-Test-BSI={test_auc:.4f} "
        f"hard_pass={passed} wall={metrics_doc['wall_seconds']}s")
    return {"run_id": run_id, "run_dir": str(run_path), "gates": gates, "metrics": metrics_doc, "hard_pass": bool(passed)}
