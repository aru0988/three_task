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

import torch
from torch.utils.data import DataLoader

from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager

from . import metrics, protocol, residual_prompt as RP, rp_pinned as RPP, rp_uncond as RPU


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
) -> dict:
    """阶段 1：训练 → 选点 → 单次 test 评估 → 保存内容寻址 Stage-1 产物（spec 7.1/8.1）。"""
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
    manager = RecordingMPTRecTrainManager(
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
    env_acc = env_accuracy_probe(model, loaders["train"], manager.env_ids, batch_size, device, log_batches)

    cfg = _stage1_cfg(
        prefix_tag=prefix_tag, budgets=budgets, model_seed=model_seed, env_seed=env_seed,
        epochs=epochs, patience=patience, batch_size=batch_size, lr=lr, uni_coe=uni_coe,
        env_coe=env_coe, reg_embedding=reg_embedding, reg_dnn=reg_dnn, embedding_size=embedding_size,
        input_size=input_size, expert_hidden=expert_hidden, tower_hidden=tower_hidden,
        dropout=dropout, vocab=vocab,
    )
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
    protocol.save_stage1(
        root, sid,
        backbone_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
        env_ids=env_ids_final,
        meta=meta,
    )
    log(f"[stage1] 完成：stage1_id={sid} best_epoch={best_epoch} "
        f"test_auc_ctr={meta['test_auc_ctr']:.4f} test_auc_cvr={meta['test_auc_cvr']:.4f} "
        f"env_acc={env_acc:.4f} wall={meta['wall_seconds']}s")
    return meta


@torch.no_grad()
def evaluate_newtask(newtask, model, loader, device, shuffler=None, split=None) -> float:
    """阶段 2 评测口径与仓库 AliCCP_NewTask.py 的 evaluation() 一致（spec 7.5）。

    `shuffler=None`（baseline/学习 correct 臂/钉死 correct 臂）时逐字保持钉死调用；
    pinned-shuffled 臂经逐 batch 条件错排分派。
    """
    newtask.eval()
    y_true, y_hat = [], []
    for step, (_, _, y, features) in enumerate(loader):
        for key in features:
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
        pred = RPP.forward_with_conditioning(newtask, dnn_input, gen_rep, spec_reps, env_embs,
                                             shuffler, split, step)
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
    variant=RP.BASELINE_VARIANT,
    prompt_reference_newtask=None,
) -> dict:
    """阶段 2：只从 --stage1-dir 加载 backbone，真冻结三件套，训练 NewTask 头，判定门禁（spec 7/8/10）。

    `variant="residual-prompt"` 时为学习门控残差 prompt 头（spec 2026-10-03 / seed-2 复现）；
    `variant="residual-prompt-pinned"` / `"residual-prompt-pinned-shuffled"` 时为 α 钉死条件对应消融头
    （spec 2026-10-05 alpha-pinned；α = 非参数 buffer、排除于优化器）；
    `variant="residual-prompt-uncond"` 时为钉死 α 无条件/全局条件臂（spec 2026-10-05 unconditional；
    条件 = 冻结常量向量 c）；默认 `baseline` 与未启用时行为一致（基线臂除恒为 "baseline" 的
    `variant` 字段外键集不变）。
    """
    t0 = time.time()
    _reset_peak_vram(device)
    log(f"[stage2] 开始：stage1_id={stage1_id} tag={tag} model_seed={model_seed} enforce_b={enforce_b}")
    if variant != RP.BASELINE_VARIANT and prompt_reference_newtask is None:
        raise ValueError("处理臂必须提供 --prompt-reference-newtask（M0/PA7 依赖参照身份，拒绝出数）")
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
    rng_before = torch.get_rng_state() if variant != RP.BASELINE_VARIANT else None
    if variant == RPP.VARIANT_PINNED_SHUFFLED:           # 钉死 α shuffled 臂：错排头 + 隔离错排器
        newtask = RPP.build_pinned_shuffled_newtask(input_size=input_size, rep_dim=rep_dim,
                                                    tower_dnn_hidden_units=list(tower_hidden),
                                                    reg_dnn=reg_dnn, device=device).to(device)
    elif variant == RPP.VARIANT_PINNED:                  # 钉死 α correct 臂
        newtask = RPP.build_pinned_newtask(input_size=input_size, rep_dim=rep_dim,
                                           tower_dnn_hidden_units=list(tower_hidden),
                                           reg_dnn=reg_dnn, device=device).to(device)
    elif variant == RPU.VARIANT_UNCOND:                  # 钉死 α 无条件/全局条件臂（常量向量 c）
        newtask = RPU.build_uncond_newtask(input_size=input_size, rep_dim=rep_dim,
                                           tower_dnn_hidden_units=list(tower_hidden),
                                           reg_dnn=reg_dnn, device=device).to(device)
    else:
        newtask = RP.build_newtask(variant, input_size=input_size, rep_dim=rep_dim,
                                   tower_dnn_hidden_units=list(tower_hidden),
                                   reg_dnn=reg_dnn, device=device).to(device)
    shuffler = RPP.ConditioningShuffler() if variant == RPP.VARIANT_PINNED_SHUFFLED else None
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=lr)
    loss_func = torch.nn.BCELoss()
    audit = None
    if variant in RPP.PINNED_VARIANTS:                    # PA1–PA4 运行期审计（钉死臂专用）
        audit = RPP.PinnedPromptAudit(newtask, rng_before=rng_before, rng_after=torch.get_rng_state(),
                                      input_size=input_size, rep_dim=rep_dim,
                                      tower_dnn_hidden_units=list(tower_hidden), reg_dnn=reg_dnn,
                                      device=device, shuffler=shuffler, optimizer=optimizer)
    elif variant == RPU.VARIANT_UNCOND:                  # UA1–UA5/UA11 运行期审计（无条件臂专用）
        audit = RPU.UncondPromptAudit(newtask, rng_before=rng_before, rng_after=torch.get_rng_state(),
                                      input_size=input_size, rep_dim=rep_dim,
                                      tower_dnn_hidden_units=list(tower_hidden), reg_dnn=reg_dnn,
                                      device=device, optimizer=optimizer)
    elif variant != RP.BASELINE_VARIANT:                 # M0/G1/G2/G3 运行期审计（学习臂专用）
        audit = RP.PromptAudit(newtask, rng_before=rng_before, rng_after=torch.get_rng_state(),
                               input_size=input_size, rep_dim=rep_dim,
                               tower_dnn_hidden_units=list(tower_hidden), reg_dnn=reg_dnn,
                               device=device)

    best_auc, best_epoch, best_weight, earlystop_count = 0.0, None, None, 0
    epoch_records = []
    for epoch in range(1, epochs + 1):
        newtask.train()
        loss_sum, steps = 0.0, 0
        for step, (_, _, y, features) in enumerate(loaders["train"]):
            for key in features:
                features[key] = features[key].to(device)
            with torch.no_grad():  # 计算图级冻结（spec 7.3 第 3 条）；backbone 恒为 eval
                dnn_input, gen_rep, spec_reps, env_embs = model.get_infos(features)
            if audit is not None and epoch == 1 and step == 0:
                audit.init_forward_check(dnn_input, gen_rep, spec_reps, env_embs)   # G2/PA4：真实首个 batch
            pred = RPP.forward_with_conditioning(newtask, dnn_input, gen_rep, spec_reps, env_embs,
                                                 shuffler, "train", step)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            if audit is not None:
                audit.after_backward(epoch=epoch, step=step)             # 逐 epoch 首个 batch 梯度探针
            optimizer.step()
            loss_sum += float(loss)
            steps += 1
        val_auc = evaluate_newtask(newtask, model, loaders["val"], device, shuffler=shuffler, split="val")
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
    test_auc = evaluate_newtask(newtask, model, loaders["test"], device, shuffler=shuffler, split="test")  # test 只评一次（spec 7.5）
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

    # ---- 处理臂：机制诊断与臂级判定（不并入 hard_pass；预注册第 4/5 节）----
    arm = prompt_probe = prompt_diagnostics = prompt_params = prompt_reference = None
    shuffle_report = None
    if variant != RP.BASELINE_VARIANT:
        alpha_final = float(newtask.prompt_gate.detach())
        prompt_probe = audit.result(alpha_final)
        if variant == RPP.VARIANT_PINNED_SHUFFLED:  # 诊断遍历同样经错排分派（train/val/test 同一原则）
            prompt_diagnostics = RPP.evaluate_prompt_diagnostics_shuffled(
                newtask, model, loaders["val"], device, shuffler, split="val")
        else:
            prompt_diagnostics = RP.evaluate_prompt_diagnostics(newtask, model, loaders["val"], device)
        prompt_params = RP.param_report(newtask)
        prompt_reference = RP.reference_head_stats(
            prompt_reference_newtask, model, loaders["val"], device, input_size=input_size,
            rep_dim=rep_dim, tower_dnn_hidden_units=list(tower_hidden), reg_dnn=reg_dnn)
        protocol_ok = all(g["verdict"] in ("PASS", "SKIP") for g in a_gates.values())
        if variant in RPP.PINNED_VARIANTS:
            if variant == RPP.VARIANT_PINNED_SHUFFLED:
                shuffle_report = shuffler.finalize()  # 全部用毕后一次性定稿（digest 稳定）
            arm = RPP.pinned_arm_verdict(auc_test=test_auc, auc_val=best_auc, probe=prompt_probe,
                                         val_stats=prompt_diagnostics, params=prompt_params,
                                         reference=prompt_reference, protocol_ok=protocol_ok,
                                         variant=variant, shuffle_report=shuffle_report,
                                         budgets=budgets, batch_size=batch_size)
        elif variant == RPU.VARIANT_UNCOND:
            arm = RPU.uncond_arm_verdict(auc_test=test_auc, auc_val=best_auc, probe=prompt_probe,
                                         val_stats=prompt_diagnostics, params=prompt_params,
                                         reference=prompt_reference, protocol_ok=protocol_ok,
                                         variant=variant)
        else:
            arm = RP.arm_verdict(auc_test=test_auc, auc_val=best_auc, probe=prompt_probe,
                                 val_stats=prompt_diagnostics, params=prompt_params,
                                 reference=prompt_reference, protocol_ok=protocol_ok)
        log(f"[stage2] variant={variant} classification={arm['classification']} "
            f"subreason={arm['subreason']} alpha_final={alpha_final:.6f}")

    # ---- 产物与 SUMMARY（spec 11）----
    if run_id is None:
        run_id = protocol.make_run_id(
            datetime.now(), prefix_tag=prefix_tag, model_seed=model_seed, tag=tag, commit=protocol.code_commit()
        )
        run_id += RPU.arm_suffix_for(variant)
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
        "variant": variant,
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
        "variant": variant,
        "commit": protocol.code_commit(),
        "git": protocol.git_state(),
    }
    gate_doc = {"run_id": run_id, "tag": tag, "enforce_b": bool(enforce_b), "gates": gates, "hard_pass": bool(passed)}
    if arm is not None:                       # 臂级判定随附落盘；不并入 hard_pass / gates 语义
        metrics_doc["prompt_hidden"] = RP.PROMPT_HIDDEN
        metrics_doc["rp_arm"] = arm
        config_doc["prompt_hidden"] = RP.PROMPT_HIDDEN
        gate_doc["residual_prompt"] = arm
    (run_path / "metrics.json").write_text(json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "config.json").write_text(json.dumps(config_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_path / "gate_report.json").write_text(json.dumps(gate_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if arm is not None:
        prompt_doc = {
            "run_id": run_id, "stage1_id": stage1_id, "variant": variant,
            "prompt_hidden": RP.PROMPT_HIDDEN,
            "construction_identity": prompt_probe["construction"],
            "init_forward": prompt_probe["init_forward"],
            "grad_probe": prompt_probe["grad_probe"],
            "alpha_final": prompt_probe["alpha_final"],
            "gate": prompt_diagnostics["gate"],
            "val_stats": prompt_diagnostics["streams"],
            "dispersion": prompt_diagnostics["dispersion"],
            "reference_dispersion": prompt_reference,
            "source_gates": {"gate_mean": gate_mean,
                             "cluster_events": art_meta.get("cluster_events", []),
                             "env_acc": art_meta.get("env_acc")},
            "params": prompt_params, "arm": arm,
        }
        if variant in RPP.PINNED_VARIANTS:     # 钉死 α 证明块（PA1/PA2 + 固定 α 证据；学习臂键集不变）
            prompt_doc["pin"] = prompt_probe["pin"]
        elif variant == RPU.VARIANT_UNCOND:    # 钉死 α 证明块 + 无条件常量证明块（UA5/CV）
            prompt_doc["pin"] = prompt_probe["pin"]
            prompt_doc["uncond"] = prompt_probe["uncond"]
        if shuffle_report is not None:        # 错排完整性/门禁随附落盘（PG1–PG7；不并入 hard_pass）
            prompt_doc["shuffle"] = {"report": shuffle_report, "gates": arm["shuffle"]["gates"],
                                     "pass": arm["shuffle"]["pass"], "base_seed": RPP.COND_PERM_SEED}
        (run_path / "prompt_report.json").write_text(
            json.dumps(prompt_doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
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
