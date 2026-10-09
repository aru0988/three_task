"""阶段 1 / 阶段 2 统一编排。数据集差异全部经 DatasetProfile 钩子注入。

固定执行顺序（位级复现依赖，不得更改）：

阶段 1：prepare → seed_model → make_env_ids → build model → manager → train
→ 载入 best_weight → post_train → cfg/stage1_id → meta → save → after_save → 摘要打印

阶段 2：prepare_stage2 → 日志/播种 → build+load+冻结 backbone（记录 sha_before、env_ids 校验）
→ build_newtask → Adam → 训练循环（best/early-stop）→ 载入 best → sha_after + 梯度断言
→ final_eval → build_gates → run_id/run_path → write_artifacts → 摘要打印

make_env_ids 使用独立 Generator、不消耗全局 RNG，因此位于 seed_model 之后
不影响模型初始化与训练的 RNG 流。
"""
from __future__ import annotations

import copy
import io
from datetime import datetime
from pathlib import Path

import torch

from benchmark.cliutil import tee_stdout
from benchmark.profile import DatasetProfile
from benchmark.protocol import (
    assert_no_grads,
    backbone_sha256,
    code_commit,
    config_hash,
    freeze_backbone,
    load_stage1,
    make_env_ids,
    make_run_id,
    save_stage1,
    seed_model,
    sha256_tensor,
    stage1_id,
)
from benchmark.recording import RecordingMPTRecTrainManager


def run_stage1(profile: DatasetProfile, root, *, epochs: int | None = None,
               device: torch.device | None = None, model=None, ctx: dict | None = None) -> dict:
    """返回统一结构 {"stage1_id", "dir", "meta"}；产物只读不覆盖（已存在则 FileExistsError）。"""
    root = Path(root)
    device = device or torch.device("cuda:0")
    epochs = profile.stage1_epochs if epochs is None else epochs
    commit, log_buffer = code_commit(), io.StringIO()

    ctx = profile.prepare(root) if ctx is None else ctx

    env_ids = make_env_ids(profile.train_size(ctx), profile.env_seed)

    seed_model(profile.model_seed)
    model = model or profile.build_model(device)
    model.to(device)
    manager = RecordingMPTRecTrainManager(model=model, env_ids=env_ids,
                                          **profile.manager_kwargs(ctx, epochs))
    with tee_stdout(log_buffer):
        manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    final_env_ids = manager.env_ids.clone()
    post = profile.post_train(model, manager, ctx, device)

    cfg = profile.build_cfg(ctx, epochs)
    cfg_sha = config_hash(cfg)
    sid = stage1_id(fingerprint_sha=profile.fingerprint_sha(ctx), model_seed=profile.model_seed,
                    epochs=epochs, cfg_sha=cfg_sha)
    meta = profile.build_meta(sid=sid, cfg=cfg, cfg_sha=cfg_sha, ctx=ctx, model=model,
                              manager=manager, env_ids=final_env_ids, post=post,
                              commit=commit, epochs=epochs)
    stage1_path = save_stage1(root, sid, backbone_state=profile.state_dict(model),
                              env_ids=final_env_ids, meta=meta)
    profile.after_save(stage1_path, log_buffer)
    profile.log_summary(sid, stage1_path, meta)
    return {"stage1_id": sid, "dir": str(stage1_path), "meta": meta}


def run_stage2(profile: DatasetProfile, root, *, stage1_dir, epochs: int | None = None,
               tag: str = "short", device: torch.device | None = None, model=None,
               now=None, log=print) -> dict:
    """返回 {"run_id", "run_dir", **write_artifacts 扩展字段}（两数据集各自保持既有字段名）。"""
    root = Path(root)
    device = device or torch.device("cuda:0")
    sid = Path(stage1_dir).name
    checkpoint = load_stage1(root, sid)                    # 唯一 backbone 来源（缺失即抛错）
    meta = checkpoint["meta"]
    epochs = profile.stage2_epochs if epochs is None else epochs
    commit, created, log_buffer = code_commit(), (now or datetime.now()), io.StringIO()

    ctx2 = profile.prepare_stage2(root, meta, device)
    profile.log_stage2_start(sid, tag, ctx2, log)

    seed_model(ctx2["model_seed"])
    backbone = model or profile.build_model(device)
    backbone.to(device)
    backbone.load_state_dict(checkpoint["backbone_state"])
    extra_facts = profile.backbone_ready(backbone, checkpoint, ctx2, log)
    freeze_backbone(backbone)
    sha_before = backbone_sha256(backbone)
    env_ids = checkpoint["env_ids"]
    env_ids_ok = sha256_tensor(env_ids) == meta["env_ids_sha256"]

    newtask = profile.build_newtask(device, ctx2)
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=profile.stage2_lr)
    loss_func = torch.nn.BCELoss()
    best_auc, best_epoch, best_state, stale, epoch_records = None, None, None, 0, []

    with tee_stdout(log_buffer):
        for epoch in range(1, epochs + 1):
            newtask.train()
            loss_sum, steps = 0.0, 0
            for _, _, y, features in ctx2["loaders"]["train"]:
                features = {key: value.to(device) for key, value in features.items()}
                with torch.no_grad():                                    # 计算图级冻结
                    dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
                pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
                if profile.stage2_cpu_loss:                              # CPU 侧损失（数据集差异经 profile 注入）
                    loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
                else:                                                    # 设备侧损失
                    loss = loss_func(pred, y.float().to(device)) + newtask.get_l2_reg()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += float(loss)
                steps += 1
            auc_val = profile.val_auc(newtask, backbone, ctx2["loaders"]["val"], device)
            epoch_records.append(profile.epoch_record(epoch, loss_sum / max(steps, 1), auc_val))
            if best_auc is None or auc_val > best_auc:
                best_auc, best_epoch, stale = auc_val, epoch, 0
                best_state = copy.deepcopy(newtask.state_dict())
            else:
                stale += 1
            stopped = stale == profile.stage2_patience
            profile.log_stage2_epoch(epoch, loss_sum / max(steps, 1), auc_val, stale, stopped, log)
            if stopped:
                break

    newtask.load_state_dict(best_state)
    sha_after = backbone_sha256(backbone)
    try:
        assert_no_grads(backbone)
        grads_all_none = True
    except AssertionError as exc:
        if not profile.stage2_catch_grads:
            raise
        grads_all_none = False
        profile.log_grads_failure(exc, log)

    final = profile.final_eval(newtask, backbone, ctx2, device)
    facts = {"sha_before": sha_before, "sha_after": sha_after, "grads_all_none": grads_all_none,
             "env_ids_ok": env_ids_ok, **extra_facts}
    gates_doc, passed = profile.build_gates(final, ctx2, meta, facts, best_auc, log)

    run_id = make_run_id(created, prefix=profile.run_id_prefix(ctx2), model_seed=ctx2["model_seed"],
                         tag=tag, commit=commit)
    run_path = Path(root) / "runs" / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    bundle = {
        "run_id": run_id, "run_path": run_path, "root": root, "stage1_id": sid, "tag": tag,
        "epochs": epochs, "created": created, "device": device, "commit": commit,
        "ctx2": ctx2, "meta": meta, "loaders": ctx2["loaders"],
        "backbone": backbone, "newtask": newtask, "env_ids": env_ids,
        "facts": facts, "epoch_records": epoch_records, "best_epoch": best_epoch, "best_auc": best_auc,
        "final": final, "gates_doc": gates_doc, "passed": passed, "log_buffer": log_buffer,
    }
    extras = profile.write_artifacts(run_path, bundle)
    profile.log_stage2_end(bundle, log)
    return {"run_id": run_id, "run_dir": str(run_path), **extras}
