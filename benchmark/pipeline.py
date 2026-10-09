"""阶段 1 / 阶段 2 统一编排。数据集差异全部经 DatasetProfile 钩子注入。

固定执行顺序（位级复现依赖，不得更改）：
prepare → seed_model → make_env_ids → build model → manager → train
→ 载入 best_weight → post_train → cfg/stage1_id → meta → save → after_save → 摘要打印

make_env_ids 使用独立 Generator、不消耗全局 RNG，因此位于 seed_model 之后
不影响模型初始化与训练的 RNG 流。
"""
from __future__ import annotations

import io
from pathlib import Path

import torch

from benchmark.cliutil import tee_stdout
from benchmark.profile import DatasetProfile
from benchmark.protocol import (
    code_commit,
    config_hash,
    make_env_ids,
    save_stage1,
    seed_model,
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
