"""DatasetProfile：数据集适配接口。

pipeline 只按固定顺序调用这些钩子，内部不出现任何数据集名判断；
每个数据集在 benchmark/datasets/<name>.py 中提供工厂函数返回本 dataclass 的实例。

ctx2（prepare_stage2 返回值）必备键：
- "loaders"：{"train", "val", "test"}，训练循环与评估共用；
- "model_seed"：阶段 2 的播种值（census 取 stage1 meta 中的值）。
其余键由数据集自定义，仅供本数据集的钩子读取。
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    model_seed: int
    env_seed: int
    stage1_epochs: int
    prepare: Callable[[Path], dict]                                  # (root) -> ctx
    build_model: Callable[[torch.device], torch.nn.Module]
    train_size: Callable[[dict], int]                                # ctx -> n 训练样本（env_ids 长度）
    manager_kwargs: Callable[[dict, int], dict]                      # (ctx, epochs) -> 记录层 manager kwargs
    post_train: Callable[[torch.nn.Module, object, dict, torch.device], dict]
    build_cfg: Callable[[dict, int], dict]                           # (ctx, epochs) -> stage1 配置
    fingerprint_sha: Callable[[dict], str]
    state_dict: Callable[[torch.nn.Module], dict]
    build_meta: Callable[..., dict]
    after_save: Callable[[Path, io.StringIO], None]                  # census 落盘 stdout.log；aliccp 为 no-op
    log_summary: Callable[[str, Path, dict], None]

    # ---- 阶段 2 ----
    stage2_epochs: int
    stage2_patience: int
    stage2_lr: float
    stage2_cpu_loss: bool                                            # census False / aliccp True
    stage2_catch_grads: bool                                         # census False（断言失败即崩）/ aliccp True（记为门禁事实）
    prepare_stage2: Callable[[Path, dict, torch.device], dict]       # (root, meta, device) -> ctx2
    log_stage2_start: Callable[[str, str, dict, Callable], None]
    backbone_ready: Callable[..., dict]                              # (backbone, checkpoint, ctx2, log) -> 额外事实
    build_newtask: Callable[[torch.device, dict], torch.nn.Module]
    val_auc: Callable[[torch.nn.Module, torch.nn.Module, object, torch.device], float]
    epoch_record: Callable[[int, float, float], dict]                # (epoch, loss, val_auc) -> 逐 epoch 记录
    log_stage2_epoch: Callable[..., None]                            # (epoch, loss, val_auc, stale, stopped, log)
    final_eval: Callable[[torch.nn.Module, torch.nn.Module, dict, torch.device], dict]
    log_grads_failure: Callable[[AssertionError, Callable], None]
    build_gates: Callable[..., tuple]                                # (final, ctx2, meta, facts, best_auc, log) -> (报告, passed)
    run_id_prefix: Callable[[dict], str]
    write_artifacts: Callable[[Path, dict], dict]                    # (run_path, bundle) -> 并入返回值的扩展字段
    log_stage2_end: Callable[[dict, Callable], None]
