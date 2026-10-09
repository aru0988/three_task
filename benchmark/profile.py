"""DatasetProfile：数据集适配接口。

pipeline 只按固定顺序调用这些钩子，内部不出现任何数据集名判断；
每个数据集在 benchmark/datasets/<name>.py 中提供工厂函数返回本 dataclass 的实例。
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
