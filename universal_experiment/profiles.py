"""census / AliCCP 数据集适配（ScreenProfile）：只保留数据与数值差异，四臂逻辑全在 runner。

显式保留的差异（不做"统一"）：
- 指纹：census = split-index sha256（断言 EXPECTED_SPLIT）；aliccp = prefix-budget sha256 + 预算。
- 缓存：census 内存张量（runner.memory_cache_split）；aliccp memmap 落盘（out/cache）。
- 机制公式：census 全量张量；aliccp 分块在线统计（数值实现不同，各自保留）。
- stage1 json schema：census epoch_records/cluster_records（offset 0）；aliccp val_epochs/clusters（offset 1）。
- 其余：模型构造参数、设备解析（aliccp 硬 cuda:0）、cpu_loss、old_task 名称、config schema。

RNG 红线：build_base / build_original_head 的调用顺序与参数必须与旧 run.py / aliccp.py 一致
（seed_model → build base → NewTask），不得"顺手统一"。
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from benchmark.datasets import aliccp as aliccp_data
from benchmark.datasets import census as census_data
from benchmark.protocol import seed_model
from multitaskrec.dataset import AliCCPDataset, CensusIncomeDataset
from multitaskrec.model import NewTask
from universal_experiment import runner as R

CENSUS_EXPECTED_BASE = "a12a5f5369a7002fb12f0ead7576e4dad3375a060ba2f90d45bfb1d566153f85"
CENSUS_EXPECTED_SPLIT = "096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c"
ALICCP_EXPECTED_BASE = "5553640bc1f43c7af0065f4f1d3f2d719022b3764751e2e4a6cb57f663f2c6ee"


@dataclass(frozen=True)
class ScreenProfile:
    """数据集适配器接口：runner 只通过这些钩子触及数据集差异。"""

    name: str
    expected_base_hash: str
    model_seed: int
    env_seed: int
    resolve_device: Callable                  # () -> torch.device
    prepare: Callable                         # (source, smoke) -> ctx
    save_pre_artifacts: Callable              # (out, ctx) -> None
    build_config: Callable                    # (ctx, run_meta) -> dict
    build_base: Callable                      # (device) -> module
    u_widths: Callable                        # (ctx, base) -> list[int]
    u_rep_dim: int
    env_id_count: Callable                    # (ctx) -> int
    task_names: list
    lr: float
    batch_size: int
    stage1_patience: int
    uni_coe: float
    env_coe: float
    record_env_acc: bool
    cluster_epoch_offset: int
    build_s1: Callable                        # (*, base_hash, universal_hash, manager, model) -> dict
    cache_split: Callable                     # (base, u, random_u, loader, device, out, split) -> (cache, envs, old)
    old_task_names: tuple
    mechanism: Callable                       # (train_cache, u, device, out, widths) -> dict
    finalize_stage1: Callable                 # (out, s1, mech, old_auc) -> None
    build_original_head: Callable             # (device) -> module
    cpu_loss: bool
    stage2_lr: float
    stage2_batch_size: int
    stage2_patience: int
    rep_dim: int


# ---------------------------------------------------------------- census
def census_prepare(source, smoke):
    train = CensusIncomeDataset(str(source / "dataset/Census-income/train.gz"), "education")
    test = CensusIncomeDataset(str(source / "dataset/Census-income/test.gz"), "education")
    val_idx, test_idx = census_data.make_split(len(test), census_data.SPLIT_SEED)
    fp = census_data.split_fingerprint(split_seed=census_data.SPLIT_SEED,
                                       stats=census_data.split_stats(val_idx, test_idx, len(train)),
                                       val_idx=val_idx, test_idx=test_idx)
    assert fp["fingerprint_sha256"] == CENSUS_EXPECTED_SPLIT
    saved = np.load(source / "artifacts/census_stage2/splits/20260929/split_indices.npz")
    assert np.array_equal(val_idx, saved["val"]) and np.array_equal(test_idx, saved["test"])
    sets = {"train": train, "val": Subset(test, val_idx.tolist()), "test": Subset(test, test_idx.tolist())}
    if smoke:
        sets = {k: Subset(v, list(range(1024))) for k, v in sets.items()}
    loaders = {k: DataLoader(v, batch_size=census_data.BATCH_SIZE, shuffle=False) for k, v in sets.items()}
    return {"loaders": loaders, "datasets": {"train": train}, "sets": sets,
            "fp": fp, "indices": (val_idx, test_idx)}


def census_save_pre_artifacts(out, ctx):
    val_idx, test_idx = ctx["indices"]
    np.savez_compressed(out / "indices.npz", val=val_idx, test=test_idx)


def census_build_config(ctx, meta):
    return {"commit": meta["commit"], "dirty": meta["dirty"], "model_seed": census_data.MODEL_SEED,
            "env_seed": census_data.ENV_SEED, "split_seed": census_data.SPLIT_SEED,
            "smoke": meta["smoke"], "fingerprint": ctx["fp"],
            "stage1_epochs": 1 if meta["smoke"] else 2, "stage2_epochs": 1 if meta["smoke"] else 5,
            "source": str(meta["source"]), "created": datetime.now().isoformat(),
            "device": str(meta["device"]), "python": os.sys.version, "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "data_sha256": {n: R.file_hash(meta["source"] / "dataset/Census-income" / n)
                            for n in ("train.gz", "test.gz")}}


def census_u_widths(ctx, base):
    feat = ctx["datasets"]["train"][0][-1]
    assert "education" not in feat
    widths = ([1 for k in feat if k not in base.embedding_network.feature_names]
              + [census_data.EMBEDDING_SIZE] * len(base.embedding_network.feature_names))
    assert sum(widths) == census_data.INPUT_SIZE
    return widths


def census_build_s1(*, base_hash, universal_hash, manager, model):
    records = census_data.stage1_epoch_records(manager)
    selected = max(records, key=lambda r: r["auc_val_income"] + r["auc_val_marital"])["epoch"]
    return dict(base_hash=base_hash, matches_historical=base_hash == CENSUS_EXPECTED_BASE,
                selected_epoch=selected, epochs=records, clusters=manager.cluster_records,
                auxiliary_steps=model.loss_history, universal_hash=universal_hash)


@torch.no_grad()
def census_mechanism(cache, u, device, out, widths):
    z = cache["u"].double(); g = cache["g"].double()
    std = z.std(0, unbiased=False)
    zc = (z - z.mean(0)) / std.clamp_min(.01)
    gc = (g - g.mean(0)) / g.std(0, unbiased=False).clamp_min(.01)
    corr = zc.T @ gc / len(z)
    probe_mask = u.sample_mask(min(4096, len(z)))
    diag = u.losses(cache["x"][:4096].to(device), cache["g"][:4096].to(device), probe_mask)
    mech = dict(u_std_mean=float(std.mean()), u_live_fraction=float((std > .01).double().mean()),
                ug_squared_correlation=float(corr.square().mean()),
                reconstruction_train_probe={k: float(v) for k, v in diag.items()})
    np.savez_compressed(out / "mechanism.npz", u_std=std.numpy(), ug_correlation=corr.numpy(),
                        n=np.array(len(z)), u_sum=z.sum(0).numpy(), g_sum=g.sum(0).numpy(),
                        u_sq=z.square().sum(0).numpy(), g_sq=g.square().sum(0).numpy(),
                        ug_sum=(z.T @ g).numpy(),
                        probe_x=cache["x"][:4096].numpy(), probe_g=g[:4096].float().numpy(),
                        probe_u=z[:4096].float().numpy(), probe_mask=probe_mask.numpy(),
                        widths=np.array(widths))
    return mech


def census_finalize_stage1(out, s1, mech, old_auc):
    s1["old_task_auc"] = old_auc
    R.dump(out / "stage1.json", s1)


def census_build_original_head(device):
    # 与旧 stage2 初始化一致：seed → 构造 MPTRec(dummy) → 构造 NewTask。
    seed_model(census_data.MODEL_SEED)
    dummy = census_data.build_mptrec(device)
    original = NewTask(census_data.INPUT_SIZE, census_data.EXPERT_HIDDEN[-1],
                       list(census_data.TOWER_HIDDEN), census_data.REG_DNN, device).to(device)
    del dummy
    return original


CENSUS_SCREEN = ScreenProfile(
    name="census", expected_base_hash=CENSUS_EXPECTED_BASE,
    model_seed=census_data.MODEL_SEED, env_seed=census_data.ENV_SEED,
    resolve_device=lambda: torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    prepare=census_prepare, save_pre_artifacts=census_save_pre_artifacts,
    build_config=census_build_config, build_base=census_data.build_mptrec,
    u_widths=census_u_widths, u_rep_dim=census_data.EXPERT_HIDDEN[-1],
    env_id_count=lambda ctx: len(ctx["sets"]["train"]),
    task_names=["Income", "Marital"],
    lr=census_data.LR, batch_size=census_data.BATCH_SIZE, stage1_patience=census_data.PATIENCE,
    uni_coe=census_data.UNI_COE, env_coe=census_data.ENV_COE,
    record_env_acc=True, cluster_epoch_offset=0,
    build_s1=census_build_s1, cache_split=R.memory_cache_split,
    old_task_names=("income", "marital"), mechanism=census_mechanism,
    finalize_stage1=census_finalize_stage1, build_original_head=census_build_original_head,
    cpu_loss=False, stage2_lr=census_data.LR, stage2_batch_size=census_data.BATCH_SIZE,
    stage2_patience=census_data.PATIENCE, rep_dim=census_data.EXPERT_HIDDEN[-1])


# ---------------------------------------------------------------- aliccp
def aliccp_prepare(source, smoke):
    budgets = ({"train": 20000, "val": 5000, "test": 10000} if smoke
               else {"train": aliccp_data.TRAIN_BUDGET, "val": aliccp_data.VAL_BUDGET,
                     "test": aliccp_data.TEST_BUDGET})
    fp = json.loads((source / "artifacts/aliccp_bench/splits/p2M-v500k-t1M/prefix_fingerprint.json")
                    .read_text(encoding="utf-8"))
    assert aliccp_data.fingerprint_digest(fp) == fp["fingerprint_sha256"]
    actual, datasets, loaders = {}, {}, {}
    for split, n in budgets.items():
        path = source / Path(aliccp_data.DATA_FILES[split])
        h = aliccp_data.prefix_sha256(path, n)
        counts = aliccp_data.scan_label_counts(path, n)
        if not smoke:
            assert h == fp["files"][split]["prefix_sha256"]
            assert aliccp_data._normalize_counts(counts) == aliccp_data._normalize_counts(fp["label_counts"][split])
        actual[split] = {"prefix_sha256": h, "label_counts": counts}
        datasets[split] = AliCCPDataset(str(path), n)
        aliccp_data.verify_label_counts(datasets[split], n, counts)
        assert set(datasets[split][0][-1]) == set(aliccp_data.build_vocab()) and "301" not in datasets[split][0][-1]
        loaders[split] = DataLoader(datasets[split], batch_size=aliccp_data.BATCH_SIZE, shuffle=False)
    return {"loaders": loaders, "datasets": datasets, "fp": fp,
            "actual_prefixes": actual, "budgets": budgets}


def aliccp_build_config(ctx, meta):
    return {"dataset": "AliCCP", "commit": meta["commit"], "dirty": meta["dirty"],
            "smoke": meta["smoke"], "model_seed": aliccp_data.MODEL_SEED,
            "env_seed": aliccp_data.ENV_SEED, "budgets": dict(ctx["budgets"]),
            "fingerprint": ctx["fp"], "source": str(meta["source"]),
            "actual_prefixes": ctx["actual_prefixes"],
            "stage1_epochs": 1 if meta["smoke"] else 3, "stage2_epochs": 1 if meta["smoke"] else 5,
            "batch_size": aliccp_data.BATCH_SIZE, "lr": aliccp_data.LR,
            "created": datetime.now().isoformat(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "u_seed": 1685480946, "mask_seed": 20261009}


def aliccp_build_base(device):
    return aliccp_data.build_mptrec(device).to(device)


def aliccp_build_s1(*, base_hash, universal_hash, manager, model):
    s1 = {"base_hash": base_hash, "universal_hash": universal_hash,
          "selected_epoch": manager.best_epoch(), "val_epochs": manager.val_epoch_aucs,
          "clusters": [{"epoch": r["epoch"], "diff_num": r["diff_num"],
                        "env_0": r["env_counts"][0], "env_1": r["env_counts"][1]}
                       for r in manager.cluster_records],
          "auxiliary_steps": model.loss_history}
    s1["matches_historical"] = base_hash == ALICCP_EXPECTED_BASE
    return s1


@torch.no_grad()
def aliccp_disk_cache(base, u, random_u, loader, device, out, split):
    """memmap 版缓存（aliccp 口径；旧 aliccp.disk_cache 等价，old 仅非 train 切分）。"""
    folder = out / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    n = len(loader.dataset)
    dims = {"x": 80, "g": 64, "s0": 64, "s1": 64, "u": 64, "r": 64, "y": None}
    arrays = {k: np.lib.format.open_memmap(folder / f"{split}_{k}.npy", mode="w+", dtype="float32",
                                           shape=(n, d) if d else (n,))
              for k, d in dims.items()}
    old = {k: np.empty(n, dtype=np.float32) for k in ("y0", "y1", "p0", "p1")} if split != "train" else {}
    offset = 0
    for y0, y1, y, features in loader:
        f = {k: v.to(device) for k, v in features.items()}
        x, g, s, envs = base.get_infos(f)
        end = offset + len(y)
        for k, v in zip(arrays, (x, g, s[0], s[1], u(x), random_u(x), y)):
            arrays[k][offset:end] = v.detach().cpu().numpy()
        if old:
            preds = base.predict(f)
            for k, v in zip(old, (y0, y1, preds[0], preds[1])):
                old[k][offset:end] = v.cpu().numpy().reshape(-1)
        offset = end
    for arr in arrays.values():
        arr.flush()
    return {k: torch.from_numpy(v) for k, v in arrays.items()}, [e.detach() for e in envs], old


@torch.no_grad()
def aliccp_mechanism(cache, u, device, out, widths):
    n = len(cache["y"]); dim = cache["u"].shape[1]
    su = torch.zeros(dim, dtype=torch.float64); sg = su.clone()
    qu = su.clone(); qg = su.clone(); cross = torch.zeros(dim, dim, dtype=torch.float64)
    for lo in range(0, n, 10000):
        z = cache["u"][lo:lo + 10000].double(); g = cache["g"][lo:lo + 10000].double()
        su += z.sum(0); sg += g.sum(0); qu += z.square().sum(0); qg += g.square().sum(0); cross += z.T @ g
    us = (qu / n - (su / n).square()).clamp_min(0).sqrt()
    gs = (qg / n - (sg / n).square()).clamp_min(0).sqrt()
    corr = (cross / n - torch.outer(su / n, sg / n)) / torch.outer(us.clamp_min(.01), gs.clamp_min(.01))
    m = u.sample_mask(min(n, 4096)); x = cache["x"][:4096]; g = cache["g"][:4096]
    diag = {k: float(v) for k, v in u.losses(x.to(device), g.to(device), m).items()}
    result = {"u_std_mean": float(us.mean()), "u_live_fraction": float((us > .01).double().mean()),
              "ug_squared_correlation": float(corr.square().mean()), "reconstruction_train_probe": diag}
    np.savez_compressed(out / "mechanism.npz", n=np.array(n), u_sum=su.numpy(), g_sum=sg.numpy(),
                        u_sq=qu.numpy(), g_sq=qg.numpy(), ug_sum=cross.numpy(), u_std=us.numpy(),
                        ug_correlation=corr.numpy(), probe_x=x.numpy(), probe_g=g.numpy(),
                        probe_u=cache["u"][:4096].numpy(), probe_mask=m.numpy(),
                        widths=np.array(widths))
    return result


def aliccp_build_original_head(device):
    # 与旧 stage2 初始化一致：seed → 构造 MPTRec(dummy) → 构造 NewTask。
    seed_model(aliccp_data.MODEL_SEED)
    dummy = aliccp_build_base(device)
    original = NewTask(80, 64, [32, 32], aliccp_data.REG_DNN, device).to(device)
    del dummy
    return original


ALICCP_SCREEN = ScreenProfile(
    name="aliccp", expected_base_hash=ALICCP_EXPECTED_BASE,
    model_seed=aliccp_data.MODEL_SEED, env_seed=aliccp_data.ENV_SEED,
    resolve_device=lambda: torch.device("cuda:0"),
    prepare=aliccp_prepare, save_pre_artifacts=lambda out, ctx: None,
    build_config=aliccp_build_config, build_base=aliccp_build_base,
    u_widths=lambda ctx, base: [5] * 16, u_rep_dim=64,
    env_id_count=lambda ctx: ctx["budgets"]["train"],
    task_names=["CTR", "CVR"],
    lr=aliccp_data.LR, batch_size=aliccp_data.BATCH_SIZE, stage1_patience=aliccp_data.STAGE1_PATIENCE,
    uni_coe=aliccp_data.UNI_COE, env_coe=aliccp_data.ENV_COE,
    record_env_acc=False, cluster_epoch_offset=1,
    build_s1=aliccp_build_s1, cache_split=aliccp_disk_cache,
    old_task_names=("ctr", "cvr"), mechanism=aliccp_mechanism,
    finalize_stage1=lambda out, s1, mech, old_auc: None,
    build_original_head=aliccp_build_original_head,
    cpu_loss=True, stage2_lr=aliccp_data.LR, stage2_batch_size=aliccp_data.BATCH_SIZE,
    stage2_patience=aliccp_data.STAGE2_PATIENCE, rep_dim=64)
