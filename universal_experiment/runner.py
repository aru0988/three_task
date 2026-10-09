"""统一四臂 runner：census / AliCCP 共用 train_head 与 run_screen 编排。

层职责：
- 本模块承载 B/U/R/G 四臂共同逻辑：缓存、训练循环、选点、冻结哈希审计、report 组装；
- 数据集差异（数据准备、指纹/预算、模型构造、缓存形态、机制公式、json schema）全部经
  profiles.ScreenProfile 注入，本模块不出现数据集专名；
- train_head 取代旧 aliccp 侧 `C.P = SimpleNamespace(...)` 的 monkeypatch，lr / batch_size /
  patience / cpu_loss 全部显式传参；函数体与旧 universal_experiment/run.py 逐语句一致。

RNG 红线：seed_model → build_base → UniversalExpert → fit_normalizer → train_two_task →
seed_model → build_original_head → 四臂的顺序与旧 run.py / aliccp.py 完全一致，不得插入任何
全局 RNG 消耗（train_head 内部只有 Adam/BCE，U 臂 shuffle 用私有 Generator）。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score  # noqa: I001  先 sklearn 后 torch（本机 DLL 加载顺序坑）
import torch

from benchmark.protocol import backbone_sha256, make_env_ids, seed_model
from benchmark.recording import RecordingMPTRecTrainManager
from universal_experiment.model import ResidualHead, UniversalExpert, UniversalStage1

SHUFFLE_SEED = 20261009


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def fit_normalizer(base, u, loader, device):
    # This iterator must not perturb the original Stage-1 random stream.
    with torch.random.fork_rng(devices=[]):
        total = torch.zeros(u.input_dim, dtype=torch.float64, device=device)
        sq = total.clone(); n = 0
        for _, _, _, features in loader:
            x = base.embedding_network({k: v.to(device) for k, v in features.items()}).double()
            total += x.sum(0); sq += x.square().sum(0); n += len(x)
        mean = total / n
        u.set_normalizer(mean.float(), (sq / n - mean.square()).clamp_min(0).sqrt().float())


@torch.no_grad()
def memory_cache_split(base, u, random_u, loader, device, out, split):
    """内存版缓存（census 口径；旧 run.cache_split 等价实现，old 仅非 train 切分）。"""
    columns = {k: [] for k in ("x", "g", "s0", "s1", "u", "r", "y")}
    old = {k: [] for k in ("y0", "y1", "p0", "p1")}
    base.eval(); u.eval(); random_u.eval()
    envs = None
    for y0, y1, y, features in loader:
        features = {k: v.to(device) for k, v in features.items()}
        x, g, specs, envs = base.get_infos(features)
        values = (x, g, specs[0], specs[1], u(x), random_u(x), y.float())
        for key, value in zip(columns, values):
            columns[key].append(value.detach().cpu())
        if split != "train":
            p = base.predict(features)
            for key, value in zip(old, (y0, y1, p[0], p[1])):
                old[key].append(value.detach().cpu().reshape(-1))
    cache = {k: torch.cat(v) for k, v in columns.items()}
    old = {k: torch.cat(v).numpy() for k, v in old.items()} if split != "train" else {}
    return cache, [e.detach() for e in envs], old


def batches(cache, device, batch_size):
    for start in range(0, len(cache["y"]), batch_size):
        yield {k: v[start:start + batch_size].to(device) for k, v in cache.items()}


def predict(head, batch, envs, arm):
    args = (batch["x"], batch["g"], [batch["s0"], batch["s1"]], envs)
    if arm == "B":
        return head(*args).reshape(-1)
    return head(*args, extra=batch[{"U": "u", "R": "r", "G": "g"}[arm]])


@torch.no_grad()
def evaluate(head, cache, envs, arm, device, batch_size):
    head.eval()
    p = torch.cat([predict(head, b, envs, arm).cpu() for b in batches(cache, device, batch_size)]).numpy()
    return float(roc_auc_score(cache["y"].numpy(), p)), p


def train_head(head, caches, envs, arm, device, epochs, out, *, lr, batch_size, patience,
               cpu_loss=False, shuffle_seed=SHUFFLE_SEED):
    """单臂头训练（旧 run.train_head 等价；超参由 profile 显式传入）。"""
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best = -1.; state = None; stale = 0; history = []; best_epoch = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train(); total = 0.; steps = 0
        for b in batches(caches["train"], device, batch_size):
            opt.zero_grad()
            prediction = predict(head, b, envs, arm)
            loss = torch.nn.functional.binary_cross_entropy(
                prediction.cpu() if cpu_loss else prediction,
                b["y"].cpu() if cpu_loss else b["y"]) + head.get_l2_reg()
            loss.backward(); opt.step()
            total += float(loss.detach()); steps += 1
        va, _ = evaluate(head, caches["val"], envs, arm, device, batch_size)
        history.append(dict(epoch=epoch, val_auc=va, loss=total / steps, steps=steps))
        print(f"{arm}: epoch={epoch} val={va:.9f}", flush=True)
        if va > best:
            best = va; best_epoch = epoch; state = copy.deepcopy(head.state_dict()); stale = 0
        else:
            stale += 1
            if stale == patience:
                break
    head.load_state_dict(state)
    result = dict(best_epoch=best_epoch, epochs=history, parameters=sum(p.numel() for p in head.parameters()),
                  wall_seconds=time.perf_counter() - start)
    raw = {}
    for split in ("val", "test"):
        score, preds = evaluate(head, caches[split], envs, arm, device, batch_size)
        result[split + "_auc"] = score
        raw[split + "_y"] = caches[split]["y"].numpy()
        raw[split + "_p"] = preds
    if arm == "U":
        for split in ("val", "test"):
            shuffled = dict(caches[split])
            order = torch.randperm(len(shuffled["y"]), generator=torch.Generator().manual_seed(shuffle_seed))
            shuffled["u"] = shuffled["u"][order]
            score, preds = evaluate(head, shuffled, envs, arm, device, batch_size)
            result[split + "_shuffled_u_auc"] = score
            raw[split + "_shuffled_u_p"] = preds
    np.savez_compressed(out / f"{arm}_predictions.npz", **raw)
    torch.save(state, out / f"{arm}_head.pt")
    dump(out / f"{arm}_metrics.json", result)
    return result


def run_screen(profile, *, source, out, smoke=False):
    """四臂 screen 全流程：固定执行顺序 + profile 钩子（旧 run.py / aliccp.py 的合并等价）。"""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if dirty and not smoke:
        raise RuntimeError("Formal experiment requires a clean worktree")
    source, out = Path(source), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    device = profile.resolve_device()
    ctx = profile.prepare(source, smoke)
    config = profile.build_config(ctx, {"commit": commit, "dirty": dirty, "smoke": smoke,
                                        "source": source, "device": device})
    dump(out / "config.json", config)
    profile.save_pre_artifacts(out, ctx)

    seed_model(profile.model_seed)
    base = profile.build_base(device).to(device)
    widths = profile.u_widths(ctx, base)
    u = UniversalExpert(widths, rep_dim=profile.u_rep_dim).to(device)
    fit_normalizer(base, u, ctx["loaders"]["train"], device)
    random_u = copy.deepcopy(u)
    model = UniversalStage1(base, u).to(device)
    env_ids = make_env_ids(profile.env_id_count(ctx), profile.env_seed)
    manager = RecordingMPTRecTrainManager(
        model, ctx["loaders"]["train"], ctx["loaders"]["val"], env_ids, profile.task_names,
        profile.lr, profile.batch_size, profile.uni_coe, profile.env_coe,
        epochs=config["stage1_epochs"], patience=profile.stage1_patience,
        record_env_acc=profile.record_env_acc, cluster_epoch_offset=profile.cluster_epoch_offset)
    print("Stage 1: supervised G/S plus detached masked-feature U", flush=True)
    manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    base_hash = backbone_sha256(base)
    s1 = profile.build_s1(base_hash=base_hash, universal_hash=backbone_sha256(u),
                          manager=manager, model=model)
    dump(out / "stage1.json", s1)
    torch.save(model.state_dict(), out / "stage1.pt")
    torch.save(random_u.state_dict(), out / "random_u.pt")
    if not smoke and base_hash != profile.expected_base_hash:
        raise RuntimeError("G/S hash differs from historical baseline; stop and audit before Stage 2")

    for module in (base, u, random_u):
        module.eval()
        for p in module.parameters():
            p.grad = None
            p.requires_grad_(False)
    caches, old_raw, envs = {}, {}, None
    for split in ("train", "val", "test"):
        print(f"Caching frozen {split} representations", flush=True)
        caches[split], envs, old = profile.cache_split(base, u, random_u, ctx["loaders"][split],
                                                       device, out, split)
        for k, v in old.items():
            old_raw[f"{split}_{k}"] = v
    np.savez_compressed(out / "old_tasks.npz", **old_raw)
    old_auc = {s: {t: float(roc_auc_score(old_raw[f"{s}_y{i}"], old_raw[f"{s}_p{i}"]))
                   for i, t in enumerate(profile.old_task_names)} for s in ("val", "test")}
    mech = profile.mechanism(caches["train"], u, device, out, widths)
    dump(out / "mechanism.json", mech)
    profile.finalize_stage1(out, s1, mech, old_auc)

    original = profile.build_original_head(device)
    results = {}
    for arm in ("B", "U", "R", "G"):
        head = copy.deepcopy(original) if arm == "B" else ResidualHead(original, profile.rep_dim).to(device)
        results[arm] = train_head(head, caches, envs, arm, device, config["stage2_epochs"], out,
                                  lr=profile.stage2_lr, batch_size=profile.stage2_batch_size,
                                  patience=profile.stage2_patience, cpu_loss=profile.cpu_loss)
        del head
    assert results["U"]["parameters"] == results["R"]["parameters"] == results["G"]["parameters"]
    assert backbone_sha256(base) == base_hash
    assert all(p.grad is None for m in (base, u, random_u) for p in m.parameters())
    deltas = {a: {s: results["U"][s + "_auc"] - results[a][s + "_auc"] for s in ("val", "test")}
              for a in ("B", "R", "G")}
    delta = deltas["B"]["test"]
    classification = "positive" if delta >= .001 else "clear_decline" if delta <= -.02 else "no_clear_improvement"
    rec = mech["reconstruction_train_probe"]
    go = (all(d["test"] >= .001 and d["val"] > 0 for d in deltas.values())
          and mech["u_live_fraction"] >= .5 and rec["reconstruction"] < rec["zero"])
    report = dict(arms=results, u_minus=deltas, classification=classification, preregistered_go=go,
                  mechanism=mech, stage1_old_tasks=old_auc, frozen_base_hash_after=backbone_sha256(base),
                  frozen_u_hash_after=backbone_sha256(u),
                  frozen_gradients_none=all(p.grad is None for m in (base, u, random_u) for p in m.parameters()),
                  stage1_matches_historical=s1["matches_historical"],
                  wall_seconds=time.perf_counter() - start,
                  prediction_hashes={p.name: file_hash(p) for p in out.glob("*.npz")})
    dump(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Universal Representation 四臂 screen 统一入口")
    parser.add_argument("--dataset", choices=["census", "aliccp"], required=True)
    parser.add_argument("--source", type=Path, required=True, help="原 checkout（数据与指纹所在处）")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    from universal_experiment import profiles

    profile = profiles.CENSUS_SCREEN if args.dataset == "census" else profiles.ALICCP_SCREEN
    run_screen(profile, source=args.source, out=args.out, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
