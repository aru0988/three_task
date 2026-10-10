"""Converged paired B/U training for Universal Representation experiments."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from multitaskrec.model import NewTask
from multitaskrec.train import MPTRecTrainManager
from universal_experiment.convergence import convergence_decision
from universal_experiment.model import UniversalExpert
from universal_experiment.paired import (
    dataset_epochs,
    make_stage1_model,
    make_stage2_heads,
    stage2_parameters,
)
from universal_experiment.run import fit_normalizer


PATIENCE = 5
EXPECTED_CENSUS_SPLIT = "096f8f16db081d7ad024b7aa450a40510a50e0acdbb439d973cc8fc9557d460c"


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_stage1(records, *, budget, patience):
    if not records:
        raise ValueError("Stage-1 produced no validation records")
    scores = [row["auc_val_0"] + row["auc_val_1"] for row in records]
    best_score = max(scores)
    best_epoch = next(row["epoch"] for row, score in zip(records, scores)
                      if score == best_score)
    stopped_early = len(records) < budget
    decision = convergence_decision(
        scores, best_epoch=best_epoch, budget=budget,
        patience=patience, stopped_early=stopped_early,
    )
    return {
        "best_epoch": best_epoch,
        "best_val_sum": best_score,
        "epochs_run": len(records),
        "convergence": asdict(decision),
    }


class PairedTrainManager(MPTRecTrainManager):
    """Original Stage-1 optimizer/loss with generic validation records."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_records = []
        self.cluster_records = []
        self._started = time.perf_counter()

    def evaluation_two_task(self, data_loader):
        scores = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.epoch_records.append({
                "epoch": len(self.epoch_records) + 1,
                "auc_val_0": float(scores[0]),
                "auc_val_1": float(scores[1]),
                "wall_seconds": time.perf_counter() - self._started,
            })
        return scores

    def cluster_2(self):
        previous = self.env_ids.clone()
        updated = super().cluster_2()
        counts = torch.bincount(updated, minlength=self.model.base.num_tasks
                                if hasattr(self.model, "base") else self.model.num_tasks)
        self.cluster_records.append({
            "epoch": len(self.epoch_records),
            "diff_num": int((previous != updated).sum()),
            "env_counts": [int(value) for value in counts.tolist()],
        })
        return updated


def prepare(dataset, source, smoke, device, model_seed=None):
    source = Path(source)
    if dataset == "census":
        from census_benchmark import protocol as protocol
        from multitaskrec.dataset import CensusIncomeDataset
        from run_census_benchmark import build_mptrec

        train = CensusIncomeDataset(str(source / "dataset/Census-income/train.gz"), "education")
        test = CensusIncomeDataset(str(source / "dataset/Census-income/test.gz"), "education")
        val_indices, test_indices = protocol.make_split(len(test), protocol.SPLIT_SEED)
        fingerprint = protocol.split_fingerprint(
            split_seed=protocol.SPLIT_SEED,
            stats=protocol.split_stats(val_indices, test_indices, len(train)),
            val_idx=val_indices,
            test_idx=test_indices,
        )
        if fingerprint["fingerprint_sha256"] != EXPECTED_CENSUS_SPLIT:
            raise RuntimeError("Census split fingerprint changed")
        sets = {
            "train": train,
            "val": Subset(test, val_indices.tolist()),
            "test": Subset(test, test_indices.tolist()),
        }
        if smoke:
            sets = {name: Subset(value, list(range(min(1024, len(value)))))
                    for name, value in sets.items()}
        loaders = {name: DataLoader(value, batch_size=protocol.BATCH_SIZE, shuffle=False)
                   for name, value in sets.items()}

        def make_u(base):
            features = train[0][-1]
            widths = ([1 for key in features if key not in base.embedding_network.feature_names]
                      + [protocol.EMBEDDING_SIZE] * len(base.embedding_network.feature_names))
            return UniversalExpert(widths), widths

        return SimpleNamespace(
            dataset=dataset, P=protocol, build=build_mptrec, sets=sets, loaders=loaders,
            task_names=["Income", "Marital"], third_task="Education",
            lr=protocol.LR, batch_size=protocol.BATCH_SIZE,
            uni_coe=protocol.UNI_COE, env_coe=protocol.ENV_COE,
            model_seed=protocol.MODEL_SEED if model_seed is None else model_seed,
            env_seed=protocol.ENV_SEED, rep_dim=protocol.EXPERT_HIDDEN[-1],
            input_size=protocol.INPUT_SIZE, tower_hidden=list(protocol.TOWER_HIDDEN),
            reg_dnn=protocol.REG_DNN, make_u=make_u,
            fingerprints=fingerprint,
        )

    from aliccp_benchmark import protocol
    from multitaskrec.dataset import AliCCPDataset
    from universal_experiment.aliccp import build

    limits = ((20000, 5000, 10000) if smoke else (2000000, 500000, 1000000))
    budgets = dict(zip(("train", "val", "test"), limits))
    fingerprint = json.loads((source / "artifacts/aliccp_bench/splits/p2M-v500k-t1M/"
                                       "prefix_fingerprint.json").read_text(encoding="utf-8"))
    sets = {}
    loaders = {}
    actual = {}
    for split, count in budgets.items():
        data_file = source / Path(protocol.DATA_FILES[split])
        prefix_hash = protocol.prefix_sha256(data_file, count)
        label_counts = protocol.scan_label_counts(data_file, count)
        if not smoke:
            if prefix_hash != fingerprint["files"][split]["prefix_sha256"]:
                raise RuntimeError(f"AliCCP {split} prefix changed")
            if protocol._normalize_counts(label_counts) != protocol._normalize_counts(
                    fingerprint["label_counts"][split]):
                raise RuntimeError(f"AliCCP {split} labels changed")
        sets[split] = AliCCPDataset(str(data_file), count)
        loaders[split] = DataLoader(sets[split], batch_size=2000, shuffle=False)
        actual[split] = {"prefix_sha256": prefix_hash, "label_counts": label_counts}

    def make_u(_base):
        return UniversalExpert([5] * 16, rep_dim=64), [5] * 16

    return SimpleNamespace(
        dataset=dataset, P=protocol, build=build, sets=sets, loaders=loaders,
        task_names=["CTR", "CVR"], third_task="BSI", lr=protocol.LR,
        batch_size=2000, uni_coe=protocol.UNI_COE, env_coe=protocol.ENV_COE,
        model_seed=protocol.MODEL_SEED if model_seed is None else model_seed,
        env_seed=protocol.ENV_SEED, rep_dim=64, input_size=protocol.INPUT_SIZE,
        tower_hidden=list(protocol.TOWER_HIDDEN), reg_dnn=protocol.REG_DNN,
        make_u=make_u,
        fingerprints={"registered": fingerprint, "actual": actual},
    )


@torch.no_grad()
def predict_old_tasks(model, loader, device):
    model.eval()
    labels = [[], []]
    predictions = [[], []]
    for y0, y1, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        output = model.predict(features)
        for index, (target, prediction) in enumerate(zip((y0, y1), output)):
            labels[index].append(target.reshape(-1).cpu())
            predictions[index].append(prediction.reshape(-1).cpu())
    return ([torch.cat(parts).numpy() for parts in labels],
            [torch.cat(parts).numpy() for parts in predictions])


def train_stage1(spec, arm, read_policy, output, epochs, patience, device):
    protocol = spec.P
    protocol.seed_model(spec.model_seed)
    base = spec.build(device).to(device)
    universal = None
    if arm == "U":
        universal, _ = spec.make_u(base)
        universal = universal.to(device)
        fit_normalizer(base, universal, spec.loaders["train"], device)
    model = make_stage1_model(base, universal, arm, read_policy).to(device)
    env_ids = protocol.make_env_ids(len(spec.sets["train"]), spec.env_seed)
    manager = PairedTrainManager(
        model, spec.loaders["train"], spec.loaders["val"], env_ids,
        spec.task_names, spec.lr, spec.batch_size, spec.uni_coe, spec.env_coe,
        epochs=epochs, patience=patience,
    )
    started = time.perf_counter()
    manager.train_two_task()
    model.load_state_dict(manager.best_weight)
    summary = summarize_stage1(manager.epoch_records, budget=epochs, patience=patience)
    raw = {}
    scores = {}
    for split in ("val", "test"):
        labels, predictions = predict_old_tasks(model, spec.loaders[split], device)
        scores[split] = {}
        for index, task in enumerate(spec.task_names):
            raw[f"{split}_y{index}"] = labels[index]
            raw[f"{split}_p{index}"] = predictions[index]
            scores[split][task] = float(roc_auc_score(labels[index].astype(int), predictions[index]))
    checkpoint = Path(output) / f"{arm}_stage1.pt"
    predictions_file = Path(output) / f"{arm}_old_tasks.npz"
    torch.save(model.state_dict(), checkpoint)
    np.savez_compressed(predictions_file, **raw)
    metrics = {
        "arm": arm,
        "read_policy": read_policy,
        "seed": spec.model_seed,
        "budget": epochs,
        "patience": patience,
        "epochs": manager.epoch_records,
        "clusters": manager.cluster_records,
        **summary,
        "old_task_auc": scores,
        "base_hash": protocol.backbone_sha256(base),
        "universal_hash": protocol.backbone_sha256(universal) if universal is not None else None,
        "checkpoint_sha256": file_hash(checkpoint),
        "predictions_sha256": file_hash(predictions_file),
        "wall_seconds": time.perf_counter() - started,
    }
    dump(Path(output) / f"{arm}_stage1.json", metrics)
    return base, universal, metrics


@torch.no_grad()
def cache_third_task(base, universal, loader, device):
    base.eval()
    if universal is not None:
        universal.eval()
    pieces = {name: [] for name in ("x", "g", "s0", "s1", "y")}
    if universal is not None:
        pieces["u"] = []
    envs = None
    for _, _, target, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, general, specifics, envs = base.get_infos(features)
        values = {"x": dnn_input, "g": general, "s0": specifics[0],
                  "s1": specifics[1], "y": target.float()}
        if universal is not None:
            values["u"] = universal(dnn_input)
        for name, value in values.items():
            pieces[name].append(value.detach().cpu())
    return {name: torch.cat(value) for name, value in pieces.items()}, [e.detach() for e in envs]


def iter_batches(cache, batch_size, device):
    for start in range(0, len(cache["y"]), batch_size):
        yield {name: value[start:start + batch_size].to(device)
               for name, value in cache.items()}


def cached_prediction(head, batch, envs, arm):
    arguments = (batch["x"], batch["g"], [batch["s0"], batch["s1"]], envs)
    if arm == "B":
        return head(*arguments).reshape(-1)
    return head(*arguments, extra=batch["u"]).reshape(-1)


@torch.no_grad()
def evaluate_cached(head, cache, envs, arm, batch_size, device):
    head.eval()
    predictions = torch.cat([
        cached_prediction(head, batch, envs, arm).cpu()
        for batch in iter_batches(cache, batch_size, device)
    ]).numpy()
    labels = cache["y"].numpy()
    return float(roc_auc_score(labels.astype(int), predictions)), labels, predictions


def train_frozen_stage2(spec, arm, base, universal, head, output, epochs, patience, device):
    for module in (base, universal):
        if module is None:
            continue
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
    caches = {}
    envs = None
    for split, loader in spec.loaders.items():
        caches[split], envs = cache_third_task(base, universal, loader, device)
    optimizer = torch.optim.Adam(head.parameters(), lr=spec.lr)
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        steps = 0
        for batch in iter_batches(caches["train"], spec.batch_size, device):
            prediction = cached_prediction(head, batch, envs, arm)
            target = batch["y"]
            if spec.dataset == "aliccp":
                loss = torch.nn.functional.binary_cross_entropy(
                    prediction.cpu(), target.cpu()) + head.get_l2_reg()
            else:
                loss = torch.nn.functional.binary_cross_entropy(
                    prediction, target) + head.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        val_auc, _, _ = evaluate_cached(
            head, caches["val"], envs, arm, spec.batch_size, device)
        history.append({"epoch": epoch, "loss": total / max(steps, 1),
                        "val_auc": val_auc,
                        "wall_seconds": time.perf_counter() - started})
        print(f"[Stage2 {arm}] epoch={epoch} val={val_auc:.9f}", flush=True)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale == patience:
                break
    head.load_state_dict(best_state)
    stopped_early = len(history) < epochs
    decision = convergence_decision(
        [row["val_auc"] for row in history], best_epoch=best_epoch,
        budget=epochs, patience=patience, stopped_early=stopped_early,
    )
    raw = {}
    metrics = {
        "arm": arm,
        "best_epoch": best_epoch,
        "epochs": history,
        "epochs_run": len(history),
        "budget": epochs,
        "patience": patience,
        "convergence": asdict(decision),
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "wall_seconds": time.perf_counter() - started,
    }
    for split in ("val", "test"):
        auc, labels, predictions = evaluate_cached(
            head, caches[split], envs, arm, spec.batch_size, device)
        metrics[f"{split}_auc"] = auc
        raw[f"{split}_y"] = labels
        raw[f"{split}_p"] = predictions
    predictions_file = Path(output) / f"{arm}_predictions.npz"
    checkpoint = Path(output) / f"{arm}_head.pt"
    np.savez_compressed(predictions_file, **raw)
    torch.save(best_state, checkpoint)
    metrics["predictions_sha256"] = file_hash(predictions_file)
    metrics["checkpoint_sha256"] = file_hash(checkpoint)
    dump(Path(output) / f"{arm}_stage2.json", metrics)
    return metrics


def run_frozen_pair(spec, read_policy, output, *, smoke, commit, dirty, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stage1_epochs = 1 if smoke else dataset_epochs(spec.dataset)
    stage2_epochs = 1 if smoke else dataset_epochs(spec.dataset)
    patience = 1 if smoke else PATIENCE
    config = {
        "dataset": spec.dataset,
        "read_policy": read_policy,
        "u_mode": "frozen",
        "seed": spec.model_seed,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "patience": patience,
        "commit": commit,
        "dirty": dirty,
        "smoke": smoke,
        "source_fingerprints": spec.fingerprints,
        "created": datetime.now().isoformat(),
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    dump(output / "config.json", config)
    stage1 = {}
    trained = {}
    for arm in ("B", "U"):
        trained[arm] = train_stage1(
            spec, arm, read_policy, output, stage1_epochs, patience, device)
        stage1[arm] = trained[arm][2]

    spec.P.seed_model(spec.model_seed)
    dummy = spec.build(device).to(device)
    original = NewTask(spec.input_size, spec.rep_dim, spec.tower_hidden,
                       spec.reg_dnn, device).to(device)
    del dummy
    baseline_head, universal_head = make_stage2_heads(original, spec.rep_dim)
    heads = {"B": baseline_head.to(device), "U": universal_head.to(device)}
    stage2 = {}
    for arm in ("B", "U"):
        base, universal, _ = trained[arm]
        stage2_parameters(heads[arm], universal, base, freeze_u=True)
        stage2[arm] = train_frozen_stage2(
            spec, arm, base, universal, heads[arm], output,
            stage2_epochs, patience, device)
    delta = {
        split: stage2["U"][f"{split}_auc"] - stage2["B"][f"{split}_auc"]
        for split in ("val", "test")
    }
    classification = ("positive" if delta["test"] >= .001 else
                      "clear_decline" if delta["test"] <= -.02 else
                      "no_clear_improvement")
    report = {
        "config": config,
        "stage1": stage1,
        "stage2": stage2,
        "u_minus_b": delta,
        "classification": classification,
        "all_converged": all(
            stage1[arm]["convergence"]["converged"]
            and stage2[arm]["convergence"]["converged"]
            for arm in ("B", "U")
        ),
    }
    dump(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("census", "aliccp"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--read-policy", choices=("no_read", "read_detached"), required=True)
    parser.add_argument("--u-mode", choices=("frozen", "trainable"), default="frozen")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.u_mode != "frozen":
        raise NotImplementedError("trainable U is enabled after the frozen read-policy screen")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    if dirty and not args.smoke:
        raise RuntimeError("formal experiment requires a clean worktree")
    if args.out.exists():
        raise FileExistsError(f"output path already exists: {args.out}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    spec = prepare(args.dataset, args.source, args.smoke, device, args.seed)
    return run_frozen_pair(spec, args.read_policy, args.out, smoke=args.smoke,
                           commit=commit, dirty=dirty, device=device)


if __name__ == "__main__":
    main()
