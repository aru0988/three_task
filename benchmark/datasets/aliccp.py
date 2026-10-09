"""AliCCP 数据集适配：常量、前缀指纹/标签计数、加载器与模型构造、阶段 1 记录探针。

唯一事实来源：docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md。
函数体搬自 aliccp_benchmark/protocol.py 与 aliccp_benchmark/bench.py，行为必须保持位级一致；
共享纯工具见 benchmark/protocol.py（不修改任何模型/训练代码）。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmark.protocol import canonical_json, sha256_bytes, stage1_id
from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec

# ---- 前缀预算（spec 5.1）----
PREFIX_TAG = "p2M-v500k-t1M"
TRAIN_BUDGET, VAL_BUDGET, TEST_BUDGET = 2_000_000, 500_000, 1_000_000
SMOKE_TRAIN_BUDGET, SMOKE_VAL_BUDGET, SMOKE_TEST_BUDGET = 20_000, 5_000, 10_000
DATA_FILES = {
    "train": "dataset/AliCCP/ctr_cvr.train",
    "val": "dataset/AliCCP/ctr_cvr.dev",
    "test": "dataset/AliCCP/ctr_cvr.test",
}
# AliCCP 无 split 种子：前缀由预算确定性定义（spec 6.1）
MODEL_SEED = 1688723512
ENV_SEED = 20261003
# ---- 训练配置（spec 8.1/8.3；论文值冻结）----
STAGE1_EPOCHS, STAGE1_PATIENCE = 3, 2
STAGE2_EPOCHS, STAGE2_PATIENCE = 5, 2
BATCH_SIZE, LR = 2000, 1e-4
UNI_COE, ENV_COE = 0.9, 0.1
REG_EMBEDDING, REG_DNN = 1e-4, 7e-6
INPUT_SIZE, EMBEDDING_SIZE = 80, 5
EXPERT_HIDDEN, TOWER_HIDDEN = (128, 64), (32, 32)
DROPOUT = (0.1, 0.3)
NUM_TASKS, NUM_ENVS = 2, 2
NEWTASK_REP_DIM = 64
# ---- 门禁阈值（spec 10；只允许在看到结果之前修改）----
AUC_FLOOR_CTR, AUC_FLOOR_CVR, AUC_FLOOR_BSI = 0.55, 0.50, 0.53
VAL_TEST_GAP_BSI = 0.05
GATE_MIN, GATE_MAX = 0.05, 0.95
ENV_SHARE_MIN = 0.05
IMPROVE_DELTA_AUC_TEST_BSI = 0.005  # provisional（spec 10.1）

ARTIFACT_ROOT = Path("artifacts/aliccp_bench")
SUMMARY_COLUMNS = [
    "run_id", "commit", "tag", "auc_val_bsi_best", "auc_test_bsi",
    "A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "B4", "stage1_id",
]


def build_vocab() -> dict:
    """AliCCP vocab 副本（去掉 101 干扰特征与 301 第三标签列）；绝不修改全局字典（spec 2.2）。"""
    vocab = AliCCP_Vocabulary_Size.copy()
    for key in ("101", "301"):
        vocab.pop(key)
    return vocab


# ---- 前缀身份（spec 5.2）----
def _prefix_budget_check(n: int) -> None:
    if n < 1:
        raise ValueError(f"预算必须为正整数: {n}")


def prefix_sha256(path, n: int) -> str:
    """sha256(表头 + 前 n 条数据行的原始字节)；数据行不足 n 时抛 ValueError。"""
    _prefix_budget_check(n)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        header = handle.readline()
        if not header:
            raise ValueError(f"文件为空: {path}")
        digest.update(header)
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（在第 {i + 1} 行截断）")
            digest.update(line)
    return digest.hexdigest()


def scan_label_counts(path, n: int) -> dict:
    """独立于加载器的原始扫描：click1 / purchase1 / bsi_pos（raw != 2）/ bsi_raw 分布。"""
    _prefix_budget_check(n)
    counts = {"n": 0, "click1": 0, "purchase1": 0, "bsi_pos": 0, "bsi_raw": {}}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（在第 {i + 1} 行截断）")
            parts = line.strip().split(",")
            click, purchase, raw = int(parts[0]), int(parts[1]), int(parts[-1])
            counts["n"] += 1
            counts["click1"] += click
            counts["purchase1"] += purchase
            counts["bsi_pos"] += 0 if raw == 2 else 1
            counts["bsi_raw"][raw] = counts["bsi_raw"].get(raw, 0) + 1
    return counts


def dataset_label_counts(dataset) -> dict:
    """从已加载的 AliCCPDataset 统计标签（bsi 为加载器变换后的值；raw 分布不可得）。"""
    click1 = purchase1 = bsi_pos = 0
    for row in dataset.data:
        click1 += int(row[0])
        purchase1 += int(row[1])
        bsi_pos += int(row[-1])
    return {"n": len(dataset.data), "click1": click1, "purchase1": purchase1, "bsi_pos": bsi_pos}


def verify_label_counts(dataset, budget: int, expected: dict) -> None:
    """A4：加载器实际样本数与标签计数必须与指纹逐项相等（比 AUC 更硬的标签对齐检查）。"""
    if len(dataset) != budget:
        raise AssertionError(f"样本数不符: len(dataset)={len(dataset)} != budget={budget}")
    actual = dataset_label_counts(dataset)
    for key in ("n", "click1", "purchase1", "bsi_pos"):
        if actual[key] != expected[key]:
            raise AssertionError(f"标签计数不符 {key}: loader={actual[key]} != fingerprint={expected[key]}")


def _row_digests(path, n: int) -> set:
    _prefix_budget_check(n)
    digests = set()
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for i in range(n):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} 数据行不足 {n}（在第 {i + 1} 行截断）")
            digests.add(int.from_bytes(hashlib.blake2b(line.strip().encode(), digest_size=8).digest(), "big"))
    return digests


def duplicate_stats(files_budgets) -> dict:
    """整行精确重复统计（切分内 + 两两跨切分）。预算相关量，随指纹固化（spec 2.5）。"""
    sets = {}
    stats = {}
    for tag, (path, budget) in files_budgets.items():
        sets[tag] = _row_digests(path, budget)
        stats[f"{tag}_within"] = int(budget) - len(sets[tag])
    for a, b in itertools.combinations(files_budgets.keys(), 2):
        stats[f"{a}_{b}"] = len(sets[a] & sets[b])
    return stats


def _header_sha256(path) -> str:
    with open(path, "rb") as handle:
        return sha256_bytes(handle.readline())


def _normalize_counts(counts: dict) -> dict:
    """JSON 往返后 bsi_raw 键会变成字符串；归一化后再比较。"""
    return {
        "n": counts["n"],
        "click1": counts["click1"],
        "purchase1": counts["purchase1"],
        "bsi_pos": counts["bsi_pos"],
        "bsi_raw": {str(k): v for k, v in counts["bsi_raw"].items()},
    }


def fingerprint_digest(fp: dict) -> str:
    body = {k: v for k, v in fp.items() if k != "fingerprint_sha256"}
    return sha256_bytes(canonical_json(body).encode())


def build_fingerprint(prefix_tag: str, files_budgets) -> dict:
    files, label_counts, budgets = {}, {}, {}
    for tag, (path, budget) in files_budgets.items():
        files[tag] = {
            "path": str(path),
            "size_bytes": os.path.getsize(path),
            "prefix_sha256": prefix_sha256(path, budget),
        }
        label_counts[tag] = scan_label_counts(path, budget)
        budgets[tag] = int(budget)
    first_tag = next(iter(files_budgets))
    fp = {
        "prefix_tag": prefix_tag,
        "budgets": budgets,
        "files": files,
        "header_sha256": _header_sha256(files_budgets[first_tag][0]),
        "label_counts": label_counts,
        "duplicate_stats": duplicate_stats(files_budgets),
    }
    fp["fingerprint_sha256"] = fingerprint_digest(fp)
    return fp


def verify_fingerprint(fp: dict) -> None:
    """A2：前缀字节、表头、标签计数与自哈希必须与指纹一致；不一致抛 AssertionError。"""
    if fp.get("fingerprint_sha256") != fingerprint_digest(fp):
        raise AssertionError("指纹自哈希不一致（字段被改动）")
    for tag, meta in fp["files"].items():
        budget = fp["budgets"][tag]
        if prefix_sha256(meta["path"], budget) != meta["prefix_sha256"]:
            raise AssertionError(f"前缀字节与指纹不符: {tag} ({meta['path']})")
        counts = scan_label_counts(meta["path"], budget)
        if _normalize_counts(counts) != _normalize_counts(fp["label_counts"][tag]):
            raise AssertionError(f"原始扫描标签计数与指纹不符: {tag}")
        if _header_sha256(meta["path"]) != fp["header_sha256"]:
            raise AssertionError(f"表头与指纹不符: {tag}")


def splits_dir(root, prefix_tag: str) -> Path:
    return Path(root) / "splits" / prefix_tag


def save_fingerprint(root, prefix_tag: str, fp: dict) -> Path:
    path = splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_fingerprint(root, prefix_tag: str) -> dict:
    path = splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
    if not path.exists():
        raise FileNotFoundError(f"前缀指纹不存在: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_fingerprint(root, prefix_tag: str, files_budgets):
    """存在则读取并校验，不存在则构建、校验、落盘。返回 (fp, created)。"""
    path = splits_dir(root, prefix_tag) / "prefix_fingerprint.json"
    if path.exists():
        fp = load_fingerprint(root, prefix_tag)
        verify_fingerprint(fp)
        return fp, False
    fp = build_fingerprint(prefix_tag, files_budgets)
    verify_fingerprint(fp)
    save_fingerprint(root, prefix_tag, fp)
    return fp, True


def run_dir(root, run_id: str) -> Path:
    return Path(root) / "runs" / run_id


def stage1_id_for(meta: dict) -> str:
    return stage1_id(fingerprint_sha=meta["fingerprint_sha256"], model_seed=meta["model_seed"],
                     epochs=meta["epochs"], cfg_sha=meta["config_hash"])


# ---- 加载器与模型构造（原 bench.py）----
def build_loaders(data_files, budgets, batch_size):
    datasets = {split: AliCCPDataset(data_files[split], budgets[split]) for split in ("train", "val", "test")}
    loaders = {
        split: DataLoader(datasets[split], batch_size=batch_size, shuffle=False, num_workers=0)
        for split in ("train", "val", "test")
    }
    return datasets, loaders


def build_mptrec(device, *, vocab=None, expert_hidden=EXPERT_HIDDEN, tower_hidden=TOWER_HIDDEN,
                 embedding_size=EMBEDDING_SIZE, input_size=INPUT_SIZE,
                 reg_embedding=REG_EMBEDDING, reg_dnn=REG_DNN, dropout=DROPOUT) -> MPTRec:
    vocab = dict(vocab) if vocab is not None else build_vocab()
    return MPTRec(
        num_tasks=NUM_TASKS,
        feature_vocabulary=vocab,
        embedding_size=embedding_size,
        input_size=input_size,
        expert_dnn_hidden_units=list(expert_hidden),
        tower_dnn_hidden_units=list(tower_hidden),
        dropout=list(dropout),
        reg_embedding=reg_embedding,
        reg_dnn=reg_dnn,
        device=device,
    )


def _budget_of(datasets, budgets) -> bool:
    return all(len(datasets[split]) == budgets[split] for split in ("train", "val", "test"))


def _versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}


def _reset_peak_vram(device) -> None:
    """torch 2.6 在 CUDA 未初始化时，_cuda_resetPeakMemoryStats 对任何实参都报 Invalid device argument；
    必须先 torch.cuda.init()（实测确认，2026-10-03 smoke）。"""
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)


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
