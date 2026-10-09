"""AliCCP 阶段 1/2 公平评测协议（唯一事实来源：docs/superpowers/specs/2026-10-03-aliccp-fair-benchmark-design.md）。

本模块只包含协议常量与纯工具：前缀指纹、种子、Stage-1 产物读写、冻结三件套、run_id 与 SUMMARY。
不修改任何模型/训练代码（spec 4.2）。
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from config import AliCCP_Vocabulary_Size

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


# ---- 种子（spec 6.1）----
def seed_model(model_seed: int) -> None:
    """只播种训练随机性来源；前缀由预算确定，不存在 split 种子。"""
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    np.random.seed(model_seed)


def make_env_ids(n: int, env_seed: int, num_envs: int = NUM_ENVS) -> torch.Tensor:
    """初始环境分配：独立 Generator，不消耗全局 RNG（spec 6.1）。长度必须 = 训练集长度。"""
    return torch.randint(0, num_envs, size=(n,), generator=torch.Generator().manual_seed(env_seed))


def build_vocab() -> dict:
    """AliCCP vocab 副本（去掉 101 干扰特征与 301 第三标签列）；绝不修改全局字典（spec 2.2）。"""
    vocab = AliCCP_Vocabulary_Size.copy()
    for key in ("101", "301"):
        vocab.pop(key)
    return vocab


# ---- 哈希工具 ----
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    t = tensor.detach().cpu().contiguous()
    return sha256_bytes(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes())


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    return sha256_bytes(canonical_json(cfg).encode())


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def git_state() -> dict:
    """运行时代码溯源：commit7 + 已跟踪文件是否存在未提交修改（未跟踪文件不计入）。"""
    commit = code_commit()
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        )
        dirty = bool(out.strip())
    except Exception:  # noqa: BLE001
        dirty = True
    return {"commit": commit, "dirty": dirty}


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


# ---- Stage-1 产物（spec 7.1：内容寻址、只读）----
def make_stage1_id(prefix_sha: str, model_seed: int, epochs: int, cfg_sha: str) -> str:
    return f"s1-{prefix_sha[:8]}-m{model_seed}-e{epochs}-{cfg_sha[:8]}"


def stage1_dir(root, stage1_id: str) -> Path:
    return Path(root) / "stage1" / stage1_id


def save_stage1(root, stage1_id: str, *, backbone_state: dict, env_ids: torch.Tensor, meta: dict) -> Path:
    out_dir = stage1_dir(root, stage1_id)
    if out_dir.exists():
        raise FileExistsError(f"stage1 产物已存在，协议禁止覆盖: {out_dir}")
    out_dir.mkdir(parents=True)
    torch.save(backbone_state, out_dir / "backbone.pt")
    torch.save(env_ids, out_dir / "env_ids.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out_dir


def load_stage1(root, stage1_id: str) -> dict:
    """阶段 2 的唯一 backbone 来源（spec 7.2）。"""
    out_dir = stage1_dir(root, stage1_id)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"stage1 产物不存在: {out_dir}")
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"stage1 meta 缺失（产物无效）: {meta_path}")
    return {
        "backbone_state": torch.load(out_dir / "backbone.pt", map_location="cpu"),
        "env_ids": torch.load(out_dir / "env_ids.pt", map_location="cpu"),
        "meta": json.loads(meta_path.read_text(encoding="utf-8")),
    }


# ---- 真冻结三件套（spec 7.3；外部参数遍历，不改模型类）----
def freeze_backbone(backbone: torch.nn.Module) -> None:
    """参数级 + 模式级；计算图级（no_grad 抽表征）由调用方保证。"""
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)


def backbone_sha256(backbone: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, param in sorted(backbone.named_parameters()):
        digest.update(name.encode())
        digest.update(sha256_tensor(param).encode())
    return digest.hexdigest()


def assert_no_grads(backbone: torch.nn.Module) -> None:
    dirty = [name for name, param in backbone.named_parameters() if param.grad is not None]
    if dirty:
        raise AssertionError(f"backbone 参数残留梯度（冻结失效）: {dirty}")


# ---- run_id 与 SUMMARY（spec 11）----
def run_dir(root, run_id: str) -> Path:
    return Path(root) / "runs" / run_id


def make_run_id(now, *, prefix_tag: str, model_seed: int, tag: str, commit: str) -> str:
    return f"{now:%Y%m%d-%H%M}-{prefix_tag}-m{model_seed}-{tag}-{commit}"


def append_summary_row(path, row: dict, columns=None) -> None:
    """只追加，永不重写已有行（spec 9.3/11.2；不得只报告成功的 run）。"""
    columns = list(columns or SUMMARY_COLUMNS)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "| " + " | ".join(columns) + " |\n" + "|" + "---|" * len(columns) + "\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")
