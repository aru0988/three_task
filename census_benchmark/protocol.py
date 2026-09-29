"""CensusIncome 阶段 2 公平评测协议（唯一事实来源：docs/superpowers/specs/2026-09-29-*.md）。"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

# ---- 三个独立种子（spec 5.2）----
SPLIT_SEED = 20260929          # 唯一决定 val/test 划分，跨 model seed 恒定
MODEL_SEED = 1685480945        # 首轮唯一 model seed（模型初始化 / batch 顺序 / dropout）
ENV_SEED = 20260929            # 只决定初始 env_ids
# ---- 首轮 smoke 配置（spec 6.1：只降 epochs，不动数据与超参）----
STAGE1_EPOCHS, STAGE2_EPOCHS, PATIENCE = 2, 5, 2
# ---- 论文超参（spec 2.2.3，本分支冻结）----
BATCH_SIZE, LR = 256, 1e-3
REG_EMBEDDING, REG_DNN = 0.006, 3e-5
UNI_COE, ENV_COE = 0.9, 0.1
INPUT_SIZE, EMBEDDING_SIZE = 123, 4
EXPERT_HIDDEN, TOWER_HIDDEN = (256, 128), (64, 32)
NUM_TASKS, NUM_ENVS = 2, 2
# ---- 门禁阈值（spec 8；只允许在看到结果之前修改）----
AUC_FLOOR, VAL_TEST_GAP, GATE_MIN, GATE_MAX, ENV_SHARE_MIN = 0.60, 0.03, 0.05, 0.95, 0.05

ARTIFACT_ROOT = Path("artifacts/census_stage2")
SUMMARY_COLUMNS = ["run_id", "commit", "auc_test_education",
                   "A1", "A2", "A4", "A5", "B1", "B2", "B3", "B4", "stage1_id"]


def seed_model(model_seed: int) -> None:
    """只播种训练随机性来源；绝不参与划分（spec 5.2）。"""
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    np.random.seed(model_seed)


def make_env_ids(n: int, env_seed: int, num_envs: int = NUM_ENVS) -> torch.Tensor:
    """初始环境分配：独立 Generator，不消耗全局 RNG（spec 5.3.2）。长度必须 = 训练集长度。"""
    return torch.randint(0, num_envs, size=(n,), generator=torch.Generator().manual_seed(env_seed))


def make_split(n_test: int, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按索引切分 test.gz，返回排序后的 (val_idx, test_idx)：与 train_test_split(dataset, ...,
    random_state=split_seed) 同一排列，但可落盘、可校验。"""
    val_idx, test_idx = train_test_split(np.arange(n_test), test_size=0.5, random_state=split_seed)
    return np.sort(val_idx), np.sort(test_idx)


def split_stats(val_idx: np.ndarray, test_idx: np.ndarray, n_train: int) -> dict:
    """A4 的三集合实测行数与交并检查。"""
    n = len(val_idx) + len(test_idx)
    return {"n_train": int(n_train), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "disjoint": bool(set(val_idx.tolist()).isdisjoint(test_idx.tolist())),
            "union_complete": bool(sorted(val_idx.tolist() + test_idx.tolist()) == list(range(n)))}


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
    except Exception:
        return "nogit"


def split_fingerprint(*, split_seed: int, stats: dict, val_idx, test_idx) -> dict:
    """确定性指纹：不含时间戳，同 split seed 必须逐字节一致（A2）。"""
    fp = {"split_seed": int(split_seed), **stats,
          "val_sha256": sha256_tensor(torch.from_numpy(val_idx)),
          "test_sha256": sha256_tensor(torch.from_numpy(test_idx)),
          "val_indices_head": val_idx[:5].tolist(), "test_indices_head": test_idx[:5].tolist()}
    fp["fingerprint_sha256"] = sha256_bytes(canonical_json(fp).encode())
    return fp


def split_dir(root: Path, split_seed: int) -> Path:
    return Path(root) / "splits" / str(split_seed)


def write_split_fingerprint(root: Path, split_seed: int, fp: dict) -> Path:
    path = split_dir(root, split_seed) / "split_fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fp, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_split_fingerprint(root: Path, split_seed: int) -> dict | None:
    path = split_dir(root, split_seed) / "split_fingerprint.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def verify_split_fingerprint(fp: dict, baseline: dict) -> bool:
    """A2：指纹只由 split seed 决定，必须与基准逐字节一致。"""
    return bool(baseline) and fp["fingerprint_sha256"] == baseline["fingerprint_sha256"]


def save_split_indices(root: Path, split_seed: int, val_idx, test_idx) -> Path:
    path = split_dir(root, split_seed) / "split_indices.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, val=val_idx, test=test_idx)
    return path


# ---- Stage-1 产物（spec 4.2：内容寻址、只读）----
def stage1_id(*, split_sha: str, model_seed: int, epochs: int, cfg_sha: str) -> str:
    """s1-<split指纹前8>-m<modelSeed>-e<epochs>-<config哈希前8>"""
    return f"s1-{split_sha[:8]}-m{model_seed}-e{epochs}-{cfg_sha[:8]}"


def stage1_dir(root: Path, sid: str) -> Path:
    return Path(root) / "stage1" / sid


def save_stage1(root: Path, sid: str, *, backbone_state: dict, env_ids: torch.Tensor, meta: dict) -> Path:
    out_dir = stage1_dir(root, sid)
    if out_dir.exists():
        raise FileExistsError(f"stage1 产物已存在，协议禁止覆盖: {out_dir}")
    out_dir.mkdir(parents=True)
    torch.save(backbone_state, out_dir / "backbone.pt")
    torch.save(env_ids, out_dir / "env_ids.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def load_stage1(root: Path, sid: str) -> dict:
    """阶段 2 的唯一 backbone 来源（spec 4.3）。"""
    out_dir = stage1_dir(root, sid)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"stage1 产物不存在: {out_dir}")
    return {"backbone_state": torch.load(out_dir / "backbone.pt", map_location="cpu"),
            "env_ids": torch.load(out_dir / "env_ids.pt", map_location="cpu"),
            "meta": json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))}


# ---- 真冻结三件套（spec 4.4、4.5；外部参数遍历，不改模型类）----
def freeze_backbone(backbone: torch.nn.Module) -> None:
    """三件套之 1（参数级）+ 2（模式级）；第 3 件（no_grad 抽表征）由调用方保证。"""
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)


def backbone_sha256(backbone: torch.nn.Module) -> str:
    """按参数名排序拼 (name, 张量哈希) 的 sha256 → A1 的 before/after 比对。"""
    digest = hashlib.sha256()
    for name, param in sorted(backbone.named_parameters()):
        digest.update(name.encode())
        digest.update(sha256_tensor(param).encode())
    return digest.hexdigest()


def assert_no_grads(backbone: torch.nn.Module) -> None:
    dirty = [name for name, param in backbone.named_parameters() if param.grad is not None]
    if dirty:
        raise AssertionError(f"backbone 参数残留梯度（冻结失效）: {dirty}")


# ---- run_id 与 SUMMARY（spec 9.1、9.2）----
def make_run_id(now, *, split_seed: int, model_seed: int, tag: str, commit: str) -> str:
    return f"{now:%Y%m%d-%H%M}-s{split_seed}-m{model_seed}-{tag}-{commit}"


def append_summary_row(path: Path, row: dict) -> None:
    """只追加，永不重写已有行（spec 7.3；不得只报告成功的 run）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("| " + " | ".join(SUMMARY_COLUMNS) + " |\n" + "|" + "---|" * len(SUMMARY_COLUMNS) + "\n",
                        encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(row[col]) for col in SUMMARY_COLUMNS) + " |\n")
