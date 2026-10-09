"""CensusIncome 数据集适配：常量、划分/指纹、加载器与模型构造。

唯一事实来源：docs/superpowers/specs/2026-09-29-*.md。函数体搬自 census_benchmark/protocol.py
与 run_census_benchmark.py，行为必须保持位级一致；共享纯工具见 benchmark/protocol.py。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from benchmark import gates
from benchmark.protocol import canonical_json, sha256_bytes, sha256_tensor
from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec

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


def build_census_loaders(split_seed: int):
    """返回 (loaders, stats, (val_idx, test_idx))：train.gz 全量训练；test.gz 按 split seed 切成
    val / test，两侧都是 `Subset(test_dataset, idx.tolist())` + `batch_size=BATCH_SIZE`（spec 2.1、4.6）。"""
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_idx, test_idx = make_split(len(test_dataset), split_seed)
    loaders = {"train": DataLoader(train_dataset, batch_size=BATCH_SIZE),
               "val": DataLoader(Subset(test_dataset, val_idx.tolist()), batch_size=BATCH_SIZE),
               "test": DataLoader(Subset(test_dataset, test_idx.tolist()), batch_size=BATCH_SIZE)}
    return loaders, split_stats(val_idx, test_idx, n_train=len(train_dataset)), (val_idx, test_idx)


def build_mptrec(device) -> MPTRec:
    """vocabulary 必须 `CensusIncome_Vocabulary_Size.copy()` 后再 `pop("education")`（spec 2.2.1）。"""
    vocabulary = CensusIncome_Vocabulary_Size.copy()
    vocabulary.pop("education")
    return MPTRec(num_tasks=NUM_TASKS, feature_vocabulary=vocabulary, embedding_size=EMBEDDING_SIZE,
                  input_size=INPUT_SIZE, expert_dnn_hidden_units=list(EXPERT_HIDDEN),
                  tower_dnn_hidden_units=list(TOWER_HIDDEN),
                  reg_embedding=REG_EMBEDDING, reg_dnn=REG_DNN, device=device)


def judge(*, backbone_sha_equal: bool, grads_all_none: bool, split_ok: bool, split_stats_ok: bool,
          env_ids_ok: bool, auc_val_income: float, auc_val_marital: float, auc_test_education: float,
          auc_val_education_best: float, gate_mean: list[float], env_shares: list[float]) -> dict:
    """A1/A2/A4/A5 + B1–B4（spec 8）。A3 按需触发，由调用方另行写入，不进入 overall_pass。"""
    outcomes = gates.evaluate([
        ("A1", backbone_sha_equal and grads_all_none,
         {"backbone_sha_equal": backbone_sha_equal, "grads_all_none": grads_all_none}),
        ("A2", split_ok, {"split_fingerprint_consistent": split_ok}),
        ("A4", split_stats_ok, {"disjoint_and_complete": split_stats_ok}),
        ("A5", env_ids_ok, {"env_ids_sha256_matches_stage1": env_ids_ok}),
        ("B1", min(auc_val_income, auc_val_marital, auc_test_education) >= AUC_FLOOR,
         {"auc_val_income": auc_val_income, "auc_val_marital": auc_val_marital,
          "auc_test_education": auc_test_education, "floor": AUC_FLOOR}),
        ("B2", abs(auc_val_education_best - auc_test_education) <= VAL_TEST_GAP,
         {"gap": abs(auc_val_education_best - auc_test_education), "limit": VAL_TEST_GAP}),
        ("B3", all(GATE_MIN <= w <= GATE_MAX for w in gate_mean), {"gate_mean": gate_mean}),
        ("B4", all(share >= ENV_SHARE_MIN for share in env_shares), {"env_shares": env_shares}),
    ])
    report = gates.render_pass_detail(outcomes)
    report["overall_pass"] = gates.hard_pass((outcome.state for outcome in outcomes), allowed=(gates.PASS,))
    report["failures"] = [outcome.gate_id for outcome in outcomes if outcome.state == gates.FAIL]
    return report
