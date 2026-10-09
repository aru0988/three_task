"""公平评测共享协议工具：两个数据集共用且实现完全相同的部分。

函数体从 census_benchmark/aliccp_benchmark 的既有实现逐字节搬移（两处原本即逐字节相同）；
改动必须保持既有产物（stage1_id / config_hash / 冻结哈希 / run_id / SUMMARY）位级一致。
数据集差异（划分、指纹、门禁、常量）见 benchmark/datasets/<name>.py。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_model(model_seed: int) -> None:
    """只播种训练随机性来源；绝不参与数据划分/前缀。"""
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    np.random.seed(model_seed)


def make_env_ids(n: int, env_seed: int, num_envs: int = 2) -> torch.Tensor:
    """初始环境分配：独立 Generator，不消耗全局 RNG。长度必须 = 训练集长度。"""
    return torch.randint(0, num_envs, size=(n,), generator=torch.Generator().manual_seed(env_seed))


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


# ---- Stage-1 产物（内容寻址、只读）----
def stage1_id(*, fingerprint_sha: str, model_seed: int, epochs: int, cfg_sha: str) -> str:
    """s1-<指纹前8>-m<modelSeed>-e<epochs>-<config哈希前8>。"""
    return f"s1-{fingerprint_sha[:8]}-m{model_seed}-e{epochs}-{cfg_sha[:8]}"


def stage1_dir(root, sid: str) -> Path:
    return Path(root) / "stage1" / sid


def save_stage1(root, sid: str, *, backbone_state: dict, env_ids: torch.Tensor, meta: dict) -> Path:
    out_dir = stage1_dir(root, sid)
    if out_dir.exists():
        raise FileExistsError(f"stage1 产物已存在，协议禁止覆盖: {out_dir}")
    out_dir.mkdir(parents=True)
    torch.save(backbone_state, out_dir / "backbone.pt")
    torch.save(env_ids, out_dir / "env_ids.pt")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return out_dir


def load_stage1(root, sid: str) -> dict:
    """阶段 2 的唯一 backbone 来源。"""
    out_dir = stage1_dir(root, sid)
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


# ---- 真冻结三件套（外部参数遍历，不改模型类）----
def freeze_backbone(backbone: torch.nn.Module) -> None:
    """参数级 + 模式级；计算图级（no_grad 抽表征）由调用方保证。"""
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)


def backbone_sha256(backbone: torch.nn.Module) -> str:
    """按参数名排序拼 (name, 张量哈希) 的 sha256 → before/after 比对。"""
    digest = hashlib.sha256()
    for name, param in sorted(backbone.named_parameters()):
        digest.update(name.encode())
        digest.update(sha256_tensor(param).encode())
    return digest.hexdigest()


def assert_no_grads(backbone: torch.nn.Module) -> None:
    dirty = [name for name, param in backbone.named_parameters() if param.grad is not None]
    if dirty:
        raise AssertionError(f"backbone 参数残留梯度（冻结失效）: {dirty}")


# ---- run_id 与 SUMMARY ----
def make_run_id(now, *, prefix: str, model_seed: int, tag: str, commit: str) -> str:
    return f"{now:%Y%m%d-%H%M}-{prefix}-m{model_seed}-{tag}-{commit}"


def append_summary_row(path, row: dict, columns) -> None:
    """只追加，永不重写已有行（不得只报告成功的 run）。"""
    columns = list(columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "| " + " | ".join(columns) + " |\n" + "|" + "---|" * len(columns) + "\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(row[col]) for col in columns) + " |\n")
