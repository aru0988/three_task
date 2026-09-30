"""零训练前置诊断：复算 NewTask 原始 router W，判定条件化 Scale+Bias 是否值得进入五 epoch。

唯一事实来源：docs/superpowers/specs/2026-09-30-stage2-cond-scale-bias-design.md

本模块的硬约束（spec 2.2）：
  * **零训练**：不构造优化器、不反传、不更新任何参数；只前向复算 router。
  * **只读 checkpoint**：`--stage1-dir` 与 `--newtask-checkpoint` 都只读，绝不写回 stage1/ 或 runs/。
  * **不改模型语义**：不 import 也不修改 `multitaskrec/model.py` 的模型类（只读 NewTask 的属性）。
  * **诊断对象 = 基线 run 训练后的新任务头**：`--newtask-checkpoint` 必填，按 baseline Stage-2 的
    model seed 与构造顺序构造出 NewTask 后 strict 载入；键/形状一旦不符即报错——宁可拒绝出数，
    也不静默地换成随机初始化的头。

用法：
    python -m census_benchmark.router_probe --stage1-dir <stage1_dir> --newtask-checkpoint <newtask.pt> --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from census_benchmark import protocol as P
from multitaskrec.model import NewTask

# ---- 预注册门槛（spec 3.2）：只允许在看到结果之前修改 ----
ROUTER_W_STD_MIN = 0.02
PROBE_DIRNAME = "probes"


def router_weights(newtask, dnn_input: torch.Tensor, env_embs: list[torch.Tensor]) -> torch.Tensor:
    """复算 `NewTask.forward` 里的原始 router：W = softmax(projection(dnn_input) @ E / T)。

    `env_embs` 由冻结 backbone 给出（`MPTRec.get_infos` 的第 4 个返回值），E = stack(env_embs, dim=1)。
    温度取自 `newtask.temperature`（协议值 150），不额外传参——避免复算与真实 forward 用错温度。
    与 forward 的逐位一致性由 test_router_probe 里的 F.softmax 探针锁定。
    """
    exist_env_embs = torch.stack(env_embs, dim=1)
    h_out = newtask.projection_network(dnn_input)
    return F.softmax(torch.mm(h_out, exist_env_embs) / newtask.temperature, dim=-1)


class RouterStats:
    """router W 的样本级统计（spec 3.1）；累加器恒在 CPU 且为 float64（显存 O(1)）。

    四项指标口径：
      * `router_w_std`      每个 env 列在样本维度的总体标准差（ddof=0）之**均值**。
      * `normalized_entropy` 每样本 H(W)/log(K) 的均值，∈[0,1]；K 为 env 数。
      * `source_share`      每 env 被 argmax 命中的样本占比（长度 K，和为 1；并列时取最小下标）。
      * `top1_share`        每样本 max_k W[b,k] 的均值（路由置信度）。
    """

    def __init__(self, num_envs: int):
        if num_envs < 2:
            raise ValueError(f"normalized_entropy 需要至少 2 个 env（log(K) 必须 > 0），收到 {num_envs}")
        self.num_envs = num_envs
        self.total = torch.zeros(num_envs, dtype=torch.float64)
        self.total_sq = torch.zeros(num_envs, dtype=torch.float64)
        self.argmax_counts = torch.zeros(num_envs, dtype=torch.float64)
        self.entropy_sum = 0.0
        self.top1_sum = 0.0
        self.count = 0

    def update(self, w: torch.Tensor) -> None:
        """`w` 是 [B, K] 的 router 概率。归约留在原设备，结果显式 .cpu()（同 GateStats 的教训）。"""
        w64 = w.detach().double()
        self.total += w64.sum(dim=0).cpu()
        self.total_sq += (w64 * w64).sum(dim=0).cpu()
        # -Σ p log p：p=0 处用 where 显式取 0（p*log p 在该点是 0*-inf=nan，但不被选中）
        safe = w64.clamp_min(0.0)
        entropy = -torch.where(safe > 0, safe * safe.log(), torch.zeros_like(safe)).sum(dim=1)
        self.entropy_sum += float(entropy.sum().cpu())
        self.top1_sum += float(w64.max(dim=1).values.sum().cpu())
        self.argmax_counts += torch.bincount(w64.argmax(dim=1).cpu(), minlength=self.num_envs).double()
        self.count += int(w64.shape[0])

    def result(self) -> dict:
        n = max(self.count, 1)
        mean = self.total / n
        std = (self.total_sq / n - mean * mean).clamp_min(0).sqrt()          # 总体 std（ddof=0）
        return {"router_w_std": float(std.mean()),
                "router_w_std_per_env": std.tolist(),
                "normalized_entropy": self.entropy_sum / n / math.log(self.num_envs),
                "source_share": (self.argmax_counts / n).tolist(),
                "top1_share": self.top1_sum / n,
                "n_samples": self.count}


def gate_verdict(router_w_std: float, threshold: float = ROUTER_W_STD_MIN) -> dict:
    """预注册门槛（spec 3.2）：`router_w_std >= 0.02`（含等号）才允许进入条件化 Scale+Bias 五 epoch。"""
    return {"rule": f"router_w_std >= {threshold}", "threshold": float(threshold),
            "observed": float(router_w_std), "proceed": bool(router_w_std >= threshold)}


def state_dict_sha256(state: dict) -> str:
    """按张量名排序拼 (name, 张量哈希) 的 sha256：证明诊断用的确实是 checkpoint 里的那个头。"""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(P.sha256_tensor(torch.as_tensor(value)).encode())
    return digest.hexdigest()


def load_newtask_state(path) -> dict:
    """只读基线 run 的 `newtask.pt`。缺失 → FileNotFoundError；不是张量 state_dict → ValueError。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"NewTask checkpoint 不存在（--newtask-checkpoint）: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"NewTask checkpoint 不是张量 state_dict: {path}（{type(state).__name__}）")
    return state


def _load_newtask_state_into(newtask, state: dict) -> None:
    """strict 载入；键或形状不符一律报错——绝不静默换成随机初始化的头。"""
    own = set(newtask.state_dict())
    missing, unexpected = sorted(own - set(state)), sorted(set(state) - own)
    if missing or unexpected:
        raise ValueError(f"NewTask checkpoint 与模型结构不兼容：missing={missing} unexpected={unexpected}")
    try:
        newtask.load_state_dict(state)
    except RuntimeError as exc:                                              # 键齐但形状不符
        raise ValueError(f"NewTask checkpoint 与模型结构不兼容：{exc}") from exc


def verify_newtask_run_config(checkpoint_path, *, stage1_id: str, model_seed: int) -> dict | None:
    """checkpoint 旁若有同 run 的 `config.json`，校验它确实是在这份 backbone / model seed 上训出来的。

    不存在则返回 None（允许只带 .pt）；存在但与 --stage1-dir 的口径不一致 → ValueError，拒绝继续。
    """
    config_path = Path(checkpoint_path).parent / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, want in (("stage1_id", stage1_id), ("model_seed", model_seed)):
        if config.get(key) != want:
            raise ValueError(f"{config_path} 的 {key}={config.get(key)} 与 {want} 不一致，拒绝继续")
    return {"config_path": str(config_path), "run_id": config.get("run_id"),
            "stage1_id": config.get("stage1_id"), "model_seed": config.get("model_seed")}


def probe_newtask(device, *, model_seed: int = P.MODEL_SEED, input_size: int = P.INPUT_SIZE,
                  rep_dim: int = P.EXPERT_HIDDEN[-1], make_backbone=None,
                  backbone_state: dict | None = None, newtask_state: dict | None = None) -> tuple:
    """复现 baseline 阶段 2 的 (backbone, NewTask)：**构造顺序即保真**。

    `run_census_benchmark.run_stage2` 的构造顺序是
        seed_model(model_seed) → build_mptrec(device) → to(device) → load_state_dict → freeze → NewTask(...)
    其中只有 `seed_model`、`build_mptrec`、`NewTask(...)` 三步消耗全局 RNG（to/load/freeze 都不消耗），
    所以这里必须保持同一顺序，否则模型结构与基线阶段 2 漂移，诊断的就不是同一个头
    （test_router_probe 用正反两例锁定该顺序）。

    `newtask_state` = 基线 run 训练后的 `newtask.pt`：构造完 NewTask 后 strict 载入，键/形状不符即报错。
    """
    if make_backbone is None:                                                # 单一事实来源，避免 vocab/超参漂移
        import run_census_benchmark
        make_backbone = run_census_benchmark.build_mptrec
    P.seed_model(model_seed)
    backbone = make_backbone(device)
    backbone.to(device)
    if backbone_state is not None:
        backbone.load_state_dict(backbone_state)
    P.freeze_backbone(backbone)
    newtask = NewTask(input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=list(P.TOWER_HIDDEN),
                      reg_dnn=P.REG_DNN, device=device).to(device)
    if newtask_state is not None:
        _load_newtask_state_into(newtask, newtask_state)
    newtask.eval()
    return backbone, newtask


@torch.no_grad()
def probe_loader(newtask, backbone, loader, device, num_envs: int = P.NUM_ENVS) -> dict:
    """在固定划分上零训练复算 router 并统计（三件套之 3：no_grad 抽表征）。"""
    newtask.eval()
    backbone.eval()
    stats = RouterStats(num_envs)
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, _, _, env_embs = backbone.get_infos(features)
        if len(env_embs) != num_envs:                                        # env 数不符 → 统计口径失真
            raise ValueError(f"env 数 {len(env_embs)} 与协议 {num_envs} 不一致，拒绝出数")
        stats.update(router_weights(newtask, dnn_input, env_embs))
    return stats.result()


def run_router_probe(root=P.ARTIFACT_ROOT, *, stage1_dir, newtask_checkpoint, split: str = "val", device=None,
                     make_backbone=None, loaders=None, stats=None, indices=None,
                     input_size: int = P.INPUT_SIZE, rep_dim: int = P.EXPERT_HIDDEN[-1], now=None) -> dict:
    """前置诊断主流程：加载固定 stage1 + 基线训练后的新任务头 → 固定划分上复算 router → 落盘门槛判定。"""
    root = Path(root)
    device = device or torch.device("cuda:0")
    sid = Path(stage1_dir).name
    checkpoint = P.load_stage1(root, sid)                                    # 只读；缺失即抛错
    meta = checkpoint["meta"]
    if meta.get("stage1_id") != sid:
        raise ValueError(f"目录名 {sid} 与 meta.stage1_id {meta.get('stage1_id')} 不一致，拒绝继续")

    # 0) 诊断对象：基线 run 训练后的新任务头（只读；缺失 / 不兼容 / 口径不符一律拒绝，spec 2.1）
    newtask_checkpoint = Path(newtask_checkpoint)
    newtask_state = load_newtask_state(newtask_checkpoint)
    checkpoint_state_sha = state_dict_sha256(newtask_state)
    newtask_run_config = verify_newtask_run_config(newtask_checkpoint, stage1_id=sid,
                                                   model_seed=meta["model_seed"])

    # 1) 同一固定划分：只由 stage1 的 split seed 决定，双向校验（A2）
    if loaders is None:
        import run_census_benchmark
        loaders, stats, indices = run_census_benchmark.build_census_loaders(meta["split_seed"])
    val_idx, test_idx = indices
    fp = P.split_fingerprint(split_seed=meta["split_seed"], stats=stats, val_idx=val_idx, test_idx=test_idx)
    split_ok = (P.verify_split_fingerprint(fp, P.load_split_fingerprint(root, meta["split_seed"]))
                and fp["fingerprint_sha256"] == meta["split_fingerprint_sha256"])
    if not split_ok:
        raise RuntimeError("划分指纹与基准/stage1 记录不一致（A2 失败），协议禁止继续")

    # 2) 冻结 backbone + 按 stage2 顺序构造新任务头并载入训练后权重（零训练；顺序见 probe_newtask）
    backbone, newtask = probe_newtask(device, model_seed=meta["model_seed"], input_size=input_size,
                                      rep_dim=rep_dim, make_backbone=make_backbone,
                                      backbone_state=checkpoint["backbone_state"],
                                      newtask_state=newtask_state)
    sha_before = P.backbone_sha256(backbone)
    # clone 后再哈希：state_dict() 返回的是参数本身，直接哈希会因引用同一块内存而恒等
    newtask_sha = state_dict_sha256({name: value.detach().clone()
                                     for name, value in newtask.state_dict().items()})
    if newtask_sha != checkpoint_state_sha:
        raise RuntimeError("strict 载入后 NewTask 与 checkpoint 不一致，拒绝出数")

    # 3) 诊断：只在固定划分上复算 router W
    metrics_out = probe_loader(newtask, backbone, loaders[split], device)

    # 4) 零训练证据：参数未变、无残留梯度
    sha_after = P.backbone_sha256(backbone)
    grads_all_none = all(param.grad is None for param in backbone.parameters())
    P.assert_no_grads(backbone)
    frozen = all(not param.requires_grad for param in backbone.parameters()) and not backbone.training

    gate = gate_verdict(metrics_out["router_w_std"])
    probe_id = f"rp-{sid}-{split}-{checkpoint_state_sha[:8]}"                 # 换 checkpoint = 换产物目录
    payload = {
        "probe_id": probe_id, "stage1_id": sid, "commit": P.code_commit(),
        "created": (now or datetime.now()).isoformat(timespec="seconds"),
        "split": split, "model_seed": meta["model_seed"], "split_seed": meta["split_seed"],
        "num_envs": P.NUM_ENVS, "temperature": int(newtask.temperature),
        "input_size": input_size, "rep_dim": rep_dim,
        "split_fingerprint_sha256": fp["fingerprint_sha256"], "split_sha256": fp[f"{split}_sha256"],
        "stage1_backbone_sha256": meta["backbone_sha256"],
        "stage1_backbone_sha256_matches": meta["backbone_sha256"] == sha_before,
        "backbone_sha256_before": sha_before, "backbone_sha256_after": sha_after,
        "newtask_checkpoint": str(newtask_checkpoint),
        "newtask_checkpoint_sha256": P.sha256_bytes(newtask_checkpoint.read_bytes()),
        "newtask_checkpoint_state_sha256": checkpoint_state_sha, "newtask_state_sha256": newtask_sha,
        "newtask_run_config": newtask_run_config,
        "frozen": bool(frozen), "grads_all_none": bool(grads_all_none), "trained_steps": 0,
        "split_ok": bool(split_ok), "metrics": metrics_out, "gate": gate,
    }
    out_dir = root / PROBE_DIRNAME / probe_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "router_probe.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
    print(f"[router-probe] probe_id={probe_id} dir={out_dir}")
    print(f"[router-probe] newtask_checkpoint={newtask_checkpoint} sha256={newtask_sha[:12]}")
    print(f"[router-probe] router_w_std={gate['observed']:.6f} "
          f"normalized_entropy={metrics_out['normalized_entropy']:.6f} "
          f"source_share={metrics_out['source_share']} top1_share={metrics_out['top1_share']:.6f} "
          f"n_samples={metrics_out['n_samples']}")
    gate_message = "PASS：允许进入条件化 Scale+Bias（5 epoch）" if gate["proceed"] else "FAIL：方向停止（spec 3.2）"
    print(f"[router-probe] gate: {gate['rule']} → {gate_message}")
    return {"probe_id": probe_id, "dir": str(out_dir), "metrics": metrics_out, "gate": gate, "payload": payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m census_benchmark.router_probe",
        description="零训练前置诊断：复算 NewTask 原始 router W，判定条件化 Scale+Bias 是否进入 5 epoch")
    parser.add_argument("--stage1-dir", type=Path, required=True, help="Stage-1 产物目录（只读）")
    parser.add_argument("--newtask-checkpoint", type=Path, required=True,
                        help="基线 run 训练后的 newtask.pt（只读；缺失或不兼容即拒绝运行）")
    parser.add_argument("--root", type=Path, default=P.ARTIFACT_ROOT, help="产物根目录（probes/ 落盘位置）")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", choices=["val", "test"], default="val", help="诊断所用固定划分（默认 val）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[router-probe] stage1_dir={args.stage1_dir} split={args.split} device={device} root={args.root}")
    run_router_probe(args.root, stage1_dir=args.stage1_dir, newtask_checkpoint=args.newtask_checkpoint,
                     split=args.split, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
