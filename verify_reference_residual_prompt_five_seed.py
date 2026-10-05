"""只读参照核验与 run-reference pin（预注册 §4.3；每个 seed 的 arm 运行之前执行）。

用法（cwd = worktree 根；主树 venv 解释器）：
    python verify_reference_residual_prompt_five_seed.py --seed 1688738016 \
        --baseline-run 20261005-XXXX-p2M-v500k-t1M-m1688738016-short-<commit>

做三件事（零训练、零写入 canonical 产物；只写 audit/five-seed/reference_pin_seed<S>.json）：
1. 从配对基线 run 的 metrics.json 取 stage1_id，加载固定 Stage-1 产物并真冻结；
2. 以与运行期 `reference_head_stats` **同口径**（val 全量、fp64 终算、unbiased=False）复算参照头
   统计：val_auc 与 pred_std；并用脚本内**独立实现**（不复用 RP 函数）逐位复核；
3. 校验参照身份：ref_val_auc == 配对基线 metrics.json:best_val_auc_bsi（逐位），
   并写出 pin JSON（含 provenance：stage1_id、baseline_run_id、newtask.pt sha256、
   backbone/env_ids/fingerprint sha、脚本 sha256）。

pin 的 baseline_auc_test / baseline_auc_val / reference_pred_std 三值即处理臂运行前必须设置到
RP_BASELINE_AUC_TEST / RP_BASELINE_AUC_VAL / RP_REFERENCE_PRED_STD 的逐位值。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from aliccp_benchmark import metrics, protocol, residual_prompt as RP  # noqa: E402
from multitaskrec.dataset import AliCCPDataset  # noqa: E402
from multitaskrec.model import MPTRec, NewTask  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def independent_reference_stats(head, backbone, loader, device) -> dict:
    """脚本内独立实现（逐字对齐参考口径，但不调用 RP.reference_head_stats）。"""
    head.eval()
    backbone.eval()
    ys, preds = [], []
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
        preds.append(head(dnn_input, gen_rep, spec_reps, env_embs).detach().float().cpu())
        ys.append(y)
    pred = torch.cat(preds).double()
    return {"val_auc": metrics.auc_score(torch.cat(ys), pred),
            "pred_std": float(pred.std(unbiased=False)),
            "pred_mean": float(pred.mean()), "n_samples": int(pred.numel())}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="只读参照核验与 run-reference pin（预注册 §4.3）")
    parser.add_argument("--root", type=str, default=str(protocol.ARTIFACT_ROOT))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="pin 输出目录（默认 <root>/audit/five-seed；dry-run 用）")
    args = parser.parse_args(argv)

    root = Path(args.root)
    device = torch.device(f"cuda:{args.gpu}")
    baseline_dir = protocol.run_dir(root, args.baseline_run)
    base_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    assert int(base_metrics["model_seed"]) == int(args.seed), "配对基线 model_seed 不符"
    stage1_id = base_metrics["stage1_id"]
    baseline_test_auc = float(base_metrics["test_auc_bsi"])
    baseline_val_auc = float(base_metrics["best_val_auc_bsi"])

    art = protocol.load_stage1(root, stage1_id)
    vocab = protocol.build_vocab()
    model = MPTRec(num_tasks=protocol.NUM_TASKS, feature_vocabulary=vocab,
                   embedding_size=protocol.EMBEDDING_SIZE, input_size=protocol.INPUT_SIZE,
                   expert_dnn_hidden_units=list(protocol.EXPERT_HIDDEN),
                   tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
                   dropout=list(protocol.DROPOUT), reg_embedding=protocol.REG_EMBEDDING,
                   reg_dnn=protocol.REG_DNN, device=device).to(device)
    model.load_state_dict(art["backbone_state"])
    protocol.freeze_backbone(model)
    assert protocol.backbone_sha256(model) == art["meta"]["backbone_sha256"], "backbone sha 与产物 meta 不符"

    dataset = AliCCPDataset(protocol.DATA_FILES["val"], protocol.VAL_BUDGET)
    loader = DataLoader(dataset, batch_size=protocol.BATCH_SIZE, shuffle=False, num_workers=0)
    assert len(dataset) == protocol.VAL_BUDGET, "val 样本数与预算不符"

    newtask_path = baseline_dir / "newtask.pt"
    rep_dim = int(list(protocol.EXPERT_HIDDEN)[-1])

    # 路径 1：运行期同口径（机制模块函数）
    same_path = RP.reference_head_stats(newtask_path, model, loader, device,
                                        input_size=protocol.INPUT_SIZE, rep_dim=rep_dim,
                                        tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
                                        reg_dnn=protocol.REG_DNN)
    # 路径 2：脚本内独立实现
    with RP.isolated_cpu_rng():
        head = NewTask(input_size=protocol.INPUT_SIZE, rep_dim=rep_dim,
                       tower_dnn_hidden_units=list(protocol.TOWER_HIDDEN),
                       reg_dnn=protocol.REG_DNN, device=device)
    head.load_state_dict(torch.load(newtask_path, map_location="cpu"), strict=True)
    head.to(device)
    independent = independent_reference_stats(head, model, loader, device)

    ref_auc = float(same_path["val_auc"])
    ref_std = float(same_path["pred_dispersion"]["pred_std"])
    checks = {
        "same_path_auc_equals_independent": ref_auc == independent["val_auc"],
        "same_path_std_equals_independent": ref_std == independent["pred_std"],
        "ref_auc_bit_equal_to_baseline_val_auc": ref_auc == baseline_val_auc,
        "n_samples_is_val_budget": independent["n_samples"] == protocol.VAL_BUDGET,
        "extra_keys_absent_in_pin_head": not hasattr(head, "prompt_gate"),
    }
    if not all(checks.values()):
        raise SystemExit(f"参照核验失败，拒绝写 pin：{checks}")

    pin = {
        "model_seed": int(args.seed),
        "stage1_id": stage1_id,
        "baseline_run_id": args.baseline_run,
        "baseline_auc_test": baseline_test_auc,
        "baseline_auc_val": baseline_val_auc,
        "reference_pred_std": ref_std,
        "reference_val_auc": ref_auc,
        "reference_pred_mean": float(same_path["pred_dispersion"]["pred_mean"]),
        "n_samples": independent["n_samples"],
        "newtask_pt_sha256": sha256_file(newtask_path),
        "backbone_sha256": art["meta"]["backbone_sha256"],
        "env_ids_sha256": art["meta"]["env_ids_sha256"],
        "fingerprint_sha256": art["meta"]["fingerprint_sha256"],
        "baseline_metrics_commit": base_metrics.get("commit"),
        "script_sha256": sha256_file(Path(__file__)),
        "checks": checks,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = Path(args.out_dir) if args.out_dir else root / "audit" / "five-seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reference_pin_seed{args.seed}.json"
    if out_path.exists():
        raise SystemExit(f"pin 已存在，禁止覆盖：{out_path}")
    out_path.write_text(json.dumps(pin, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pin] seed={args.seed} stage1_id={stage1_id} baseline={args.baseline_run}")
    print(f"[pin] baseline_auc_test={baseline_test_auc!r} baseline_auc_val={baseline_val_auc!r}")
    print(f"[pin] reference_pred_std={ref_std!r} (ref_val_auc={ref_auc!r}, == baseline val: True)")
    print(f"[pin] checks: {json.dumps(checks, ensure_ascii=False)}")
    print(f"[pin] newtask.pt sha256={pin['newtask_pt_sha256']}")
    print(f"[pin] written: {out_path}")
    print("[pin] 处理臂 env（逐位复制）：")
    print(f'  $env:RP_BASELINE_AUC_TEST = "{baseline_test_auc!r}"')
    print(f'  $env:RP_BASELINE_AUC_VAL = "{baseline_val_auc!r}"')
    print(f'  $env:RP_REFERENCE_PRED_STD = "{ref_std!r}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
