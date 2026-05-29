"""Run Stage 2 from cached Stage 1 checkpoint, supporting all 5 fusion modes.

Usage:
    python run_newtask_from_ckpt.py --dataset AliCCP --mode prompt --seed 1688723512 --gpu 0
    python run_newtask_from_ckpt.py --dataset AliCCP --mode fw --all-seeds --gpu 0
    python run_newtask_from_ckpt.py --dataset AliCCP --all-modes --all-seeds --gpu 0
"""
import argparse
import copy
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size, CensusIncome_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset, CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from utils.task_correlation import get_task_correlation

warnings.filterwarnings("ignore")

ALL_MODES = ["fw", "tes", "prompt", "tcprompt_fixed", "tcprompt_learnable", "cgr", "affinity_gate", "kl_prompt"]


@torch.no_grad()
def evaluate(newtask, mptrec, data_loader):
    newtask.eval()
    device = next(newtask.parameters()).device
    y_true, y_hat = [], []
    for _, _, y, features in data_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        y_true.append(y)
        y_hat.append(pred)
    y_true = torch.cat(y_true)
    y_hat = torch.cat(y_hat)
    return roc_auc_score(y_true.int(), y_hat.cpu())


def get_config(dataset):
    if dataset == "CensusIncome":
        return {
            "vocabulary": CensusIncome_Vocabulary_Size,
            "remove_key": "education",
            "embedding_size": 4, "input_size": 123, "rep_dim": 128,
            "expert_hidden": (256, 128), "tower_hidden": (64, 32),
            "reg_embedding": 0.006, "reg_dnn": 3e-5, "lr": 1e-3,
            "batch_size": 256, "epochs": 30, "patience": 5,
            "ckpt_dir": "checkpoints/tcprompt_exp",
            "train_path": "dataset/Census-income/train.gz",
            "test_path": "dataset/Census-income/test.gz",
            "new_task_idx": 2,
        }
    elif dataset == "AliCCP":
        return {
            "vocabulary": AliCCP_Vocabulary_Size,
            "remove_keys": ["101", "301"],
            "embedding_size": 5, "input_size": 80, "rep_dim": 64,
            "expert_hidden": (128, 64), "tower_hidden": (32, 32),
            "reg_embedding": 0.0001, "reg_dnn": 7e-6, "lr": 1e-4,
            "batch_size": 2000, "epochs": 30, "patience": 5,
            "ckpt_dir": "checkpoints/tcprompt_exp_aliccp",
            "train_path": "dataset/AliCCP/ctr_cvr.train",
            "test_path": "dataset/AliCCP/ctr_cvr.test",
            "new_task_idx": 2, "dropout": [0.1, 0.3],
            "train_size": 5_000_000, "val_size": 500_000, "test_size": 5_000_000,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def make_newtask(cfg, mode, device):
    if mode in ("tcprompt_fixed", "tcprompt_learnable"):
        rho_vector = get_task_correlation(
            "CensusIncome" if "Census" in cfg.get("train_path", "") else "AliCCP",
            new_task_idx=cfg["new_task_idx"],
        )
        return NewTask(
            input_size=cfg["input_size"], rep_dim=cfg["rep_dim"],
            tower_dnn_hidden_units=list(cfg["tower_hidden"]),
            reg_dnn=cfg["reg_dnn"], device=device,
            fusion_mode="tcprompt", rho_vector=rho_vector, lambda_init=0.5,
            lambda_learnable=(mode == "tcprompt_learnable"),
        )
    else:
        return NewTask(
            input_size=cfg["input_size"], rep_dim=cfg["rep_dim"],
            tower_dnn_hidden_units=list(cfg["tower_hidden"]),
            reg_dnn=cfg["reg_dnn"], device=device,
            fusion_mode=mode,
        )


def run_single(dataset, seed, mode, gpu, kl_beta=0.1):
    cfg = get_config(dataset)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{gpu}")

    # --- Load data ---
    if dataset == "CensusIncome":
        train_ds = CensusIncomeDataset(cfg["train_path"], cfg["remove_key"])
        test_ds = CensusIncomeDataset(cfg["test_path"], cfg["remove_key"])
        val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=seed)
    else:
        train_ds = AliCCPDataset(cfg["train_path"], cfg["train_size"])
        test_ds = AliCCPDataset(cfg["test_path"], cfg["test_size"])
        val_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", cfg["val_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    # --- Prepare vocab ---
    vocab = cfg["vocabulary"].copy()
    if "remove_key" in cfg:
        vocab.pop(cfg["remove_key"])
    for k in cfg.get("remove_keys", []):
        if k in vocab:
            vocab.pop(k)

    # --- Build MPTRec, load checkpoint ---
    m_kwargs = dict(
        num_tasks=2, feature_vocabulary=vocab, embedding_size=cfg["embedding_size"],
        input_size=cfg["input_size"], expert_dnn_hidden_units=cfg["expert_hidden"],
        tower_dnn_hidden_units=cfg["tower_hidden"],
        reg_embedding=cfg["reg_embedding"], reg_dnn=cfg["reg_dnn"], device=device,
    )
    if "dropout" in cfg:
        m_kwargs["dropout"] = cfg["dropout"]

    mptrec = MPTRec(**m_kwargs).to(device)
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"stage1_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    mptrec.load_state_dict(torch.load(ckpt_path))
    mptrec.eval()

    # --- Create NewTask ---
    newtask = make_newtask(cfg, mode, device).to(device)

    # --- Stage 2 ---
    print(f"\n[Stage 2] dataset={dataset} seed={seed} mode={mode}")
    if mode == "kl_prompt":
        print(f"  kl_beta={kl_beta}")
    print(f"  Params: lr={cfg['lr']}, epochs={cfg['epochs']}, patience={cfg['patience']}")
    sys.stdout.flush()

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=cfg["lr"])
    loss_func = nn.BCELoss()
    best_val_auc, best_weight, earlystop_count = 0, None, 0

    # Routing warm-up: first 5 epochs linearly anneal gate from neutral (0.5) to full
    warmup_epochs = 5

    for epoch in range(1, cfg["epochs"] + 1):
        # Update gate warmup alpha
        if mode == "cgr":
            newtask.gate_warmup_alpha = min(1.0, epoch / warmup_epochs)
        newtask.train()
        for _, _, y, features in train_loader:
            for key in features.keys():
                features[key] = features[key].to(device)
            dnn_input, uni_rep, prop_reps, env_embeddings = mptrec.get_infos(features)
            pred = newtask(dnn_input, uni_rep, prop_reps, env_embeddings)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            if mode == "kl_prompt":
                loss += kl_beta * newtask.kl_loss.to(loss.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auc = evaluate(newtask, mptrec, val_loader)
        arrow = " *" if val_auc > best_val_auc else ""
        print(f"  Epoch {epoch:2d}/{cfg['epochs']}  val_auc={val_auc:.4f}{arrow}")
        sys.stdout.flush()

        if val_auc > best_val_auc:
            earlystop_count = 0
            best_val_auc = val_auc
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            if earlystop_count >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch}")
                sys.stdout.flush()
                break

    newtask.load_state_dict(best_weight)
    test_auc = evaluate(newtask, mptrec, test_loader)

    lambda_val = "---"
    if mode in ("tcprompt_fixed", "tcprompt_learnable"):
        lambda_val = f"{newtask.tc_fusion.get_lambda():.4f}"
    kl_val = f"{newtask.kl_loss.item():.4f}" if mode == "kl_prompt" else "---"

    print(f"RESULT|dataset={dataset}|seed={seed}|mode={mode}|"
          f"val={best_val_auc:.4f}|test={test_auc:.4f}|lambda={lambda_val}|kl={kl_val}")
    return test_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["CensusIncome", "AliCCP"])
    parser.add_argument("--mode", type=str, default="prompt",
                        choices=ALL_MODES)
    parser.add_argument("--all-modes", action="store_true",
                        help="Run all 5 fusion modes")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true",
                        help="Run all 3 standard seeds")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--kl_beta", type=float, default=0.1,
                        help="KL regularization weight for kl_prompt mode (default 0.1)")
    args = parser.parse_args()

    if args.dataset == "CensusIncome":
        seeds = [1685480945, 1688723512, 1689453621]
    else:
        seeds = [1688723512, 1688723740, 1688738016]

    if not args.all_seeds:
        if args.seed is None:
            parser.error("Must specify --seed or --all-seeds")
        seeds = [args.seed]

    modes = ALL_MODES if args.all_modes else [args.mode]

    all_results = []
    for mode in modes:
        mode_results = []
        for seed in seeds:
            auc = run_single(args.dataset, seed, mode, args.gpu, args.kl_beta)
            mode_results.append(auc)
        mean = np.mean(mode_results)
        std = np.std(mode_results, ddof=1) if len(mode_results) > 1 else 0
        print(f"SUMMARY|mode={mode}|mean={mean:.4f}|std={std:.4f}|"
              f"seeds={','.join(f'{r:.4f}' for r in mode_results)}")
        all_results.append((mode, mean, std, mode_results))

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"FINAL SUMMARY — {args.dataset} ({len(seeds)} seeds × {len(modes)} modes)")
        print(f"{'='*70}")
        print(f"{'Mode':<22} {'Test AUC':<16} {'Δ vs best'}")
        print(f"{'-'*50}")
        best_mean = max(r[1] for r in all_results)
        for mode, mean, std, _ in all_results:
            delta = mean - best_mean
            print(f"{mode:<22} {mean:.4f} ± {std:.4f}   {delta:+.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
