"""
T4 experiment: compare 2-source vs 3-source attention for learning T4.

Usage:
    # Full sweep
    python run_t4_experiment.py --all --all-seeds --gpu 0

    # Stage 1 only
    python run_t4_experiment.py --stage1 --t4 sex --all-seeds --gpu 0
    python run_t4_experiment.py --stage1 --t4 race --all-seeds --gpu 0

    # Stage 2 only (requires Stage 1 ckpts)
    python run_t4_experiment.py --stage2 --t4 sex --all-seeds --gpu 0
    python run_t4_experiment.py --stage2 --t4 race --all-seeds --gpu 0
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

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncome4TaskDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager

warnings.filterwarnings("ignore")

CENSUS_SEEDS = [1685480945, 1688723512, 1689453621]
ALL_MODES = ["prompt", "kl_prompt", "fw"]
CKPT_DIR = "checkpoints/t4_experiment"
os.makedirs(CKPT_DIR, exist_ok=True)


def get_cfg(t4):
    vocab = CensusIncome_Vocabulary_Size.copy()
    vocab.pop("education")
    if t4 in vocab:
        vocab.pop(t4)
    return dict(
        vocabulary=vocab,
        embedding_size=4, input_size=123, rep_dim=128,
        expert_hidden=(256, 128), tower_hidden=(64, 32),
        reg_embedding=0.006, reg_dnn=3e-5,
        lr_stage1=1e-3, lr_stage2=1e-3,
        batch_size=256, epochs_stage1=10, epochs_stage2=30,
        patience=5,
        train_path="dataset/Census-income/train.gz",
        test_path="dataset/Census-income/test.gz",
    )


def run_stage1(seed, t4, gpu):
    print(f"\n[Stage1] t4={t4} seed={seed}")
    sys.stdout.flush()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{gpu}")

    cfg = get_cfg(t4)
    train_ds = CensusIncome4TaskDataset(cfg["train_path"], t4, return_t4=False)
    test_ds = CensusIncome4TaskDataset(cfg["test_path"], t4, return_t4=False)
    val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    mptrec = MPTRec(
        num_tasks=3,
        feature_vocabulary=cfg["vocabulary"],
        embedding_size=cfg["embedding_size"],
        input_size=cfg["input_size"],
        expert_dnn_hidden_units=cfg["expert_hidden"],
        tower_dnn_hidden_units=cfg["tower_hidden"],
        reg_embedding=cfg["reg_embedding"],
        reg_dnn=cfg["reg_dnn"],
        device=device,
    ).to(device)

    env_ids = torch.randint(0, 3, size=(len(train_ds),))
    train_manager = MPTRecTrainManager(
        model=mptrec, train_loader=train_loader, val_loader=val_loader,
        env_ids=env_ids, task_name=['Income', 'Marital', 'Education'],
        lr=cfg["lr_stage1"], batch_size=cfg["batch_size"],
        uni_coe=0.9, env_coe=0.1, epochs=cfg["epochs_stage1"],
    )
    train_manager.train_three_task()
    mptrec.load_state_dict(train_manager.best_weight)

    ckpt_path = os.path.join(CKPT_DIR, f"backbone_3task_t4_{t4}_seed{seed}.pt")
    torch.save(mptrec.state_dict(), ckpt_path)
    print(f"  Saved: {ckpt_path}")

    aucs = train_manager.evaluation_three_task(test_loader)
    print(f"  T1_Income={aucs[0]:.4f}  T2_Marital={aucs[1]:.4f}  T3_Education={aucs[2]:.4f}")
    sys.stdout.flush()
    return ckpt_path


@torch.no_grad()
def evaluate_t4(newtask, mptrec, data_loader):
    newtask.eval()
    device = next(newtask.parameters()).device
    t_true, t_hat = [], []
    for _, _, _, y_t4, features in data_loader:
        for k in features.keys():
            features[k] = features[k].to(device)
        dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
        pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
        t_true.append(y_t4)
        t_hat.append(pred)
    return roc_auc_score(torch.cat(t_true).int(), torch.cat(t_hat).cpu())


def run_stage2(seed, t4, cond, mode, gpu):
    num_src = 2 if cond == "A" else 3
    print(f"\n[Stage2] t4={t4} cond={cond}({num_src}src) mode={mode} seed={seed}")
    sys.stdout.flush()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{gpu}")

    cfg = get_cfg(t4)
    train_ds = CensusIncome4TaskDataset(cfg["train_path"], t4)
    test_ds = CensusIncome4TaskDataset(cfg["test_path"], t4)
    val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"])

    mptrec = MPTRec(
        num_tasks=3,
        feature_vocabulary=cfg["vocabulary"],
        embedding_size=cfg["embedding_size"],
        input_size=cfg["input_size"],
        expert_dnn_hidden_units=cfg["expert_hidden"],
        tower_dnn_hidden_units=cfg["tower_hidden"],
        reg_embedding=cfg["reg_embedding"],
        reg_dnn=cfg["reg_dnn"],
        device=device,
    ).to(device)

    ckpt_path = os.path.join(CKPT_DIR, f"backbone_3task_t4_{t4}_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run Stage 1 first.")
    mptrec.load_state_dict(torch.load(ckpt_path))
    mptrec.eval()

    newtask = NewTask(
        input_size=cfg["input_size"], rep_dim=cfg["rep_dim"],
        tower_dnn_hidden_units=list(cfg["tower_hidden"]),
        reg_dnn=cfg["reg_dnn"], device=device,
        fusion_mode=mode, num_source_tasks=num_src,
    ).to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=cfg["lr_stage2"])
    loss_func = nn.BCELoss()
    best_val, best_w, earlystop = 0, None, 0

    for epoch in range(1, cfg["epochs_stage2"] + 1):
        newtask.train()
        for _, _, _, y_t4, features in train_loader:
            for k in features.keys():
                features[k] = features[k].to(device)
            dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y_t4.float()) + newtask.get_l2_reg()
            if mode == "kl_prompt":
                loss += 0.1 * newtask.kl_loss.to(loss.device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auc = evaluate_t4(newtask, mptrec, val_loader)
        arrow = " *" if val_auc > best_val else ""
        print(f"  Epoch {epoch:2d}/{cfg['epochs_stage2']}  val_auc={val_auc:.4f}{arrow}")
        sys.stdout.flush()

        if val_auc > best_val:
            earlystop, best_val, best_w = 0, val_auc, copy.deepcopy(newtask.state_dict())
        else:
            earlystop += 1
            if earlystop >= cfg["patience"]:
                print(f"  Early stop at epoch {epoch}")
                sys.stdout.flush()
                break

    newtask.load_state_dict(best_w)
    test_auc = evaluate_t4(newtask, mptrec, test_loader)
    print(f"RESULT|t4={t4}|cond={cond}|mode={mode}|seed={seed}|test={test_auc:.4f}")
    sys.stdout.flush()
    return test_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--t4", type=str, choices=["sex", "race"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        for t4 in ["sex", "race"]:
            for seed in CENSUS_SEEDS:
                run_stage1(seed, t4, args.gpu)
            for cond in ["A", "B"]:
                for mode in ALL_MODES:
                    for seed in CENSUS_SEEDS:
                        run_stage2(seed, t4, cond, mode, args.gpu)
        print("\n" + "=" * 60)
        print("T4 EXPERIMENT COMPLETE")
        print("=" * 60)
        return

    seeds = CENSUS_SEEDS if args.all_seeds else [args.seed]

    if args.stage1:
        for seed in seeds:
            run_stage1(seed, args.t4, args.gpu)

    if args.stage2:
        for cond in ["A", "B"]:
            for mode in ALL_MODES:
                for seed in seeds:
                    run_stage2(seed, args.t4, cond, mode, args.gpu)


if __name__ == "__main__":
    main()
