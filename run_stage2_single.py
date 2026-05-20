"""Run Stage 2 for a single fusion mode and seed. Prints result immediately."""
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
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager
from utils.task_correlation import get_task_correlation

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["fw", "tes", "prompt", "tcprompt_fixed", "tcprompt_learnable"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load data
    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=args.seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    ci_vocabulary = CensusIncome_Vocabulary_Size.copy()
    ci_vocabulary.pop("education")
    device = torch.device(f"cuda:{args.gpu}")

    ckpt_path = f"checkpoints/tcprompt_exp/stage1_seed{args.seed}.pt"

    # Create and load MPTRec
    mptrec = MPTRec(
        num_tasks=2, feature_vocabulary=ci_vocabulary, embedding_size=4,
        input_size=123, expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32), reg_embedding=0.006, reg_dnn=3e-5,
        device=device,
    )
    mptrec.to(device)

    # Train Stage 1 if no checkpoint
    if not os.path.exists(ckpt_path):
        print(f"[seed={args.seed}] Training Stage 1...", flush=True)
        train_manager = MPTRecTrainManager(
            model=mptrec, train_loader=train_loader, val_loader=val_loader,
            env_ids=env_ids, task_name=["Income", "Marital"], lr=1e-3,
            batch_size=256, uni_coe=0.9, env_coe=0.1, epochs=10,
        )
        train_manager.train_two_task()
        mptrec.load_state_dict(train_manager.best_weight)
        torch.save(mptrec.state_dict(), ckpt_path)
    else:
        print(f"[seed={args.seed}] Loading Stage 1 from {ckpt_path}", flush=True)
        mptrec.load_state_dict(torch.load(ckpt_path))

    # Create NewTask
    fusion_mode = args.mode
    if fusion_mode == "tcprompt_learnable":
        actual_mode = "tcprompt"
        lambda_learnable = True
    elif fusion_mode == "tcprompt_fixed":
        actual_mode = "tcprompt"
        lambda_learnable = False
    else:
        actual_mode = fusion_mode
        lambda_learnable = False

    if actual_mode == "tcprompt":
        rho_vector = get_task_correlation("CensusIncome", new_task_idx=2)
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode="tcprompt",
            rho_vector=rho_vector, lambda_init=0.5,
            lambda_learnable=lambda_learnable,
        )
    else:
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode=actual_mode,
        )
    newtask.to(device)

    # Train Stage 2
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()
    best_val_auc, best_weight, earlystop_count = 0, None, 0

    for epoch in range(1, 31):
        newtask.train()
        for _, _, y, features in train_loader:
            for key in features.keys():
                features[key] = features[key].to(device)
            dnn_input, uni_rep, prop_reps, env_embeddings = mptrec.get_infos(features)
            pred = newtask(dnn_input, uni_rep, prop_reps, env_embeddings)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_auc = evaluate(newtask, mptrec, val_loader)
        if val_auc > best_val_auc:
            earlystop_count = 0
            best_val_auc = val_auc
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            if earlystop_count >= 5:
                break

    newtask.load_state_dict(best_weight)
    test_auc = evaluate(newtask, mptrec, test_loader)

    lambda_val = "---"
    if actual_mode == "tcprompt":
        lambda_val = f"{newtask.tc_fusion.get_lambda():.4f}"

    print(f"RESULT|seed={args.seed}|mode={args.mode}|val={best_val_auc:.4f}|test={test_auc:.4f}|lambda={lambda_val}", flush=True)


if __name__ == "__main__":
    main()
