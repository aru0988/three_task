"""Run Stage 2 (new-task) for a single fusion mode and seed on AliCCP."""
import copy
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager
from utils.task_correlation import get_task_correlation

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

CKPT_DIR = "checkpoints/tcprompt_exp_aliccp"
DATA_SIZE = 5_000_000


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
    parser.add_argument("--train-size", type=int, default=DATA_SIZE)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"stage1_seed{args.seed}.pt")

    # Load data
    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", args.train_size)
    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 500_000)
    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 5_000_000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    ali_vocabulary = AliCCP_Vocabulary_Size.copy()
    ali_vocabulary.pop("101")
    ali_vocabulary.pop("301")
    device = torch.device(f"cuda:{args.gpu}")
    input_size = 80
    rep_dim = 64

    # Create MPTRec
    mptrec = MPTRec(
        num_tasks=2, feature_vocabulary=ali_vocabulary, embedding_size=5,
        input_size=input_size, expert_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32], dropout=[0.1, 0.3],
        reg_embedding=0.0001, reg_dnn=7e-6, device=device,
    )
    mptrec.to(device)

    # Train or load Stage 1
    if not os.path.exists(ckpt_path):
        print(f"[seed={args.seed}] Training Stage 1...", flush=True)
        train_manager = MPTRecTrainManager(
            model=mptrec, train_loader=train_loader, val_loader=val_loader,
            env_ids=env_ids, task_name=["CTR", "CVR"], lr=1e-4,
            batch_size=2000, uni_coe=0.9, env_coe=0.1, epochs=10,
        )
        train_manager.train_two_task()
        mptrec.load_state_dict(train_manager.best_weight)
        torch.save(mptrec.state_dict(), ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}", flush=True)
    else:
        print(f"[seed={args.seed}] Loading Stage 1 from {ckpt_path}", flush=True)
        mptrec.load_state_dict(torch.load(ckpt_path))

    # Create NewTask with specified fusion mode
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
        rho_vector = get_task_correlation("AliCCP", new_task_idx=2)
        newtask = NewTask(
            input_size=input_size, rep_dim=rep_dim,
            tower_dnn_hidden_units=[32, 32], reg_dnn=7e-6, device=device,
            fusion_mode="tcprompt", rho_vector=rho_vector, lambda_init=0.5,
            lambda_learnable=lambda_learnable,
        )
    else:
        newtask = NewTask(
            input_size=input_size, rep_dim=rep_dim,
            tower_dnn_hidden_units=[32, 32], reg_dnn=7e-6, device=device,
            fusion_mode=actual_mode,
        )
    newtask.to(device)

    # Stage 2 training
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-4)
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

    print(f"RESULT|seed={args.seed}|mode={args.mode}|"
          f"val={best_val_auc:.4f}|test={test_auc:.4f}|lambda={lambda_val}", flush=True)


if __name__ == "__main__":
    main()
