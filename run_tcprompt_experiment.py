"""
TC-Prompt Experiment: Compare fusion strategies for new-task generalization.
Optimized version: reuses Stage 1 pretrained weights across fusion modes.

Runs MPT-Rec Stage 1 (pretraining) once per seed, then Stage 2 with:
  - fw:           Fixed Weights (equal, non-learnable)
  - tes:          Task Embedding Similarity (task-level)
  - prompt:       Original MPT-Rec instance-level attention
  - tcprompt_fixed:  TC-Prompt with fixed lambda=0.5
  - tcprompt_learnable: TC-Prompt with learnable lambda (init=0.5)
"""
import argparse
import copy
import os
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

CHECKPOINT_DIR = "checkpoints/tcprompt_exp"


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


def train_newtask(newtask, mptrec, train_loader, val_loader, device,
                  epochs=30, patience=5, lr=1e-3):
    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    best_auc, best_weight, earlystop_count = 0, None, 0

    for epoch in range(1, epochs + 1):
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

        auc_val = evaluate(newtask, mptrec, val_loader)
        if auc_val > best_auc:
            earlystop_count = 0
            best_auc = auc_val
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            if earlystop_count >= patience:
                break

    newtask.load_state_dict(best_weight)
    return best_auc


def run_stage1(seed, gpu):
    """Run Stage 1 pretraining once. Returns (mptrec, data_loaders)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    ci_vocabulary = CensusIncome_Vocabulary_Size.copy()
    ci_vocabulary.pop("education")
    device = torch.device(f"cuda:{gpu}")

    mptrec = MPTRec(
        num_tasks=2, feature_vocabulary=ci_vocabulary, embedding_size=4,
        input_size=123, expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32), reg_embedding=0.006, reg_dnn=3e-5,
        device=device,
    )
    mptrec.to(device)

    train_manager = MPTRecTrainManager(
        model=mptrec, train_loader=train_loader, val_loader=val_loader,
        env_ids=env_ids, task_name=["Income", "Marital"], lr=1e-3,
        batch_size=256, uni_coe=0.9, env_coe=0.1, epochs=10,
    )
    train_manager.train_two_task()
    mptrec.load_state_dict(train_manager.best_weight)

    return mptrec, train_loader, val_loader, test_loader, device


def run_stage2(mptrec, train_loader, val_loader, test_loader, device,
               fusion_mode, rho_vector, lambda_init, lambda_learnable):
    """Run Stage 2 new-task training with given fusion mode."""
    if fusion_mode in ("tcprompt_fixed", "tcprompt_learnable"):
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode="tcprompt",
            rho_vector=rho_vector, lambda_init=lambda_init,
            lambda_learnable=lambda_learnable,
        )
    else:
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode=fusion_mode,
        )
    newtask.to(device)

    val_auc = train_newtask(newtask, mptrec, train_loader, val_loader, device)
    test_auc = evaluate(newtask, mptrec, test_loader)

    lambda_val = None
    if fusion_mode in ("tcprompt_fixed", "tcprompt_learnable"):
        lambda_val = newtask.tc_fusion.get_lambda()

    return val_auc, test_auc, lambda_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Load cached Stage 1 weights")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    seeds = [1685480945, 1688723512, 1689453621]
    rho_vector = get_task_correlation("CensusIncome", new_task_idx=2)

    configs = [
        ("fw", "Fixed Weights", False, 0.0),
        ("tes", "Task Emb Similarity", False, 0.0),
        ("prompt", "Prompt-tuning (original)", False, 0.0),
        ("tcprompt_fixed", "TC-Prompt (lambda=0.5)", False, 0.5),
        ("tcprompt_learnable", "TC-Prompt (lambda learn)", True, 0.5),
    ]

    print("=" * 72)
    print("TC-Prompt Experiment: CensusIncome T3 (Education)")
    print(f"Seeds: {seeds}")
    print(f"Task correlations: Income-Edu={rho_vector[0]:.3f}, "
          f"Marital-Edu={rho_vector[1]:.3f}")
    print("=" * 72)

    all_results = {}

    for seed_idx, seed in enumerate(seeds):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"stage1_seed{seed}.pt")

        if args.skip_stage1 and os.path.exists(ckpt_path):
            print(f"\n[Seed {seed}] Loading cached Stage 1 from {ckpt_path}")
            mptrec = MPTRec(
                num_tasks=2, feature_vocabulary=CensusIncome_Vocabulary_Size,
                embedding_size=4, input_size=123, expert_dnn_hidden_units=(256, 128),
                tower_dnn_hidden_units=(64, 32), reg_embedding=0.006, reg_dnn=3e-5,
                device=torch.device(f"cuda:{args.gpu}"),
            )
            mptrec.to(torch.device(f"cuda:{args.gpu}"))
            train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
            test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
            val_dataset, test_dataset = train_test_split(
                test_dataset, test_size=0.5, random_state=seed
            )
            train_loader = DataLoader(train_dataset, batch_size=256)
            val_loader = DataLoader(val_dataset, batch_size=256)
            test_loader = DataLoader(test_dataset, batch_size=256)
            device = torch.device(f"cuda:{args.gpu}")
            mptrec.load_state_dict(torch.load(ckpt_path))
        else:
            print(f"\n[Seed {seed}] Running Stage 1 pretraining...")
            mptrec, train_loader, val_loader, test_loader, device = run_stage1(
                seed, args.gpu
            )
            torch.save(mptrec.state_dict(), ckpt_path)
            print(f"  Stage 1 checkpoint saved to {ckpt_path}")

        for mode, label, is_learnable, lam_init in configs:
            val_auc, test_auc, lam = run_stage2(
                mptrec, train_loader, val_loader, test_loader, device,
                mode, rho_vector, lam_init, is_learnable,
            )

            if label not in all_results:
                all_results[label] = {"val": [], "test": [], "lam": []}
            all_results[label]["val"].append(val_auc)
            all_results[label]["test"].append(test_auc)
            if lam is not None:
                all_results[label]["lam"].append(lam)

            lam_str = f"lambda={lam:.4f}" if lam is not None else ""
            print(f"  {label:<30} val={val_auc:.4f}  test={test_auc:.4f}  {lam_str}")

    # Print final summary
    print("\n" + "=" * 72)
    print("FINAL RESULTS — CensusIncome T3 (Education)")
    print("=" * 72)
    print(f"{'Fusion Strategy':<30} {'Test AUC':<20} {'Val AUC':<20} {'Lambda':<10}")
    print("-" * 80)

    for label, res in all_results.items():
        test_mean = np.mean(res["test"])
        test_std = np.std(res["test"])
        val_mean = np.mean(res["val"])
        val_std = np.std(res["val"])
        lam_str = f"{np.mean(res['lam']):.4f}" if res["lam"] else "---"
        print(f"{label:<30} {test_mean:.4f}+-{test_std:.4f}    "
              f"{val_mean:.4f}+-{val_std:.4f}    {lam_str}")

    # Compare best vs prompt baseline
    prompt_test = np.mean(all_results["Prompt-tuning (original)"]["test"])
    best_label = max(all_results, key=lambda k: np.mean(all_results[k]["test"]))
    best_test = np.mean(all_results[best_label]["test"])
    delta = best_test - prompt_test

    print(f"\nBest: {best_label} (Test AUC={best_test:.4f})")
    print(f"Delta over original prompt: {delta:+.4f}")
    if delta > 0:
        print(">>> TC-Prompt shows POSITIVE improvement over original MPT-Rec.")
    else:
        print(">>> TC-Prompt did NOT improve over original MPT-Rec.")


if __name__ == "__main__":
    main()
