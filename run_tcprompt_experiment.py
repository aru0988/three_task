"""
TC-Prompt Experiment: Compare fusion strategies for new-task generalization.

Runs MPT-Rec Stage 1 (pretraining) + Stage 2 (new-task) on CensusIncome T3=Education
with multiple fusion modes, multi-seed evaluation.

Fusion modes:
  - fw:           Fixed Weights (equal, non-learnable)
  - tes:          Task Embedding Similarity (task-level)
  - prompt:       Original MPT-Rec instance-level attention (baseline)
  - tcprompt_05:  TC-Prompt with fixed λ=0.5
  - tcprompt_L:   TC-Prompt with learnable λ (init=0.5)
"""
import argparse
import copy
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


def run_experiment(seed, gpu, fusion_mode, lambda_init=0.5, lambda_learnable=False):
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

    # Stage 1: Pretraining
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

    # Stage 2: New-task generalization
    if fusion_mode == "tcprompt_fixed" or fusion_mode == "tcprompt_learnable":
        rho_vector = get_task_correlation("CensusIncome", new_task_idx=2)
        is_learnable = (fusion_mode == "tcprompt_learnable")
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode="tcprompt",
            rho_vector=rho_vector, lambda_init=lambda_init,
            lambda_learnable=is_learnable,
        )
    else:
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode=fusion_mode,
        )
    newtask.to(device)

    val_auc = train_newtask(newtask, mptrec, train_loader, val_loader, device)
    test_auc = evaluate(newtask, mptrec, test_loader)

    if fusion_mode.startswith("tcprompt"):
        lambda_val = newtask.tc_fusion.get_lambda()
        return val_auc, test_auc, lambda_val
    return val_auc, test_auc, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    seeds = [1685480945, 1688723512, 1689453621]
    configs = [
        ("fw", "Fixed Weights", False, 0.0),
        ("tes", "Task Emb Similarity", False, 0.0),
        ("prompt", "Prompt-tuning (original)", False, 0.0),
        ("tcprompt_fixed", "TC-Prompt (λ=0.5 fixed)", False, 0.5),
        ("tcprompt_learnable", "TC-Prompt (λ learnable)", True, 0.5),
    ]

    print("=" * 72)
    print("TC-Prompt Experiment: CensusIncome T3 (Education)")
    print("=" * 72)
    print(f"{'Fusion Strategy':<28} {'Val AUC':<12} {'Test AUC':<12} {'λ final':<10}")
    print("-" * 72)

    all_results = {}

    for mode, label, is_learnable, lam_init in configs:
        val_aucs, test_aucs, lambdas = [], [], []
        print(f"\n--- {label} ---")

        for seed in seeds:
            val_auc, test_auc, lam = run_experiment(
                seed, args.gpu, mode, lambda_init=lam_init,
                lambda_learnable=is_learnable,
            )
            val_aucs.append(val_auc)
            test_aucs.append(test_auc)
            if lam is not None:
                lambdas.append(lam)
            print(f"  seed={seed}: val={val_auc:.4f}, test={test_auc:.4f}", end="")
            if lam is not None:
                print(f", λ={lam:.4f}")
            else:
                print()

        mean_val = np.mean(val_aucs)
        std_val = np.std(val_aucs)
        mean_test = np.mean(test_aucs)
        std_test = np.std(test_aucs)
        lam_str = f"{np.mean(lambdas):.4f}" if lambdas else "---"

        all_results[label] = {
            "val_mean": mean_val, "val_std": std_val,
            "test_mean": mean_test, "test_std": std_test,
            "lambda": lam_str,
        }
        print(f"  MEAN: val={mean_val:.4f}±{std_val:.4f}, test={mean_test:.4f}±{std_test:.4f}")

    print("\n" + "=" * 72)
    print("FINAL RESULTS — CensusIncome T3 (Education)")
    print("=" * 72)
    print(f"{'Fusion Strategy':<28} {'Test AUC':<18} {'λ':<10}")
    print("-" * 56)
    for label, res in all_results.items():
        print(f"{label:<28} {res['test_mean']:.4f}±{res['test_std']:.4f}   {res['lambda']:<10}")

    # Find best
    best_label = max(all_results, key=lambda k: all_results[k]["test_mean"])
    best_auc = all_results[best_label]["test_mean"]
    prompt_auc = all_results["Prompt-tuning (original)"]["test_mean"]
    delta = best_auc - prompt_auc
    print(f"\nBest: {best_label} (AUC={best_auc:.4f})")
    print(f"Δ over original prompt: {delta:+.4f}")

    if delta > 0:
        print("TC-Prompt shows POSITIVE improvement over original MPT-Rec.")
    else:
        print("TC-Prompt did NOT improve over original MPT-Rec.")


if __name__ == "__main__":
    main()
