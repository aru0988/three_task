"""
Train Stage 2 (CensusIncome T3) with additional diagnostics:
  1. Record lambda gradient norm per epoch (for TC-Prompt modes)
  2. Record attention weight statistics (mean, variance per source task)
  3. Save trained model for later projection analysis
"""
import copy
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from utils.task_correlation import get_task_correlation

warnings.filterwarnings("ignore")

OUTPUT_DIR = "analysis/output"


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


def train_with_diagnostics(seed, gpu, mode, save_weights=True):
    """Train Stage 2 and collect diagnostic metrics."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(
        test_dataset, test_size=0.5, random_state=seed
    )
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    ci_vocabulary = CensusIncome_Vocabulary_Size.copy()
    ci_vocabulary.pop("education")
    device = torch.device(f"cuda:{gpu}")

    # Load Stage 1
    mptrec = MPTRec(
        num_tasks=2, feature_vocabulary=ci_vocabulary, embedding_size=4,
        input_size=123, expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32), reg_embedding=0.006, reg_dnn=3e-5,
        device=device,
    )
    mptrec.to(device)
    mptrec.load_state_dict(
        torch.load(f"checkpoints/tcprompt_exp/stage1_seed{seed}.pt")
    )
    mptrec.eval()

    # Task embedding analysis from Stage 1
    E_T1 = mptrec.env_embedding_network(torch.tensor([0]).to(device)).squeeze()
    E_T2 = mptrec.env_embedding_network(torch.tensor([1]).to(device)).squeeze()
    cos_E12 = F.cosine_similarity(E_T1.unsqueeze(0), E_T2.unsqueeze(0)).item()

    rho_vec = get_task_correlation("CensusIncome", new_task_idx=2)
    print(f"\n{'='*60}")
    print(f"Diagnostic Training: seed={seed}, mode={mode}")
    print(f"Stage 1 task embedding cos_sim(T1,T2) = {cos_E12:.4f}")
    print(f"Label Pearson ρ(T1,T2)=0.178, ρ(T1,T3)={rho_vec[0]:.3f}, "
          f"ρ(T2,T3)={rho_vec[1]:.3f}")

    # Create NewTask
    if mode == "tcprompt_learnable":
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode="tcprompt",
            rho_vector=rho_vec, lambda_init=0.5, lambda_learnable=True,
        )
    elif mode == "tcprompt_fixed":
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode="tcprompt",
            rho_vector=rho_vec, lambda_init=0.5, lambda_learnable=False,
        )
    else:
        newtask = NewTask(
            input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
            reg_dnn=3e-5, device=device, fusion_mode=mode,
        )
    newtask.to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()
    best_val_auc, best_weight, earlystop_count = 0, None, 0

    # Diagnostic tracking
    lambda_vals = []
    lambda_grads = []
    attn_mean_history = []
    attn_var_history = []

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

        # Record lambda gradient norm (TC-Prompt only)
        if newtask.fusion_mode == "tcprompt":
            lambda_vals.append(newtask.tc_fusion.get_lambda())
            if newtask.tc_fusion.lambda_corr.grad is not None:
                lambda_grads.append(
                    newtask.tc_fusion.lambda_corr.grad.abs().item()
                )
            else:
                lambda_grads.append(0.0)

        val_auc = evaluate(newtask, mptrec, val_loader)
        if val_auc > best_val_auc:
            earlystop_count = 0
            best_val_auc = val_auc
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            if earlystop_count >= 5:
                break

    # Load best weights and evaluate
    newtask.load_state_dict(best_weight)
    test_auc = evaluate(newtask, mptrec, test_loader)

    # Final attention weight analysis
    newtask.eval()
    all_W = []
    exist_env_embs = torch.stack(
        [mptrec.env_embedding_network(torch.tensor([i]).to(device)).squeeze()
         for i in range(2)], dim=1
    )

    n_samples = 0
    for _, _, _, features in test_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, _, _, _ = mptrec.get_infos(features)
        if newtask.fusion_mode == "tcprompt":
            W = newtask.tc_fusion(dnn_input, exist_env_embs).squeeze(-1)
        else:
            h_p = newtask.projection_network(dnn_input)
            W = torch.mm(h_p, exist_env_embs) / newtask.temperature
            W = F.softmax(W, dim=-1)
        all_W.append(W.detach().cpu())
        n_samples += W.shape[0]
        if n_samples >= 5000:
            break
    all_W = torch.cat(all_W, dim=0)

    mean_W = all_W.mean(dim=0)
    var_W = all_W.var(dim=0)

    # Report
    report = {
        "seed": seed, "mode": mode, "test_auc": test_auc,
        "best_val_auc": best_val_auc,
        "cos_E12": cos_E12,
        "rho_T1_new": float(rho_vec[0]), "rho_T2_new": float(rho_vec[1]),
        "mean_W_T1": float(mean_W[0]), "mean_W_T2": float(mean_W[1]),
        "var_W_T1": float(var_W[0]), "var_W_T2": float(var_W[1]),
        "lambda_final": lambda_vals[-1] if lambda_vals else None,
        "lambda_grad_final": lambda_grads[-1] if lambda_grads else None,
        "lambda_history": lambda_vals,
        "lambda_grad_history": lambda_grads,
    }

    print(f"\n{'='*60}")
    print(f"RESULTS for seed={seed}, mode={mode}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Mean attention weights: T1={mean_W[0]:.4f}, T2={mean_W[1]:.4f}")
    print(f"  Attention variance: T1={var_W[0]:.6f}, T2={var_W[1]:.6f}")
    print(f"  Task emb cos_sim: {cos_E12:.4f}")
    print(f"  Label ρ: (T1,T3)={rho_vec[0]:.3f}, (T2,T3)={rho_vec[1]:.3f}")

    if lambda_vals:
        print(f"  λ trajectory: {[f'{v:.4f}' for v in lambda_vals]}")
        print(f"  λ gradient trajectory: "
              f"{[f'{g:.6f}' for g in lambda_grads]}")
        mean_grad = np.mean(lambda_grads) if lambda_grads else 0
        dnn_grad_norm = 0.0
        for p in newtask.tower_network.parameters():
            if p.grad is not None:
                dnn_grad_norm += p.grad.norm().item()
        print(f"  Mean |∂L/∂λ| = {mean_grad:.6f}")
        print(f"  DNN param grad norm (tower) ≈ {dnn_grad_norm:.4f}")
        if dnn_grad_norm > 0:
            ratio = mean_grad / dnn_grad_norm
            print(f"  Ratio |∂L/∂λ| / ||∂L/∂θ_tower|| = {ratio:.8f}")
            print(f"  => λ gradient is {ratio*100:.4f}% of tower gradient")

    if save_weights:
        save_path = os.path.join(OUTPUT_DIR, f"newtask_{mode}_seed{seed}.pt")
        torch.save(best_weight, save_path)
        print(f"  Weights saved to {save_path}")

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1685480945)
    parser.add_argument("--mode", type=str, default="tcprompt_learnable",
                        choices=["prompt", "tcprompt_fixed", "tcprompt_learnable"])
    args = parser.parse_args()

    report = train_with_diagnostics(args.seed, args.gpu, args.mode)

    # Comparative analysis printout
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    print(f"1. Task embedding cos_sim = {report['cos_E12']:.4f}")
    print(f"   vs label Pearson ρ(T1,T2) = 0.178")
    print(f"   => Stage 1 task embeddings {'DO' if abs(report['cos_E12']) > 0.1 else 'do NOT strongly'} "
          f"reflect label correlation")

    print(f"\n2. Mean attention: T1={report['mean_W_T1']:.4f}, T2={report['mean_W_T2']:.4f}")
    if abs(report['mean_W_T1'] - report['rho_T1_new']) < 0.1:
        print(f"   Mean W_T1 ({report['mean_W_T1']:.4f}) is close to ρ(T1,T3) ({report['rho_T1_new']:.3f})")
        print(f"   => Attention weights correlate with task label correlation (IMPLICIT ENCODING)")
    else:
        print(f"   Mean W_T1 ({report['mean_W_T1']:.4f}) differs from ρ(T1,T3) ({report['rho_T1_new']:.3f})")

    print(f"\n3. Attention variance: T1={report['var_W_T1']:.6f}, T2={report['var_W_T2']:.6f}")
    if report['var_W_T1'] > 0.001:
        print(f"   Variance > 0 → instance-level differentiation is active")
    else:
        print(f"   Variance ≈ 0 → attention is effectively task-level (not instance-level)")

    if report['lambda_grad_final'] is not None:
        print(f"\n4. λ gradient: final |∂L/∂λ| = {report['lambda_grad_final']:.6f}")
        if report['lambda_grad_final'] < 0.001:
            print(f"   ∂L/∂λ ≈ 0 → λ is insensitive to loss → correlation prior is redundant")
        else:
            print(f"   ∂L/∂λ > 0 → λ is being actively optimized")


if __name__ == "__main__":
    main()
