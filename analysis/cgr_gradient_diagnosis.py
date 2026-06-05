"""
CGR gradient diagnosis: compare gradient variance of projection network
between original Prompt and CGR.

Usage:
    python analysis/cgr_gradient_diagnosis.py --dataset CensusIncome --seed 1685480945 --gpu 0
    python analysis/cgr_gradient_diagnosis.py --dataset AliCCP --seed 1688723512 --gpu 0
"""
import argparse, copy, os, sys, warnings
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CensusIncome_Vocabulary_Size, AliCCP_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset, AliCCPDataset
from multitaskrec.model import MPTRec, NewTask

warnings.filterwarnings("ignore")


def run_gradient_analysis(dataset, seed, gpu, fusion_mode):
    """Run Stage 2 and record gradient statistics for projection network."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    device = torch.device(f"cuda:{gpu}")

    if dataset == "CensusIncome":
        cfg = dict(emb_size=4, input_size=123, rep_dim=128,
                   expert_hidden=(256, 128), tower_hidden=(64, 32),
                   reg_emb=0.006, reg_dnn=3e-5, lr=1e-3,
                   bs=256, epochs=30, patience=5,
                   ckpt_dir="checkpoints/tcprompt_exp")
        train_ds = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
        test_ds = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
        val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=seed)
        vocab = CensusIncome_Vocabulary_Size.copy()
        vocab.pop("education")
        dropout = None
    else:
        cfg = dict(emb_size=5, input_size=80, rep_dim=64,
                   expert_hidden=(128, 64), tower_hidden=(32, 32),
                   reg_emb=0.0001, reg_dnn=7e-6, lr=1e-4,
                   bs=2000, epochs=30, patience=5,
                   ckpt_dir="checkpoints/tcprompt_exp_aliccp")
        train_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 5_000_000)
        test_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 5_000_000)
        val_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 500_000)
        vocab = AliCCP_Vocabulary_Size.copy()
        vocab.pop("101"); vocab.pop("301")
        dropout = [0.1, 0.3]

    train_loader = DataLoader(train_ds, batch_size=cfg["bs"])
    val_loader = DataLoader(val_ds, batch_size=cfg["bs"])
    test_loader = DataLoader(test_ds, batch_size=cfg["bs"])

    # Build and load Stage 1 checkpoint
    mptrec = MPTRec(num_tasks=2, feature_vocabulary=vocab,
                    embedding_size=cfg["emb_size"], input_size=cfg["input_size"],
                    expert_dnn_hidden_units=cfg["expert_hidden"],
                    tower_dnn_hidden_units=cfg["tower_hidden"],
                    reg_embedding=cfg["reg_emb"], reg_dnn=cfg["reg_dnn"],
                    device=device, dropout=dropout).to(device)
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"stage1_seed{seed}.pt")
    mptrec.load_state_dict(torch.load(ckpt_path))
    mptrec.eval()

    # Build NewTask with specified fusion mode
    newtask = NewTask(input_size=cfg["input_size"], rep_dim=cfg["rep_dim"],
                      tower_dnn_hidden_units=list(cfg["tower_hidden"]),
                      reg_dnn=cfg["reg_dnn"], device=device,
                      fusion_mode=fusion_mode).to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=cfg["lr"])
    loss_func = nn.BCELoss()
    best_val, best_w, earlystop = 0, None, 0

    # Gradient tracking
    gradient_norms = []  # list of per-batch gradient norms
    param_name = "projection_network"

    for epoch in range(1, cfg["epochs"] + 1):
        newtask.train()
        epoch_grads = []
        for _, _, y, features in train_loader:
            for k in features.keys():
                features[k] = features[k].to(device)
            dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()

            # Record gradient norm of projection network parameters
            total_norm = 0.0
            for name, p in newtask.named_parameters():
                if param_name in name and p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            epoch_grads.append(total_norm)

            optimizer.step()

        gradient_norms.append(epoch_grads)

        # Validation
        newtask.eval()
        y_true, y_hat = [], []
        for _, _, y, features in val_loader:
            for k in features.keys(): features[k] = features[k].to(device)
            dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
            y_true.append(y); y_hat.append(pred.detach())
        val_auc = roc_auc_score(torch.cat(y_true).int(), torch.cat(y_hat).cpu())

        if val_auc > best_val:
            earlystop, best_val, best_w = 0, val_auc, copy.deepcopy(newtask.state_dict())
        else:
            earlystop += 1
            if earlystop >= cfg["patience"]:
                break

    # Test
    newtask.load_state_dict(best_w)
    newtask.eval()
    y_true, y_hat = [], []
    for _, _, y, features in test_loader:
        for k in features.keys(): features[k] = features[k].to(device)
        dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
        pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
        y_true.append(y); y_hat.append(pred.detach())
    test_auc = roc_auc_score(torch.cat(y_true).int(), torch.cat(y_hat).cpu())

    # Compute gradient statistics across all epochs
    all_grads = [g for epoch_g in gradient_norms for g in epoch_g]
    grad_mean = np.mean(all_grads)
    grad_std = np.std(all_grads)
    grad_cv = grad_std / max(grad_mean, 1e-8)  # Coefficient of variation

    print(f"GRADIENT|dataset={dataset}|seed={seed}|mode={fusion_mode}|"
          f"test_auc={test_auc:.4f}|"
          f"grad_mean={grad_mean:.6f}|grad_std={grad_std:.6f}|grad_cv={grad_cv:.4f}")

    # Per-epoch gradient stats
    print(f"\nPer-epoch gradient norm stats for {fusion_mode}:")
    for ep, epoch_g in enumerate(gradient_norms, 1):
        ep_mean = np.mean(epoch_g)
        ep_std = np.std(epoch_g)
        print(f"  Epoch {ep:2d}: mean={ep_mean:.6f} ± {ep_std:.6f}")

    return test_auc, grad_mean, grad_std, grad_cv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["CensusIncome", "AliCCP"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    print(f"\n{'='*60}")
    print(f"Running gradient analysis: {args.dataset} seed={args.seed}")
    print(f"{'='*60}")

    # Run Prompt
    print(f"\n>>> Mode: prompt")
    p_auc, p_mean, p_std, p_cv = run_gradient_analysis(
        args.dataset, args.seed, args.gpu, "prompt"
    )

    # Run CGR
    print(f"\n>>> Mode: cgr")
    c_auc, c_mean, c_std, c_cv = run_gradient_analysis(
        args.dataset, args.seed, args.gpu, "cgr"
    )

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"GRADIENT ANALYSIS SUMMARY — {args.dataset} seed={args.seed}")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Prompt':<18} {'CGR':<18} {'Delta / Ratio'}")
    print(f"{'-'*60}")
    print(f"{'Test AUC':<20} {p_auc:.4f}              {c_auc:.4f}              {c_auc - p_auc:+.4f}")
    print(f"{'Grad Mean':<20} {p_mean:.6f}          {c_mean:.6f}          {c_mean/p_mean:.2f}x")
    print(f"{'Grad Std':<20} {p_std:.6f}          {c_std:.6f}          {c_std/p_std:.2f}x")
    print(f"{'Grad CV (sigma/mu)':<20} {p_cv:.4f}              {c_cv:.4f}              {c_cv/p_cv:.2f}x")
