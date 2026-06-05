"""
CGR diagnostic: extract g (confidence) statistics during training.
Usage:
    python analysis/cgr_diagnosis.py --dataset CensusIncome --seed 1685480945 --gpu 0
    python analysis/cgr_diagnosis.py --dataset AliCCP --seed 1688723512 --gpu 0
"""
import argparse, copy, os, sys, warnings
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CensusIncome_Vocabulary_Size, AliCCP_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset, AliCCPDataset
from multitaskrec.model import MPTRec, NewTask

warnings.filterwarnings("ignore")


@torch.no_grad()
def evaluate(newtask, mptrec, loader):
    newtask.eval()
    device = next(newtask.parameters()).device
    y_true, y_hat = [], []
    for _, _, y, features in loader:
        for k in features.keys():
            features[k] = features[k].to(device)
        dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
        pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
        y_true.append(y)
        y_hat.append(pred)
    return roc_auc_score(torch.cat(y_true).int(), torch.cat(y_hat).cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["CensusIncome", "AliCCP"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    if args.dataset == "CensusIncome":
        cfg = dict(vocab=CensusIncome_Vocabulary_Size, remove_key="education",
                   emb_size=4, input_size=123, rep_dim=128, expert_hidden=(256, 128),
                   tower_hidden=(64, 32), reg_emb=0.006, reg_dnn=3e-5, lr=1e-3,
                   bs=256, epochs=30, patience=5,
                   ckpt_dir="checkpoints/tcprompt_exp")
        train_ds = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
        test_ds = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
        val_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=args.seed)
    else:
        cfg = dict(vocab=AliCCP_Vocabulary_Size, remove_keys=["101", "301"],
                   emb_size=5, input_size=80, rep_dim=64, expert_hidden=(128, 64),
                   tower_hidden=(32, 32), reg_emb=0.0001, reg_dnn=7e-6, lr=1e-4,
                   bs=2000, epochs=30, patience=5, dropout=[0.1, 0.3],
                   ckpt_dir="checkpoints/tcprompt_exp_aliccp")
        train_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 5_000_000)
        test_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 5_000_000)
        val_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 500_000)

    train_loader = DataLoader(train_ds, batch_size=cfg["bs"])
    val_loader = DataLoader(val_ds, batch_size=cfg["bs"])
    test_loader = DataLoader(test_ds, batch_size=cfg["bs"])

    vocab = cfg["vocab"].copy()
    if "remove_key" in cfg:
        vocab.pop(cfg["remove_key"])
    for k in cfg.get("remove_keys", []):
        if k in vocab: vocab.pop(k)

    mptrec = MPTRec(num_tasks=2, feature_vocabulary=vocab,
                    embedding_size=cfg["emb_size"], input_size=cfg["input_size"],
                    expert_dnn_hidden_units=cfg["expert_hidden"],
                    tower_dnn_hidden_units=cfg["tower_hidden"],
                    reg_embedding=cfg["reg_emb"], reg_dnn=cfg["reg_dnn"],
                    device=device, dropout=cfg.get("dropout")).to(device)
    ckpt_path = os.path.join(cfg["ckpt_dir"], f"stage1_seed{args.seed}.pt")
    mptrec.load_state_dict(torch.load(ckpt_path))
    mptrec.eval()

    newtask = NewTask(input_size=cfg["input_size"], rep_dim=cfg["rep_dim"],
                      tower_dnn_hidden_units=list(cfg["tower_hidden"]),
                      reg_dnn=cfg["reg_dnn"], device=device,
                      fusion_mode="cgr").to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=cfg["lr"])
    loss_func = nn.BCELoss()
    best_val, best_w, earlystop = 0, None, 0
    g_means, g_stds = [], []
    warmup_epochs = 5

    for epoch in range(1, cfg["epochs"] + 1):
        # Routing warmup anneal
        newtask.gate_warmup_alpha = min(1.0, epoch / warmup_epochs)
        newtask.train()
        epoch_g = []
        for _, _, y, features in train_loader:
            for k in features.keys():
                features[k] = features[k].to(device)
            dnn_in, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_in, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Extract g (decoupled: task-embedding + shared-rep statistics)
            with torch.no_grad():
                exist = torch.stack(env_embs, dim=1)
                new_emb = newtask.env_embedding_network(newtask.new_env_idx).squeeze(0)
                src_mean = exist.mean(dim=1)
                src_var = exist.var(dim=1)
                gen_mean = gen_rep.mean(dim=0)
                gen_var = gen_rep.var(dim=0)
                g_in = torch.cat([new_emb, src_mean, src_var, gen_mean, gen_var], dim=-1)
                g_in = g_in.unsqueeze(0).expand(dnn_in.shape[0], -1)
                g_raw = newtask.confidence_mlp(g_in)
                g_val = newtask.gate_warmup_alpha * g_raw + (1 - newtask.gate_warmup_alpha) * 0.5
                epoch_g.append(g_val.mean().item())

        g_means.append(np.mean(epoch_g))
        g_stds.append(np.std(epoch_g))

        val_auc = evaluate(newtask, mptrec, val_loader)
        if val_auc > best_val:
            earlystop, best_val, best_w = 0, val_auc, copy.deepcopy(newtask.state_dict())
        else:
            earlystop += 1
            if earlystop >= cfg["patience"]:
                break

    newtask.load_state_dict(best_w)
    test_auc = evaluate(newtask, mptrec, test_loader)

    # Print diagnostic results
    final_g_mean = np.mean(g_means[-3:])  # last 3 epochs
    avg_g_variance = np.mean(g_stds[-3:])
    print(f"CGR_DIAG|dataset={args.dataset}|seed={args.seed}|"
          f"test_auc={test_auc:.4f}|g_mean_final={final_g_mean:.4f}|"
          f"g_std_final={avg_g_variance:.4f}")

    # Print per-epoch g trajectory
    for ep, (m, s) in enumerate(zip(g_means, g_stds), 1):
        print(f"  Epoch {ep:2d}: g_mean={m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
