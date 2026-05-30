"""
Diagnose AliCCP Stage1 checkpoint quality differences across seeds.
Usage: python analysis/diagnose_ckpt.py --gpu 0
"""
import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import MPTRec

warnings.filterwarnings("ignore")

CKPT_DIR = "checkpoints/tcprompt_exp_aliccp"
SEEDS = [1688723512, 1688723740, 1688738016]


def load_model(seed, device):
    vocab = AliCCP_Vocabulary_Size.copy()
    vocab.pop("101")
    vocab.pop("301")
    mptrec = MPTRec(
        num_tasks=2, feature_vocabulary=vocab, embedding_size=5,
        input_size=80, expert_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32), dropout=[0.1, 0.3],
        reg_embedding=0.0001, reg_dnn=7e-6, device=device,
    ).to(device)
    ckpt = torch.load(f"{CKPT_DIR}/stage1_seed{seed}.pt", map_location=device)
    mptrec.load_state_dict(ckpt)
    mptrec.eval()
    return mptrec


@torch.no_grad()
def extract_reps(model, loader, device):
    """Extract gen_rep and spec_reps for all samples."""
    gen_reps, spec0_reps, spec1_reps = [], [], []
    for _, _, _, features in loader:
        for k in features.keys():
            features[k] = features[k].to(device)
        dnn_input, gen_rep, spec_reps, _ = model.get_infos(features)
        gen_reps.append(gen_rep.cpu())
        spec0_reps.append(spec_reps[0].cpu())
        spec1_reps.append(spec_reps[1].cpu())
    return {
        "gen": torch.cat(gen_reps, dim=0),
        "spec0": torch.cat(spec0_reps, dim=0),
        "spec1": torch.cat(spec1_reps, dim=0),
    }


def diagnose():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")

    # Load a subset of data for representation extraction
    train_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 50000)
    train_loader = DataLoader(train_ds, batch_size=2000)

    models = {}
    reps = {}
    for seed in SEEDS:
        print(f"Loading seed={seed}...")
        models[seed] = load_model(seed, device)
        reps[seed] = extract_reps(models[seed], train_loader, device)

    # --- 1. Weight Statistics ---
    print("\n" + "=" * 70)
    print("1. WEIGHT STATISTICS")
    print("=" * 70)
    for seed in SEEDS:
        m = models[seed]
        weights = []
        for name, p in m.named_parameters():
            weights.append(p.data.cpu().numpy().flatten())
        all_w = np.concatenate(weights)
        print(f"  seed={seed}: mean={all_w.mean():.6f}, std={all_w.std():.6f}, "
              f"frac_zero={(np.abs(all_w) < 1e-6).mean():.4f}")

    # --- 2. Embedding Weight Norms ---
    print("\n" + "=" * 70)
    print("2. EMBEDDING LAYER WEIGHT NORMS")
    print("=" * 70)
    for seed in SEEDS:
        m = models[seed]
        emb_norms = {}
        for name, emb in m.embedding_network.embedding_dict.items():
            emb_norms[name] = emb.weight.norm(2).item()
        print(f"  seed={seed}: avg_emb_norm={np.mean(list(emb_norms.values())):.4f}, "
              f"max={np.max(list(emb_norms.values())):.4f}, "
              f"min={np.min(list(emb_norms.values())):.4f}")

    # --- 3. Expert Weight Similarity ---
    print("\n" + "=" * 70)
    print("3. SHARED EXPERT LAST LAYER WEIGHT COSINE SIMILARITY")
    print("=" * 70)
    shared_weights = {}
    for seed in SEEDS:
        m = models[seed]
        # Last linear layer of shared expert
        for layer in m.shared_expert_network.mlp:
            if hasattr(layer, 'weight'):
                shared_weights[seed] = layer.weight.data.cpu()
        # Only keep last linear layer
        break

    # Re-collect correctly
    for seed in SEEDS:
        m = models[seed]
        last_linear = None
        for module in m.shared_expert_network.mlp:
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        shared_weights[seed] = last_linear.weight.data.cpu()

    for i, s1 in enumerate(SEEDS):
        for s2 in SEEDS[i+1:]:
            w1 = shared_weights[s1].flatten().unsqueeze(0)
            w2 = shared_weights[s2].flatten().unsqueeze(0)
            cos = F.cosine_similarity(w1, w2).item()
            print(f"  cos_sim(seed{s1}, seed{s2}) = {cos:.4f}")

    # --- 4. Representation Statistics ---
    print("\n" + "=" * 70)
    print("4. REPRESENTATION STATISTICS (on 50K samples)")
    print("=" * 70)
    for seed in SEEDS:
        r = reps[seed]
        gen_norm = r["gen"].norm(dim=1).mean().item()
        s0_norm = r["spec0"].norm(dim=1).mean().item()
        s1_norm = r["spec1"].norm(dim=1).mean().item()
        gen_s0_cos = F.cosine_similarity(r["gen"], r["spec0"], dim=1).mean().item()
        gen_s1_cos = F.cosine_similarity(r["gen"], r["spec1"], dim=1).mean().item()
        s0_s1_cos = F.cosine_similarity(r["spec0"], r["spec1"], dim=1).mean().item()
        gen_std = r["gen"].std(dim=0).mean().item()
        print(f"  seed={seed}:")
        print(f"    ||gen||={gen_norm:.4f}, ||spec0||={s0_norm:.4f}, ||spec1||={s1_norm:.4f}")
        print(f"    cos(gen,spec0)={gen_s0_cos:.4f}, cos(gen,spec1)={gen_s1_cos:.4f}, cos(spec0,spec1)={s0_s1_cos:.4f}")
        print(f"    gen_active_dims={(r['gen'].std(dim=0) > 0.01).sum().item()}/{r['gen'].shape[1]}")

    # --- 5. Env Embedding and Task Embedding ---
    print("\n" + "=" * 70)
    print("5. ENV/TASK EMBEDDINGS")
    print("=" * 70)
    for seed in SEEDS:
        m = models[seed]
        env_emb = m.env_embedding_network.weight.data.cpu()
        cos_env = F.cosine_similarity(env_emb[0].unsqueeze(0), env_emb[1].unsqueeze(0)).item()
        print(f"  seed={seed}: cos_sim(env_T0, env_T1)={cos_env:.4f}, "
              f"||env_T0||={env_emb[0].norm().item():.4f}, ||env_T1||={env_emb[1].norm().item():.4f}")

    # --- 6. Gate Network Output Distribution ---
    print("\n" + "=" * 70)
    print("6. GATE NETWORK OUTPUT (gen vs spec weight)")
    print("=" * 70)
    for seed in SEEDS:
        m = models[seed]
        gate_weights = []
        for _, _, _, features in train_loader:
            for k in features.keys():
                features[k] = features[k].to(device)
            dnn_input = m.embedding_network(features)
            gate_out = m.gate_networks[0](dnn_input)
            gate_weights.append(gate_out.cpu())
        all_gates = torch.cat(gate_weights, dim=0)
        gen_weight = all_gates[:, 1].mean().item()  # gate[:,1] = gen_rep weight
        spec_weight = all_gates[:, 0].mean().item() # gate[:,0] = spec weight
        print(f"  seed={seed}: mean_gate_gen_weight={gen_weight:.4f}, "
              f"mean_gate_spec_weight={spec_weight:.4f}, "
              f"gate_std={(all_gates.std(dim=0)[0]):.4f}")

    # --- 7. Prediction Confidence ---
    print("\n" + "=" * 70)
    print("7. PREDICTION CONFIDENCE ON VALIDATION SET")
    print("=" * 70)
    val_ds = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 10000)
    val_loader = DataLoader(val_ds, batch_size=2000)
    for seed in SEEDS:
        m = models[seed]
        all_preds = [[], []]
        for y0, y1, _, features in val_loader:
            for k in features.keys():
                features[k] = features[k].to(device)
            pred = m.predict(features)
            all_preds[0].append(pred[0].cpu())
            all_preds[1].append(pred[1].cpu())
        p0 = torch.cat(all_preds[0])
        p1 = torch.cat(all_preds[1])
        # Confidence = how far from 0.5
        conf0 = (p0 - 0.5).abs().mean().item()
        conf1 = (p1 - 0.5).abs().mean().item()
        # Entropy-like measure
        bce_like = -(p0 * p0.log() + (1-p0) * (1-p0).log()).mean().item()
        print(f"  seed={seed}: T0_conf={conf0:.4f}, T1_conf={conf1:.4f}, "
              f"T0_entropy={bce_like:.4f}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    diagnose()
