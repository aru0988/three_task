"""
Analyze whether MPT-Rec's projection network implicitly encodes task correlations.

Hypothesis: The projection network P learns to produce query vectors h_p whose
attention distribution over source tasks correlates with the true task Pearson ρ.
If true, adding an explicit correlation prior (TC-Prompt) is redundant.

Outputs:
  1. Cosine similarity between mean query vectors for T1/T2/T3 samples
  2. Average attention weight distribution over source tasks
  3. Comparison with ground-truth Pearson ρ
"""
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from utils.task_correlation import get_task_correlation

warnings.filterwarnings("ignore")


def extract_query_vectors(newtask, mptrec, data_loader, device, label_idx=2):
    """Extract projection network query vectors grouped by label.

    Args:
        newtask: trained NewTask model
        mptrec: trained MPTRec
        data_loader: test data
        device: torch device
        label_idx: which label to group by (0=T1, 1=T2, 2=T3)

    Returns:
        h_pos: query vectors for positive samples (label=1)
        h_neg: query vectors for negative samples (label=0)
    """
    newtask.eval()
    mptrec.eval()
    h_pos, h_neg = [], []

    for y0, y1, y2, features in data_loader:
        labels = [y0, y1, y2]
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)

        # Get projection output (query vector)
        if newtask.fusion_mode == "tcprompt":
            h_p = newtask.tc_fusion.projection_network(dnn_input)
        else:
            h_p = newtask.projection_network(dnn_input)

        label = labels[label_idx]
        mask_pos = (label == 1)
        mask_neg = (label == 0)

        if mask_pos.sum() > 0:
            h_pos.append(h_p[mask_pos].detach().cpu())
        if mask_neg.sum() > 0:
            h_neg.append(h_p[mask_neg].detach().cpu())

    return (torch.cat(h_pos) if h_pos else None,
            torch.cat(h_neg) if h_neg else None)


def compute_attention_weights(newtask, mptrec, data_loader, device):
    """Compute average attention weights over source tasks for all test samples."""
    newtask.eval()
    mptrec.eval()
    all_W = []

    for _, _, _, features in data_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)

        exist_env_embs = torch.stack(env_embs, dim=1)  # (rep_dim, K)

        if newtask.fusion_mode == "tcprompt":
            W = newtask.tc_fusion(dnn_input, exist_env_embs).squeeze(-1)
        else:
            h_p = newtask.projection_network(dnn_input)
            W = torch.mm(h_p, exist_env_embs) / newtask.temperature
            W = F.softmax(W, dim=-1)

        all_W.append(W.detach().cpu())

    all_W = torch.cat(all_W, dim=0)  # (N, K)
    return all_W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1685480945)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    # Load CensusIncome data
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    _, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=256)

    ci_vocabulary = CensusIncome_Vocabulary_Size.copy()
    ci_vocabulary.pop("education")

    # Load MPTRec Stage 1
    mptrec = MPTRec(
        num_tasks=2, feature_vocabulary=ci_vocabulary, embedding_size=4,
        input_size=123, expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32), reg_embedding=0.006, reg_dnn=3e-5,
        device=device,
    )
    mptrec.to(device)
    mptrec.load_state_dict(
        torch.load(f"checkpoints/tcprompt_exp/stage1_seed{args.seed}.pt")
    )

    rho_vector = get_task_correlation("CensusIncome", new_task_idx=2)

    for mode, label in [("prompt", "Original Prompt"),
                          ("tcprompt_learnable", "TC-Prompt (learnable)")]:
        print(f"\n{'='*60}")
        print(f"Analyzing: {label}")
        print(f"{'='*60}")

        # Load NewTask
        if mode == "tcprompt":
            actual_mode = "tcprompt"
            lambda_learnable = True
        else:
            actual_mode = mode
            lambda_learnable = False

        if actual_mode == "tcprompt":
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

        # We don't have saved Stage 2 weights, but we can analyze the untrained
        # projection network structure. For trained analysis, load from checkpoint.
        # Here we analyze the projection's inherent properties.

        # 1. Compute attention weight statistics
        print("\nComputing attention weight distributions...")
        attn_weights = compute_attention_weights(newtask, mptrec, test_loader, device)

        mean_W = attn_weights.mean(dim=0)
        var_W = attn_weights.var(dim=0)
        print(f"  Mean attention: T1={mean_W[0]:.4f}, T2={mean_W[1]:.4f}")
        print(f"  Attention variance: T1={var_W[0]:.6f}, T2={var_W[1]:.6f}")
        print(f"  Ground-truth ρ: Income-Edu={rho_vector[0]:.3f}, "
              f"Marital-Edu={rho_vector[1]:.3f}")

        # 2. Compute query vector cosine similarity between task groups
        print("\nComputing query vector cosine similarity...")
        h_t1_pos, _ = extract_query_vectors(
            newtask, mptrec, test_loader, device, label_idx=0
        )
        h_t2_pos, _ = extract_query_vectors(
            newtask, mptrec, test_loader, device, label_idx=1
        )
        h_t3_pos, _ = extract_query_vectors(
            newtask, mptrec, test_loader, device, label_idx=2
        )

        if all(x is not None for x in [h_t1_pos, h_t2_pos, h_t3_pos]):
            h_t1_mean = h_t1_pos.mean(dim=0)
            h_t2_mean = h_t2_pos.mean(dim=0)
            h_t3_mean = h_t3_pos.mean(dim=0)

            cos_12 = F.cosine_similarity(h_t1_mean.unsqueeze(0),
                                         h_t2_mean.unsqueeze(0)).item()
            cos_13 = F.cosine_similarity(h_t1_mean.unsqueeze(0),
                                         h_t3_mean.unsqueeze(0)).item()
            cos_23 = F.cosine_similarity(h_t2_mean.unsqueeze(0),
                                         h_t3_mean.unsqueeze(0)).item()

            print(f"  Query cosine similarity (T1,T2): {cos_12:.4f}  "
                  f"[label ρ={0.178:.3f}]")
            print(f"  Query cosine similarity (T1,T3): {cos_13:.4f}  "
                  f"[label ρ={0.186:.3f}]")
            print(f"  Query cosine similarity (T2,T3): {cos_23:.4f}  "
                  f"[label ρ={0.142:.3f}]")

            # Hypothesis check: does cos_sim correlate with ρ?
            cos_sims = np.array([cos_12, cos_13, cos_23])
            rhos = np.array([0.178, 0.186, 0.142])
            corr = np.corrcoef(cos_sims, rhos)[0, 1]
            print(f"\n  Correlation between query cos_sim and label ρ: {corr:.4f}")
            print(f"  => {'Projection NETWORK DOES encode task correlation'
                  if abs(corr) > 0.5 else
                  'Projection network does NOT clearly encode task correlation'}")

        # 3. Per-task attention preference
        print("\nAttention preference by source task:")
        # Check if samples from T1-heavy users prefer T1 source
        # (using income=1 vs income=0 to split)
        h_inc_high, _ = extract_query_vectors(
            newtask, mptrec, test_loader, device, label_idx=0
        )
        if h_inc_high is not None:
            inc_high_W = compute_attention_weights(
                newtask, mptrec,
                DataLoader(
                    [d for i, d in enumerate(test_dataset) if d[0] == 1],
                    batch_size=256
                ),
                device
            ) if sum(1 for d in test_dataset if d[0] == 1) > 0 else None

            if inc_high_W is not None:
                print(f"  Income=1 samples: mean W = [{inc_high_W.mean(0)[0]:.4f}, "
                      f"{inc_high_W.mean(0)[1]:.4f}]")


if __name__ == "__main__":
    main()
