"""
Compute Silhouette Score for representation cluster quality evaluation.
Quantitatively compares MPT-Rec and PLE representation disentanglement.
"""
import argparse
import warnings

import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size, CensusIncome_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset, CensusIncomeDataset
from multitaskrec.model import PLE, MPTRec

warnings.filterwarnings("ignore")


def extract_representations(model, data_loader, device, max_batches=50):
    """Extract task-sharing, T1-specific, and T2-specific representations."""
    gen_reps, spec_rep_0, spec_rep_1 = [], [], []
    for batch_idx, (_, _, _, features) in enumerate(data_loader):
        if batch_idx >= max_batches:
            break
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, _ = model.get_infos(features)
        gen_reps.append(gen_rep.detach().cpu())
        spec_rep_0.append(spec_reps[0].detach().cpu())
        spec_rep_1.append(spec_reps[1].detach().cpu())

    return (
        torch.cat(gen_reps).numpy(),
        torch.cat(spec_rep_0).numpy(),
        torch.cat(spec_rep_1).numpy(),
    )


def compute_silhouette(representations, sample_size=5000):
    """Compute Silhouette Score for three representation clusters."""
    gen_rep, spec_0, spec_1 = representations
    n = min(sample_size, len(gen_rep), len(spec_0), len(spec_1))

    rng = np.random.RandomState(42)
    idx = rng.choice(len(gen_rep), size=n, replace=False)

    X = np.concatenate(
        [gen_rep[idx], spec_0[idx], spec_1[idx]], axis=0
    )
    labels = np.concatenate(
        [np.zeros(n), np.ones(n), 2 * np.ones(n)]
    )

    return silhouette_score(X, labels, random_state=42)


def run_censusincome(seed, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    _, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=2000)

    ci_vocabulary = CensusIncome_Vocabulary_Size.copy()
    ci_vocabulary.pop("education")
    device = torch.device(f"cuda:{gpu}")

    results = {}

    # PLE
    ple = PLE(
        num_tasks=2, input_size=123, feature_vocabulary=ci_vocabulary,
        embedding_size=4, shared_expert_num=1, specific_expert_num=1,
        num_levels=2, expert_dnn_hidden_units=(256,),
        tower_dnn_hidden_units=(64, 32), reg_embedding=3e-4, reg_dnn=3e-4,
    )
    ple.load_state_dict(torch.load(f"baseline/ple/CensusIncome_{seed}.pt"))
    ple.to(device)
    ple.eval()
    reps_ple = extract_representations(ple, train_loader, device)
    results["PLE"] = compute_silhouette(reps_ple)

    # MPT-Rec with different α values
    for alpha_tag, pt_file in [("α=0.1", f"CensusIncome_{seed}.pt"),
                                 ("α=0.3", f"CensusIncome_{seed}_a03.pt")]:
        mptrec = MPTRec(
            num_tasks=2, feature_vocabulary=ci_vocabulary, embedding_size=4,
            input_size=123, expert_dnn_hidden_units=(256, 128),
            tower_dnn_hidden_units=(64, 32), reg_embedding=0.006, reg_dnn=3e-5,
            device=device,
        )
        mptrec.to(device)
        try:
            mptrec.load_state_dict(torch.load(pt_file))
        except FileNotFoundError:
            print(f"Warning: {pt_file} not found, using α=0.1 checkpoint")
            mptrec.load_state_dict(torch.load(f"CensusIncome_{seed}.pt"))
        mptrec.eval()
        reps_mptrec = extract_representations(mptrec, train_loader, device)
        results[f"MPT-Rec ({alpha_tag})"] = compute_silhouette(reps_mptrec)

    return results


def run_aliccp(seed, gpu):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 1_000_000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    device = torch.device(f"cuda:{gpu}")

    results = {}

    # PLE
    ple = PLE(
        num_tasks=2, input_size=80, feature_vocabulary=AliCCP_Vocabulary_Size,
        embedding_size=5, shared_expert_num=1, specific_expert_num=1,
        num_levels=2, expert_dnn_hidden_units=(128,),
        tower_dnn_hidden_units=(32, 32), reg_embedding=1e-6, reg_dnn=1e-6,
        dropout=(0.1, 0.3),
    )
    ple.load_state_dict(torch.load(f"baseline/ple/AliCCP_{seed}.pt"))
    ple.to(device)
    ple.eval()
    reps_ple = extract_representations(ple, train_loader, device)
    results["PLE"] = compute_silhouette(reps_ple)

    # MPT-Rec with different α values
    for alpha_tag, pt_file in [("α=0.1", f"AliCCP_{seed}.pt"),
                                 ("α=0.3", f"AliCCP_{seed}_a03.pt")]:
        mptrec = MPTRec(
            num_tasks=2, feature_vocabulary=AliCCP_Vocabulary_Size, embedding_size=5,
            input_size=80, expert_dnn_hidden_units=(128, 64),
            tower_dnn_hidden_units=(32, 32), dropout=(0.1, 0.3),
            reg_embedding=1e-4, reg_dnn=7e-6, device=device,
        )
        mptrec.to(device)
        try:
            mptrec.load_state_dict(torch.load(pt_file))
        except FileNotFoundError:
            print(f"Warning: {pt_file} not found, using α=0.1 checkpoint")
            mptrec.load_state_dict(torch.load(f"AliCCP_{seed}.pt"))
        mptrec.eval()
        reps_mptrec = extract_representations(mptrec, train_loader, device)
        results[f"MPT-Rec ({alpha_tag})"] = compute_silhouette(reps_mptrec)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CensusIncome",
                        choices=["CensusIncome", "AliCCP"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1685480945)
    args = parser.parse_args()

    if args.dataset == "CensusIncome":
        results = run_censusincome(args.seed, args.gpu)
    else:
        results = run_aliccp(1688723740, args.gpu)

    print("\n" + "=" * 60)
    print(f"Silhouette Scores — {args.dataset}")
    print("=" * 60)
    for model_name, score in results.items():
        print(f"  {model_name:<25} {score:.4f}")


if __name__ == "__main__":
    main()
