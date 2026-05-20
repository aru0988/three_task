import argparse
import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import AliCCP_Vocabulary_Size, CensusIncome_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset, CensusIncomeDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager


@torch.no_grad()
def evaluation_newtask(newtask, mptrec, data_loader):
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


def run_censusincome(alpha, gpu, seed):
    """Run MPT-Rec on CensusIncome with given alpha, then evaluate T3."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    env_ids = torch.randint(0, 2, size=(len(train_dataset),))

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
    # Override alpha in GAN training
    train_manager._original_alpha = alpha
    train_manager.train_two_task()
    mptrec.load_state_dict(train_manager.best_weight)
    auc_t1, auc_t2 = train_manager.evaluation_two_task(test_loader)

    newtask = NewTask(input_size=123, rep_dim=128, tower_dnn_hidden_units=[64, 32],
                      reg_dnn=3e-5, device=device)
    newtask.to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()
    best_auc, best_weight, earlystop_count = 0, None, 0

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

        auc_val = evaluation_newtask(newtask, mptrec, val_loader)
        if auc_val > best_auc:
            earlystop_count, best_auc = 0, auc_val
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            if earlystop_count == 5:
                break

    newtask.load_state_dict(best_weight)
    auc_t3 = evaluation_newtask(newtask, mptrec, test_loader)
    return auc_t1, auc_t2, auc_t3


def run_aliccp(alpha, gpu, seed):
    """Run MPT-Rec on AliCCP with given alpha, then evaluate T3."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_dataset = AliCCPDataset("dataset/AliCCP/train.gz", data_size=5_000_000)
    test_dataset = AliCCPDataset("dataset/AliCCP/test.gz", data_size=5_000_000)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)
    env_ids = torch.randint(0, 3, size=(len(train_dataset),))

    device = torch.device(f"cuda:{gpu}")
    mptrec = MPTRec(
        num_tasks=3, feature_vocabulary=AliCCP_Vocabulary_Size, embedding_size=5,
        input_size=80, expert_dnn_hidden_units=(128, 64),
        tower_dnn_hidden_units=(32, 32), reg_embedding=0.006, reg_dnn=3e-5,
        dropout=[0.1, 0.3], device=device,
    )
    mptrec.to(device)

    train_manager = MPTRecTrainManager(
        model=mptrec, train_loader=train_loader, val_loader=val_loader,
        env_ids=env_ids, task_name=["CTR", "CVR", "BSI"], lr=1e-4,
        batch_size=2000, uni_coe=0.9, env_coe=0.1, epochs=10,
    )
    train_manager._original_alpha = alpha
    train_manager.train_three_task()
    mptrec.load_state_dict(train_manager.best_weight)
    auc_t1, auc_t2, auc_t3_old = train_manager.evaluation_three_task(test_loader)

    # T3 (BSI) already trained, return its performance
    return auc_t1, auc_t2, auc_t3_old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="AliCCP", choices=["CensusIncome", "AliCCP"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1688723512)
    args = parser.parse_args()

    alpha_list = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    results = {}

    for alpha in alpha_list:
        print(f"\n{'='*60}")
        print(f"Running α = {alpha}")
        print(f"{'='*60}")

        if args.dataset == "CensusIncome":
            t1, t2, t3 = run_censusincome(alpha, args.gpu, args.seed)
        else:
            t1, t2, t3 = run_aliccp(alpha, args.gpu, args.seed)

        results[alpha] = (t1, t2, t3)
        print(f"α={alpha}: T1={t1:.4f}, T2={t2:.4f}, T3={t3:.4f}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'α':<10} {'AUC/T1':<12} {'AUC/T2':<12} {'AUC/T3':<12}")
    print("-" * 46)
    for alpha in alpha_list:
        t1, t2, t3 = results[alpha]
        print(f"{alpha:<10} {t1:<12.4f} {t2:<12.4f} {t3:<12.4f}")


if __name__ == "__main__":
    main()
