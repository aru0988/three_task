"""消融实验: Figure 4-6

Figure 4: 表征分离对已有任务影响 (variant → T1/T2 AUC)
Figure 5: 表征分离对新任务影响 (variant → T3 AUC via fine-tune)
Figure 6: Prompt机制有效性 (fusion_mode → T3 AUC via fine-tune)

Usage:
  # Figure 4+5: ablation variants
  python run_ablation.py --dataset CensusIncome --variant full --seed 1685480945
  python run_ablation.py --dataset CensusIncome --variant share_only --seed 1685480945
  python run_ablation.py --dataset CensusIncome --variant specific_only --seed 1685480945
  python run_ablation.py --dataset CensusIncome --variant no_gan --seed 1685480945

  # Figure 6: prompt fusion modes (need pretrained full model first)
  python run_ablation.py --dataset CensusIncome --variant full --fusion_mode fw --seed 1685480945
  python run_ablation.py --dataset CensusIncome --variant full --fusion_mode tes --seed 1685480945
  python run_ablation.py --dataset AliCCP --variant full --fusion_mode fw --seed 1688723512
  python run_ablation.py --dataset AliCCP --variant full --fusion_mode tes --seed 1688723512
"""
import argparse
import copy
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.append('')

from config import CensusIncome_Vocabulary_Size, AliCCP_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset, AliCCPDataset
from multitaskrec.model import MPTRec, NewTask
from multitaskrec.train import MPTRecTrainManager
from multitaskrec.utils import count_params


@torch.no_grad()
def evaluate_two_task(model, data_loader, device):
    """Evaluate T1 and T2 AUC after 2-task pretraining (Figure 4)"""
    model.eval()
    y_true_0, y_true_1, y_hat_0, y_hat_1 = [], [], [], []
    for y_0, y_1, _, features in data_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        pred = model.predict(features)
        y_true_0.append(y_0)
        y_true_1.append(y_1)
        y_hat_0.append(pred[0])
        y_hat_1.append(pred[1])
    yt0 = torch.cat(y_true_0)
    yt1 = torch.cat(y_true_1)
    yh0 = torch.cat(y_hat_0)
    yh1 = torch.cat(y_hat_1)
    auc0 = roc_auc_score(yt0.int(), yh0.cpu())
    auc1 = roc_auc_score(yt1.int(), yh1.cpu())
    return auc0, auc1


@torch.no_grad()
def evaluate_new_task(newtask, mptrec, data_loader, device):
    """Evaluate T3 AUC after fine-tuning (Figure 5/6)"""
    newtask.eval()
    mptrec.eval()
    y_true, y_hat = [], []
    for _, _, y, features in data_loader:
        for key in features.keys():
            features[key] = features[key].to(device)
        dnn_input, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        y_true.append(y)
        y_hat.append(pred)
    yt = torch.cat(y_true)
    yh = torch.cat(y_hat)
    return roc_auc_score(yt.int(), yh.cpu())


def run_censusincome(args):
    device = torch.device(f"cuda:{args.gpu}")
    task1, task2, task3 = "Income", "Marital", "Education"
    ci_vocab = CensusIncome_Vocabulary_Size.copy()
    ci_vocab.pop("education")

    train_dataset = CensusIncomeDataset("dataset/Census-income/train.gz", "education")
    test_dataset = CensusIncomeDataset("dataset/Census-income/test.gz", "education")
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=256)
    val_loader = DataLoader(val_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    mptrec = MPTRec(
        num_tasks=2,
        feature_vocabulary=ci_vocab,
        embedding_size=4,
        input_size=123,
        expert_dnn_hidden_units=[256, 128],
        tower_dnn_hidden_units=[64, 32],
        reg_embedding=args.reg_embedding,
        reg_dnn=args.reg_dnn,
        variant=args.variant,
        device=device,
    )
    mptrec.to(device)

    count_params(mptrec)

    print(f"\n{'='*60}")
    print(f"CensusIncome variant={args.variant} fusion_mode={args.fusion_mode} seed={args.seed}")
    print(f"{'='*60}")

    # Phase 1: 2-task pretraining
    print("-" * 32, "Multi-task pre-training phase", "-" * 32)
    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=[task1, task2],
        lr=1e-3,
        batch_size=256,
        uni_coe=args.uni_coe,
        env_coe=args.env_coe,
        epochs=10,
    )
    train_manager.train_two_task()
    mptrec.load_state_dict(train_manager.best_weight)

    # Figure 4: evaluate T1/T2 on test set
    auc_t1, auc_t2 = evaluate_two_task(mptrec, test_loader, device)
    print(f"Figure4: AUC-Test-{task1}={auc_t1:.4f}, AUC-Test-{task2}={auc_t2:.4f}")

    # Phase 2: New task generalization (Figure 5/6)
    if args.skip_finetune:
        return {"T1": auc_t1, "T2": auc_t2}

    print("-" * 32, "New task generalization phase", "-" * 32)
    newtask = NewTask(
        input_size=123,
        rep_dim=128,
        tower_dnn_hidden_units=[64, 32],
        reg_dnn=args.reg_dnn,
        fusion_mode=args.fusion_mode,
        device=device,
    )
    newtask.to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-3)
    loss_func = nn.BCELoss()
    epochs = 30
    patience = 5
    earlystop_count = 0
    best_auc_score = 0
    best_weight = None

    for epoch in range(1, epochs + 1):
        newtask.train()
        for _, _, y, features in train_loader:
            for key in features.keys():
                features[key] = features[key].to(device)
            dnn_input, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        auc_val = evaluate_new_task(newtask, mptrec, val_loader, device)
        print(f"AUC-Val-{task3}:{auc_val:.4f}")
        if auc_val > best_auc_score:
            earlystop_count = 0
            best_auc_score = auc_val
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            print(f"EarlyStopping count {earlystop_count}")
            if earlystop_count == patience:
                print(f"EarlyStopping at epoch {epoch}")
                break

    newtask.load_state_dict(best_weight)
    auc_t3 = evaluate_new_task(newtask, mptrec, test_loader, device)
    print(f"Figure5/6: AUC-Test-{task3}={auc_t3:.4f}")

    return {"T1": auc_t1, "T2": auc_t2, "T3": auc_t3}


def run_aliccp(args):
    device = torch.device(f"cuda:{args.gpu}")
    task1, task2, task3 = "CTR", "CVR", "BSI"

    train_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.train", 5000000)
    val_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.dev", 500000)
    test_dataset = AliCCPDataset("dataset/AliCCP/ctr_cvr.test", 5000000)
    train_loader = DataLoader(train_dataset, batch_size=2000,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=2000,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=2000,
                             num_workers=2, pin_memory=True, persistent_workers=True)
    env_ids = torch.randint(0, 2, (len(train_dataset),))

    ali_vocab = AliCCP_Vocabulary_Size.copy()
    ali_vocab.pop("101")
    ali_vocab.pop("301")

    mptrec = MPTRec(
        num_tasks=2,
        feature_vocabulary=ali_vocab,
        embedding_size=5,
        input_size=80,
        expert_dnn_hidden_units=[128, 64],
        tower_dnn_hidden_units=[32, 32],
        dropout=[0.1, 0.3],
        reg_embedding=args.reg_embedding,
        reg_dnn=args.reg_dnn,
        variant=args.variant,
        device=device,
    )
    mptrec.to(device)

    count_params(mptrec)

    print(f"\n{'='*60}")
    print(f"AliCCP variant={args.variant} fusion_mode={args.fusion_mode} seed={args.seed}")
    print(f"{'='*60}")

    # Phase 1: 2-task pretraining
    print("-" * 32, "Multi-task pre-training phase", "-" * 32)
    train_manager = MPTRecTrainManager(
        model=mptrec,
        train_loader=train_loader,
        val_loader=val_loader,
        env_ids=env_ids,
        task_name=[task1, task2],
        lr=1e-4,
        batch_size=2000,
        uni_coe=args.uni_coe,
        env_coe=args.env_coe,
        epochs=10,
    )
    train_manager.train_two_task()
    mptrec.load_state_dict(train_manager.best_weight)

    # Figure 4: evaluate T1/T2
    auc_t1, auc_t2 = evaluate_two_task(mptrec, test_loader, device)
    print(f"Figure4: AUC-Test-{task1}={auc_t1:.4f}, AUC-Test-{task2}={auc_t2:.4f}")

    if args.skip_finetune:
        return {"T1": auc_t1, "T2": auc_t2}

    # Phase 2: New task generalization (Figure 5/6)
    print("-" * 32, "New task generalization phase", "-" * 32)
    newtask = NewTask(
        input_size=80,
        rep_dim=64,
        tower_dnn_hidden_units=[32, 32],
        reg_dnn=args.reg_dnn,
        fusion_mode=args.fusion_mode,
        device=device,
    )
    newtask.to(device)

    optimizer = torch.optim.Adam(params=newtask.parameters(), lr=1e-4)
    loss_func = nn.BCELoss()
    epochs = 30
    patience = 5
    earlystop_count = 0
    best_auc_score = 0
    best_weight = None

    for epoch in range(1, epochs + 1):
        newtask.train()
        for _, _, y, features in train_loader:
            for key in features.keys():
                features[key] = features[key].to(device)
            dnn_input, gen_rep, spec_reps, env_embs = mptrec.get_infos(features)
            pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
            loss = loss_func(pred.cpu(), y.float()) + newtask.get_l2_reg()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        auc_val = evaluate_new_task(newtask, mptrec, val_loader, device)
        print(f"AUC-Val-{task3}:{auc_val:.4f}")
        if auc_val > best_auc_score:
            earlystop_count = 0
            best_auc_score = auc_val
            best_weight = copy.deepcopy(newtask.state_dict())
        else:
            earlystop_count += 1
            print(f"EarlyStopping count {earlystop_count}")
            if earlystop_count == patience:
                print(f"EarlyStopping at epoch {epoch}")
                break

    newtask.load_state_dict(best_weight)
    auc_t3 = evaluate_new_task(newtask, mptrec, test_loader, device)
    print(f"Figure5/6: AUC-Test-{task3}={auc_t3:.4f}")

    return {"T1": auc_t1, "T2": auc_t2, "T3": auc_t3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CensusIncome",
                        choices=["CensusIncome", "AliCCP"])
    parser.add_argument("--variant", type=str, default="full",
                        choices=["full", "share_only", "specific_only", "no_gan"])
    parser.add_argument("--fusion_mode", type=str, default="prompt",
                        choices=["prompt", "fw", "tes"])
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Only do 2-task pretraining (Figure 4)")
    parser.add_argument("--uni_coe", type=float, default=0)
    parser.add_argument("--env_coe", type=float, default=0)
    parser.add_argument("--reg_embedding", type=float, default=0.006)
    parser.add_argument("--reg_dnn", type=float, default=3e-5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1685480945)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "CensusIncome":
        result = run_censusincome(args)
    else:
        result = run_aliccp(args)

    print(f"\n{'='*60}")
    print(f"RESULT: {args.dataset} variant={args.variant} fusion_mode={args.fusion_mode} seed={args.seed}")
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
