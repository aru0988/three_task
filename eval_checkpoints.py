"""评估所有 checkpoint，记录 AUC 均值和标准差"""
import sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append('')

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import SharedBottom, MMOE, PLE, STEM

SEEDS = [1685480945, 1685463909, 1685477428]


@torch.no_grad()
def evaluate_model(model, test_loader, task_num, device):
    model.eval()
    y_true = [[] for _ in range(task_num)]
    y_hat = [[] for _ in range(task_num)]
    for y_0, y_1, y_2, features in test_loader:
        y = [y_0, y_1, y_2]
        for key in features.keys():
            features[key] = features[key].to(device)
        pred = model(features)
        for i in range(task_num):
            y_true[i].append(y[i])
            y_hat[i].append(pred[i])
    aucs = []
    for i in range(task_num):
        yt = torch.cat(y_true[i])
        yh = torch.cat(y_hat[i])
        aucs.append(roc_auc_score(yt.int(), yh.cpu()))
    return aucs


def make_test_loader(seed, batch_size=256):
    test_dataset = CensusIncomeDataset('dataset/Census-income/test.gz', 'education')
    _, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    return DataLoader(test_dataset, batch_size=batch_size)


def eval_sharedbottom(seed, device):
    ci_vocab = CensusIncome_Vocabulary_Size.copy()
    ci_vocab.pop("education")
    model = SharedBottom(
        num_tasks=3, feature_vocabulary=ci_vocab, embedding_size=4,
        input_size=123, shared_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32), reg_embedding=3e-4, reg_dnn=3e-4
    )
    model.load_state_dict(torch.load(f'baseline/sharedbottom/census_income_{seed}.pt'))
    model.to(device)
    test_loader = make_test_loader(seed)
    return evaluate_model(model, test_loader, 3, device)


def eval_mmoe(seed, device):
    ci_vocab = CensusIncome_Vocabulary_Size.copy()
    ci_vocab.pop("education")
    model = MMOE(
        num_tasks=3, num_experts=3, feature_vocabulary=ci_vocab, embedding_size=4,
        input_size=123, expert_dnn_hidden_units=(256, 128),
        tower_dnn_hidden_units=(64, 32), reg_embedding=3e-4, reg_dnn=3e-4
    )
    model.load_state_dict(torch.load(f'baseline/mmoe/census_income_{seed}.pt'))
    model.to(device)
    test_loader = make_test_loader(seed)
    return evaluate_model(model, test_loader, 3, device)


def eval_ple(seed, device):
    ci_vocab = CensusIncome_Vocabulary_Size.copy()
    ci_vocab.pop("education")
    model = PLE(
        num_tasks=3, input_size=123, feature_vocabulary=ci_vocab, embedding_size=4,
        shared_expert_num=1, specific_expert_num=1, num_levels=2,
        expert_dnn_hidden_units=(256,), tower_dnn_hidden_units=(64, 32),
        reg_embedding=3e-4, reg_dnn=3e-4
    )
    model.load_state_dict(torch.load(f'baseline/ple/CensusIncome_{seed}.pt'))
    model.to(device)
    test_loader = make_test_loader(seed)
    return evaluate_model(model, test_loader, 3, device)


def eval_stem(seed, device):
    ci_vocab = CensusIncome_Vocabulary_Size.copy()
    ci_vocab.pop("education")
    model = STEM(
        task_num=3, shared_expert_num=1, specific_expert_num=1,
        feature_vocabulary=ci_vocab, embedding_size=4, input_size=123,
        expert_dnn_hidden_unit=[256, 128], tower_dnn_hidden_unit=[64, 32],
        reg_embedding=3e-4, reg_dnn=3e-4
    )
    model.load_state_dict(torch.load(f'baseline/stem/census_income_{seed}.pt'))
    model.to(device)
    test_loader = make_test_loader(seed)
    return evaluate_model(model, test_loader, 3, device)


def run_eval(name, eval_fn, device):
    print(f"\n{'='*60}")
    print(f"{name} 3-task (T1=Income, T2=Marital, T3=Education)")
    print(f"{'='*60}")
    all_aucs = []
    for seed in SEEDS:
        try:
            aucs = eval_fn(seed, device)
            all_aucs.append(aucs)
            print(f"Seed {seed}: Income={aucs[0]:.4f}, Marital={aucs[1]:.4f}, Education={aucs[2]:.4f}")
        except Exception as e:
            print(f"Seed {seed}: FAILED - {e}")
    if all_aucs:
        arr = np.array(all_aucs)
        m, s = arr.mean(axis=0), arr.std(axis=0)
        print(f"Mean: Income={m[0]:.4f}±{s[0]:.4f}, Marital={m[1]:.4f}±{s[1]:.4f}, Education={m[2]:.4f}±{s[2]:.4f}")
    return all_aucs


def main():
    device = torch.device("cuda:0")
    run_eval("SharedBottom", eval_sharedbottom, device)
    run_eval("MMOE", eval_mmoe, device)
    run_eval("PLE", eval_ple, device)
    run_eval("STEM", eval_stem, device)


if __name__ == '__main__':
    main()
