"""评估所有 5 个 seed 的 checkpoint，记录 AUC 均值和标准差"""
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

    test_dataset = CensusIncomeDataset('dataset/Census-income/test.gz', 'education')
    _, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
    test_loader = DataLoader(test_dataset, batch_size=256)

    aucs = evaluate_model(model, test_loader, 3, device)
    return aucs


def main():
    device = torch.device("cuda:0")
    seeds = [1685480945, 1685463909, 1685477428, 1685459668, 1685496394]
    task_names = ['Income', 'Marital', 'Education']

    print("=" * 72)
    print("SharedBottom 3-task (T1=Income, T2=Marital, T3=Education)")
    print("=" * 72)
    all_aucs = []
    for seed in seeds:
        try:
            aucs = eval_sharedbottom(seed, device)
            all_aucs.append(aucs)
            print(f"Seed {seed}: AUC-Income={aucs[0]:.4f}, AUC-Marital={aucs[1]:.4f}, AUC-Education={aucs[2]:.4f}")
        except Exception as e:
            print(f"Seed {seed}: FAILED - {e}")

    if all_aucs:
        all_aucs = np.array(all_aucs)
        mean_aucs = all_aucs.mean(axis=0)
        std_aucs = all_aucs.std(axis=0)
        print("-" * 72)
        print(f"Mean: AUC-Income={mean_aucs[0]:.4f}±{std_aucs[0]:.4f}, AUC-Marital={mean_aucs[1]:.4f}±{std_aucs[1]:.4f}, AUC-Education={mean_aucs[2]:.4f}±{std_aucs[2]:.4f}")


if __name__ == '__main__':
    main()
