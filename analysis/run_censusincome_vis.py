"""Figure 7: CensusIncome t-SNE representation visualization.
Compares MPT-Rec vs PLE generic/specific representations.
"""
import sys
import warnings
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, '.')

from config import CensusIncome_Vocabulary_Size
from multitaskrec.dataset import CensusIncomeDataset
from multitaskrec.model import PLE, MPTRec
from multitaskrec.train import TrainManager, MPTRecTrainManager

warnings.filterwarnings('ignore')

seed = 1685480945
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("=" * 60)
print("Figure 7: CensusIncome t-SNE Visualization")
print("=" * 60)

train_dataset = CensusIncomeDataset('dataset/Census-income/train.gz', 'education')
test_dataset = CensusIncomeDataset('dataset/Census-income/test.gz', 'education')
val_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, random_state=seed)
train_loader = DataLoader(train_dataset, batch_size=256)
val_loader = DataLoader(val_dataset, batch_size=256)
test_loader = DataLoader(test_dataset, batch_size=256)

ci_vocabulary = CensusIncome_Vocabulary_Size.copy()
ci_vocabulary.pop("education")

device = torch.device("cuda:0")

# ===== MPT-Rec =====
print("\n--- Training MPT-Rec ---")
model = MPTRec(
    num_tasks=2,
    feature_vocabulary=ci_vocabulary,
    embedding_size=4,
    input_size=123,
    expert_dnn_hidden_units=(256, 128),
    tower_dnn_hidden_units=(64, 32),
    reg_embedding=0.006,
    reg_dnn=1e-5,
    device=device
)
model.to(device)

train_manager = MPTRecTrainManager(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    env_ids=torch.randint(0, 2, (len(train_dataset),)),
    task_name=['Income', 'Marital'],
    lr=1e-3,
    batch_size=256,
    uni_coe=0,
    env_coe=0,
    epochs=10,
)
train_manager.train_two_task()
model.load_state_dict(train_manager.best_weight)

print("Extracting MPT-Rec representations...")
generic_rep_list, proprietary_rep_0_list, proprietary_rep_1_list = [], [], []
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    _, gen_rep, spec_reps, _ = model.get_infos(features)
    generic_rep_list.append(gen_rep)
    proprietary_rep_0_list.append(spec_reps[0])
    proprietary_rep_1_list.append(spec_reps[1])
mptrec_gen = torch.cat(generic_rep_list).detach().cpu()
mptrec_spec0 = torch.cat(proprietary_rep_0_list).detach().cpu()
mptrec_spec1 = torch.cat(proprietary_rep_1_list).detach().cpu()

# ===== PLE =====
print("\n--- Loading PLE ---")
model_ple = PLE(
    num_tasks=3,
    input_size=123,
    feature_vocabulary=ci_vocabulary,
    embedding_size=4,
    shared_expert_num=1,
    specific_expert_num=1,
    num_levels=2,
    expert_dnn_hidden_units=(256, ),
    tower_dnn_hidden_units=(64, 32),
    reg_embedding=3e-4,
    reg_dnn=3e-4
)
model_ple.load_state_dict(torch.load(f'baseline/ple/CensusIncome_{seed}.pt'))
model_ple.to(device)

print("Extracting PLE representations...")
generic_rep_list, proprietary_rep_0_list, proprietary_rep_1_list = [], [], []
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    generic_rep, proprietary_reps = model_ple.get_reps(features)
    generic_rep_list.append(generic_rep)
    proprietary_rep_0_list.append(proprietary_reps[0])
    proprietary_rep_1_list.append(proprietary_reps[1])
ple_gen = torch.cat(generic_rep_list).detach().cpu()
ple_spec0 = torch.cat(proprietary_rep_0_list).detach().cpu()
ple_spec1 = torch.cat(proprietary_rep_1_list).detach().cpu()

# ===== t-SNE =====
print("\n--- Running t-SNE ---")
point_size = 20000

# MPT-Rec
rep_mptrec = torch.cat([mptrec_gen[:point_size], mptrec_spec0[:point_size], mptrec_spec1[:point_size]])
tsne = manifold.TSNE(n_components=2, random_state=2023, perplexity=30)
emb_2d = tsne.fit_transform(rep_mptrec.numpy())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(emb_2d[:point_size, 0], emb_2d[:point_size, 1], c='green', s=1, label='Generic')
axes[0].scatter(emb_2d[point_size:2*point_size, 0], emb_2d[point_size:2*point_size, 1], c='#FF6666', s=1, label='Specific (Income)')
axes[0].scatter(emb_2d[2*point_size:, 0], emb_2d[2*point_size:, 1], c='#0066CC', s=1, label='Specific (Marital)')
axes[0].set_title('MPT-Rec Representations', fontsize=12, fontweight='bold')
axes[0].legend(markerscale=5)

# PLE
rep_ple = torch.cat([ple_gen[:point_size], ple_spec0[:point_size], ple_spec1[:point_size]])
emb_2d = tsne.fit_transform(rep_ple.numpy())

axes[1].scatter(emb_2d[:point_size, 0], emb_2d[:point_size, 1], c='green', s=1, label='Generic')
axes[1].scatter(emb_2d[point_size:2*point_size, 0], emb_2d[point_size:2*point_size, 1], c='#FF6666', s=1, label='Specific (Income)')
axes[1].scatter(emb_2d[2*point_size:, 0], emb_2d[2*point_size:, 1], c='#0066CC', s=1, label='Specific (Marital)')
axes[1].set_title('PLE Representations', fontsize=12, fontweight='bold')
axes[1].legend(markerscale=5)

plt.suptitle('CensusIncome: Representation Separation (Figure 7)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/fig7_censusincome_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis/fig7_censusincome_tsne.png")
print("Done!")
