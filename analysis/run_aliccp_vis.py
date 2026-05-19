"""Figure 7: AliCCP t-SNE representation visualization.
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
from torch.utils.data import DataLoader

sys.path.insert(0, '.')

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import PLE, MPTRec
from multitaskrec.train import MPTRecTrainManager

warnings.filterwarnings('ignore')

seed_mptrec = 1688723512
seed_ple = 1688723740
np.random.seed(seed_mptrec)
torch.manual_seed(seed_mptrec)
torch.cuda.manual_seed(seed_mptrec)
torch.cuda.manual_seed_all(seed_mptrec)

print("=" * 60)
print("Figure 7: AliCCP t-SNE Visualization")
print("=" * 60)

train_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.train', 1000000)
val_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.dev', 100000)
test_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.test', 1000000)
train_loader = DataLoader(train_dataset, batch_size=2000)
val_loader = DataLoader(val_dataset, batch_size=2000)
test_loader = DataLoader(test_dataset, batch_size=2000)
env_ids = torch.randint(0, 2, (len(train_dataset),))

device = torch.device("cuda:0")

ali_vocab = AliCCP_Vocabulary_Size.copy()
ali_vocab.pop("101")
ali_vocab.pop("301")

# ===== MPT-Rec =====
print("\n--- Training MPT-Rec ---")
model = MPTRec(
    num_tasks=2,
    feature_vocabulary=ali_vocab,
    embedding_size=5,
    input_size=80,
    expert_dnn_hidden_units=(128, 64),
    tower_dnn_hidden_units=(32, 32),
    dropout=(0.1, 0.3),
    reg_embedding=0.0001,
    reg_dnn=7e-6,
    device=device
)
model.to(device)

train_manager = MPTRecTrainManager(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    env_ids=env_ids,
    task_name=['CTR', 'CVR'],
    lr=1e-4,
    batch_size=2000,
    uni_coe=0.9,
    env_coe=0.1,
    epochs=10,
)
train_manager.train_two_task()
model.load_state_dict(train_manager.best_weight)

print("Extracting MPT-Rec representations...")
generic_rep_list, proprietary_rep_0_list, proprietary_rep_1_list = [], [], []
count = 0
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    _, gen_rep, spec_reps, _ = model.get_infos(features)
    generic_rep_list.append(gen_rep)
    proprietary_rep_0_list.append(spec_reps[0])
    proprietary_rep_1_list.append(spec_reps[1])
    count += 1
    if count == 100:
        break
mptrec_gen = torch.cat(generic_rep_list).detach().cpu()
mptrec_spec0 = torch.cat(proprietary_rep_0_list).detach().cpu()
mptrec_spec1 = torch.cat(proprietary_rep_1_list).detach().cpu()

# ===== PLE =====
print("\n--- Loading PLE ---")
torch.manual_seed(seed_ple)
torch.cuda.manual_seed(seed_ple)
torch.cuda.manual_seed_all(seed_ple)
np.random.seed(seed_ple)

model_ple = PLE(
    num_tasks=3,
    input_size=80,
    feature_vocabulary=ali_vocab,
    embedding_size=5,
    shared_expert_num=1,
    specific_expert_num=1,
    num_levels=2,
    expert_dnn_hidden_units=(128, ),
    tower_dnn_hidden_units=(32, 32),
    reg_embedding=1e-6,
    reg_dnn=1e-6,
    dropout=(0.1, 0.3),
)
model_ple.load_state_dict(torch.load(f'baseline/ple/AliCCP_{seed_ple}.pt'))
model_ple.to(device)

print("Extracting PLE representations...")
generic_rep_list, proprietary_rep_0_list, proprietary_rep_1_list = [], [], []
count = 0
for _, _, _, features in train_loader:
    for key in features.keys():
        features[key] = features[key].to(device)
    generic_rep, proprietary_reps = model_ple.get_reps(features)
    generic_rep_list.append(generic_rep)
    proprietary_rep_0_list.append(proprietary_reps[0])
    proprietary_rep_1_list.append(proprietary_reps[1])
    count += 1
    if count == 100:
        break
ple_gen = torch.cat(generic_rep_list).detach().cpu()
ple_spec0 = torch.cat(proprietary_rep_0_list).detach().cpu()
ple_spec1 = torch.cat(proprietary_rep_1_list).detach().cpu()

# ===== t-SNE =====
print("\n--- Running t-SNE ---")
point_size = 40000

# MPT-Rec
rep_mptrec = torch.cat([mptrec_gen[:point_size], mptrec_spec0[:point_size], mptrec_spec1[:point_size]])
tsne = manifold.TSNE(n_components=2, random_state=2023, perplexity=30)
emb_2d = tsne.fit_transform(rep_mptrec.numpy())

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
axes[0].scatter(emb_2d[:point_size, 0], emb_2d[:point_size, 1], c='green', s=1, label='Generic')
axes[0].scatter(emb_2d[point_size:2*point_size, 0], emb_2d[point_size:2*point_size, 1], c='#FF6666', s=1, label='Specific (CTR)')
axes[0].scatter(emb_2d[2*point_size:, 0], emb_2d[2*point_size:, 1], c='#0066CC', s=1, label='Specific (CVR)')
axes[0].set_title('MPT-Rec Representations', fontsize=12, fontweight='bold')
axes[0].legend(markerscale=5)

# PLE
rep_ple = torch.cat([ple_gen[:point_size], ple_spec0[:point_size], ple_spec1[:point_size]])
emb_2d = tsne.fit_transform(rep_ple.numpy())

axes[1].scatter(emb_2d[:point_size, 0], emb_2d[:point_size, 1], c='green', s=1, label='Generic')
axes[1].scatter(emb_2d[point_size:2*point_size, 0], emb_2d[point_size:2*point_size, 1], c='#FF6666', s=1, label='Specific (CTR)')
axes[1].scatter(emb_2d[2*point_size:, 0], emb_2d[2*point_size:, 1], c='#0066CC', s=1, label='Specific (CVR)')
axes[1].set_title('PLE Representations', fontsize=12, fontweight='bold')
axes[1].legend(markerscale=5)

plt.suptitle('AliCCP: Representation Separation (Figure 7)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('analysis/fig7_aliccp_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved analysis/fig7_aliccp_tsne.png")
print("Done!")
