import sys
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('')

from config import AliCCP_Vocabulary_Size
from multitaskrec.dataset import AliCCPDataset
from multitaskrec.model import PLE
from multitaskrec.train import TrainManager

warnings.filterwarnings('ignore')


def main():
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    task_num = 3
    ali_vocab = AliCCP_Vocabulary_Size.copy()
    ali_vocab.pop("101")
    ali_vocab.pop("301")
    model = PLE(
        num_tasks=task_num,
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
    device = torch.device("cuda:0")
    model.to(device)

    from multitaskrec.utils import count_params
    count_params(model)

    train_manager = TrainManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task_name=['CTR', 'CVR', 'BSI'],
        lr=1e-4,
    )
    train_manager.train_multi_task(task_num)
    torch.save(train_manager.best_weight, f'baseline/ple/AliCCP_{seed}.pt')

    model.load_state_dict(train_manager.best_weight)
    auc_test = train_manager.evaluation_multi_task(test_loader, task_num)
    print('AUC-Test-CTR:{:.4f}, AUC-Test-CVR:{:.4f}, AUC-Test-BSI:{:.4f}'.format(auc_test[0], auc_test[1], auc_test[2]))


if __name__ == '__main__':
    train_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.train', 5000000)
    val_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.dev', 500000)
    test_dataset = AliCCPDataset('dataset/AliCCP/ctr_cvr.test', 5000000)
    train_loader = DataLoader(train_dataset, batch_size=2000)
    val_loader = DataLoader(val_dataset, batch_size=2000)
    test_loader = DataLoader(test_dataset, batch_size=2000)

    for seed in [1688723740]:
        main()
