# MPT-Rec — three_task（新任务泛化）

论文 *Efficient Multi-task Prompt Tuning for Recommendation*（[arXiv:2408.17214](https://arxiv.org/abs/2408.17214)，TOIS 2025）中**新任务泛化**协议的可运行复现。

代码基于官方仓库 [BAI-LAB/MPT-Rec](https://github.com/BAI-LAB/MPT-Rec) 的 `three_task` 分支，目录结构保持一致，在此基础上修复了若干导致无法运行的 bug（详见下方"与官方代码的差异"）。

## 分支约定

| 分支 | 用途 |
|---|---|
| `master` | 干净基线：结构与论文一致、可直接运行。**新实验一律从这里拉分支** |
| `archive/exploration` | 历史探索归档（TC-Prompt / CGR / affinity gate / KL-Prompt / T4 实验 / 论文草稿），仅作存档，不再维护 |
| 其他分支 | 从 `master` 拉出的想法分支：验证有效 → 合并回 `master`；无效 → 删除 |

## 环境

- Python 3.10 + PyTorch（CUDA），依赖见 `.venv/`
- 数据集放在 `dataset/` 下：`Census-income/`（train.gz / test.gz）、`AliCCP/`（ctr_cvr.train / dev / test）。ByteRec 官方未提供数据，相关脚本无法运行

## 运行

```bash
# 单机单卡，--gpu 指定卡号

# ===== 预训练（阶段 1，两任务）=====
python CensusIncome_MPTRec.py --seed 1685480945
python AliCCP_MPTRec.py      --seed 1688723512

# ===== 新任务泛化（阶段 1+2 在一个脚本内）=====
python CensusIncome_NewTask.py --seed 1685480945     # 新任务 = education
python AliCCP_NewTask.py       --seed 1688723512     # 新任务 = BSI(301)

# ===== Baseline（阶段 1 保存权重 → 阶段 2 加载冻结）=====
python -m baseline.sharedbottom.CensusIncome_SharedBottom
python -m baseline.sharedbottom.CensusIncome_NewTask
# MMOE / PLE / STEM / SingleTask / SparseSharing / CsRec 同构
```

## 阶段协议

- **阶段 1**：两任务联合训练，MPT-Rec 用生成对抗方式分离任务共享/特异表征（论文 Eq.(1)–(7)）
- **阶段 2**：冻结预训练参数，只为新任务训练投影网络 + 门控 + tower；环境标签由**共享表征的预测损失**取 argmin 得到（论文 Eq.(6)）
- MPT-Rec 的 `uni_coe=0.9 / env_coe=0.1` 对应论文 Eq.(7) 的平衡权重 α=0.1

## 与官方代码的差异（均为运行修复）

1. `SingleTaskTrainManager` / `SparseSharingTrainManager`：补上损失累计与返回值，修复崩溃
2. `SparseSharing` / `CsRec`：mask 的 tensor/ndarray 混算、`torch.load` 设备、`np.where(p > 1, …)` 恒假的修复
3. `AliCCP_NewTask.py`：官方 `newtask(**output)` 对 tuple 解包会报错，改为按位置传参
4. CensusIncome 系列：`Vocabulary_Size` 改为 `.copy()` 后再 `pop`，避免污染全局字典
5. `SingleTask.functional_forward` 与 `STEM.get_reps` 的错误引用/缺失参数修复
6. `NewTask` / `MPTRec` 用 `register_buffer` 保存固定的任务索引张量
7. 各 baseline 脚本补齐权重保存路径、CUDA 设备与数据路径
8. **启用 GAN 损失**：官方 `three_task` 的 `MPTRecTrainManager` 把式(7) 的 GAN 损失注释掉了（只留 fused 预测损失），本仓库按论文 Eq.(7) 启用 `uni_coe*(gen) + env_coe*env`，与 `two_task` 分支的实现一致

## 引用

```bibtex
@article{bai2024efficient,
  title={Efficient Multi-task Prompt Tuning for Recommendation},
  author={Bai, Ting and Huang, Le and Yu, Yue and Yang, Cheng and Hou, Cheng and Zhao, Zhe and Shi, Chuan},
  journal={ACM Transactions on Information Systems},
  year={2025}
}
```
