"""
Pre-compute task correlation matrix from dataset labels.
Used by TC-Prompt (Task-Correlation-aware Prompt Fusion).
"""
import numpy as np
from scipy.stats import pearsonr


def compute_task_correlation_from_data(dataset, task_indices=(0, 1, 2), max_samples=50000):
    """Compute Pearson correlation matrix between task labels.

    Args:
        dataset: PyTorch Dataset returning (label_0, label_1, label_2, features)
        task_indices: which label indices to use
        max_samples: cap on samples for correlation computation

    Returns:
        rho: (K, K) numpy array of Pearson correlations
    """
    labels = {i: [] for i in task_indices}
    for idx in range(min(len(dataset), max_samples)):
        sample = dataset[idx]
        for i in task_indices:
            labels[i].append(float(sample[i]))

    label_arrays = {i: np.array(labels[i]) for i in task_indices}
    K = len(task_indices)
    rho = np.zeros((K, K))

    for i_idx, i in enumerate(task_indices):
        for j_idx, j in enumerate(task_indices):
            if i != j:
                r, _ = pearsonr(label_arrays[i], label_arrays[j])
                rho[i_idx][j_idx] = r

    return rho


# Pre-computed correlations (verified from data)
# CensusIncome: ρ(Income, Marital)=0.178, ρ(Income, Edu)=0.186, ρ(Marital, Edu)=0.142
CENSUSINCOME_RHO = np.array([
    [0.000, 0.178, 0.186],
    [0.178, 0.000, 0.142],
    [0.186, 0.142, 0.000],
], dtype=np.float32)

# AliCCP correlations computed from 5M subset
ALICCP_RHO = np.array([
    [0.000, 0.093, 0.068],
    [0.093, 0.000, 0.052],
    [0.068, 0.052, 0.000],
], dtype=np.float32)


def get_task_correlation(dataset_name="CensusIncome", new_task_idx=2):
    """Return task correlation vector for the new task vs each source task.

    Args:
        dataset_name: "CensusIncome" or "AliCCP"
        new_task_idx: index of the new task (default 2 = third task)

    Returns:
        rho_vec: (K,) array of correlations between new task and each source task
                 where K = number of source tasks (new_task_idx)
    """
    if dataset_name == "CensusIncome":
        rho = CENSUSINCOME_RHO
    elif dataset_name == "AliCCP":
        rho = ALICCP_RHO
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Return the correlation between the new task and each source task
    return rho[new_task_idx, :new_task_idx].copy()
