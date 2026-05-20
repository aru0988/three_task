"""
TC-Prompt: Task-Correlation-aware Prompt Fusion.

Replaces the original instance-level attention in MPT-Rec Stage 2 with
a fusion mechanism that incorporates task correlation priors:

    W_i = Softmax( h_p · E_i / τ  +  λ · ρ_new,i )

where:
  - h_p = projection(dnn_input)    instance-level query
  - E_i = frozen source task embeddings
  - τ   = temperature (default 150)
  - λ   = learnable scalar balancing correlation prior
  - ρ   = pre-computed Pearson correlation (new_task, source_task_i)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TCPromptFusion(nn.Module):
    """Task-Correlation-aware Prompt Fusion.

    Extends MPT-Rec's instance-level attention with a task-correlation prior
    that biases the attention distribution toward tasks more correlated with
    the new task.  When λ = 0, this reduces to the original MPT-Rec attention.
    """

    def __init__(self, input_size, rep_dim, num_source_tasks,
                 rho_vector, temperature=150.0, lambda_init=0.5,
                 lambda_learnable=True):
        """
        Args:
            input_size: dimension of dnn_input
            rep_dim: dimension of representation space
            num_source_tasks: K = number of source tasks
            rho_vector: (K,) float32 array of Pearson correlations
                        between new_task and each source_task_i
            temperature: softmax temperature τ
            lambda_init: initial value for λ
            lambda_learnable: whether λ is trainable
        """
        super().__init__()
        self.rep_dim = rep_dim
        self.num_source_tasks = num_source_tasks
        self.temperature = temperature

        # Projection network — same architecture as original MPT-Rec
        self.projection_network = nn.Sequential(
            nn.Linear(input_size, rep_dim // 2, bias=False),
            nn.ReLU(),
            nn.Linear(rep_dim // 2, rep_dim, bias=False),
            nn.LayerNorm(rep_dim),
        )

        # λ: balances instance-level attention vs correlation prior
        if lambda_learnable:
            self.lambda_corr = nn.Parameter(torch.tensor(lambda_init))
        else:
            self.register_buffer(
                'lambda_corr', torch.tensor(lambda_init)
            )

        # Task correlation prior (fixed, not optimized)
        self.register_buffer(
            'rho', torch.tensor(rho_vector, dtype=torch.float32)
        )

    def forward(self, dnn_input, exist_env_embs):
        """Compute attention weights and fused representation.

        Args:
            dnn_input: (B, input_size) raw input features
            exist_env_embs: (K, rep_dim) frozen source task embeddings

        Returns:
            W: (B, K, 1) attention weights
        """
        B = dnn_input.shape[0]

        # Instance-level query
        h_p = self.projection_network(dnn_input)  # (B, rep_dim)

        # Instance-level similarity scores
        instance_scores = torch.mm(h_p, exist_env_embs) / self.temperature  # (B, K)

        # Task-correlation prior (broadcast to batch)
        corr_scores = self.rho.unsqueeze(0).expand(B, -1)  # (B, K)

        # Combined scores: W_i ∝ exp(instance_score_i + λ · ρ_i)
        combined_scores = instance_scores + self.lambda_corr * corr_scores
        W = F.softmax(combined_scores, dim=-1).unsqueeze(2)  # (B, K, 1)

        return W

    def get_lambda(self):
        """Return current λ value (for logging)."""
        return self.lambda_corr.item()
