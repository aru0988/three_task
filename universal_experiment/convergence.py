"""Small, auditable convergence rules shared by paired B/U runners."""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ConvergenceDecision:
    converged: bool
    reason: str
    best_epoch: int
    epochs_run: int
    budget: int
    patience: int


def convergence_decision(
    val_curve: Sequence[float], *, best_epoch: int, budget: int,
    patience: int, stopped_early: bool,
) -> ConvergenceDecision:
    """Classify only observed stopping behavior; never extrapolate a curve."""
    epochs_run = len(val_curve)
    if not val_curve:
        raise ValueError("val_curve must not be empty")
    if not 1 <= best_epoch <= epochs_run:
        raise ValueError("best_epoch must index an observed epoch")
    if budget < epochs_run:
        raise ValueError("budget cannot be smaller than epochs_run")
    if patience < 1:
        raise ValueError("patience must be positive")

    if not stopped_early:
        reason = "best_epoch_at_cap" if best_epoch == budget else "budget_exhausted_before_patience"
        return ConvergenceDecision(False, reason, best_epoch, epochs_run, budget, patience)
    if best_epoch > epochs_run - 2:
        return ConvergenceDecision(False, "best_epoch_too_late", best_epoch, epochs_run, budget, patience)
    return ConvergenceDecision(True, "patience_exhausted", best_epoch, epochs_run, budget, patience)
