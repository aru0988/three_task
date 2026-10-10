"""Independent raw-artifact verifier for paired B/U experiments."""

import argparse
import json
from pathlib import Path

import numpy as np

from universal_experiment.verify import rank_auc


def verify(folder, *, write=True):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    checks = {}
    scores = {}
    reference_labels = {}
    for arm in ("B", "U"):
        raw = np.load(folder / f"{arm}_predictions.npz")
        scores[arm] = {}
        metrics = report["stage2"][arm]
        for split in ("val", "test"):
            labels = raw[f"{split}_y"]
            predictions = raw[f"{split}_p"]
            score = rank_auc(labels, predictions)
            scores[arm][split] = score
            checks[f"{arm}_{split}_auc"] = abs(score - metrics[f"{split}_auc"]) < 1e-12
            if split in reference_labels:
                checks[f"{arm}_{split}_labels"] = np.array_equal(
                    labels, reference_labels[split])
            reference_labels[split] = labels
        selected = max(metrics["epochs"], key=lambda row: row["val_auc"])
        checks[f"{arm}_selection"] = (
            selected["epoch"] == metrics["best_epoch"]
            and abs(selected["val_auc"] - metrics["val_auc"]) < 1e-12
        )
        checks[f"{arm}_convergence_recorded"] = (
            bool(metrics["convergence"]["converged"])
            == bool(report["stage2"][arm]["convergence"]["converged"])
        )

    for key in ("seed", "budget", "patience"):
        checks[f"stage1_matched_{key}"] = (
            report["stage1"]["B"][key] == report["stage1"]["U"][key]
        )
    for key in ("budget", "patience"):
        checks[f"stage2_matched_{key}"] = (
            report["stage2"]["B"][key] == report["stage2"]["U"][key]
        )
    deltas = {split: scores["U"][split] - scores["B"][split]
              for split in ("val", "test")}
    checks["val_delta"] = abs(deltas["val"] - report["u_minus_b"]["val"]) < 1e-12
    checks["test_delta"] = abs(deltas["test"] - report["u_minus_b"]["test"]) < 1e-12
    expected_class = ("positive" if deltas["test"] >= .001 else
                      "clear_decline" if deltas["test"] <= -.02 else
                      "no_clear_improvement")
    checks["classification"] = expected_class == report["classification"]
    expected_converged = all(
        report[phase][arm]["convergence"]["converged"]
        for phase in ("stage1", "stage2") for arm in ("B", "U")
    )
    checks["all_converged"] = expected_converged == report["all_converged"]
    checks["clean_formal_start"] = (
        not report["config"]["dirty"] or report["config"]["smoke"]
    )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "independent_auc": scores,
        "independent_u_minus_b": deltas,
        "failures": [name for name, passed in checks.items() if not passed],
    }
    if write:
        (folder / "verification.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"paired verification failed: {result['failures']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
