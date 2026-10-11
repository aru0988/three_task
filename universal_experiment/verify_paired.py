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
        if 'base_hash_before' in metrics:
            checks[f'{arm}_base_frozen'] = metrics['base_hash_before'] == metrics['base_hash_after']
            checks[f'{arm}_base_matches_stage1'] = metrics['base_hash_before'] == report['stage1'][arm].get('base_hash')
            if arm == 'U':
                checks['U_hash_matches_stage1'] = metrics['universal_hash_before'] == report['stage1'][arm]['universal_hash']
                if metrics.get('trainable_u'):
                    checks['U_updated'] = metrics['universal_hash_before'] != metrics['universal_hash_after']
                else:
                    checks['U_frozen'] = metrics['universal_hash_before'] == metrics['universal_hash_after']
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
    config = report['config']
    if 'seed' in config:
        for arm in ('B', 'U'):
            checks[f'{arm}_config_seed'] = report['stage1'][arm]['seed'] == config['seed']
            for phase in ('stage1', 'stage2'):
                checks[f'{arm}_{phase}_config_budget'] = report[phase][arm]['budget'] == config[f'{phase}_epochs']
                checks[f'{arm}_{phase}_config_patience'] = report[phase][arm]['patience'] == config['patience']
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
    from universal_experiment.paired_run import file_hash
    for phase in ('stage1', 'stage2'):
        for arm in ('B', 'U'):
            metrics = report[phase][arm]
            suffixes = ('stage1.pt', 'old_tasks.npz') if phase == 'stage1' else ('head.pt', 'predictions.npz')
            for suffix, key in zip(suffixes, ('checkpoint_sha256', 'predictions_sha256')):
                if key in metrics:
                    checks[f'{arm}_{phase}_{key}'] = file_hash(folder / f'{arm}_{suffix}') == metrics[key]
            if phase == 'stage1' and 'old_task_auc' in metrics:
                with np.load(folder / f'{arm}_old_tasks.npz') as old_raw:
                    for split, tasks in metrics['old_task_auc'].items():
                        for i, (task, score) in enumerate(tasks.items()):
                            checks[f'{arm}_old_{split}_{task}_auc'] = abs(rank_auc(
                                old_raw[f'{split}_y{i}'], old_raw[f'{split}_p{i}']) - score) < 1e-12
    if report['stage2']['U'].get('trainable_u'):
        checks['U_selected_stage2_checkpoint'] = file_hash(folder / 'U_stage2_universal.pt') == report['stage2']['U']['universal_checkpoint_sha256']
    old_after = report.get('old_tasks_after_stage2')
    if old_after is not None:
        path = folder / 'U_old_tasks_after_stage2.npz'
        checks['old_after_hash'] = file_hash(path) == old_after['predictions_sha256']
        with np.load(path) as raw, np.load(folder / 'U_old_tasks.npz') as before:
            for split, tasks in old_after['auc'].items():
                for i, (task, score) in enumerate(tasks.items()):
                    key = f'{split}_y{i}'
                    checks[f'old_after_{split}_{task}_labels'] = np.array_equal(raw[key], before[key])
                    recomputed = rank_auc(raw[key], raw[f'{split}_p{i}'])
                    checks[f'old_after_{split}_{task}_auc'] = abs(recomputed - score) < 1e-12
                    expected_delta = recomputed - report['stage1']['U']['old_task_auc'][split][task]
                    checks[f'old_after_{split}_{task}_delta'] = abs(expected_delta - old_after['delta_from_stage1'][split][task]) < 1e-12
                    if config.get('u_mode') == 'frozen' or config.get('read_policy') == 'no_read':
                        checks[f'old_after_{split}_{task}_unchanged'] = np.array_equal(raw[f'{split}_p{i}'], before[f'{split}_p{i}'])
    if 'u_mode' in report['config'] and 'baseline_code_identity' in report['config']:
        checks['old_after_present'] = old_after is not None
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
