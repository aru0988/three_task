# Universal Representation: Paired B/U Experiments

This branch compares the original MPT-Rec baseline **B**, whose structure contains no Universal Representation module, with one U-based modification at a time. Four-arm B/U/R/G experiments live only on `exp/universal-representation`.

## Protocol and baseline reuse

B is trained once per exact protocol identity and then reused. Reuse requires the same dataset/task order, split fingerprint, seeds, code and hyperparameters, initialization, budgets, patience, validation selection and saved artifact hashes. The machine-readable registry is `results/universal/paired-baselines.json`. Any mismatch creates a new baseline entry rather than silently reusing a result.

Current Census protocol: seed 1685480945, split seed 20260929, 199523/49881/49881 examples, Stage-1 and Stage-2 maximum 20 epochs, patience 5, validation-only selection. The canonical original MPT-Rec B has Education validation/test AUC 0.8606132635/0.8599184895 and Income/Marital test AUC 0.9455806928/0.9909109553.

Classification is appended without changing historical decisions: U−B test AUC >= +0.001 is positive; −0.02 < delta < +0.001 is no clear improvement; delta <= −0.02 is clear decline. Convergence, old-task behavior and cost are reported separately.

## Census no-read, frozen U

Evidence: `results/universal/paired-census-no-read-frozen-seed1685480945/`; clean training commit `b619f19`; independent verifier passed every check.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8592235244 | 0.8573714985 | 19 | 20 |

U−B is −0.0013897391 validation and −0.0025469910 test: no clear improvement. B converged by patience; U reached the budget before patience. Old tasks do not read U, so B and U have identical base hashes and old-task predictions. This variant is closed unless a new mechanism hypothesis is registered.

## Census detached old-task read, frozen U

Evidence: `results/universal/paired-census-read-detached-frozen-seed1685480945/`; clean training commit `427282d`; independent verifier passed every check.

| Arm | Education val AUC | Education test AUC | Best epoch | Epochs run |
| --- | ---: | ---: | ---: | ---: |
| B | 0.8606132635 | 0.8599184895 | 13 | 18 |
| U | 0.8622726923 | 0.8614862850 | 20 | 20 |

U−B is +0.0016594288 validation and +0.0015677956 test: positive on this single seed. Income changes by −0.0007808833 and Marital by +0.0009854415 relative to B, so neither old task declines by 0.001. U reaches its best validation score at the 20-epoch cap and is not declared converged. Relative to no-read U, detached-read improves Education test AUC by +0.0041147866; this descriptive cross-screen contrast motivates retaining detached read for the next frozen-versus-trainable-U test.

## Reproduction and verification

```powershell
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m unittest universal_experiment.test_convergence universal_experiment.test_paired universal_experiment.test_paired_run universal_experiment.test_verify_paired -v
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m universal_experiment.verify_paired results\universal\paired-census-no-read-frozen-seed1685480945
& 'D:\MPT-Rec-three_task\MPT-Rec\.venv\Scripts\python.exe' -m universal_experiment.verify_paired results\universal\paired-census-read-detached-frozen-seed1685480945
```

The next authorized comparison is detached-read U with frozen versus trainable Stage-2 U. Each U variant is judged against the same canonical B; B is not retrained when the manifest identity matches.
