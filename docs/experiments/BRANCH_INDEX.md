# Experiment branch inventory and direction map

Snapshot: 2026-10-08, local `origin/exp/*` refs. This is a provenance inventory, not a claim that experiments have been migrated. There are **39** remote experiment refs, each listed exactly once below. `master`, `main`, infrastructure baselines, and `archive/exploration` are outside this experiment-direction inventory. No remote ref has yet been deleted by this consolidation.

## Stage-2 Residual Prompt (12)

Maintained target: `exp/stage2-residual-prompt-consolidated`. Dataset, seed, budget, and causal-control runs belong in one ledger, but divergent code is not automatically equivalent.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/stage2-residual-prompt-gate` | `e0ad6afbf61de037819f64b74021ff7ea95384ca` |
| `origin/exp/aliccp-stage2-residual-prompt-gate` | `3e2f0833fdb8a4966d5d815b41a8cefcfb0887d6` |
| `origin/exp/aliccp-stage2-residual-prompt-seed-replication` | `a40836e3d6a7f73c4943bada98139d414957301a` |
| `origin/exp/aliccp-stage2-residual-prompt-longer-budget` | `e46e5d28ea572c1d6d5ace68b8100b848fe374e7` |
| `origin/exp/aliccp-stage2-residual-prompt-20epoch` | `10e86dc6cbd047b103211ab75dd99c1c68db3526` |
| `origin/exp/aliccp-stage2-residual-prompt-five-seed` | `28aa1feae1fea0762136beebdfe6eb8387edb170` |
| `origin/exp/aliccp-stage2-residual-prompt-shuffled-condition` | `fbfff0945c77c74866955238f67e9aeaa0774670` |
| `origin/exp/aliccp-stage2-residual-prompt-alpha-pinned-condition` | `f08ae6e459ae9fe048daf11eda43799d19ccc7d1` |
| `origin/exp/aliccp-stage2-residual-prompt-unconditional-control` | `6375bbcb36f5df849b9ae64bced1b434641d1429` |
| `origin/exp/aliccp-stage2-residual-prompt-alpha-dynamics` | `2a940b44a54470d6dbf2225669e560617ff36133` |
| `origin/exp/census-stage2-residual-prompt-seed-recheck` | `41feb29067dc2dec94e330c6b7393762e8cd2f78` |
| `origin/exp/census-stage2-residual-prompt-3seed-expansion` | `dac28dd0aa697152584f817c93f951d5f7cabcbf` |

## Stage-2 Null Expert (4)

Maintained target: `exp/stage2-null-expert-consolidated`.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/stage2-null-expert` | `87e2b8dc9ab8d631b1f17e4c1f436a01d730ff42` |
| `origin/exp/census-stage2-null-expert-multiseed` | `1196acd04c4e5d2846e0644d6345532f3632d148` |
| `origin/exp/census-stage2-null-expert-longer-budget` | `6263d6d00b971404480480bb16fec467c262a6ba` |
| `origin/exp/aliccp-stage2-null-expert` | `aba8ee910fc4122e067c9cd45a4a7c4a1cbbd1e9` |

## Stage-2 attenuation (5)

Maintained target: `exp/stage2-attenuation-consolidated`. Fixed attenuation, learnable attenuation, and the specific-path control are distinct comparisons in one direction, not interchangeable runs.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/aliccp-stage2-attenuation-seed-replication` | `f589e6179d68669d71aa21e6313455e7976f85d4` |
| `origin/exp/aliccp-stage2-attenuation-longer-budget` | `b515f8c323d4155c6faa771b34e61aa02a257b9b` |
| `origin/exp/aliccp-stage2-learnable-attenuation` | `ac26eea10f02bffa613e754b2b3abddb82e1bb61` |
| `origin/exp/aliccp-stage2-specific-attenuation-control` | `331442077f378bb41cf1fef946b8b57e1c2cb95c` |
| `origin/exp/census-stage2-attenuation-transfer` | `b990e11154dfb21540fc3cece7333fedbf1a2abc` |

## Stage-1 normalized clustering (4)

Maintained target: `exp/stage1-normalized-clustering-consolidated`.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/aliccp-stage1-normalized-env-clustering` | `2058de843282de2dd7113c95cedb7d5f2e7de7a6` |
| `origin/exp/aliccp-stage1-normalized-clustering-seed-replication` | `7ab445cb4126800c5d5e0decebb923c186780b23` |
| `origin/exp/aliccp-stage1-normalized-clustering-seed3` | `f8ff4cf0735188551f2c4fa7c45b8458b6f56436` |
| `origin/exp/aliccp-stage1-normalized-clustering-seeds45` | `3dd9bd333ea05422e5d83f8b5f8e0cfa1856bf22` |

## Parameter-efficient ensemble (2)

Maintained target: `exp/aliccp-parameter-efficient-ensemble-consolidated`.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/aliccp-parameter-efficient-ensemble` | `54b81683bfb121a67156be3d0cb7e9b071be64e6` |
| `origin/exp/aliccp-parameter-efficient-ensemble-seeds2-5` | `8da8eb1b175ccec38887fecdbd1b9ca669286a32` |

## Latent cross-task prediction diagnostic (2)

Maintained target: `exp/aliccp-latent-cross-task-diagnostic-consolidated`. Diagnostic AUC must not be reported as MPT-Rec new-task improvement.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/aliccp-latent-cross-task-diagnostic` | `72b13e2e2a8e9de7010a014524cecf62786e3d8d` |
| `origin/exp/aliccp-latent-cross-task-diagnostic-seeds2-3` | `8900a8bb6e13de3db21cfa8d6cf3eb7a8aff96bd` |

## Historical affinity gate (2)

Maintained target: `exp/stage2-affinity-gate-consolidated`.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/stage2-affinity-gate-repro` | `88d787c69055090358efeb2584a51999f931ba3b` |
| `origin/exp/stage2-affinity-gate-corrected` | `78073393b72b1496efc4a16b122c4e331af7e908` |

## Historical CGR gate (2)

Maintained target: `exp/stage2-cgr-consolidated`.

| Source remote ref | Tip SHA |
|---|---|
| `origin/exp/stage2-cgr-repro` | `f3612467db39065fdab4adf85b0d33b0f890809b` |
| `origin/exp/stage2-cgr-instance` | `3702dc12c29f807ad8aa2e5500f56b0739f461f2` |

## Distinct single-branch directions (6)

These are already one branch per direction; no new branch is created merely to rename them.

| Direction | Retained remote ref | Tip SHA |
|---|---|---|
| Generalization-driven environment weighting | `origin/exp/aliccp-generalization-env-weighting` | `06639d482574819ee4bd60e576ee40e60b1655b2` |
| Incremental utility verifier | `origin/exp/aliccp-incremental-utility-verifier` | `275ee35e9ed94ecd76eb5e7a6ca118b689168275` |
| Stage-1 gradient conflict audit | `origin/exp/stage1-gradient-conflict-audit` | `b19c8c3887f382096c7ed23e0dc2f044ff53822c` |
| Stage-2 attention environment prior | `origin/exp/stage2-attn-env-prior` | `d1a680f4f96ca3f90ac58e71686ab4e4620e090a` |
| Stage-2 conditional scale/bias | `origin/exp/stage2-cond-scale-bias` | `bd8aebad80a43c35d8251360f5f7244c6d2b66fc` |
| Stage-2 LoRA adapter | `origin/exp/stage2-lora-adapter` | `034d580185ca244b38effcd903fdc7c5b318d3c9` |

## Safety observations

- Several extensions are **not** descendants of their same-direction predecessors: RP seed-replication/five-seed, Census RP seed-recheck/three-seed, ensemble seed1/seeds2-5, and latent diagnostic seed1/seeds2-3. A later tip alone is therefore not a sufficient archive of earlier documentation.
- `origin/exp/aliccp-stage1-normalized-clustering-seed3` **is** an ancestor of `origin/exp/aliccp-stage1-normalized-clustering-seeds45`.
- Existing local worktrees with uncommitted files at snapshot time: `MPT-Rec`, `MPT-Rec-exp-aliccp-atten-seed2`, `MPT-Rec-exp-aliccp-stage2-residual`, and `MPT-Rec-exp-aliccp-training-state-controller`. They must not be cleaned or overwritten.
- The local training-state-controller worktree has no matching `origin/exp/*` ref in this snapshot and is not silently counted among the 39 remote refs.
