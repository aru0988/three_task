# Direction-level experiment branch consolidation

## Goal and scope

Replace the current seed-, budget-, dataset-, and ablation-named branch sprawl with one maintained branch per improvement direction. Begin with Stage-2 Residual Prompt (RP), then repeat the verified procedure for other directions. Do not merge `master`, rewrite historical commits, alter historical preregistration verdicts, or discard a valid result.

## Chosen approach

The maintained RP branch is `exp/stage2-residual-prompt-consolidated`, based on the audited AliCCP five-seed RP branch. It has one canonical experiment ledger at `docs/experiments/residual-prompt.md`, with sections for initial screens, paired seeds, longer budgets, causal ablations, CensusIncome, and terminal judgments. Each row records the source branch and immutable commit, dataset, seed, budget, paired control, validation/test metrics and deltas, mechanism gates, historical verdict, later threshold reclassification, raw-artifact availability, and independent-verifier status. Missing evidence is marked unavailable, never inferred.

Shared RP implementation appears once. Distinct controls become explicit configuration or narrowly named modules/tests only when a code diff and test prove they are behaviorally distinct. Where immediate unification would alter an experiment's semantics, the ledger points to an immutable source tag; the historical implementation remains recoverable there and is not falsely described as reproduced by the maintained branch.

After exact coverage checks, create `archive/experiment/<old-branch-name>` tags pointing to each historical remote tip. Verify every tag resolves to the original SHA. Only then may redundant remote branches be removed; keep the new maintained branch and any still-active direction branch. Local worktrees and uncommitted files are never removed by this operation.

## Alternatives considered

1. Keep all branches and add only an index: lowest risk, but fails the branch-count objective.
2. Merge every historical branch wholesale: preserves commit ancestry, but imports conflicting code and summaries and can silently change experiment semantics.
3. **Selected:** one maintained direction branch plus exact source tags and an audited ledger. This reduces active branch count without losing provenance, while code is unified only where tests establish equivalence.

## Verification and failure handling

Build an inventory directly from remote refs, commit trees, and source documents before migration. For each ledger entry, compare recorded numbers against its source summary and, when available, raw artifacts. Do not treat a source branch name or Claude Code prose as numerical evidence. Compare source and destination code hashes for files claimed identical. Run the relevant existing tests after each compatible code import. If a source artifact is missing or inconsistent, retain its tag and mark the entry `UNVERIFIED`; do not promote its numerical claim or delete its branch. A result that was already invalid is omitted from numerical aggregation and identified by source reference only when needed for audit history.

Before remote branch deletion, require all of the following: the ledger enumerates every source branch, each source tip has a verified archive tag, every valid unique record has a destination row or a clearly linked preserved source artifact, the maintained branch is pushed, and independent source-to-ledger checks pass. No force-pushes, no `master` merge, and no deletion of raw or local worktree content.

## Rollout

First complete RP across AliCCP and CensusIncome. Then consolidate Null Expert, attenuation, Stage-1 normalized clustering, parameter-efficient ensemble, and latent cross-task diagnostic separately. Other one-off directions remain independent until an inventory shows that two branches truly belong to the same mechanism. The final branch index lists one maintained ref per direction and archive tags for historical experiment points.
