# Direction-level experiment branch consolidation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Account for every remote experiment branch, publish one maintained branch and one experiment ledger per improvement direction, and retire redundant remote branch names only after exact archival and coverage checks.

**Architecture:** Read-only inventory of `origin/exp/*` drives a direction map. Each direction branch preserves one shared implementation, explicit variant differences, a ledger that cites source commit SHAs, and archive tags that retain old tips. Separate infrastructure baselines and `master` remain untouched.

**Tech Stack:** Git refs/worktrees, Markdown ledgers, existing Python benchmark/tests, PowerShell for read-only inventory, `apply_patch` for documentation/code edits.

---

### Task 1: Freeze and classify remote branch inventory

**Files:** Create `docs/experiments/BRANCH_INDEX.md` on the RP consolidated branch.

- [ ] Run `git for-each-ref --format='%(refname:short) %(objectname)' refs/remotes/origin/exp` and save the exact branch/SHA inventory in the index.
- [ ] Classify every ref exactly once: RP, Null Expert, attenuation, normalized clustering, ensemble, latent diagnostic, historical affinity/CGR, or distinct singleton.
- [ ] Check `git worktree list --porcelain` and `git status --short` in attached worktrees; flag branches with uncommitted content as retained.
- [ ] Compare each multi-branch pair with `git merge-base --is-ancestor` and `git diff --name-status`; record non-ancestral divergences.
- [ ] Check index completeness by comparing the sorted set of branch names in the document against `git for-each-ref` output; commit the index.

### Task 2: Consolidate RP ledger and compatible code

**Files:** Create `docs/experiments/residual-prompt.md`; modify only compatible RP benchmark files and their tests on `exp/stage2-residual-prompt-consolidated`.

- [ ] Read each RP source summary/spec and `git ls-tree` for its tracked artifacts; extract numerical claims only from source records or raw artifacts.
- [ ] Add a ledger row for initial, seed expansion, longer-budget, and ablation experiments on AliCCP and CensusIncome, with immutable source SHA and verification state.
- [ ] Compare code files claimed identical by `git show <ref>:<path>` object hashes. Import distinct behavior only if tests can preserve old behavior behind an explicit option; otherwise link the archived implementation rather than copying duplicate code.
- [ ] Run existing RP unit tests and source-to-ledger numerical checks; commit after all rows are accounted for.

### Task 3: Consolidate Null Expert and attenuation directions

**Files:** Create one maintained branch and `docs/experiments/<direction>.md` for each direction; update `BRANCH_INDEX.md`.

- [ ] For Null Expert, reconcile original CensusIncome, CensusIncome multi-seed/long-budget, and AliCCP mechanism-gate records without blending incompatible baselines.
- [ ] For attenuation, separate fixed attenuation, learnable attenuation, and specific-control outcomes inside one direction ledger, retaining dataset and budget distinctions.
- [ ] Run direction-specific existing tests and check recorded AUC deltas against source records before each branch push.

### Task 4: Consolidate normalized clustering, ensemble, and latent diagnostic

**Files:** One maintained branch and one ledger per direction; update `BRANCH_INDEX.md`.

- [ ] Include every original and seed-extension source commit, preserving preregistered thresholds and later reclassification separately.
- [ ] Keep mechanism repair, ensemble utility, and latent diagnostic claims separate from actual Stage-2 model improvement claims.
- [ ] Run the relevant current unit tests and verify each ledger's source count/seed set before push.

### Task 5: Resolve historical duplicate names and preserve distinct singletons

**Files:** Direction ledgers for affinity and CGR; `BRANCH_INDEX.md` entries for all singleton directions.

- [ ] Compare historical/reproduction/corrected affinity and CGR code and results; maintain one direction branch each, with historical failures and corrected runs explicitly labeled.
- [ ] Leave separate singleton directions separate. Give each one an index entry and source-commit link; do not create a new branch merely to rename it.
- [ ] Verify all original `origin/exp/*` refs occur once and only once in the index.

### Task 6: Archive, push, and retire redundant refs

**Files:** Final `BRANCH_INDEX.md` and direction ledgers; Git refs.

- [ ] For each old ref eligible for retirement, create `archive/experiment/<old-branch-name>` tag at its exact original SHA; verify tag SHA equals source SHA.
- [ ] Push every maintained direction branch and required tags; verify remote refs with `git ls-remote`.
- [ ] Re-run all ledger/source and branch/tag coverage checks. If any check fails, keep the affected branch and mark it unresolved.
- [ ] Delete only verified redundant remote branch refs, never tags, worktrees, `master`, `main`, baseline infrastructure, or local uncommitted content.
- [ ] Report final maintained branches, retired branch names, archive tags, unresolved branches, tests, and any result marked unverified.
