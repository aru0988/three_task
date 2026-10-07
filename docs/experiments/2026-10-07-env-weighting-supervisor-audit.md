# Environment-weighting supervisor pre-run audit (2026-10-07)

Scope: AliCCP direction 2 only. This is a pre-run review, not a training result or a GO authorization.

## Independently checked evidence

- Fair-base worktree branch: `exp/aliccp-generalization-env-weighting`, initial commit `8133d32`.
- Canonical seed `1688723512` Stage-1 `meta.json` (`s1-5c060b9c-m1688723512-e3-3a30e2c0`) reports `env_0=566`, `env_1=1,999,434`, total 2,000,000; `git.dirty=false`. I reopened the raw metadata and recomputed the shares: 0.0283% / 99.9717%. Weighting on this raw partition is **NO-GO** because the small environment cannot support a reliable environment-specific generalization estimate.
- The train-prefix fingerprint reports 566 purchase positives in 2,000,000 rows. The previously independently audited normalized-partition cross-tab reports 564/2 purchase positives for seed 1 and 565/1 for seed 2. These cross-tabs are prior evidence, not a new raw-label recomputation in this review. Balanced environment counts alone therefore do not establish CVR-signal identifiability.
- The current CC Switch configuration reports current provider `DeepSeek` (`cn_official`) and model mapping name `deepseek-flash`. This is configuration evidence, not a claim about every historical call.

## Audit-script review before any full run

The uncommitted `aliccp_benchmark/audit_env_weighting.py` is not yet approved as an audit verifier:

1. `PRECEDENT_BLOB` is displayed but `parity_check()` executes `git show` through a movable branch ref. The current branch-path blob independently resolves to `1a68c0b48dad7b30b849a6cb9582fc17041cfbd1`; execution must assert that hash or read the pinned blob itself.
2. `ess_floor_at_clip` puts the high weight on the majority environment. For shares 0.48/0.52 and clip endpoints 10/0.1, the two ESS fractions are 0.490403 and 0.529595. The smaller value occurs when the minority receives the high weight. Test both assignments and take the minimum before calling it a floor.
3. The script writes its `--out` JSON. Describe it as non-mutating with respect to Stage-1 artifacts, not literally read-only.
4. The documented direct-file CLI conflicts with its relative package import. Invoke as `python -m aliccp_benchmark.audit_env_weighting` in the project runtime and test `--help` there.

No full audit output was accepted or added to a summary. A run of the unreviewed script was interrupted before an output JSON appeared. Do not infer GO from its unfinished execution.

## Release conditions for the next step

The preregistration must name the exact weighting signal and task, train-internal holdout or cross-fitting rule, per-environment/per-fold positive and negative support thresholds, frozen Stage-1 and data identities, and the three arms `pure fair baseline`, `normalized + uniform`, `normalized + learned`. The latter two arms must differ only in weighting. Official validation/test labels may be used for final evaluation, never for learning or choosing weights. If the proposed signal cannot be estimated in every required environment, record NO-GO and advance to direction 3 without training this direction.
