# Stage-2 Null Expert: consolidated experiment ledger

This document groups the CensusIncome and AliCCP Null Expert experiments without pooling their AUCs. Each delta is against the paired baseline for the same dataset, seed, Stage-1 state, split, and budget. Historical preregistered verdicts are preserved; the later `Δtest ≥ +0.001` classification is separate. The full original branch-tip SHA for each source appears below.

| Source experiment | Original branch tip | Historical decision |
|---|---|---|
| CensusIncome initial seed | `exp/stage2-null-expert` @ `87e2b8dc9ab8d631b1f17e4c1f436a01d730ff42` | Original absolute Education AUC target 0.8521 missed; mechanism active; initial direction stopped. |
| CensusIncome five short-budget seeds | `exp/census-stage2-null-expert-multiseed` @ `1196acd04c4e5d2846e0644d6345532f3632d148` | 2/5 later positive, 3/5 no-clear; CI contains zero; attribution/headroom not established. |
| CensusIncome five 10-epoch pairs | `exp/census-stage2-null-expert-longer-budget` @ `6263d6d00b971404480480bb16fec467c262a6ba` | `NOT_PERSIST` under frozen V1–V5: mean delta below +0.001. |
| AliCCP initial seed | `exp/aliccp-stage2-null-expert` @ `aba8ee910fc4122e067c9cd45a4a7c4a1cbbd1e9` | `MECHANISM_FAIL`: Null top-1 rate exactly zero; positive AUC delta cannot be attributed to Null selection. |

## CensusIncome Education: five canonical short-budget pairs

Source: five-seed spec §12.2 on `1196acd`. Each row is a separate paired comparison; test and best-val AUC are not interchangeable.

| Seed | Baseline test AUC | Null test AUC | Δtest | Baseline / Null best-val AUC | Δval | Later class |
|---|---:|---:|---:|---|---:|---|
| 1685480945 | 0.8500685307175756 | 0.8513648272440768 | +0.001296296526501206 | 0.8527881905614896 / 0.853020431042571 | +0.00023224048108139161 | `POSITIVE` |
| 1685463909 | 0.845256873871603 | 0.8447283846542692 | −0.0005284892173337274 | 0.8476002975447581 / 0.8473110085312799 | −0.0002892890134781334 | `NO_CLEAR` |
| 1685477428 | 0.8427796966516823 | 0.8456572919817906 | +0.00287759533010834 | 0.8436747451693688 / 0.8476304556509646 | +0.00395571048159582 | `POSITIVE` |
| 1685459668 | 0.8432785266659345 | 0.8440976853967682 | +0.0008191587308337134 | 0.8460871904349292 / 0.8453586690532408 | −0.0007285213816884406 | `NO_CLEAR` |
| 1685496394 | 0.8474462695821553 | 0.8476911350393439 | +0.0002448654571886033 | 0.84798146522571 / 0.8473791525613715 | −0.0006023126643385224 | `NO_CLEAR` |

Five-seed mean `Δtest = +0.0009418853654596271`, sample SD `0.00127822927632415`, reported Student-t 95% CI `[−0.0006449914677945057,+0.0025287621987137602]`. Null was active in all five seeds (reported top-1 range 0.301–0.452); the three `NO_CLEAR` seeds have `headroom_remains=false` in the source. Initial seed's historical conclusion remains a failure against its original **absolute** 0.8521 target even though its later relative classification is positive.

## CensusIncome: paired 10-epoch persistence test

Source: longer-budget spec §11 and terminal verdict on `6263d6d`. Ten runs (five seeds × two arms), 10 epochs with patience 3, one run per arm/seed, no tuning. The per-seed order matches the short-budget table above.

| Seed | Δtest at 10 epochs | Later class |
|---|---:|---|
| 1685480945 | +0.001296296526501206 | `POSITIVE` |
| 1685463909 | +0.0018317538029973823 | `POSITIVE` |
| 1685477428 | +0.0015037888571964864 | `POSITIVE` |
| 1685459668 | −0.0009452475261504389 | `NO_CLEAR` |
| 1685496394 | −0.0003256030423448575 | `NO_CLEAR` |

Mean `Δtest = +0.0006721977236399557`, sample SD `0.0012285508673826697`, reported 95% CI `[−0.000853005106583309,+0.0021974005538632204]`. V1 failed because mean < +0.001; V2–V5 and all identity checks passed. Thus the historical conclusion is **`NOT_PERSIST`**, not invalid. Null remained active (top-1 range 0.3244–0.4810). The two no-clear seeds had no supported headroom; the apparent change from 2/5 to 3/5 positives did not rescue the mean or CI.

## AliCCP BSI: failed mechanism attribution

Source: AliCCP Null spec on `aba8ee9`. Baseline test/val AUC was `0.5988392178311113 / 0.5781533414372665`; Null arm test/val AUC was `0.6126857532817985 / 0.5919398413138859`. The paired test delta is `+0.0138465354506872` (later numerical class `POSITIVE`), but the **Null top-1 rate was 0.0000**. Its learned key stayed below the one-third selection threshold, so this run did not establish the proposed “choose no old task” mechanism. The preregistered stop rule classified it as `MECHANISM_FAIL`; it was not expanded and must not be reported as a Null Expert gain.

## Provenance and implementation status

The current maintained branch inherits the CensusIncome five-seed implementation. AliCCP and 10-epoch variants have distinct source code and are not yet claimed executable from this branch. Their original full commits remain in Git; experiment-specific raw JSON and prediction arrays are Git-ignored local artifacts, so GitHub alone does not permit a fresh raw-AUC recomputation. Historical independent verifier outcomes and raw checks are documented in their source specifications; this consolidation does not replace those checks. No original result, including invalid or negative results, has been overwritten.
