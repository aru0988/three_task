# Stage-2 attenuation: consolidated experiment ledger

This ledger groups five remote experiment branches under the attenuation hypothesis. Fixed specific-path attenuation, a learned coefficient, a Null-gain attribution control, longer-budget persistence, and CensusIncome transfer are **different comparisons**. Their metrics are not pooled. All deltas are treatment minus the paired baseline of the same dataset/seed/split/budget. Original preregistered verdicts are retained alongside the later relative classification (`POSITIVE` when `Δtest ≥ +0.001`; `NO_CLEAR` when `−0.02 < Δtest < +0.001`).

| Source experiment | Original branch tip | Historical conclusion |
|---|---|---|
| AliCCP fixed specific-path attenuation / Null attribution | `exp/aliccp-stage2-specific-attenuation-control` @ `331442077f378bb41cf1fef946b8b57e1c2cb95c` | `ATTENUATION_SUPPORTED`: reproduced 91.764% of the observed Null-arm gain at seed 1688723512. |
| AliCCP fixed attenuation second seed | `exp/aliccp-stage2-attenuation-seed-replication` @ `f589e6179d68669d71aa21e6313455e7976f85d4` | `STABILITY_SUPPORTED` for two 5-epoch seeds under original threshold. |
| AliCCP fixed attenuation 10 epochs | `exp/aliccp-stage2-attenuation-longer-budget` @ `b515f8c323d4155c6faa771b34e61aa02a257b9b` | `NOT_PERSIST`: test effect nearly zero and val direction negative. |
| AliCCP learnable coefficient | `exp/aliccp-stage2-learnable-attenuation` @ `ac26eea10f02bffa613e754b2b3abddb82e1bb61` | `IDENTITY_PRESERVED`; gradient available, but coefficient movement too small for intended mechanism. |
| CensusIncome fixed attenuation transfer | `exp/census-stage2-attenuation-transfer` @ `b990e11154dfb21540fc3cece7333fedbf1a2abc` | `TRANSFER_NOT_SUPPORTED`; test and val deltas both negative. |

## AliCCP BSI fixed attenuation

The coefficient was frozen at `c = 0.6972233730330467`. Source `3314420` reports zero added trainable parameters and a three-way baseline/Null/attenuation comparison. Source `f589e61` adds the second canonical seed; source `b515f8c` tests persistence without reselecting the coefficient.

| Seed / budget | Baseline test / best-val AUC | Attenuation test / best-val AUC | Δtest | Δval | Later class; original verdict |
|---|---|---|---:|---:|---|
| 1688723512 / 5 epochs | 0.5988392178311113 / 0.5781533414372665 | 0.6115453624066993 / 0.5914326183692061 | +0.012706144575588052 | +0.013279276931939643 | `POSITIVE`; attribution `ATTENUATION_SUPPORTED` (91.764% of Null gain). |
| 1688723740 / 5 epochs | 0.5974422649550507 / see source | 0.6051559662367624 / see source | +0.007713701281711671 | +0.004483122552346286 | `POSITIVE`; `STABILITY_SUPPORTED`; A3 run-to-run check remained SKIP. |
| 1688723740 / 10 epochs | See source 10-epoch pair | See source 10-epoch pair | +0.0005273306552253665 | −0.002653370173158476 | `NO_CLEAR`; `NOT_PERSIST`, identity checks 8/8 passed. |

The two 5-epoch paired deltas average `+0.010209922928649862` test and `+0.008881199742142964` val, but the source explicitly warns that two model seeds do not estimate same-seed run noise. At 10 epochs the test gain shrank by `−0.007186370626486305` relative to the seed-2 five-epoch result. This rules out a persistence claim under the frozen protocol; it does not invalidate the short-budget observations.

## Learnable coefficient and CensusIncome transfer

AliCCP learned-coefficient run (`ac26eea`) had BSI test/val AUC `0.5996733678081829 / 0.5791006721251453`, versus baseline `0.5988392178311113 / 0.5781533414372665`. Reported deltas were `+0.0008341499770716521` test (`NO_CLEAR`) and `+0.0009473306878787779` val. The first-step gradient test passed, but best-epoch coefficient remained exactly 1.0; the original verdict is `IDENTITY_PRESERVED`, not successful attenuation.

CensusIncome transfer (`b990e11`) used the AliCCP-derived coefficient without retuning. Education test AUC was `0.8489935694070636` versus baseline `0.8500685307175756`, giving `Δtest = −0.0010749613105119904` (`NO_CLEAR`); reported `Δval = −0.000969247326740863`. Both original transfer criteria failed. The source discloses a first execution with a rounded PowerShell coefficient as **invalid**; only the corrected single run is used for the result. The invalid execution remains disclosed but is excluded from aggregation.

## Code and raw-data status

This maintained branch currently starts from the AliCCP seed-replication code. It does not claim that all other variants or CensusIncome run from one unified executable yet; their original exact commits remain the source implementation. Individual raw `metrics.json` and prediction artifacts are Git-ignored locally, so GitHub alone contains committed summaries/specifications rather than all raw predictions. No outcome has been converted into a stronger claim by consolidation.
