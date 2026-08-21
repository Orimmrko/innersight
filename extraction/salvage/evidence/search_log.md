# What was searched, and what was found

Run 2026-08-08 over four folders on the Mac, all four mounted read-only into
the session. Recorded so nobody spends an hour repeating it.

## Folders searched

| Folder | What it is |
|---|---|
| `~/Developer/innersight_long` | the abandoned GNN / embeddings / Qdrant / Flask version |
| `~/Developer/innersight2` | the 18-slice rebuild, reached Slice 5 |
| `~/Developer/innersight` | the current 7-stage flat rewrite — the destination |
| `~/Developer/innersight_dataset` | the raw r4.2 release, the 30-user sample, and the answer key |

`~/Developer/innersight.egg-info` was not searched separately: it is a
setuptools packaging stub pointing at one of the above, and holds no data.

## Query 1 — binary artifact formats

Searched every folder for `*.parquet`, `*.pkl`, `*.feather`, `*.h5`, `*.npy`,
`*.npz`, excluding `.venv` and `.git`.

**Result: zero matches. Not one file in any of the four folders.**

## Query 2 — the directories such a file would live in

Searched for any directory named `feature_store*`, `processed`, `cache`,
`artifacts`, `outputs`, `interim`.

**Result:** only `innersight/outputs` (empty, created by Stage 0) and
`innersight_long/backend/.pytest_cache/v/cache` (pytest's own).
No `feature_store/` anywhere — which is the default output directory
`innersight2`'s `build_features.py` writes to, so that script has never been
run to completion on this machine.

## Query 3 — every file over 200 KB

To catch an artifact under a name nobody would guess.

**Result:** in `innersight_long/backend/checkpoints/` — `node2vec_embeddings.pt`,
`metapath2vec_embeddings.pt`, `model_graphsage.pt`, `gnn_embeddings.pt`,
`best_model.pt`, `standardizer.pt`, `alerts.json`, plus per-split graph tensors
under `graphs/`, `n2v/`, `m2v/`, `graphsage/`, `flat/`. All of it belongs to the
abandoned design; none of it is a `(user, day)` count table. Everything else
over 200 KB was `.venv` contents or the HTML explainers.

## Query 4 — code that writes such a file

Grepped for `to_parquet`, `read_parquet`, `.pkl`, `cache_dir`, `feather`.

**Result:** `innersight_long/backend/innersight/data/feature_store.py` (a
generic save/load helper, no cached output beside it) and, in `innersight2`,
`scripts/build_features.py` → `feature_store/daily_features.parquet` plus the
Slice 3/5 scripts that read it. The code exists; its output does not.

## Corroboration

The project memory from the innersight2 sessions already recorded, under
"still open": *"Michael must run the full 16 GB pipeline himself; every number
so far is from the 30-user sample."* The filesystem agrees with the notes.

## Conclusion

**FOUND ARTIFACT: none.** Checks A–F had nothing to run against, so they were
instead run against the table the salvaged code produces — see
`checks_a_to_f.txt` and `kpc0073_check_e.txt`.
