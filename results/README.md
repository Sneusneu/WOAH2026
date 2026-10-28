# Results

Every prediction record contains an email ID and boolean labels only. Raw bodies, subjects, sender addresses, original IDs, source records, and model error text were omitted. Run manifests retain aggregate run metadata and timestamps.

- `mixed100/` contains the three 2026-06-09 model outputs used for the paper’s human-comparison headline metrics.
- `dideco60/2026-05-29-flat/` and `dideco60/2026-06-01/` preserve both 60-email runs. The June 1 source manifest lists flat runs even though flat and hierarchical output files are present.
- `dideco60/gold_standard.json` is the 60-email historical gold with voter-roster and per-item voter-count metadata removed.
- `nazario120/2026-06-01/` contains the six 120-email outputs and inter-model metrics; there is no matching human gold for that corpus in the source folder.
- `computed_metrics.json` is regenerated from the mixed-100 annotations, human labels, and the evaluator. `dideco60/metrics.json` and `nazario120/metrics.json` contain the corresponding corpus-specific summaries.
- `reported_tables/` contains the available source tables. They are not a complete set of seven separate table files; the raw predictions and computed metrics are the reproducible values available in this folder.
