# WOAH 2026 release

This paper measures agreement between LLM annotators and human labels for online manipulation in phishing and abuse email corpora. It compares flat and hierarchical prompting, examines human agreement, and evaluates models against held-out human annotations.

The release contains code, taxonomy, anonymized labels, ID-only model predictions, and supporting evaluation artifacts. It contains no email bodies, subjects, sender addresses, or source corpus files, to protect people represented in the corpora and respect each source dataset’s terms.

## Layout

- `prompts/` — flat and hierarchical DSPy strategies.
- `taxonomy/` — 23 sub-label definitions and revised annotator guidelines.
- `annotation-tool/` — Go annotation CLI source and module files; compiled binaries and email data are omitted.
- `corpus/` — preparation scripts, ID/source/body-hash manifest, and corpus access notes.
- `annotations/` — three pseudonymized human label files and majority-vote gold, keyed by email ID.
- `eval/` — OpenRouter runner, Cohen’s kappa, Krippendorff’s alpha, held-out benchmark, F1, and plotting scripts.
- `results/` — ID-and-prediction-only model outputs, run manifests, and reported table files.

## Source corpora

Obtain Nazario from [its corpus site](https://monkey.org/~jose/phishing/), TExtPhish from its [gated Hugging Face dataset page](https://huggingface.co/datasets/TExtPhish/TExtPhish), and Enron from [CMU](https://www.cs.cmu.edu/~enron/). Keep downloaded corpora and generated email text outside this repository. TExtPhish requires access approval and has separate CC BY-NC-ND 4.0 terms; the source corpora are not covered by `LICENSE-DATA`. See `corpus/README.md` for normalization, seed, manifest, and preparation commands.

## Reproduction

Install the dependencies with `python -m pip install -r requirements.txt`. Reproduce agreement and held-out human metrics from the bundled ID-only labels and mixed 100-email predictions by running from this directory:

```sh
python eval/full_iaa_report.py --human annotations/annotator1.json annotations/annotator2.json annotations/annotator3.json --llm results/mixed100/claude_sonnet4.json results/mixed100/gemma3_27b.json results/mixed100/qwen2_5_7b.json
python eval/held_out_comparison.py --human annotations/annotator1.json --human annotations/annotator2.json --human annotations/annotator3.json
python eval/evaluate_mixed100.py
python eval/plot_annotations.py --human annotations/annotator1.json --llm1 results/mixed100/claude_sonnet4.json --llm2 results/mixed100/gemma3_27b.json --llm3 results/mixed100/qwen2_5_7b.json --out results/mixed100/figures
python eval/kappa_eval.py --results-dir results/dideco60/2026-06-01 --gold results/dideco60/gold_standard.json --output /tmp/dideco60_metrics.json
```

To call OpenRouter, set `OPENROUTER_API_KEY` in the environment, rebuild the sample to a path outside the repository (see `corpus/README.md`), and run:

```sh
python eval/woah_experiment_runner.py --input /tmp/woah_annotation_sample.json --models gpt4o claude llama --strategies flat hierarchical
```

The runner uses temperature `0.0`. Provider models may change behind a route; use the checked-in predictions for the reported outputs. `results/computed_metrics.json` records the mixed-100 checks: human alpha 0.604 (explicit 0.516, implicit 0.655); top-level macro kappa Claude 0.648, Gemma 0.689, Qwen 0.456; sub-label macro kappa Claude 0.469, Gemma 0.372, Qwen 0.232. The held-out script reports 0.299 under the strict rule and 0.382 under the lenient rule.

## Model routes and run dates

| Experiment | Saved run date | Route evidence |
|---|---|---|
| DIDECO, 60 emails | 2026-05-29 (`results_60_flat`); 2026-06-01 (`results`) | The `results` manifest identifies GPT-4o, Claude, and Llama but records no full route. The result filenames say `claude_sonnet4`; `annotate_sublabels.py` specifies `openrouter/anthropic/claude-sonnet-4-5`; `woah_experiment_runner.py` specifies `openrouter/anthropic/claude-haiku-4-5`. The actual Claude route for these 60-email files is unresolved. The June 1 manifest lists only flat runs although flat and hierarchical files are present. |
| Nazario, 120 emails | 2026-06-01 | Runner configuration records `openrouter/openai/gpt-4o`, `openrouter/anthropic/claude-haiku-4-5`, and `openrouter/meta-llama/llama-3.3-70b-instruct`; the Claude filenames also say Haiku. |
| Mixed, 100 emails | 2026-06-09 | Export metadata records `openrouter/anthropic/claude-sonnet-4-5`, `openrouter/google/gemma-3-27b-it`, and `openrouter/qwen/qwen-2.5-7b-instruct`. |

The 60-email manifests do not resolve the Sonnet/Haiku discrepancy, so this release reports both pieces of evidence without harmonizing them. The source code and manifests record run dates as shown above.

## License

Code is MIT-licensed in `LICENSE`. Annotations, taxonomy definitions, and guidelines are CC BY 4.0 under `LICENSE-DATA`; source email corpora are excluded.

```bibtex
@inproceedings{popovic2026reliablellmannnotators,
  title     = {How Reliable Are LLM Annotators for Online Manipulation? Evidence from Phishing and Abuse Corpora},
  author    = {Popovic and Parmentier and Lauer and Riquet and Meyer},
  booktitle = {Proceedings of the Workshop on Online Abuse and Harms},
  year      = {2026}
}
```
