# Corpus preparation

No source email text is distributed here. Obtain each source corpus directly from its provider and keep downloaded data and generated samples outside this repository.

- **Nazario:** download the selected yearly mbox files from [the corpus site](https://monkey.org/~jose/phishing/). The preparation script defaults to 2021–2024 and seed 42.
- **TExtPhish:** request access to the email-level training CSV from [the official Hugging Face dataset page](https://huggingface.co/datasets/TExtPhish/TExtPhish). Access is gated and its own CC BY-NC-ND 4.0 and usage terms apply; this release does not redistribute it.
- **Enron:** download the May 7, 2015 archive from [CMU’s Enron corpus page](https://www.cs.cmu.edu/~enron/). The script accepts an extracted maildir.

## Sample normalization and manifest

`manifest.csv` was generated from the exact 100-record `annotation_sample.json` used for the human annotations. It records the annotation email ID, source name, and SHA-256 of the final normalized body. The source-native message IDs are unavailable: the original builder replaces them with sequential `naz_NNNN`, `txp_NNNN`, and `enr_NNNN` IDs.

Normalization is source-specific, as implemented in `build_annotation_sample.py`: all bodies collapse runs of three or more newlines to two, collapse repeated spaces and tabs to one, replace non-ASCII and non-printable characters with spaces, and trim surrounding whitespace. Nazario HTML is stripped when detected; TExtPhish Reddit preambles and leading quote markers are removed; Enron MIME quoted-printable encoding and forwarded RFC headers are handled before common cleanup. Each manifest hash is over the UTF-8 bytes of the final normalized `text` value.

The mixed sample builder uses seed 42 by default and verifies the generated IDs, sources, and body hashes against `manifest.csv` before writing. It refuses to write email text inside the release repository. The exact original TExtPhish CSV snapshot and source-native IDs were not present, so a rebuild may fail verification if provider data or ordering differs. A mismatch is intentionally fatal.

Run the preparation scripts with output paths outside this repository, then build the mixed sample:

```sh
python corpus/prepare_nazario.py --local-dir /path/to/nazario_mbox --out /tmp/nazario_sample.json --sample 120 --seed 42
python corpus/prepare_enron.py --enron-dir /path/to/enron/maildir --out /tmp/enron_sample.json --n 100 --seed 42
python corpus/build_annotation_sample.py --nazario /tmp/nazario_sample.json --textphish-csv /path/to/TExtPhish/train.csv --enron /tmp/enron_sample.json --out /tmp/woah_annotation_sample.json --seed 42
```

The manifest is an identity check for the sample; the repository contains only IDs, source names, and body hashes.

## Source-native ID recovery status

The local workspace has Nazario mbox files and a prepared Nazario sample, but it does not have the TExtPhish source CSV or the Enron maildir/archive. Because two raw sources are missing, source-native ID recovery was stopped without changing `manifest.csv` or `build_annotation_sample.py`. No source-native IDs or unmatched-hash claims have been inferred; the existing IDs remain the sequential annotation IDs. Revisit this item when the TExtPhish CSV and Enron source files are available.
