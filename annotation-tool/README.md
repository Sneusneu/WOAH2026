# WOAH annotation tool

A keyboard-driven TUI for binary persuasion-intent annotation. It works on one intent at a time and can resume saved sessions. No dataset or compiled binary is included.

## Build

From the repository root:

```sh
cd annotation-tool
go build -o /tmp/woah .
```

## Usage

Pass a dataset stored outside this repository, the taxonomy file, an anonymized annotator ID, and an output path outside the repository:

```sh
/tmp/woah annotate \
  --dataset /path/to/private_sample.json \
  --intents ../taxonomy/intents.json \
  --annotator annotator1 \
  --output /path/outside/repo/annotator1.json
```

The dataset must use the tool's expected `id`, `source`, and `text` fields. Do not add the dataset or annotation-session metadata to this repository. The public annotation files contain only email IDs and binary labels; annotator identities and timing metadata were removed.

## Keybindings

| Key | Action |
|---|---|
| `y` / `n` | Record yes / no and advance |
| `s` | Skip for a later session |
| `←` / `→` | Navigate in the current chunk |
| `↑` / `↓` / `j` / `k` / `PgUp` / `PgDn` | Scroll the email body |
| `q` | Save and quit |
