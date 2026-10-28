#!/usr/bin/env python3
"""
prepare_nazario.py — Download and prepare the Nazario phishing corpus
for the WOAH 2026 annotation pipeline.

Downloads selected mbox files from https://monkey.org/~jose/phishing/,
parses them, cleans the email text, samples N emails, and outputs
nazario_emails.json in the same format as emails.json (DIDECO pipeline format).

Usage:
  python prepare_nazario.py --out nazario_emails.json --sample 120
  python prepare_nazario.py --out nazario_emails.json --sample 120 --years 2022 2023 2024
  python prepare_nazario.py --local-dir ./nazario_mbox --out nazario_emails.json --sample 120

Downloads ~50–80MB of mbox files. Use --local-dir if you already have them.
"""

import os
import re
import json
import email
import mailbox
import random
import hashlib
import argparse
import tempfile
import urllib.request
from pathlib import Path
from email import policy
from email.parser import BytesParser


# ── Corpus files ──────────────────────────────────────────────────────────────

BASE_URL = "https://monkey.org/~jose/phishing/"

# Available files — we skip the very old ones (pre-2015) and private-phishing4
# as they're either noisy or restricted. Years 2019+ are most relevant for
# modern spear-phishing patterns matching DIDECO's LLM-generated emails.
AVAILABLE_FILES = {
    "2015": "phishing-2015",
    "2016": "phishing-2016",
    "2017": "phishing-2017",
    "2018": "phishing-2018",
    "2019": "phishing-2019",
    "2020": "phishing-2020",
    "2021": "phishing-2021",
    "2022": "phishing-2022",
    "2023": "phishing-2023",
    "2024": "phishing-2024",
}

DEFAULT_YEARS = ["2021", "2022", "2023", "2024"]

# ── Text extraction ───────────────────────────────────────────────────────────

def get_text_from_message(msg) -> str:
    """Extract clean plain text from an email.Message object."""
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if "attachment" in cd:
                continue
            if ct == "text/plain":
                try:
                    charset = part.get_content_charset() or "utf-8"
                    body += part.get_payload(decode=True).decode(charset, errors="replace")
                except Exception:
                    pass
            elif ct == "text/html" and not body:
                # fallback: strip HTML tags if no plain text found
                try:
                    charset = part.get_content_charset() or "utf-8"
                    raw = part.get_payload(decode=True).decode(charset, errors="replace")
                    body += re.sub(r"<[^>]+>", " ", raw)
                except Exception:
                    pass
    else:
        try:
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(charset, errors="replace")
                if msg.get_content_type() == "text/html":
                    body = re.sub(r"<[^>]+>", " ", body)
        except Exception:
            body = str(msg.get_payload())

    return body


def clean_text(text: str) -> str:
    """Normalise whitespace, remove quoted headers and forwarding artifacts."""
    # Remove common email forwarding headers
    text = re.sub(r"-{3,}.*?Original Message.*?-{3,}", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(>.*\n?)+", "", text, flags=re.MULTILINE)        # quoted lines
    text = re.sub(r"\n{3,}", "\n\n", text)                             # excessive newlines
    text = re.sub(r"[ \t]{2,}", " ", text)                             # multiple spaces
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)                       # non-ASCII
    return text.strip()


def extract_email_record(msg, source_file: str, idx: int) -> dict | None:
    """Convert a mailbox.Message to a pipeline-ready dict. Returns None if unusable."""
    subject = str(msg.get("subject", "")).strip()
    from_   = str(msg.get("from", "")).strip()
    date_   = str(msg.get("date", "")).strip()

    # Decode encoded subject
    try:
        decoded_parts = email.header.decode_header(subject)
        subject = " ".join(
            part.decode(enc or "utf-8") if isinstance(part, bytes) else part
            for part, enc in decoded_parts
        )
    except Exception:
        pass

    body = get_text_from_message(msg)
    body = clean_text(body)

    # Quality filters
    if len(body) < 50:      # too short to be meaningful
        return None
    if len(body) > 8000:    # truncate very long emails
        body = body[:8000] + "\n[TRUNCATED]"

    # Deduplification key
    content_hash = hashlib.md5((subject + body[:200]).encode()).hexdigest()[:8]
    uid = f"naz_{source_file}_{idx:04d}_{content_hash}"

    return {
        "id":      uid,
        "subject": subject,
        "body":    body,
        "source":  "Nazario",
        "meta": {
            "from":   from_,
            "date":   date_,
            "file":   source_file,
            "index":  idx,
        }
    }


# ── Download ──────────────────────────────────────────────────────────────────

def download_file(filename: str, out_dir: Path) -> Path | None:
    url  = BASE_URL + filename
    dest = out_dir / filename
    if dest.exists():
        print(f"  ✓ Already downloaded: {filename}")
        return dest
    print(f"  ↓ Downloading {filename} from {url} ...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        dest.write_bytes(data)
        print(f" {len(data) / 1e6:.1f}MB")
        return dest
    except Exception as e:
        print(f" FAILED: {e}")
        return None


# ── Parse mbox ────────────────────────────────────────────────────────────────

def parse_mbox(filepath: Path) -> list[dict]:
    """Parse an mbox file and return cleaned email records."""
    records = []
    source  = filepath.stem

    try:
        mb = mailbox.mbox(str(filepath))
        msgs = list(mb)
    except Exception as e:
        print(f"  ⚠  Could not parse {filepath.name}: {e}")
        return []

    seen_hashes = set()
    for i, msg in enumerate(msgs):
        rec = extract_email_record(msg, source, i)
        if rec is None:
            continue
        # dedup
        h = hashlib.md5((rec["subject"] + rec["body"][:100]).encode()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        records.append(rec)

    print(f"  Parsed {filepath.name}: {len(msgs)} raw → {len(records)} usable")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Nazario corpus for WOAH pipeline")
    parser.add_argument("--out",       default="nazario_emails.json",
                        help="Output JSON file (pipeline-ready)")
    parser.add_argument("--sample",    type=int, default=120,
                        help="Number of emails to sample (default: 120)")
    parser.add_argument("--years",     nargs="+", default=DEFAULT_YEARS,
                        choices=list(AVAILABLE_FILES.keys()),
                        help="Which yearly files to use (default: 2021-2024)")
    parser.add_argument("--local-dir", default=None,
                        help="Path to directory with already-downloaded mbox files "
                             "(skips download step)")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--no-download", action="store_true",
                        help="Parse only — don't attempt downloads")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.out).resolve()
    if output_path == repo_root or repo_root in output_path.parents:
        raise SystemExit("Refusing to write email text inside the release repository.")

    random.seed(args.seed)

    print("=" * 60)
    print("  NAZARIO CORPUS PREPARATION — WOAH 2026")
    print("=" * 60)
    print(f"  Target: {args.sample} emails | Years: {args.years}")

    # Resolve source directory
    if args.local_dir:
        source_dir = Path(args.local_dir)
        print(f"\n  Using local files from: {source_dir}")
    else:
        source_dir = Path(tempfile.gettempdir()) / "woah_nazario_mbox"
        source_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Download directory: {source_dir}/")

    # Download / locate files
    mbox_files = []
    for year in args.years:
        fname = AVAILABLE_FILES[year]
        fpath = source_dir / fname
        if args.local_dir or args.no_download:
            if fpath.exists():
                mbox_files.append(fpath)
            else:
                print(f"  ⚠  Not found locally: {fpath}")
        else:
            result = download_file(fname, source_dir)
            if result:
                mbox_files.append(result)

    if not mbox_files:
        print("\n  ✗ No mbox files available. Run without --no-download or check --local-dir.")
        return

    # Parse all files
    print(f"\n  Parsing {len(mbox_files)} mbox files...")
    all_records = []
    for fpath in mbox_files:
        all_records.extend(parse_mbox(fpath))

    print(f"\n  Total usable emails: {len(all_records)}")

    if len(all_records) < args.sample:
        print(f"  ⚠  Only {len(all_records)} emails available, "
              f"reducing sample to {len(all_records)}")
        args.sample = len(all_records)

    # Sample
    sampled = random.sample(all_records, args.sample)

    # Assign clean sequential IDs (keeping meta for traceability)
    for i, rec in enumerate(sampled, 1):
        rec["pipeline_id"] = f"naz_{i:03d}"

    # Stats
    print(f"\n  Sample stats:")
    sources = {}
    for r in sampled:
        s = r["meta"]["file"]
        sources[s] = sources.get(s, 0) + 1
    for s, c in sorted(sources.items()):
        print(f"    {s}: {c} emails")

    body_lens = [len(r["body"]) for r in sampled]
    print(f"  Body length: min={min(body_lens)} max={max(body_lens)} "
          f"mean={sum(body_lens)//len(body_lens)}")

    # Save — use pipeline_id as the id field for compatibility
    output = []
    for rec in sampled:
        output.append({
            "id":      rec["pipeline_id"],
            "subject": rec["subject"],
            "body":    rec["body"],
            "source":  "Nazario",
            "original_id": rec["id"],
            "meta":    rec["meta"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Saved {len(output)} emails → {output_path}")
    print(f"\n  Next step:")
    print(f"    python woah_experiment_runner.py --input {args.out} \\")
    print(f"      --output-dir results_nazario \\")
    print(f"      --models gpt4o claude llama --strategies flat")
    print("=" * 60)


if __name__ == "__main__":
    main()
