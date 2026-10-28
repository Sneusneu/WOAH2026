#!/usr/bin/env python3
"""
prepare_enron.py — Sample legitimate emails from Enron for the
classification pilot experiment (WOAH 2026 short paper).

Produces enron_sample.json in the same format as emails.json,
ready to feed into woah_experiment_runner.py for intent annotation.

Usage (from extracted tgz):
  python prepare_enron.py --enron-dir ./maildir --out enron_sample.json --n 100

Usage (from Kaggle CSV):
  python prepare_enron.py --csv emails.csv --out enron_sample.json --n 100
"""

import os
import re
import json
import email
import random
import argparse
import hashlib
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

# Match DIDECO body length distribution (50–600 words)
MIN_WORDS = 50
MAX_WORDS = 600
RANDOM_SEED = 42


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_body(text: str) -> str:
    # Remove forwarding artifacts
    text = re.sub(r"-{3,}.*?Forwarded.*?-{3,}", " ", text,
                  flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^>.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def extract_from_mbox(path: Path) -> dict | None:
    """Parse a single Enron mbox file into a record."""
    try:
        raw = path.read_bytes()
        msg = email.message_from_bytes(raw)
    except Exception:
        return None

    subject = str(msg.get("subject", "")).strip()
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    charset = part.get_content_charset() or "utf-8"
                    body += part.get_payload(decode=True).decode(
                        charset, errors="replace")
                except Exception:
                    pass
    else:
        try:
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode(charset, errors="replace")
        except Exception:
            return None

    body = clean_body(body)
    word_count = len(body.split())

    if word_count < MIN_WORDS or word_count > MAX_WORDS:
        return None
    if len(body) < 100:
        return None

    uid = hashlib.md5((subject + body[:100]).encode()).hexdigest()[:10]
    return {
        "id":      f"enr_{uid}",
        "subject": subject,
        "body":    body,
        "source":  "Enron",
        "label":   0,  # legitimate email
        "meta":    {"path": str(path), "words": word_count}
    }


def extract_from_csv(csv_path: str, n: int) -> list[dict]:
    """Extract from Kaggle-style emails.csv — streaming, no full load."""
    import csv, sys
    csv.field_size_limit(sys.maxsize)
    records = []
    seen = set()
    target = n * 3  # collect 3x then stop early

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i % 50000 == 0 and i > 0:
                print(f"    Scanned {i:,} rows, {len(records)} candidates...")
            body = clean_body(row.get("message", ""))
            word_count = len(body.split())
            if word_count < MIN_WORDS or word_count > MAX_WORDS:
                continue
            subject = row.get("subject", "").strip()
            if any(x in body.lower() for x in ["mime-version", "content-type",
                                                "attachment", "<?xml"]):
                continue
            uid = hashlib.md5((subject + body[:100]).encode()).hexdigest()[:10]
            if uid in seen:
                continue
            seen.add(uid)
            records.append({
                "id":      f"enr_{uid}",
                "subject": subject,
                "body":    body,
                "source":  "Enron",
                "label":   0,
                "meta":    {"words": word_count}
            })
            if len(records) >= target:
                break
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sample Enron legitimate emails for classification pilot")
    parser.add_argument("--enron-dir", default=None,
                        help="Path to extracted maildir/ from enron_mail_20150507.tgz")
    parser.add_argument("--csv", default=None,
                        help="Path to Kaggle emails.csv")
    parser.add_argument("--out", default="enron_sample.json")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of emails to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.out).resolve()
    if output_path == repo_root or repo_root in output_path.parents:
        raise SystemExit("Refusing to write email text inside the release repository.")

    random.seed(args.seed)

    if not args.enron_dir and not args.csv:
        print("Error: provide --enron-dir or --csv")
        return

    print(f"Sampling {args.n} Enron emails (seed={args.seed})...")

    records = []

    if args.csv:
        print(f"  Reading from CSV: {args.csv}")
        records = extract_from_csv(args.csv, args.n)
        print(f"  Candidates: {len(records)}")

    elif args.enron_dir:
        print(f"  Scanning maildir: {args.enron_dir}")
        all_files = list(Path(args.enron_dir).rglob("*"))
        # Skip sent/sent_items folders to avoid replies to phishing
        all_files = [f for f in all_files if f.is_file()
                     and "sent" not in str(f).lower()
                     and "." not in f.name]  # Enron files have no extension
        random.shuffle(all_files)
        print(f"  Total candidate files: {len(all_files)}")

        seen = set()
        for fpath in all_files:
            rec = extract_from_mbox(fpath)
            if rec is None:
                continue
            uid = rec["id"]
            if uid in seen:
                continue
            seen.add(uid)
            records.append(rec)
            if len(records) >= args.n * 5:
                break
        print(f"  Usable records: {len(records)}")

    if len(records) < args.n:
        print(f"  Warning: only {len(records)} usable emails found, "
              f"reducing n to {len(records)}")
        args.n = len(records)

    sampled = random.sample(records, args.n)

    # Assign clean sequential IDs
    for i, rec in enumerate(sampled, 1):
        rec["pipeline_id"] = f"enr_{i:03d}"
        rec["id"] = rec["pipeline_id"]

    # Stats
    word_counts = [r["meta"]["words"] for r in sampled]
    print(f"\n  Sampled: {len(sampled)} emails")
    print(f"  Word count: min={min(word_counts)} "
          f"max={max(word_counts)} "
          f"mean={sum(word_counts)//len(word_counts)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved -> {output_path}")
    print(f"\n  Next:")
    print(f"    python woah_experiment_runner.py --input {args.out} \\")
    print(f"      --output-dir results_enron \\")
    print(f"      --models claude --strategies flat")


if __name__ == "__main__":
    main()
