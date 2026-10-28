#!/usr/bin/env python3
"""
prepare_textphish.py — Sample emails from TExtPhish for annotation.
Strips Reddit user preamble before the actual phishing email content.

Usage:
  python3 prepare_textphish.py \
    --csv ~/.cache/huggingface/.../email-level/train.csv \
    --out textphish_sample.json --n 33 --seed 42
"""

import csv
import sys
import re
import json
import random
import hashlib
import argparse
from pathlib import Path

MIN_WORDS = 30
MAX_WORDS = 600


def strip_reddit_preamble(text: str) -> str:
    """
    TExtPhish emails are Reddit posts where users paste phishing emails
    they received, often with 1-3 sentences of context first.
    This function detects and removes that preamble.
    """
    lines = text.split("\n")

    # Patterns that strongly indicate the email body is starting
    EMAIL_START_PATTERNS = [
        r"^(dear|hello|hi|good\s+(morning|afternoon|evening|day))\b",
        r"^(greetings|attention|urgent|important notice)\b",
        r"^(from|to|subject|date)\s*:",
        r"^my name is\b",
        r"^i am (writing|contacting|reaching out)\b",
        r"^you (have been|are|were)\b",
        r"^your (account|password|email|information|details)\b",
        r"^we (have|are|would like|need|detected|noticed)\b",
        r"^it has come to our\b",
        r"^this is (a|an|to)\b.*official",
        r"^(congratulations|you('ve| have) (won|been selected))\b",
        r"^(warning|alert|notice|notification)\s*[:\-!]",
        r"^please (be advised|note|verify|confirm|click|update)\b",
        r"^as (per|a result of|part of)\b",
        r"^i am (the|a) (ceo|director|manager|attorney|lawyer|officer)\b",
        r"^(first|last)\s+\w+",  # "First Last" name pattern common in TExtPhish
    ]

    # Patterns that indicate Reddit commentary (not phishing email)
    REDDIT_PATTERNS = [
        r"\b(got this|received this|found this|sharing this)\b",
        r"\b(is this (legit|real|a scam|phishing))\b",
        r"\b(my (mom|dad|friend|wife|husband|boss|colleague|coworker))\b",
        r"\b(thought (it was|this was|you('d| would)))\b",
        r"\b(just (got|received|wanted to share))\b",
        r"\b(can (someone|anyone) (help|tell me|explain))\b",
        r"\b(here('s| is) (the|an?) (email|message|text|one))\b",
        r"\b(copy.paste|copy paste)\b",
        r"\bfull (email|message|text) (below|follows)\b",
        r"\b(posted|sharing|submitting) (this|it) (here|for)\b",
        r"^\s*(edit|update|tldr|tl;dr)\s*:",
        r"\b(reddit|subreddit|r\/\w+)\b",
        r"\bOP\b",
    ]

    compiled_starts  = [re.compile(p, re.IGNORECASE) for p in EMAIL_START_PATTERNS]
    compiled_reddit  = [re.compile(p, re.IGNORECASE) for p in REDDIT_PATTERNS]

    def is_email_start(line):
        s = line.strip()
        return any(p.match(s) for p in compiled_starts)

    def is_reddit(line):
        return any(p.search(line) for p in compiled_reddit)

    # Find the first line that looks like the phishing email starts
    best_start = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if is_email_start(s):
            best_start = i
            break
        # Quoted blocks often mark the start of the pasted email
        if s.startswith('"') or s.startswith('>'):
            best_start = i
            break

    if best_start > 0:
        remaining = "\n".join(lines[best_start:]).strip()
        if len(remaining.split()) >= 20:
            return remaining

    # Fallback: skip leading lines that look like Reddit commentary
    non_empty = [l for l in lines if l.strip()]
    if non_empty and is_reddit(non_empty[0]):
        skip = 0
        for line in non_empty:
            if is_reddit(line) and len(line.split()) < 25:
                skip += 1
            else:
                break
        if 0 < skip < len(non_empty):
            remaining = "\n".join(non_empty[skip:]).strip()
            if len(remaining.split()) >= 20:
                return remaining

    return text


def clean_body(text: str) -> str:
    text = strip_reddit_preamble(str(text))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="textphish_sample.json")
    parser.add_argument("--n", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.out).resolve()
    if output_path == repo_root or repo_root in output_path.parents:
        raise SystemExit("Refusing to write email text inside the release repository.")

    random.seed(args.seed)
    csv.field_size_limit(sys.maxsize)

    records = []
    seen = set()

    with open(args.csv, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        print(f"  Columns: {header}")

        for row in reader:
            if len(row) < 2:
                continue
            content = row[1].strip() if len(row) > 1 else ""
            orig_label = row[2].strip() if len(row) > 2 else ""

            if not content or content.lower() in ("nan", ""):
                continue

            body = clean_body(content)
            word_count = len(body.split())

            if word_count < MIN_WORDS or word_count > MAX_WORDS:
                continue

            uid = hashlib.md5(body[:100].encode()).hexdigest()[:10]
            if uid in seen:
                continue
            seen.add(uid)

            records.append({
                "id":     f"txp_{uid}",
                "subject": "",
                "body":   body,
                "source": "TExtPhish",
                "label":  1,
                "meta":   {"words": word_count, "orig_label": orig_label}
            })

            if len(records) >= args.n * 5:
                break

    print(f"  Candidates: {len(records)}")

    if len(records) < args.n:
        print(f"  Warning: only {len(records)} usable — reducing n to {len(records)}")
        args.n = len(records)

    sampled = random.sample(records, args.n)
    for i, rec in enumerate(sampled, 1):
        rec["pipeline_id"] = f"txp_{i:03d}"
        rec["id"] = rec["pipeline_id"]

    word_counts = [r["meta"]["words"] for r in sampled]
    print(f"  Sampled: {len(sampled)} emails")
    print(f"  Words: min={min(word_counts)} max={max(word_counts)} "
          f"mean={sum(word_counts)//len(word_counts)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    main()
