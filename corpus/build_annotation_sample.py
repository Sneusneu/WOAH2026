#!/usr/bin/env python3
"""
build_annotation_sample.py — Build the 100-email annotation sample.
Mix of Nazario (33) + TExtPhish (33) + Enron (34).

Usage:
  python3 build_annotation_sample.py \
    --nazario nazario_sample.json \
    --textphish-csv ~/.cache/huggingface/hub/datasets--TExtPhish--TExtPhish/snapshots/f3ee7212619e0afd682a1101828221959bf42d1b/email-level/train.csv \
    --enron enron_sample.json \
    --out annotation_sample.json
"""

import json, re, random, csv, sys, hashlib, quopri, argparse
from pathlib import Path
csv.field_size_limit(sys.maxsize)

SOURCE_NAMES = {'Nazario': 'nazario', 'TExtPhish': 'textphish', 'Enron': 'enron'}


def verify_manifest(rows, manifest_path):
    with open(manifest_path, newline='', encoding='utf-8') as f:
        expected = {r['message_id']: (r['source'], r['sha256_normalized_body'])
                    for r in csv.DictReader(f)}
    actual = {r['id']: (SOURCE_NAMES[r['source']],
                        hashlib.sha256(r['text'].encode('utf-8')).hexdigest())
              for r in rows}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(k for k in set(actual) & set(expected)
                         if actual[k] != expected[k])
        raise SystemExit(f'Manifest mismatch: missing={len(missing)}, '
                         f'extra={len(extra)}, changed={len(changed)}')


def decode_mime(text):
    try:
        text = re.sub(r'=\r?\n', '', text)
        return quopri.decodestring(text.encode('latin-1', errors='replace')).decode('latin-1', errors='replace')
    except Exception:
        return re.sub(r'=([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), text)


def strip_enron_headers(text):
    rfc = re.compile(
        r'^(Message-ID|Date|From|To|Subject|Cc|Bcc|Mime-Version|'
        r'Content-Type|Content-Transfer-Encoding|X-From|X-To|X-cc|'
        r'X-bcc|X-Folder|X-Origin|X-FileName)\s*:',
        re.IGNORECASE | re.MULTILINE)
    lines = text.split('\n')
    in_headers = False
    for i, line in enumerate(lines):
        if rfc.match(line.strip()):
            in_headers = True
        elif in_headers and line.strip() == '':
            body = '\n'.join(lines[i+1:]).strip()
            if len(body.split()) >= 8:
                return strip_enron_headers(body)
    forward = re.compile(
        r'[\w\s./]+@[\w_]+\s*\n\s*\d{2}/\d{2}/\d{4}[^\n]*\n'
        r'(\s*(To|cc|From|Subject|Sent by)[^\n]*\n)+\s*\n', re.IGNORECASE)
    m = forward.search(text)
    if m:
        body = text[m.end():].strip()
        if len(body.split()) >= 8:
            return strip_enron_headers(body)
    return text


def is_still_headers(text):
    first = '\n'.join(text.split('\n')[:3])
    return bool(re.match(r'^(To|From|Subject|Cc|Message-ID|Date|[\w\s./]+@[\w_]+)\s*[:\n]',
                         first, re.IGNORECASE))


def is_garbled(text):
    words = text.split()
    if not words: return True
    single = sum(1 for w in words[:30] if len(w) == 1)
    return (single / min(len(words), 30)) > 0.35



def is_html(text):
    """Detect emails that are raw HTML/CSS rather than plain text."""
    html_markers = ['<html', '<body', '<div', '<table', '<td', '<tr',
                    'box-sizing', 'margin: 0', 'padding: 0', 'font-family:',
                    'background-color:', '@media', '<!DOCTYPE']
    text_lower = text.lower()
    hits = sum(1 for m in html_markers if m.lower() in text_lower)
    return hits >= 3


def strip_html_tags(text):
    """Strip HTML tags and decode entities for HTML-formatted emails."""
    import html
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    return text

def strip_reddit_preamble(text):
    lines = text.split('\n')
    non_empty = [l for l in lines if l.strip()]
    if not non_empty: return text
    EMAIL_START = [
        r"^(dear|hello|hi|good\s+(morning|afternoon|evening|day))\b",
        r"^(greetings|attention|urgent|important notice)\b",
        r"^my name is\b",
        r"^i (am|have) (writing|contacting|discovered|noticed|found)\b",
        r"^you (have been|are|were|need to|must)\b",
        r"^your (account|password|email|information|device|computer)\b",
        r"^we (have|are|detected|noticed|found|want to)\b",
        r"^(congratulations|warning|alert|notice|important)\b",
        r"^please (be advised|note|verify|confirm|click|update|pay)\b",
        r"^i possess\b",
    ]
    REDDIT = [
        r"\b(got this|received this|found this|sharing this)\b",
        r"\b(is this (legit|real|a scam|phishing|fake))\b",
        r"\b(my (mom|dad|friend|wife|husband|boss|colleague|coworker|parent))\b",
        r"\b(just (got|received|wanted to share))\b",
        r"\b(here'?s? (the|an?) (email|message|text|one|scam))\b",
        r"\b(reddit|subreddit)\b",
        r"\bnone of my business\b",
        r"\bwanted to (share|post|show)\b",
    ]
    cs = [re.compile(p, re.IGNORECASE) for p in EMAIL_START]
    cr = [re.compile(p, re.IGNORECASE) for p in REDDIT]
    for i, line in enumerate(non_empty):
        if any(p.match(line.strip()) for p in cs):
            if i > 0:
                rem = '\n'.join(non_empty[i:]).strip()
                if len(rem.split()) >= 15: return rem
            break
    skip = 0
    for line in non_empty:
        if any(p.search(line) for p in cr) and len(line.split()) < 40:
            skip += 1
        else:
            break
    if 0 < skip < len(non_empty):
        rem = '\n'.join(non_empty[skip:]).strip()
        if len(rem.split()) >= 15: return rem
    return text


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', str(text))
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nazario',      required=True)
    parser.add_argument('--textphish-csv', required=True)
    parser.add_argument('--enron',        required=True)
    parser.add_argument('--out', required=True,
                        help='Output path outside the release repository (contains email text)')
    parser.add_argument('--manifest', default=str(Path(__file__).with_name('manifest.csv')))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(args.out).resolve()
    if output_path == repo_root or repo_root in output_path.parents:
        raise SystemExit('Refusing to write email text inside the release repository; '
                         'choose an external output path.')

    TARGET = {'Nazario': 33, 'TExtPhish': 33, 'Enron': 34}
    combined = []

    # ── Nazario ──
    with open(args.nazario) as f:
        naz_data = json.load(f)
    random.shuffle(naz_data)
    for e in naz_data:
        if sum(1 for x in combined if x['source']=='Nazario') >= TARGET['Nazario']:
            break
        body = clean_text(e.get('body', ''))
        if is_html(body):
            body = strip_html_tags(body)
            body = clean_text(body)
        if len(body.split()) >= 20 and not is_garbled(body):
            combined.append({'source': 'Nazario', 'body': body})
    print(f"Nazario: {sum(1 for e in combined if e['source']=='Nazario')}")

    # ── TExtPhish — read from CSV, more candidates ──
    seen = set()
    with open(args.textphish_csv, encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)
        for row in reader:
            if sum(1 for e in combined if e['source']=='TExtPhish') >= TARGET['TExtPhish']:
                break
            if len(row) < 2: continue
            content = row[1].strip()
            if not content or content.lower() == 'nan': continue
            body = strip_reddit_preamble(clean_text(content))
            # Strip leading > quote markers
            body = re.sub(r'^>\s*', '', body, flags=re.MULTILINE).strip()
            if is_garbled(body): continue
            if len(body.split()) < 25 or len(body.split()) > 500: continue
            uid = hashlib.md5(body[:100].encode()).hexdigest()[:10]
            if uid in seen: continue
            seen.add(uid)
            combined.append({'source': 'TExtPhish', 'body': body})
    print(f"TExtPhish: {sum(1 for e in combined if e['source']=='TExtPhish')}")

    # ── Enron ──
    with open(args.enron) as f:
        enron_all = json.load(f)
    random.shuffle(enron_all)
    for e in enron_all:
        if sum(1 for x in combined if x['source']=='Enron') >= TARGET['Enron']:
            break
        raw = clean_text(e.get('body', ''))
        body = decode_mime(raw)
        body = strip_enron_headers(body)
        body = clean_text(body)
        if is_still_headers(body): continue
        if body.count('=09') + body.count('=20') > 5: continue
        if len(body.split()) >= 20:
            combined.append({'source': 'Enron', 'body': body})
    print(f"Enron: {sum(1 for e in combined if e['source']=='Enron')}")
    print(f"Total: {len(combined)}")

    # ── Shuffle + assign IDs ──
    random.shuffle(combined)
    output = []
    sc = {'Nazario': 0, 'TExtPhish': 0, 'Enron': 0}
    pm = {'Nazario': 'naz', 'TExtPhish': 'txp', 'Enron': 'enr'}
    for e in combined:
        src = e['source']
        sc[src] += 1
        output.append({'id': f"{pm[src]}_{sc[src]:04d}", 'source': src, 'text': e['body']})

    verify_manifest(output, args.manifest)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(output)} emails → {output_path}")


if __name__ == '__main__':
    main()
# This is a patch marker - see fix below
