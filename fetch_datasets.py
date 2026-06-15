#!/usr/bin/env python3
"""
fetch_datasets.py
=================
Build the authentic sample bank from the ORIGINAL datasets — nothing is
generated. Run this ONCE on a machine that has internet (and/or your local
SQuAD files); it writes `sample_bank.json`, which the app loads automatically.

Routing (the answer to "where does each dataset go?"):

    Wiki40B  ->  track "quality"        ->  Quality tab
    SQuAD    ->  track "coverage_time"  ->  Semantic Coverage + Comprehension Time

Targets: up to 1000 passages per language per track, each passage <= 2000 words.

----------------------------------------------------------------------------
Quick start — download EVERYTHING online (no files needed)
----------------------------------------------------------------------------
  pip install datasets
  python fetch_datasets.py --online --per-lang 1000 --max-words 2000

That single command pulls, for all three languages:
  • Wiki40B  (Hugging Face, google/wiki40b)           -> Quality track
  • SQuAD    (en: official; ar: Arabic-SQuAD + ARCD;   -> Coverage + Time track
              tr: SQuAD-TR)                            directly from source
caps each passage at <= 2000 words, samples up to 1000 per language per track,
preserves the passages already in the bank, and writes sample_bank.json.

Variations:
  python fetch_datasets.py --online --no-wiki          # SQuAD only (skip Wiki40B)
  python fetch_datasets.py --online --squad-ar my.json # override one language with a local file

You can still point --squad-en/tr/ar at your own JSON/JSONL files; both the
official nested SQuAD JSON (data -> paragraphs -> qas) and the flat schema
(id, context, question, answer_text, language, title) are accepted. Missing
files and failed downloads are reported and skipped — the script never crashes.

Routing: Wiki40B -> Quality tab; SQuAD -> Semantic Coverage + Comprehension Time.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import ssl
import urllib.request
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample_bank.json")

WIKI_MARKERS = ["_START_ARTICLE_", "_START_SECTION_", "_START_PARAGRAPH_", "_NEWLINE_"]


def clean_wiki(text: str) -> str:
    for m in WIKI_MARKERS:
        text = text.replace(m, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def wcount(text: str) -> int:
    return len(re.sub(r"[^\w\s]", " ", text).split())


# ---------------------------------------------------------------------------
# Wiki40B  ->  quality
# ---------------------------------------------------------------------------
def load_wiki40b(langs, per_lang, max_words, min_words=40):
    from datasets import load_dataset
    out = []
    for lang in langs:
        print(f"📥 Wiki40B [{lang}] from Hugging Face …")
        ds = load_dataset("google/wiki40b", lang, split="train", streaming=True)
        kept = 0
        for i, row in enumerate(ds):
            if kept >= per_lang:
                break
            text = clean_wiki(row.get("text", ""))
            w = wcount(text)
            if min_words <= w <= max_words:
                kept += 1
                title = text.split(".")[0][:60]
                out.append({"id": f"wiki_{lang}_{kept:04d}", "language": lang,
                            "source": "Wiki40B", "track": "quality",
                            "domain": "Wiki40B", "title": title, "text": text, "qa": []})
        print(f"   kept {kept} passages")
    return out


# ---------------------------------------------------------------------------
# SQuAD family  ->  coverage_time
# ---------------------------------------------------------------------------
def _read_local(path):
    if not os.path.exists(path):
        print(f"   ⚠ file not found: {path}  — skipping.")
        return None
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        print(f"   ⚠ file is empty: {path}  — skipping.")
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows or None


def _answer_of(rec):
    if rec.get("answer_text"):
        return rec["answer_text"]
    ans = rec.get("answers")
    if isinstance(ans, dict) and ans.get("text"):
        return ans["text"][0] if isinstance(ans["text"], list) else ans["text"]
    if isinstance(ans, list) and ans:
        a = ans[0]
        return a.get("text", "") if isinstance(a, dict) else str(a)
    return rec.get("answer", "")


def _iter_qa(parsed):
    """Yield {context, question, answer, title} from either the flat schema
    (id, context, question, answer_text/answers, title) or the official nested
    SQuAD JSON (data -> paragraphs -> qas -> answers)."""
    # nested SQuAD / Arabic-SQuAD / ARCD / SQuAD-TR official format
    if isinstance(parsed, dict) and "data" in parsed:
        for article in parsed["data"]:
            title = article.get("title", "")
            for para in article.get("paragraphs", []):
                ctx = para.get("context", "")
                for qa in para.get("qas", []):
                    if qa.get("is_impossible"):
                        continue
                    q = qa.get("question", "")
                    answers = qa.get("answers") or qa.get("plausible_answers") or []
                    ans = answers[0]["text"] if answers and isinstance(answers[0], dict) else ""
                    if ctx and q and ans:
                        yield {"context": ctx, "question": q, "answer": ans, "title": title}
        return
    # flat list of records
    if isinstance(parsed, list):
        records = parsed
    elif isinstance(parsed, dict):
        # a single flat record, or a mapping id -> record
        if parsed.get("context") and parsed.get("question"):
            records = [parsed]
        else:
            records = [v for v in parsed.values() if isinstance(v, dict)]
    else:
        records = []
    for r in records:
        if not isinstance(r, dict):
            continue
        ctx = r.get("context", "")
        q = r.get("question", "")
        a = _answer_of(r)
        if ctx and q and a:
            yield {"context": ctx, "question": q, "answer": a, "title": r.get("title", "")}


def _group_by_context(qa_iter, lang, per_lang, max_words, start_id=0, existing=None):
    """Group normalized {context,question,answer,title} into passages."""
    by_ctx = existing if existing is not None else {}
    for rec in qa_iter:
        ctx = rec["context"]
        slot = by_ctx.setdefault(ctx, {"title": "", "qa": []})
        slot["title"] = rec["title"] or slot["title"]
        slot["qa"].append({"question": rec["question"], "answer": rec["answer"]})
    out = []
    ctxs = list(by_ctx.items())
    random.shuffle(ctxs)
    for ctx, slot in ctxs:
        if len(out) >= per_lang:
            break
        if wcount(ctx) > max_words:
            continue
        seen, qa = set(), []
        for p in slot["qa"]:
            k = p["question"][:60]
            if k not in seen:
                seen.add(k)
                qa.append(p)
        out.append({"id": f"sq_{lang}_{start_id+len(out)+1:04d}", "language": lang,
                    "source": "SQuAD", "track": "coverage_time", "domain": "SQuAD",
                    "title": slot["title"] or f"SQuAD {lang} #{start_id+len(out)+1}",
                    "text": ctx, "qa": qa})
    return out


def load_squad_local(path, lang, per_lang, max_words):
    print(f"📥 SQuAD [{lang}] from {path} …")
    parsed = _read_local(path)
    if parsed is None:
        return []
    out = _group_by_context(_iter_qa(parsed), lang, per_lang, max_words)
    if not out:
        print(f"   ⚠ no usable QA found in {path} (unrecognised format) — skipping.")
        return []
    print(f"   kept {len(out)} passages")
    return out


# ---------------------------------------------------------------------------
# Online SQuAD — direct download of the ORIGINAL files (no datasets scripts)
# ---------------------------------------------------------------------------
ONLINE_SQUAD_URLS = {
    "en": ["https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"],
    "ar": ["https://raw.githubusercontent.com/husseinmozannar/SOQAL/master/data/Arabic-SQuAD.json",
           "https://raw.githubusercontent.com/husseinmozannar/SOQAL/master/data/arcd.json"],
    "tr": ["https://raw.githubusercontent.com/boun-tabi/squad-tr/beta/data/squad-tr-train-v1.0.0.json.gz",
           "https://raw.githubusercontent.com/boun-tabi/squad-tr/beta/data/squad-tr-dev-v1.0.0.json.gz"],
}


def _read_bytes(url, timeout):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (mindmap-lab)"})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def _http_get(url, timeout=180):
    raw = _read_bytes(url, timeout)
    # If GitHub returned a Git-LFS pointer instead of the file, refetch via media host.
    if raw[:200].lstrip().startswith(b"version https://git-lfs") and "raw.githubusercontent.com" in url:
        media = url.replace("raw.githubusercontent.com", "media.githubusercontent.com/media")
        raw = _read_bytes(media, timeout)
    # Gunzip if needed (by extension or by gzip magic bytes).
    if url.endswith(".gz") or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def load_squad_online(lang, per_lang, max_words):
    by_ctx: dict = {}
    for url in ONLINE_SQUAD_URLS.get(lang, []):
        if len(by_ctx) >= per_lang * 3:   # plenty already buffered
            break
        print(f"🌐 SQuAD [{lang}] downloading {url.split('/')[-1]} …")
        try:
            parsed = _http_get(url)
        except Exception as exc:
            print(f"   ⚠ download failed ({exc}); trying next source if any.")
            continue
        before = len(by_ctx)
        _group_by_context(_iter_qa(parsed), lang, 10 ** 9, max_words, existing=by_ctx)
        print(f"   +{len(by_ctx) - before} passages buffered")
    if not by_ctx:
        print(f"   ✗ no Arabic/Turkish/English SQuAD could be fetched for {lang}.")
        return []
    out = _group_by_context(iter(()), lang, per_lang, max_words, existing=by_ctx)
    print(f"   kept {len(out)} passages for {lang}")
    return out


def load_squad_hf_en(per_lang, max_words):
    from datasets import load_dataset
    print("📥 SQuAD [en] from Hugging Face (rajpurkar/squad) …")
    ds = load_dataset("rajpurkar/squad", split="train")
    by_ctx = defaultdict(lambda: {"title": "", "qa": []})
    for r in ds:
        slot = by_ctx[r["context"]]
        slot["title"] = r.get("title", "")
        if r["answers"]["text"]:
            slot["qa"].append({"question": r["question"], "answer": r["answers"]["text"][0]})
    out = []
    items = list(by_ctx.items())
    random.shuffle(items)
    for ctx, slot in items:
        if len(out) >= per_lang or wcount(ctx) > max_words:
            continue
        out.append({"id": f"sq_en_{len(out)+1:04d}", "language": "en",
                    "source": "SQuAD", "track": "coverage_time", "domain": "SQuAD",
                    "title": slot["title"], "text": ctx, "qa": slot["qa"]})
    print(f"   kept {len(out)} passages")
    return out


def merge_keep_existing(new_bank):
    """Preserve any authentic passages already in sample_bank.json."""
    existing = []
    if os.path.exists(OUT):
        try:
            existing = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            existing = []
    seen = {(b["language"], b["text"][:80]) for b in new_bank}
    for b in existing:
        key = (b["language"], b["text"][:80])
        if key not in seen and b.get("source") in ("Wiki40B", "SQuAD"):
            seen.add(key)
            new_bank.append(b)
    return new_bank


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true",
                    help="download EVERYTHING: Wiki40B (HF) + SQuAD/Arabic-SQuAD/SQuAD-TR (direct). No files needed.")
    ap.add_argument("--no-wiki", action="store_true", help="with --online, skip Wiki40B (SQuAD only)")
    ap.add_argument("--wiki40b", action="store_true", help="fetch Wiki40B (en/tr/ar) from HF")
    ap.add_argument("--wiki-langs", default="en,tr,ar")
    ap.add_argument("--squad-en"); ap.add_argument("--squad-tr"); ap.add_argument("--squad-ar")
    ap.add_argument("--squad-hf", action="store_true", help="pull English SQuAD from HF")
    ap.add_argument("--per-lang", type=int, default=1000)
    ap.add_argument("--max-words", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    squad_paths = [("en", args.squad_en), ("tr", args.squad_tr), ("ar", args.squad_ar)]
    given = [(l, p) for l, p in squad_paths if p]

    # Fail fast on bad local paths BEFORE importing datasets / starting any stream.
    missing = [p for _, p in given if not os.path.exists(p)]
    if missing:
        print("✗ These --squad-* files were not found:")
        for p in missing:
            print(f"    {p}")
        print(f"\n  Current directory: {os.getcwd()}")
        print("  Either pass real paths, or just use:  python fetch_datasets.py --online")
        print("  (downloads everything, no local files needed). Nothing was written.")
        return 1

    if not (args.online or args.wiki40b or given or args.squad_hf):
        ap.error("Nothing to do. Use --online to download everything, or pass "
                 "--wiki40b / --squad-en/tr/ar. See --help.")

    bank = []
    given_langs = {l for l, _ in given}

    # Wiki40B (HF parquet) for the quality track
    if args.online and not args.no_wiki or args.wiki40b:
        try:
            bank += load_wiki40b(args.wiki_langs.split(","), args.per_lang, args.max_words)
        except ImportError:
            print("✗ Wiki40B needs the 'datasets' library:  pip install datasets")
        except Exception as exc:
            print(f"⚠ Wiki40B fetch failed ({exc}); continuing with SQuAD only.")

    # SQuAD online (skips any language supplied as a local file)
    if args.online:
        for lang in ("en", "ar", "tr"):
            if lang in given_langs:
                continue
            try:
                bank += load_squad_online(lang, args.per_lang, args.max_words)
            except Exception as exc:
                print(f"⚠ Online SQuAD [{lang}] failed: {exc}")

    # SQuAD from local files (overrides)
    for lang, path in given:
        try:
            bank += load_squad_local(path, lang, args.per_lang, args.max_words)
        except Exception as exc:
            print(f"⚠ Could not load {lang} SQuAD from {path}: {exc}")

    if args.squad_hf and not args.online and "en" not in given_langs:
        try:
            bank += load_squad_hf_en(args.per_lang, args.max_words)
        except Exception as exc:
            print(f"⚠ English SQuAD (HF) failed: {exc}")

    if not bank:
        print("✗ No passages were loaded. Existing sample_bank.json left unchanged.")
        return 1

    bank = merge_keep_existing(bank)
    json.dump(bank, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)

    by = defaultdict(lambda: defaultdict(int))
    for b in bank:
        by[b["language"]][b["source"]] += 1
    print("\n✅ wrote", OUT, "with", len(bank), "passages")
    for lang in sorted(by):
        print(f"   {lang}: " + ", ".join(f"{s}={n}" for s, n in by[lang].items()))
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
