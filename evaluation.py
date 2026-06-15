"""
evaluation.py
=============
The thesis evaluation tracks, implemented faithfully to the paper:

  • Quality (human + Claude judge) — five binary criteria
      SC Structural Coherence · SA Semantic Accuracy · CC Concept Centrality
      BC Branch Completeness   · GC Graph Clarity
    Each map gets 0/1 per criterion; quality% of a map = (#Good / 5) × 100;
    the running quality is the mean of map quality% (per language-model cell).
    A multilingual LLM "Claude judge" can rate the same five criteria.

  • Semantic coverage — Auto-QA and Human-QA tracks
      coverage = (correct answers) / (total questions), per Jain et al. (2024):
        SemCov(s,t) = (1/|QA(t)|) · Σ 1[ Equivalent( answer_from_mindmap(qi), gold_i ) ]
      Auto-QA  : questions generated from the passage by the LLM.
      Human-QA : human-curated questions (SQuAD / Arabic-SQuAD / SQuAD-TR).
    Answers are produced using ONLY the mind map; equivalence is decided by a
    deterministic matcher (substring / token-overlap / numeric) backed by an
    LLM equivalence judge.

  • Comprehension time — handled in the UI (timer over ⟨q,s⟩ / ⟨q,t⟩ / ⟨q,s+t⟩);
    this module provides the answer-correctness helper.

Reference results reported in the thesis are exposed for display only.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

import prompts
from llm import Provider, chat

# ---------------------------------------------------------------------------
# Five binary criteria (verbatim definitions from the paper)
# ---------------------------------------------------------------------------
CRITERIA = [
    ("SC", "Structural Coherence",
     "Logical organization and hierarchy of nodes and branches."),
    ("SA", "Semantic Accuracy",
     "The information matches the source text without hallucination."),
    ("CC", "Concept Centrality",
     "The main concept is correctly identified as the root."),
    ("BC", "Branch Completeness",
     "All important information is included in the branches."),
    ("GC", "Graph Clarity",
     "Visual clarity and readability — no excessive density or sparsity."),
]
CRITERIA_CODES = [c[0] for c in CRITERIA]

# Thesis reference results (for display alongside the live experiment).
REFERENCE = {
    "quality": {  # % Good (majority vote), per language-model
        "gemini": {"ar": 91.9, "en": 86.0, "tr": 82.1},
        "qwen": {"ar": 86.7, "en": 87.0, "tr": 66.0},
    },
    "auto_qa_avg": {"gemini": 71.0, "qwen": None},
    "human_qa_avg": {"gemini": 64.8, "qwen": 47.4},
    "comprehension_reduction": {"en": 35.2, "ar": 38.6, "tr": 36.1},  # % faster
}


def quality_percent(ratings: dict) -> float:
    """A single map's quality% = (#Good / 5) × 100."""
    good = sum(1 for c in CRITERIA_CODES if ratings.get(c))
    return round(100.0 * good / len(CRITERIA_CODES), 1)


def aggregate_quality(maps: list[dict]) -> float:
    """Running quality = mean of per-map quality% (unweighted)."""
    if not maps:
        return 0.0
    return round(sum(quality_percent(m) for m in maps) / len(maps), 1)


# ---------------------------------------------------------------------------
# Claude judge (multilingual LLM-as-judge over the five criteria)
# ---------------------------------------------------------------------------
JUDGE_PROMPT = """You are a careful, multilingual evaluator of mind maps. You will be given a SOURCE TEXT and a MIND MAP derived from it. Judge the mind map on five binary criteria. Use the source's own language; do not translate.

Criteria (answer 1 = Good, 0 = Bad for each):
- SC Structural Coherence: logical organization and hierarchy of nodes and branches.
- SA Semantic Accuracy: the information matches the source text without hallucination.
- CC Concept Centrality: the main concept is correctly identified as the root.
- BC Branch Completeness: all important information is included in the branches.
- GC Graph Clarity: visual clarity and readability, with no excessive density or sparsity.

Output ONLY valid JSON, no markdown fences, in exactly this form:
{"SC":{"score":1,"reason":"..."},"SA":{"score":0,"reason":"..."},"CC":{"score":1,"reason":"..."},"BC":{"score":1,"reason":"..."},"GC":{"score":1,"reason":"..."}}

SOURCE TEXT:
{source}

MIND MAP:
{mindmap}
"""


@dataclass
class JudgeResult:
    ratings: dict          # {"SC": bool, ...}
    reasons: dict          # {"SC": "..."}
    raw: str = ""
    error: str = ""

    @property
    def quality(self) -> float:
        return quality_percent(self.ratings)


def claude_judge(provider: Provider, key: str, source: str, mindmap_repr: str) -> JudgeResult:
    prompt = JUDGE_PROMPT.replace("{source}", source).replace("{mindmap}", mindmap_repr)
    try:
        raw = chat(provider, key, "You are a precise evaluator. Output only JSON.",
                   prompt, temperature=0.0, max_tokens=800)
    except Exception as exc:
        return JudgeResult({}, {}, error=str(exc))
    obj = _json_block(raw)
    if not obj:
        return JudgeResult({}, {}, raw=raw, error="Could not parse judge JSON.")
    ratings, reasons = {}, {}
    for c in CRITERIA_CODES:
        cell = obj.get(c, {})
        if isinstance(cell, dict):
            ratings[c] = bool(int(cell.get("score", 0)))
            reasons[c] = str(cell.get("reason", ""))
        else:
            ratings[c] = bool(int(cell)) if str(cell).isdigit() else False
            reasons[c] = ""
    return JudgeResult(ratings, reasons, raw=raw)


# ---------------------------------------------------------------------------
# Answer matching / equivalence
# ---------------------------------------------------------------------------
_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower().strip()
    s = _DIAC.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    s = re.sub(r"[^\w\u0600-\u06FF\s]", " ", s)
    return " ".join(s.split())


def _nums(s: str):
    return set(re.findall(r"\d+(?:[.,]\d+)?", s or ""))


def deterministic_match(gold: str, pred: str) -> bool:
    """Substring / token-overlap / numeric matching (the paper's fast checks)."""
    if not gold or not pred:
        return False
    if "<unknown>" in pred.lower() or "unanswerable" in pred.lower():
        return False
    g, p = _norm(gold), _norm(pred)
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    gt, pt = set(g.split()), set(p.split())
    if gt:
        overlap = len(gt & pt) / len(gt)
        if overlap >= 0.6:
            return True
    gn, pn = _nums(gold), _nums(pred)
    if gn and gn <= pn:
        return True
    return False


def llm_equivalent(provider: Provider, key: str, question: str, a1: str, a2: str) -> bool:
    """LLM equivalence judge (EQUIVALENCE_QA_PROMPT)."""
    user = (prompts.EQUIVALENCE_QA_PROMPT + f"\n\nQuestion: {question}\n"
            f"Answer 1: {a1}\nAnswer 2: {a2}\nConclusion:")
    try:
        out = chat(provider, key, "Output only Yes or No.", user,
                   temperature=0.0, max_tokens=10)
    except Exception:
        return False
    return out.strip().lower().startswith("y")


def answer_from_mindmap(provider: Provider, key: str, mindmap_repr: str, question: str) -> str:
    user = prompts.QA_VALIDITY_PROMPT.format(data=mindmap_repr, question=question)
    try:
        return chat(provider, key, "Answer using only the provided information.",
                    user, temperature=0.0, max_tokens=256).strip()
    except Exception as exc:
        return f"<error: {exc}>"


# ---------------------------------------------------------------------------
# Auto-QA / Human-QA coverage
# ---------------------------------------------------------------------------
@dataclass
class QAItem:
    question: str
    gold: str
    map_answer: str
    correct: bool
    method: str            # "deterministic" | "llm" | "miss"


@dataclass
class CoverageResult:
    track: str             # "Auto-QA" | "Human-QA"
    items: list
    coverage: float        # %
    error: str = ""


def generate_auto_qa(provider: Provider, key: str, text: str, limit: int = 5) -> list[dict]:
    try:
        raw = chat(provider, key, "Output only the requested JSON.",
                   prompts.AUTO_QA_PROMPT.format(text=text),
                   temperature=0.2, max_tokens=900)
    except Exception:
        return []
    obj = _json_block(raw) or {}
    pairs = obj.get("qa_pairs", []) if isinstance(obj, dict) else []
    out = []
    for p in pairs[:limit]:
        q, a = p.get("question"), p.get("answer")
        if q and a:
            out.append({"question": q, "answer": a})
    return out


def run_coverage(provider: Provider, key: str, mindmap_repr: str,
                 qa_pairs: list[dict], track: str,
                 use_llm_judge: bool = True) -> CoverageResult:
    """Answer each question with the mind map only, then judge equivalence."""
    items: list[QAItem] = []
    for qa in qa_pairs:
        q, gold = qa["question"], qa["answer"]
        pred = answer_from_mindmap(provider, key, mindmap_repr, q)
        if deterministic_match(gold, pred):
            items.append(QAItem(q, gold, pred, True, "deterministic"))
        elif use_llm_judge and llm_equivalent(provider, key, q, gold, pred):
            items.append(QAItem(q, gold, pred, True, "llm"))
        else:
            items.append(QAItem(q, gold, pred, False, "miss"))
    cov = round(100.0 * sum(i.correct for i in items) / len(items), 1) if items else 0.0
    return CoverageResult(track, items, cov)


# ---------------------------------------------------------------------------
# Demo fallbacks (no key needed)
# ---------------------------------------------------------------------------
def demo_judge(source: str, mindmap_repr: str) -> JudgeResult:
    ratings = {"SC": True, "SA": True, "CC": True, "BC": False, "GC": True}
    reasons = {
        "SC": "Hierarchy is logical: root → themes → facts.",
        "SA": "Leaf values match the source; no invented facts.",
        "CC": "The main entity is correctly the root.",
        "BC": "One minor detail from the source is missing from the branches.",
        "GC": "Balanced tree, readable density.",
    }
    return JudgeResult(ratings, reasons, raw="(demo judge)")


def demo_coverage(qa_pairs: list[dict], track: str) -> CoverageResult:
    items = []
    for i, qa in enumerate(qa_pairs):
        correct = (i % 4 != 3)  # ~75% correct, realistic
        pred = qa["answer"] if correct else "<unknown>"
        items.append(QAItem(qa["question"], qa["answer"], pred, correct,
                            "deterministic" if correct else "miss"))
    cov = round(100.0 * sum(i.correct for i in items) / len(items), 1) if items else 0.0
    return CoverageResult(track, items, cov)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _json_block(text: str):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def mindmap_repr(result) -> str:
    """Best textual representation of a mind map for QA / judging."""
    if getattr(result, "mindmap_json", None):
        return json.dumps(result.mindmap_json, ensure_ascii=False, indent=2)
    return getattr(result, "mermaid_code", "") or ""


# ---------------------------------------------------------------------------
# Claude QA reviewer — the thesis role of Claude in Auto-QA
# ("Qwen-generated questions reviewed by Claude Sonnet 4"). Claude reviews the
# AUTO-GENERATED questions and keeps only the valid ones before answering.
# ---------------------------------------------------------------------------
QA_REVIEWER_PROMPT = """You are a QA quality reviewer. Review the following question-answer pairs generated from a text passage.

TEXT PASSAGE:
{text}

GENERATED QA PAIRS:
{qa_pairs_json}

TASK:
For each QA pair, check:
1. Is the question answerable from the text?
2. Is the answer correct and found in the text?
3. Is the question clear and well-formed?

OUTPUT FORMAT (JSON only):
{{
    "reviewed_pairs": [
        {{"question": "...", "answer": "...", "is_valid": true/false, "reason": "..."}}
    ],
    "valid_count": <number>,
    "total_count": <number>
}}

JSON:"""


def review_auto_qa(provider: Provider, key: str, text: str, qa_pairs: list):
    """Claude reviews auto-generated questions; returns (kept_pairs, review_rows)."""
    import json as _json
    prompt = QA_REVIEWER_PROMPT.format(
        text=text, qa_pairs_json=_json.dumps(qa_pairs, ensure_ascii=False))
    try:
        raw = chat(provider, key, "Output only the requested JSON.", prompt,
                   temperature=0.0, max_tokens=900)
    except Exception:
        return qa_pairs, [{"question": p["question"], "answer": p["answer"],
                           "is_valid": True, "reason": "(reviewer unavailable)"} for p in qa_pairs]
    obj = _json_block(raw) or {}
    rows = obj.get("reviewed_pairs") if isinstance(obj, dict) else None
    if not rows:
        return qa_pairs, [{"question": p["question"], "answer": p["answer"],
                           "is_valid": True, "reason": "(unparsed review)"} for p in qa_pairs]
    kept = [{"question": r.get("question"), "answer": r.get("answer")}
            for r in rows if r.get("is_valid")]
    if not kept:                      # never drop everything
        kept = qa_pairs
    return kept, rows


def demo_review_auto_qa(qa_pairs: list):
    rows = []
    for i, p in enumerate(qa_pairs):
        valid = (i % 5 != 4)          # ~80% kept
        rows.append({"question": p["question"], "answer": p["answer"],
                     "is_valid": valid,
                     "reason": "Answerable and found in the text." if valid
                     else "Answer not clearly grounded in the passage."})
    kept = [p for p, r in zip(qa_pairs, rows) if r["is_valid"]] or qa_pairs
    return kept, rows
