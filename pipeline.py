"""
pipeline.py
===========
Orchestrates the mind-map pipeline and records every step so the UI can show
exactly what prompt was sent and what the model returned.

Pipeline order (faithful to the notebooks, with the placeholder wiring fixed):

  WITHOUT three critics  (fast path)
    1. Generate mind map    (lang system prompt + input text -> chosen model)
    2. Repair JSON          (utility model -> valid JSON)
    3. Convert to Mermaid   (utility model -> mindmap diagram)

  WITH three critics  (full quality gate)
    1. Generate mind map
    2. Repair JSON
    3. Extract paths            (shared by the global + factual critics)
    4. Critic 1 - Local Structure   : leaf values specific?      -> yes/no
    5. Critic 2 - Global Structure  : TOC/paths informative?      -> useful yes/no
    6. Critic 3 - Factual           : 6a bullet points (numbered source sentences)
                                       6b attribute each path -> [n] / [NA]
                                       6c validate -> ACCEPT / REJECT (zero NA)
    7. AND gate              : accepted = local AND global AND factual
    8. Convert to Mermaid

Each step is captured as a ``Step`` dataclass. The orchestrator never raises on
a model error; it records the error on the step and continues, so a live demo
in front of an audience degrades gracefully instead of crashing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import prompts
from llm import Provider, chat

# A small, neutral system message for the single-block pipeline prompts.
_UTILITY_SYSTEM = "Follow the instructions exactly and output only what is requested."


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------
@dataclass
class Step:
    number: str               # "1", "4", "6a" ...
    key: str                  # machine id, e.g. "local_critic"
    title: str
    role: str                 # "Generation" | "Repair" | "Critic" | "Render"
    model_label: str          # which model ran this step
    system_prompt: str = ""   # exact system message sent
    user_prompt: str = ""     # exact user message sent
    output: str = ""          # raw model output
    verdict: Optional[bool] = None   # critics only: pass/fail
    verdict_label: str = ""          # e.g. "useful: yes"
    error: str = ""
    note: str = ""            # extra UI note (e.g. demo data)


@dataclass
class Result:
    steps: list[Step] = field(default_factory=list)
    mindmap_json: Optional[dict] = None
    mermaid_code: str = ""
    accepted: Optional[bool] = None   # None when critics are off
    critics_summary: dict = field(default_factory=dict)
    demo: bool = False

    def step(self, key: str) -> Optional[Step]:
        for s in self.steps:
            if s.key == key:
                return s
        return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def extract_mindmap_json(text: str) -> Optional[dict]:
    """Pull the JSON object out of a 'MindMap ... END_THOUGHT' block.

    Falls back to the last balanced {...} object in the text.
    """
    if not text:
        return None
    # 1) MindMap ... END_THOUGHT block (the format the prompts request)
    m = re.search(r"MindMap\s*(\{.*\})\s*END_THOUGHT", text, re.DOTALL)
    if m:
        obj = _safe_json(m.group(1))
        if obj is not None:
            return obj
    # 2) fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        obj = _safe_json(m.group(1))
        if obj is not None:
            return obj
    # 3) first balanced object
    obj = _safe_json(_first_balanced_object(text))
    return obj


def _safe_json(s: Optional[str]) -> Optional[dict]:
    if not s:
        return None
    try:
        val = json.loads(s)
        return val if isinstance(val, (dict, list)) else None
    except Exception:
        return None


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
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
                return text[start : i + 1]
    return None


# ---- Mermaid cleaning (ported from the AR mermaid-conversion notebook) ----
def _clean_text(text: str) -> str:
    text = re.sub(r"\broot\b", "", text, flags=re.IGNORECASE)
    text = text.replace("(", "").replace(")", "")
    text = text.replace(":", " -")
    for ch in "[]{};<>|\"'`#&":
        text = text.replace(ch, "")
    return " ".join(text.split())


def clean_mermaid_code(mermaid_code: str) -> str:
    """Strip fences/special chars and keep indentation, so mermaid.js renders."""
    if not mermaid_code:
        return ""
    code = mermaid_code.strip()
    code = re.sub(r"^```(?:mermaid)?", "", code).strip()
    code = re.sub(r"```$", "", code).strip()

    lines = code.split("\n")
    cleaned: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        if line.strip() == "mindmap":
            cleaned.append("mindmap")
            continue
        indent = len(line) - len(line.lstrip())
        if "root((" in line:
            mt = re.search(r"root\(\((.+?)\)\)", line)
            content = _clean_text(mt.group(1)) if mt else _clean_text(line.strip())
            cleaned.append(" " * indent + f"root(({content}))")
        else:
            cleaned.append(" " * indent + _clean_text(line.strip()))
    out = "\n".join(cleaned)
    if not out.lstrip().startswith("mindmap"):
        out = "mindmap\n" + out
    return out


def json_to_mermaid_local(obj, _depth: int = 0, _root: bool = True) -> str:
    """Deterministic JSON->Mermaid fallback (no model needed).

    Used in demo mode and as a safety net if the Mermaid model step fails.
    """
    lines: list[str] = ["mindmap"] if _root else []

    def label_of(node: dict) -> str:
        for k in ("label", "canonical", "id", "name"):
            if isinstance(node.get(k), str) and node[k].strip():
                return node[k].strip()
        return "node"

    def walk(node, depth):
        pad = "  " * depth
        if isinstance(node, dict):
            label_keys = ("label", "canonical", "id", "name", "title")
            lbl = next((node[k] for k in label_keys
                        if isinstance(node.get(k), str) and node[k].strip()), None)
            if lbl is not None and ("children" in node or len(node) <= 2):
                lbl = _clean_text(lbl)
                lines.append(f"{pad}root(({lbl}))" if depth == 1 else f"{pad}{lbl}")
                for c in (node.get("children") or []):
                    walk(c, depth + 1)
            else:
                # key -> value mapping (the notebook schema)
                for k, v in node.items():
                    if k in label_keys:
                        continue
                    key_lbl = _clean_text(str(k))
                    lines.append(f"{pad}root(({key_lbl}))" if depth == 1 else f"{pad}{key_lbl}")
                    walk(v, depth + 1)
        elif isinstance(node, list):
            for c in node:
                walk(c, depth)
        elif isinstance(node, str):
            lines.append(f"{pad}{_clean_text(node)}")

    root = obj.get("root", obj) if isinstance(obj, dict) else obj
    walk(root, 1)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(
    *,
    input_text: str,
    language: str,
    gen_provider: Provider,
    gen_key: str,
    util_provider: Provider,
    util_key: str,
    use_critics: bool,
    progress: Optional[Callable[[str], None]] = None,
) -> Result:
    """Execute the pipeline live. Records every step; never raises on LLM error."""
    res = Result()

    def say(msg: str):
        if progress:
            progress(msg)

    def run_util(key, title, role, number, system, user, parser=None):
        step = Step(number=number, key=key, title=title, role=role,
                    model_label=util_provider.label, system_prompt=system, user_prompt=user)
        try:
            step.output = chat(util_provider, util_key, system, user)
        except Exception as exc:
            step.error = str(exc)
        res.steps.append(step)
        return step

    # --- Step 1: Generation ---
    say("Generating mind map…")
    sys_prompt = prompts.get_mindmap_system_prompt(language, input_text)
    gen = Step(number="1", key="generate", title="Mind map generation",
               role="Generation", model_label=gen_provider.label,
               system_prompt=sys_prompt, user_prompt=input_text)
    try:
        gen.output = chat(gen_provider, gen_key, sys_prompt, input_text)
    except Exception as exc:
        gen.error = str(exc)
    res.steps.append(gen)

    raw_for_repair = gen.output or "{}"

    # --- Step 2: JSON repair ---
    say("Repairing JSON…")
    repair = run_util(
        "json_repair", "JSON repair", "Repair", "2",
        _UTILITY_SYSTEM,
        prompts.build_pipeline_prompt("json_repair", broken_json=raw_for_repair),
    )
    repaired_obj = extract_mindmap_json(repair.output) or extract_mindmap_json(gen.output)
    res.mindmap_json = repaired_obj
    repaired_json_str = json.dumps(repaired_obj, ensure_ascii=False, indent=2) if repaired_obj else (repair.output or raw_for_repair)

    if use_critics:
        # --- Step 3: Paths extraction ---
        say("Extracting paths…")
        paths = run_util(
            "paths_extraction", "Path extraction", "Repair", "3",
            _UTILITY_SYSTEM,
            prompts.build_pipeline_prompt("paths_extraction", json_structure=repaired_json_str),
        )
        paths_text = paths.output

        # --- Step 4: Local Structure Critic ---
        say("Critic 1 / 3 — Local structure…")
        local = run_util(
            "local_critic", "Critic 1 · Local Structure", "Critic", "4",
            _UTILITY_SYSTEM,
            prompts.build_pipeline_prompt("local_structure_critic", json_structure=repaired_json_str),
        )
        local.verdict = _passed(local.output, "answer:", "yes")
        local.verdict_label = "all leaf values specific"

        # --- Step 5: Global Structure Critic ---
        say("Critic 2 / 3 — Global structure…")
        glob = run_util(
            "global_critic", "Critic 2 · Global Structure", "Critic", "5",
            _UTILITY_SYSTEM,
            prompts.build_pipeline_prompt("global_structure_critic", paths=paths_text),
        )
        glob.verdict = _passed(glob.output, "useful:", "yes")
        glob.verdict_label = "TOC is informative"

        # --- Step 6: Factual Critic (composite) ---
        say("Critic 3 / 3 — Factual grounding…")
        bullets = run_util(
            "factual_bullets", "Critic 3a · Bullet points", "Critic", "6a",
            _UTILITY_SYSTEM,
            prompts.build_pipeline_prompt("bullet_points", input_text=input_text),
        )
        attribution = run_util(
            "factual_attribution", "Critic 3b · Path attribution", "Critic", "6b",
            _UTILITY_SYSTEM,
            prompts.build_pipeline_prompt("factual_critic", bullet_points=bullets.output, paths=paths_text),
        )
        validator = run_util(
            "factual_validator", "Critic 3c · Factual validator", "Critic", "6c",
            _UTILITY_SYSTEM,
            prompts.build_pipeline_prompt("factual_validator", paths_with_citations=attribution.output),
        )
        validator.verdict = "accept" in (validator.output or "").lower()
        validator.verdict_label = "every path grounded (zero [NA])"

        local_ok = bool(local.verdict)
        global_ok = bool(glob.verdict)
        factual_ok = bool(validator.verdict)
        res.critics_summary = {"local": local_ok, "global": global_ok, "factual": factual_ok}
        res.accepted = local_ok and global_ok and factual_ok

    # --- Final step: Mermaid render ---
    say("Converting to Mermaid…")
    mer = run_util(
        "mermaid", "Mermaid conversion", "Render", "7" if use_critics else "3",
        _UTILITY_SYSTEM,
        prompts.build_pipeline_prompt("mermaid", json_structure=repaired_json_str),
    )
    mermaid_code = clean_mermaid_code(mer.output)
    if (not mermaid_code or "mindmap" not in mermaid_code) and repaired_obj:
        mermaid_code = json_to_mermaid_local(repaired_obj)
        mer.note = "Model output unusable; rendered with the deterministic JSON→Mermaid fallback."
    res.mermaid_code = mermaid_code
    say("Done.")
    return res


def _passed(text: str, key: str, want: str) -> bool:
    """Look for the LAST occurrence of `key` and check it equals `want`."""
    if not text:
        return False
    low = text.lower()
    idx = low.rfind(key.lower())
    if idx == -1:
        # critic sometimes answers bare yes/no
        return want in low.split()
    tail = low[idx + len(key):].strip()
    token = tail.split()[0] if tail.split() else ""
    return token.startswith(want)
