"""
app.py
======
Multilingual Mind-Map Generation & Evaluation Lab.

Tabs
  1. Generate & Compare   — paragraph -> mind map, with/without three critics,
                            one model or BOTH models side by side, plus a
                            Wiki40B / SQuAD sample picker. Every run is kept.
  2. Quality              — five binary criteria (SC SA CC BC GC) rated by you
                            (thesis: human evaluators); optional LLM cross-check; accumulates to a
                            running quality %.
  3. Semantic Coverage    — Auto-QA and Human-QA; answers come from the mind map
                            only; accumulates to a running coverage %.
  4. Comprehension Time   — timed study over <q,s> / <q,t> / <q,s+t>; accumulates
                            to mean response time and % reduction.

Demo mode (no key) shows the full flow with authentic example outputs so nothing
breaks during a presentation; add a key to run any track live.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html as _html
import json
import re
import time

import streamlit as st
import streamlit.components.v1 as components

import prompts
import demo as demo_mod
import samples as S
import evaluation as E
from llm import PROVIDERS, resolve_api_key, with_model, with_endpoint
from pipeline import run_pipeline

st.set_page_config(page_title="Mind-Map Generation & Evaluation Lab",
                   page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root{ --ink:#1b1b2f; --indigo:#4338ca; --indigo-soft:#eef0ff;
  --emerald:#0f8a5f; --emerald-soft:#e7f6ef; --rust:#b3401f; --rust-soft:#fbe9e3;
  --line:#e6e6f0; --muted:#6b6b86; --paper:#f7f7fc; --amber:#9a6a00; }
html, body, [class*="css"]{ font-family:'Inter',system-ui,sans-serif; }
.block-container{ padding-top:1.6rem; max-width:1180px; }
.app-title{ font-family:'Spectral',serif; font-weight:700; font-size:2.0rem;
  color:var(--ink); letter-spacing:-.01em; margin:0 0 .1rem; line-height:1.1; }
.app-sub{ color:var(--muted); font-size:1.0rem; margin:0 0 .3rem; }
.app-cite{ color:var(--muted); font-size:.8rem; font-style:italic; }
.mode-pill{ display:inline-block; padding:.16rem .55rem; border-radius:999px;
  font-size:.72rem; font-weight:600; }
.mode-live{ background:var(--emerald-soft); color:var(--emerald); border:1px solid #bfe6d4; }
.mode-demo{ background:#fff6e6; color:var(--amber); border:1px solid #f0dca8; }
.gate{ border-radius:12px; padding:.7rem 1rem; margin:.2rem 0 .8rem; font-weight:600; }
.gate-pass{ background:var(--emerald-soft); color:var(--emerald); border:1px solid #bfe6d4; }
.gate-fail{ background:var(--rust-soft); color:var(--rust); border:1px solid #eecbbf; }
.gate-off{ background:var(--paper); color:var(--muted); border:1px solid var(--line); }
.verdict{ font-size:.76rem; font-weight:600; padding:.1rem .45rem; border-radius:6px; }
.v-pass{ background:var(--emerald-soft); color:var(--emerald); }
.v-fail{ background:var(--rust-soft); color:var(--rust); }
.rolechip{ font-size:.7rem; font-weight:600; color:var(--indigo);
  background:var(--indigo-soft); padding:.1rem .45rem; border-radius:6px; }
.modelchip{ font-size:.72rem; color:var(--muted); font-family:'IBM Plex Mono',monospace; }
.sec-h{ font-family:'Spectral',serif; font-weight:600; font-size:1.25rem;
  color:var(--ink); margin:1.0rem 0 .3rem; }
.demo-note{ font-size:.76rem; color:var(--amber); background:#fff6e6;
  border:1px solid #f0dca8; padding:.28rem .55rem; border-radius:8px;
  display:inline-block; margin-bottom:.35rem; }
code, pre, .stCodeBlock{ font-family:'IBM Plex Mono',monospace !important; }
hr{ border:none; border-top:1px solid var(--line); margin:1.0rem 0; }
.run-head{ display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; margin:.1rem 0 .5rem; }
.run-id{ font-family:'IBM Plex Mono',monospace; font-weight:600; font-size:.8rem; color:var(--ink); }
.cfg{ font-size:.72rem; font-weight:600; color:var(--ink); background:var(--paper);
  border:1px solid var(--line); padding:.1rem .45rem; border-radius:6px; }
.run-card{ border:1px solid var(--line); border-radius:14px; padding:.9rem 1rem;
  background:#fff; box-shadow:0 1px 2px rgba(20,20,50,.04); }
.bigstat{ font-family:'Spectral',serif; font-weight:700; font-size:2.4rem;
  color:var(--indigo); line-height:1; }
.bigstat small{ font-family:'Inter'; font-size:.8rem; color:var(--muted);
  font-weight:500; display:block; margin-top:.2rem; }
.statcard{ border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem;
  background:#fff; text-align:center; }
table.tbl{ width:100%; border-collapse:collapse; font-size:.82rem;
  border:1px solid var(--line); border-radius:10px; overflow:hidden; }
table.tbl th{ background:var(--paper); color:var(--muted); font-weight:600;
  text-align:center; padding:.45rem .5rem; border-bottom:1px solid var(--line); }
table.tbl td{ text-align:center; padding:.4rem .5rem; border-bottom:1px solid var(--line); color:var(--ink); }
table.tbl td.l{ text-align:left; }
table.tbl tr:last-child td{ border-bottom:none; }
table.tbl .ok{ color:var(--emerald); font-weight:700; }
table.tbl .no{ color:var(--rust); font-weight:700; }
table.tbl .muted{ color:var(--muted); }
.crit-name{ font-weight:600; color:var(--ink); }
.crit-code{ font-family:'IBM Plex Mono',monospace; color:var(--indigo); font-weight:600; }
.crit-desc{ font-size:.8rem; color:var(--muted); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LANG_NAME = {"en": "English", "tr": "Turkish", "ar": "Arabic"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_secrets() -> dict:
    try:
        return dict(st.secrets)
    except Exception:
        return {}


def _esc(s) -> str:
    return _html.escape(str(s))


def _tree_html_from_json(obj, rtl=False) -> str:
    """A dependency-free nested-list view of a mind map JSON (any shape)."""
    LABEL_KEYS = ("label", "canonical", "name", "title", "id")

    def node_html(node, depth=0):
        if depth > 8:
            return ""
        if isinstance(node, dict):
            # schema A: explicit label (+ optional children)
            lbl = next((node[k] for k in LABEL_KEYS
                        if isinstance(node.get(k), str) and node[k].strip()), None)
            if lbl is not None and ("children" in node or len(node) <= 2):
                kids = node.get("children") or []
                inner = "".join(node_html(c, depth + 1) for c in kids)
                return f"<li><span class='tk'>{_esc(lbl)}</span>{_ul(inner)}</li>"
            # schema B: key -> value mapping
            parts = []
            for k, v in node.items():
                if k in LABEL_KEYS:
                    continue
                parts.append(f"<li><span class='tk'>{_esc(k)}</span>"
                             f"{_ul(node_html(v, depth + 1))}</li>")
            return "".join(parts)
        if isinstance(node, list):
            return "".join(node_html(c, depth + 1) for c in node)
        return f"<li><span class='tv'>{_esc(node)}</span></li>"

    def _ul(inner):
        return f"<ul>{inner}</ul>" if inner else ""

    root = obj.get("root", obj) if isinstance(obj, dict) else obj
    body = node_html(root, 0)
    d = "rtl" if rtl else "ltr"
    return (f"<div class='tree' dir='{d}'><ul>{body}</ul></div>"
            if body else "")


def _tree_html_from_mermaid(code, rtl=False) -> str:
    """Fallback: parse the mermaid indentation into a nested list."""
    lines = [l for l in (code or "").split("\n") if l.strip()
             and l.strip().lower() != "mindmap"]
    if not lines:
        return ""
    items = []
    for l in lines:
        indent = len(l) - len(l.lstrip())
        txt = l.strip()
        m = re.search(r"\(\((.+?)\)\)|\[(.+?)\]|\((.+?)\)", txt)
        if m:
            txt = next(g for g in m.groups() if g)
        items.append((indent, txt))
    html_out, stack = [], []
    for indent, txt in items:
        while stack and indent <= stack[-1]:
            html_out.append("</ul></li>")
            stack.pop()
        html_out.append(f"<li><span class='tk'>{_esc(txt)}</span><ul>")
        stack.append(indent)
    html_out.append("</ul></li>" * len(stack))
    d = "rtl" if rtl else "ltr"
    return f"<div class='tree' dir='{d}'><ul>{''.join(html_out)}</ul></div>"


_TREE_CSS = """
<style>
.tree ul{list-style:none;margin:0;padding-left:18px;border-left:1px dashed #d7d9ee;}
.tree>ul{padding-left:4px;border-left:none;}
.tree li{margin:3px 0;position:relative;}
.tree .tk{font-weight:600;color:#2b2b50;background:#eef0ff;border:1px solid #d9dcf5;
  border-radius:6px;padding:1px 7px;display:inline-block;}
.tree .tv{color:#33334d;background:#f4f7f3;border:1px solid #dbe7d6;border-radius:6px;
  padding:1px 7px;display:inline-block;}
.tree[dir=rtl] ul{padding-left:0;padding-right:18px;border-left:none;border-right:1px dashed #d7d9ee;}
.tree[dir=rtl]>ul{padding-right:4px;border-right:none;}
</style>"""


def render_mindmap(code, *, json_obj=None, rtl=False, height=460, key="mm"):
    """Render a mind map as a Mermaid diagram, with an automatic, dependency-free
    nested-tree fallback if Mermaid cannot load (offline/blocked) or fails."""
    safe = json.dumps(code or "", ensure_ascii=False)
    direction = "rtl" if rtl else "ltr"
    fallback = ""
    if json_obj is not None:
        fallback = _tree_html_from_json(json_obj, rtl)
    if not fallback:
        fallback = _tree_html_from_mermaid(code, rtl)
    fb_json = json.dumps(_TREE_CSS + fallback, ensure_ascii=False)

    tpl = f"""
    <div dir="{direction}" style="width:100%; overflow:auto; background:#fff;
         border:1px solid #e6e6f0; border-radius:12px; padding:12px;">
      <pre class="mermaid" id="m_{key}" style="margin:0;">{_esc(code or '')}</pre>
      <div id="fb_{key}" style="display:none;"></div>
    </div>
    <script>
    (function() {{
      var code = {safe};
      var fbHtml = {fb_json};
      var box = document.getElementById('m_{key}');
      var fb  = document.getElementById('fb_{key}');
      var done = false;
      function showFallback() {{
        if (done) return; done = true;
        if (box) box.style.display = 'none';
        fb.innerHTML = fbHtml; fb.style.display = 'block';
      }}
      function rendered() {{
        return box && box.querySelector('svg');
      }}
      function run() {{
        try {{
          mermaid.initialize({{ startOnLoad:false, securityLevel:'loose', theme:'neutral',
            themeVariables:{{ fontFamily:'Inter, sans-serif', fontSize:'14px',
              primaryColor:'#eef0ff', primaryBorderColor:'#4338ca', lineColor:'#9aa0c7' }} }});
          mermaid.run({{ nodes:[box] }})
            .then(function() {{ if (!rendered()) showFallback(); done = true; }})
            .catch(function() {{ showFallback(); }});
        }} catch (e) {{ showFallback(); }}
      }}
      function load(src, next) {{
        var s = document.createElement('script');
        s.src = src; s.onload = run; s.onerror = next;
        document.head.appendChild(s);
      }}
      // try two CDNs, then fall back to the tree
      load('https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js', function() {{
        load('https://unpkg.com/mermaid@10/dist/mermaid.min.js', showFallback);
      }});
      // safety timeout: if nothing rendered, show the tree
      setTimeout(function() {{ if (!rendered()) showFallback(); }}, 4000);
    }})();
    </script>"""
    components.html(tpl, height=height, scrolling=True)


# backwards-compatible alias
def mermaid(code, *, rtl=False, height=460, key="mm", json_obj=None):
    render_mindmap(code, json_obj=json_obj, rtl=rtl, height=height, key=key)


def status_dot(ok: bool) -> str:
    return "🟢" if ok else "⚪️"


def pct_card(value, label):
    v = "—" if value is None else f"{value:.1f}%"
    return f'<div class="statcard"><div class="bigstat">{v}<small>{label}</small></div></div>'


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
secrets = get_secrets()
with st.sidebar:
    st.markdown("### Configuration")
    language = st.radio("Language", ["en", "tr", "ar"],
                        format_func=lambda c: LANG_NAME[c], horizontal=True)
    is_rtl = language == "ar"

    gen_choice = st.radio(
        "Generation model",
        ["gemini", "qwen", "both"],
        format_func=lambda f: {"gemini": "Gemini 2.5 Flash",
                               "qwen": "Qwen2.5-7B-Instruct",
                               "both": "Both (side by side)"}[f])

    qwen_host = st.selectbox(
        "Qwen host", ["qwen_openrouter", "qwen_deepinfra", "qwen_cerebras", "qwen_custom"],
        format_func=lambda k: PROVIDERS[k].label,
        disabled=(gen_choice == "gemini"))
    qwen_base_url = ""
    if qwen_host == "qwen_custom":
        qwen_base_url = st.text_input(
            "Server base URL (OpenAI-compatible)",
            placeholder="https://api.deepinfra.com/v1/openai",
            help="e.g. DeepInfra: https://api.deepinfra.com/v1/openai · "
                 "Together: https://api.together.xyz/v1 · "
                 "HF router: https://router.huggingface.co/v1")

    use_critics = st.toggle("Use the three critics", value=True,
                            help="Local · Global · Factual quality gate (AND).")

    with st.expander("Advanced", expanded=False):
        gemini_model = st.text_input(
            "Gemini model", value="gemini-2.5-flash",
            help="Editable: Google retires models periodically. "
                 "gemini-2.0-flash was retired June 2026.")
        qwen_model_override = st.text_input(
            "Qwen model override (optional)", value="",
            help="Leave blank to use the host default. OpenRouter free: "
                 "qwen/qwen-2.5-7b-instruct:free  ·  paid: drop ':free'.")
        util_key_name = st.selectbox(
            "Utility model (repair · critics · Mermaid · QA)",
            list(PROVIDERS.keys()),
            index=list(PROVIDERS.keys()).index("qwen_openrouter"),
            format_func=lambda k: PROVIDERS[k].label)
        judge_model = st.text_input("Claude judge model", value="claude-sonnet-4-6",
                                    help="Anthropic model string for the quality judge.")
    gemini_provider = with_model(PROVIDERS["gemini_flash"], gemini_model.strip() or "gemini-2.5-flash")
    qwen_provider = PROVIDERS[qwen_host]
    if qwen_host == "qwen_custom" or qwen_model_override.strip() or qwen_base_url.strip():
        qwen_provider = with_endpoint(
            qwen_provider,
            base_url=qwen_base_url.strip() or None,
            model=qwen_model_override.strip() or None)
    if util_key_name == "gemini_flash":
        util_provider = gemini_provider
    elif util_key_name == qwen_host:
        util_provider = qwen_provider
    else:
        util_provider = PROVIDERS[util_key_name]
    judge_provider = with_model(PROVIDERS["claude_judge"], judge_model.strip() or "claude-sonnet-4-6")

    # which generation providers are active
    gen_providers = []
    if gen_choice in ("gemini", "both"):
        gen_providers.append(gemini_provider)
    if gen_choice in ("qwen", "both"):
        gen_providers.append(qwen_provider)

    st.markdown("---")
    st.markdown("#### API keys")
    st.caption("From `st.secrets` or pasted here for this session only.")
    relevant = {}
    for p in gen_providers + [util_provider, judge_provider]:
        relevant[p.secret_name] = p
    session_keys = {}
    for secret_name, prov in relevant.items():
        existing = bool(resolve_api_key(prov, secrets))
        entered = st.text_input(f"{status_dot(existing)} {secret_name}", type="password",
                                placeholder="found in secrets" if existing else "paste key (optional)",
                                key=f"key_{secret_name}")
        session_keys[secret_name] = resolve_api_key(prov, secrets, override=entered)

    gen_keys_ok = all(session_keys.get(p.secret_name) for p in gen_providers) and \
        bool(session_keys.get(util_provider.secret_name))
    gen_live = gen_keys_ok

    if gen_live:
        st.success("Generation: Live mode.")
    else:
        st.warning("Generation: Demo mode. Add keys to run live.")


def key_for(provider) -> str:
    return session_keys.get(provider.secret_name, "")


def is_live(provider) -> bool:
    return bool(key_for(provider))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
mode_html = ('<span class="mode-pill mode-live">LIVE</span>' if gen_live
             else '<span class="mode-pill mode-demo">DEMO</span>')
st.markdown(
    f'<div class="app-title">Mind-Map Generation &amp; Evaluation Lab {mode_html}</div>'
    '<div class="app-sub">Generation (with/without three critics, one or both '
    'models) and the thesis evaluation tracks: quality, semantic coverage, and '
    'comprehension time — English · Turkish · Arabic.</div>'
    '<div class="app-cite">Companion to “Generating Mind Maps from Textual '
    'Content: Multilingual Text Processing and Evaluation Metrics with LLMs.”</div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
st.session_state.setdefault("runs", [])
st.session_state.setdefault("run_counter", 0)
st.session_state.setdefault("quality_records", [])   # human ratings
st.session_state.setdefault("judge_records", [])     # Claude judge
st.session_state.setdefault("coverage_records", [])  # Auto/Human-QA
st.session_state.setdefault("comp_records", [])      # comprehension time


def family_label(provider):
    return {"gemini": "Gemini 2.5 Flash", "qwen": "Qwen2.5-7B",
            "claude": "Claude"}.get(provider.family, provider.family)


TRACK_NAME = {"quality": "Quality", "coverage_time": "Coverage & Time"}


def add_run(res, provider, mode, input_text, sample_id, track):
    st.session_state["run_counter"] += 1
    seq = st.session_state["run_counter"]
    st.session_state["runs"].append({
        "id": seq, "seq": seq, "result": res,
        "language": language, "is_rtl": is_rtl,
        "model_family": provider.family, "model_label": provider.label,
        "model_short": family_label(provider),
        "use_critics": use_critics, "input_text": input_text,
        "sample_id": sample_id, "mode": mode, "track": track})


def run_label(run) -> str:
    crit = "3 critics" if run["use_critics"] else "no critics"
    return (f'RUN {run["seq"]} · {run["model_short"]} · {LANG_NAME[run["language"]]} · '
            f'{crit} · {run["mode"]}')


def find_run(seq):
    for r in st.session_state["runs"]:
        if r["seq"] == seq:
            return r
    return None


def generate_one(provider, input_text, sample_id, track):
    if is_live(provider) and is_live(util_provider):
        res = run_pipeline(
            input_text=input_text, language=language,
            gen_provider=provider, gen_key=key_for(provider),
            util_provider=util_provider, util_key=key_for(util_provider),
            use_critics=use_critics)
        add_run(res, provider, "LIVE", input_text, sample_id, track)
    else:
        res = demo_mod.build_demo_result(
            input_text=input_text, language=language,
            gen_provider=provider, util_provider=util_provider,
            use_critics=use_critics)
        add_run(res, provider, "DEMO", input_text, sample_id, track)


# ===========================================================================
# TABS
# ===========================================================================
tab_gen, tab_quality, tab_cov, tab_time, tab_lib = st.tabs(
    ["① Generate (text → map)", "② Quality · Wiki40B (5 human criteria)",
     "③ Semantic Coverage · SQuAD (Auto/Human-QA)", "④ Comprehension Time · SQuAD",
     "📚 Prompt library"])


# ---------------------------------------------------------------------------
# TAB 1 — Generate & Compare
# ---------------------------------------------------------------------------
def render_run(run, rid, *, compact=False):
    res = run["result"]
    head_l, head_r = st.columns([5, 1])
    with head_l:
        crit = "3 critics" if run["use_critics"] else "no critics"
        mc = "mode-live" if run["mode"] == "LIVE" else "mode-demo"
        st.markdown(
            f'<div class="run-head"><span class="run-id">RUN {run["seq"]}</span>'
            f'<span class="mode-pill {mc}">{run["mode"]}</span>'
            f'<span class="cfg">{run["model_short"]}</span>'
            f'<span class="cfg">{LANG_NAME[run["language"]]}</span>'
            f'<span class="cfg">{crit}</span></div>', unsafe_allow_html=True)
    with head_r:
        if st.button("✕", key=f"rm_{rid}", help="Remove this run"):
            st.session_state["runs"] = [r for r in st.session_state["runs"] if r["id"] != rid]
            st.rerun()

    if res.accepted is None:
        st.markdown('<div class="gate gate-off">No critics — rendered directly after repair.</div>',
                    unsafe_allow_html=True)
    else:
        cs = res.critics_summary
        chips = " ".join(f'<span class="verdict {"v-pass" if cs[k] else "v-fail"}">'
                         f'{k.capitalize()}: {"PASS" if cs[k] else "FAIL"}</span>'
                         for k in ("local", "global", "factual"))
        gc, lab = ("gate-pass", "✓ ACCEPTED") if res.accepted else ("gate-fail", "✗ REJECTED")
        st.markdown(f'<div class="gate {gc}">{lab} &nbsp; {chips}</div>', unsafe_allow_html=True)

    if res.demo:
        st.markdown('<span class="demo-note">Demo diagram</span>', unsafe_allow_html=True)
    if res.mermaid_code:
        n = res.mermaid_code.count("\n") + 1
        cap, per = (560, 26) if compact else (760, 32)
        render_mindmap(res.mermaid_code, json_obj=res.mindmap_json, rtl=run["is_rtl"],
                       height=min(cap, max(280, per * n)), key=f"mm_{rid}")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("⬇ .mmd", res.mermaid_code,
                               file_name=f"mindmap_{run['language']}_run{run['seq']}.mmd",
                               key=f"dlm_{rid}", use_container_width=True)
        with d2:
            if res.mindmap_json is not None:
                st.download_button("⬇ .json",
                                   json.dumps(res.mindmap_json, ensure_ascii=False, indent=2),
                                   file_name=f"mindmap_{run['language']}_run{run['seq']}.json",
                                   key=f"dlj_{rid}", use_container_width=True)
    else:
        st.info("No diagram produced — see steps below.")

    with st.expander("Detailed steps · prompts & outputs"):
        for s in res.steps:
            v = "  ·  ✓ pass" if s.verdict is True else "  ·  ✗ fail" if s.verdict is False else \
                "  ·  ⚠ error" if s.error else ""
            with st.expander(f"Step {s.number} — {s.title}{v}", expanded=False):
                st.markdown(f'<span class="rolechip">{s.role}</span> &nbsp; '
                            f'<span class="modelchip">{_html.escape(s.model_label)}</span>',
                            unsafe_allow_html=True)
                if s.verdict is not None:
                    vc, vt = ("v-pass", "PASS") if s.verdict else ("v-fail", "FAIL")
                    st.markdown(f'<span class="verdict {vc}">{vt} — {_html.escape(s.verdict_label)}</span>',
                                unsafe_allow_html=True)
                if s.note:
                    st.markdown(f'<span class="demo-note">{_html.escape(s.note)}</span>', unsafe_allow_html=True)
                if s.error:
                    st.error(s.error)
                tp, to = st.tabs(["Prompt sent", "Model output"])
                with tp:
                    if s.system_prompt:
                        st.markdown("**System**"); st.code(s.system_prompt, language="markdown")
                    st.markdown("**User**"); st.code(s.user_prompt, language="markdown")
                with to:
                    if s.output:
                        st.code(s.output, language="json" if s.key in ("json_repair", "factual_validator") else "text")
                    else:
                        st.caption("No output.")


def comparison_table(runs) -> str:
    rows = ""
    for run in runs:
        res = run["result"]
        crit = "Yes" if run["use_critics"] else "No"
        if res.accepted is None:
            lgf = '<td class="muted">—</td>' * 3
            acc = '<span class="muted">n/a</span>'
        else:
            cs = res.critics_summary
            lgf = "".join(f'<td class="{ "ok" if cs[k] else "no"}">{"✓" if cs[k] else "✗"}</td>'
                          for k in ("local", "global", "factual"))
            acc = '<span class="ok">ACCEPTED</span>' if res.accepted else '<span class="no">REJECTED</span>'
        rows += (f'<tr><td>#{run["seq"]}</td><td>{run["mode"]}</td><td>{run["model_short"]}</td>'
                 f'<td>{run["language"].upper()}</td><td>{crit}</td>{lgf}<td>{acc}</td></tr>')
    return ('<table class="tbl"><thead><tr><th>Run</th><th>Mode</th><th>Model</th>'
            '<th>Lang</th><th>Critics</th><th>Local</th><th>Global</th><th>Factual</th>'
            f'<th>Result</th></tr></thead><tbody>{rows}</tbody></table>')


with tab_gen:
    st.caption("This tab only generates the mind map (up to the diagram). "
               "The dataset you pick decides where the map is evaluated: "
               "**Wiki40B → Quality**; **SQuAD → Semantic Coverage + Comprehension Time**.")

    # ----- sample picker with dataset + length filters -----
    cnt = S.counts().get(language, {})
    cnt_txt = " · ".join(f"{k}: {v}" for k, v in cnt.items()) or "none"
    st.markdown(f'<span class="modelchip">Bank for {LANG_NAME[language]} → {cnt_txt}</span>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1.4, 1.4, 1.2])
    with f1:
        src_choice = st.selectbox(
            "Dataset", ["Wiki40B (→ Quality)", "SQuAD (→ Coverage & Time)"],
            help="Wiki40B routes to Quality; SQuAD carries its questions to "
                 "Coverage and Comprehension Time.")
        src_key = "Wiki40B" if src_choice.startswith("Wiki40B") else "SQuAD"
    with f2:
        max_w = st.select_slider("Max length (words)", [80, 200, 500, 1000, 2000], value=2000)
    with f3:
        st.write(""); st.write("")
        st.caption("≤ 2000 words supported")

    # Wiki40B set may be empty before fetch_datasets.py is run -> fall back to Curated
    pool = S.filter_samples(language=language, source=src_key, max_words=max_w)
    if src_key == "Wiki40B" and not pool:
        pool = S.filter_samples(language=language, source="Curated", max_words=max_w)
        if pool:
            st.info("No Wiki40B passages in the bank yet for this language — showing "
                    "curated quality examples. Run `python fetch_datasets.py --online` to load Wiki40B.")
    if src_key == "SQuAD" and not pool:
        st.warning("No SQuAD passages in the bank for this language yet. "
                   "Run `python fetch_datasets.py --online` to download SQuAD/Arabic-SQuAD/SQuAD-TR.")

    opt_labels = {s.id: f"[{s.length_band} · {s.words}w] {s.title}"
                  + (f"  ·  {len(s.qa)} Q" if s.qa else "") for s in pool}
    pick_l, pick_r = st.columns([4, 1])
    with pick_l:
        opts = ["(type my own)"] + list(opt_labels.keys())
        chosen = st.selectbox(f"Sample ({len(pool)} available)", opts,
                              format_func=lambda k: "✍️ Type my own text"
                              if k == "(type my own)" else opt_labels[k])
    with pick_r:
        load = st.button("Load sample", use_container_width=True,
                         disabled=(chosen == "(type my own)"))

    if st.session_state.get("_gen_lang") != language:
        st.session_state["_gen_lang"] = language
        st.session_state["gen_text"] = pool[0].text if pool else ""
        st.session_state["gen_sample_id"] = pool[0].id if pool else None
    if load and chosen != "(type my own)":
        s = S.get(chosen)
        st.session_state["gen_text"] = s.text
        st.session_state["gen_sample_id"] = s.id

    gen_text = st.text_area("Input paragraph", key="gen_text", height=150)
    if is_rtl:
        st.markdown("<style>textarea{direction:rtl; text-align:right;}</style>", unsafe_allow_html=True)

    active_sample_id = st.session_state.get("gen_sample_id")
    active_sample = S.get(active_sample_id) if active_sample_id else None
    if active_sample and active_sample.text.strip() != gen_text.strip():
        active_sample = active_sample_id = None

    # routing for this generation: dataset sample -> its track; free text -> Quality
    gen_track = active_sample.track if active_sample else "quality"
    route_txt = ("→ Quality" if gen_track == "quality"
                 else "→ Semantic Coverage + Comprehension Time")
    st.markdown(f'<span class="cfg">This map will route {route_txt}</span>',
                unsafe_allow_html=True)

    label = "with" if use_critics else "without"
    models_txt = " + ".join(family_label(p) for p in gen_providers)
    if st.button(f"▶  Generate · {models_txt} · {label} three critics",
                 type="primary", use_container_width=True):
        if not gen_text.strip():
            st.error("Please enter a paragraph or load a sample.")
        else:
            with st.spinner("Running the pipeline…"):
                for p in gen_providers:
                    generate_one(p, gen_text.strip(), active_sample_id, gen_track)

    runs_all = st.session_state["runs"]
    if runs_all:
        st.markdown("---")
        b1, b2, b3 = st.columns([3, 2, 1.4])
        with b1:
            st.markdown(f'<div class="sec-h">Runs · {len(runs_all)} kept</div>', unsafe_allow_html=True)
        with b2:
            per_row = st.radio("Per row", [1, 2, 3], index=1, horizontal=True,
                               key="per_row", label_visibility="collapsed")
        with b3:
            if st.button("🗑 Clear all runs", use_container_width=True):
                st.session_state["runs"] = []
                st.rerun()
        if len(runs_all) >= 2:
            st.markdown(comparison_table(runs_all), unsafe_allow_html=True)
            st.write("")
        ordered = list(reversed(runs_all))
        for i in range(0, len(ordered), per_row):
            chunk = ordered[i:i + per_row]
            cols = st.columns(len(chunk), gap="medium")
            for col, run in zip(cols, chunk):
                with col:
                    st.markdown('<div class="run-card">', unsafe_allow_html=True)
                    render_run(run, run["id"], compact=(per_row > 1))
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Generate a mind map to begin. Runs are kept here for side-by-side comparison.")


# ---------------------------------------------------------------------------
# Shared: run selector for evaluation tabs
# ---------------------------------------------------------------------------
def select_run(key, track=None):
    runs = st.session_state["runs"]
    if track:
        runs = [r for r in runs if r.get("track") == track]
    if not runs:
        if track:
            need = ("a Wiki40B sample (or your own text)" if track == "quality"
                    else "a SQuAD sample")
            st.info(f"No mind maps routed here yet. In tab ① generate from {need}.")
        else:
            st.info("Generate a mind map in tab ① first, then evaluate it here.")
        return None
    options = [r["seq"] for r in reversed(runs)]
    seq = st.selectbox("Mind map to evaluate", options,
                       format_func=lambda s: run_label(find_run(s)), key=key)
    return find_run(seq)


def show_map_small(run, key):
    res = run["result"]
    if res.mermaid_code:
        n = res.mermaid_code.count("\n") + 1
        render_mindmap(res.mermaid_code, json_obj=res.mindmap_json, rtl=run["is_rtl"],
                       height=min(440, max(240, 24 * n)), key=key)


# ---------------------------------------------------------------------------
# TAB 2 — Quality (five HUMAN criteria; optional LLM cross-check)
# ---------------------------------------------------------------------------
with tab_quality:
    st.markdown('<div class="sec-h">Five binary quality criteria</div>', unsafe_allow_html=True)
    crit_rows = "".join(
        f'<tr><td class="crit-code">{c}</td><td class="l"><span class="crit-name">{n}</span>'
        f'<div class="crit-desc">{d}</div></td></tr>' for c, n, d in E.CRITERIA)
    st.markdown(f'<table class="tbl"><thead><tr><th>Code</th><th class="l">Criterion</th>'
                f'</tr></thead><tbody>{crit_rows}</tbody></table>', unsafe_allow_html=True)
    st.caption("Each criterion is binary (Good = 1 / Bad = 0). A map's quality% = "
               "(# Good ÷ 5) × 100; the running quality is the mean across rated maps.")
    st.info("Per the thesis, the five criteria are rated by **human evaluators** "
            "(two per language, majority vote). Claude's role in the thesis is an "
            "*auxiliary* one — applying the three critics, computing semantic "
            "coverage, and repairing JSON for visualization — not judging these five "
            "criteria. The LLM cross-check below is therefore an optional aid, not "
            "part of the thesis quality protocol.")

    run = select_run("q_run", track="quality")
    if run:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Mind map**")
            show_map_small(run, key="qmap")
            with st.expander("Source paragraph"):
                st.markdown(f'<div dir="{"rtl" if run["is_rtl"] else "ltr"}" '
                            f'style="white-space:pre-wrap">{_html.escape(run["input_text"])}</div>',
                            unsafe_allow_html=True)
        with right:
            st.markdown("**Your rating (human)**")
            human = {}
            for c, n, d in E.CRITERIA:
                human[c] = st.toggle(f"{c} — {n}", value=True, key=f"hr_{run['seq']}_{c}")
            qp = E.quality_percent(human)
            st.markdown(pct_card(qp, "this map · human"), unsafe_allow_html=True)
            if st.button("💾 Save human rating", key=f"saveq_{run['seq']}", use_container_width=True):
                st.session_state["quality_records"].append(
                    {"seq": run["seq"], "model": run["model_short"], "language": run["language"],
                     "ratings": dict(human)})
                st.success("Saved.")

        st.markdown("---")
        st.markdown("**Optional LLM cross-check** &nbsp; "
                    "<span class='crit-desc'>(auxiliary — not the thesis protocol)</span>",
                    unsafe_allow_html=True)
        jcol1, jcol2 = st.columns([1, 2])
        with jcol1:
            live_j = is_live(judge_provider)
            st.caption(("Live · " + judge_provider.model) if live_j else "Demo (no Anthropic key)")
            if st.button("⚖️ Run Claude judge", key=f"judge_{run['seq']}", use_container_width=True):
                repr_ = E.mindmap_repr(run["result"])
                if live_j:
                    jr = E.claude_judge(judge_provider, key_for(judge_provider),
                                        run["input_text"], repr_)
                else:
                    jr = E.demo_judge(run["input_text"], repr_)
                st.session_state[f"jr_{run['seq']}"] = jr
        jr = st.session_state.get(f"jr_{run['seq']}")
        with jcol2:
            if jr:
                if jr.error:
                    st.error(jr.error)
                else:
                    chips = " ".join(f'<span class="verdict {"v-pass" if jr.ratings.get(c) else "v-fail"}">'
                                     f'{c}: {"1" if jr.ratings.get(c) else "0"}</span>' for c in E.CRITERIA_CODES)
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown(pct_card(jr.quality, "this map · Claude judge"), unsafe_allow_html=True)
                    with st.expander("Judge reasons"):
                        for c in E.CRITERIA_CODES:
                            st.markdown(f"**{c}** — {_html.escape(jr.reasons.get(c, ''))}")
                    if st.button("💾 Save judge result", key=f"savej_{run['seq']}"):
                        st.session_state["judge_records"].append(
                            {"seq": run["seq"], "model": run["model_short"],
                             "language": run["language"], "ratings": dict(jr.ratings)})
                        st.success("Saved.")
            else:
                st.caption("Run the judge to see per-criterion verdicts and reasons.")

    # running aggregates
    st.markdown("---")
    st.markdown('<div class="sec-h">Running quality results</div>', unsafe_allow_html=True)
    qrecs = st.session_state["quality_records"]
    jrecs = st.session_state["judge_records"]
    a, b, c = st.columns(3)
    with a:
        st.markdown(pct_card(E.aggregate_quality(qrecs) if qrecs else None,
                             f"Human · {len(qrecs)} maps"), unsafe_allow_html=True)
    with b:
        st.markdown(pct_card(E.aggregate_quality(jrecs) if jrecs else None,
                             f"Claude judge · {len(jrecs)} maps"), unsafe_allow_html=True)
    with c:
        ref = E.REFERENCE["quality"]
        st.markdown('<div class="statcard"><div class="bigstat" style="font-size:1.1rem">'
                    f'G {ref["gemini"][language]}% · Q {ref["qwen"][language]}%'
                    f'<small>thesis quality · {LANG_NAME[language]}</small></div></div>',
                    unsafe_allow_html=True)

    def per_criterion_table(recs, title):
        if not recs:
            return ""
        rows = ""
        for cprefix in E.CRITERIA_CODES:
            vals = [1 if r["ratings"].get(cprefix) else 0 for r in recs]
            pct = 100.0 * sum(vals) / len(vals)
            rows += f'<tr><td class="crit-code">{cprefix}</td><td>{pct:.1f}%</td></tr>'
        avg = E.aggregate_quality(recs)
        rows += f'<tr><td><b>Avg</b></td><td><b>{avg:.1f}%</b></td></tr>'
        return (f'<div class="crit-desc" style="margin:.4rem 0 .2rem">{title}</div>'
                f'<table class="tbl"><thead><tr><th>Criterion</th><th>% Good</th></tr>'
                f'</thead><tbody>{rows}</tbody></table>')

    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown(per_criterion_table(qrecs, "Per-criterion % Good — human"), unsafe_allow_html=True)
    with pc2:
        st.markdown(per_criterion_table(jrecs, "Per-criterion % Good — Claude judge"), unsafe_allow_html=True)
    if qrecs or jrecs:
        if st.button("🗑 Reset quality records"):
            st.session_state["quality_records"] = []
            st.session_state["judge_records"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 3 — Semantic Coverage (Auto-QA / Human-QA)
# ---------------------------------------------------------------------------
def coverage_items_table(cov) -> str:
    rows = ""
    for it in cov.items:
        mark = '<span class="ok">✓</span>' if it.correct else '<span class="no">✗</span>'
        rows += (f'<tr><td class="l">{_html.escape(it.question)}</td>'
                 f'<td class="l">{_html.escape(it.gold)}</td>'
                 f'<td class="l">{_html.escape(it.map_answer)}</td>'
                 f'<td>{mark}</td><td class="muted">{it.method}</td></tr>')
    return ('<table class="tbl"><thead><tr><th class="l">Question</th><th class="l">Gold answer</th>'
            '<th class="l">Answer from mind map</th><th>Match</th><th>Method</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


with tab_cov:
    st.markdown('<div class="sec-h">Semantic coverage (QA-based)</div>', unsafe_allow_html=True)
    st.caption("Coverage = correct answers ÷ total questions. Answers are produced "
               "from the mind map ONLY; agreement uses deterministic matching "
               "(substring · token-overlap · numeric), as in the thesis. In **Auto-QA**, "
               "Claude Sonnet 4 first **reviews the generated questions** and keeps only "
               "the valid ones — this is Claude's actual judging role in the thesis.")
    st.latex(r"\mathrm{SemCov}(s,t)=\frac{1}{|QA(t)|}\sum_{q_i\in QA(t)} \mathbf{1}\big[\mathrm{Equivalent}(Q(s,q_i),\,a_i)\big]")

    run = select_run("c_run", track="coverage_time")
    if run:
        show_map_small(run, key="cmap")
        repr_ = E.mindmap_repr(run["result"])
        live_u = is_live(util_provider)
        st.caption(("Live · " + util_provider.label) if live_u else "Demo (no utility-model key)")

        track = st.radio("Track", ["Auto-QA", "Human-QA"], horizontal=True, key="cov_track")
        sample = S.get(run["sample_id"]) if run["sample_id"] else None

        if track == "Human-QA" and (not sample or not sample.qa):
            st.warning("Human-QA needs a SQuAD-style sample (with gold questions). "
                       "Generate a run from a SQuAD sample in tab ①.")
        else:
            n_q = st.slider("Questions to evaluate", 1, 5, 3, key="cov_nq")
            reviewer = judge_provider if is_live(judge_provider) else util_provider
            if track == "Auto-QA":
                st.caption("Claude reviewer: " + (("Live · " + reviewer.label)
                           if is_live(reviewer) else "Demo"))
            if st.button(f"▶ Run {track}", key=f"cov_run_{run['seq']}", use_container_width=True):
                with st.spinner(f"Running {track}…"):
                    review_rows = None
                    if track == "Auto-QA":
                        if live_u:
                            qa = E.generate_auto_qa(util_provider, key_for(util_provider),
                                                    run["input_text"], limit=n_q)
                            if not qa and sample and sample.qa:
                                qa = sample.qa[:n_q]
                            # thesis step: Claude reviews the generated questions
                            if is_live(reviewer):
                                qa, review_rows = E.review_auto_qa(
                                    reviewer, key_for(reviewer), run["input_text"], qa)
                            else:
                                qa, review_rows = E.demo_review_auto_qa(qa)
                            cov = E.run_coverage(util_provider, key_for(util_provider),
                                                 repr_, qa, "Auto-QA")
                        else:
                            qa = (sample.qa[:n_q] if sample and sample.qa
                                  else [{"question": "What is the main subject of the text?",
                                         "answer": run["input_text"].split(".")[0][:60]}])
                            qa, review_rows = E.demo_review_auto_qa(qa)
                            cov = E.demo_coverage(qa, "Auto-QA")
                    else:  # Human-QA (gold human questions; no review needed)
                        qa = sample.qa[:n_q]
                        if live_u:
                            cov = E.run_coverage(util_provider, key_for(util_provider),
                                                 repr_, qa, "Human-QA")
                        else:
                            cov = E.demo_coverage(qa, "Human-QA")
                    st.session_state[f"cov_{run['seq']}_{track}"] = cov
                    st.session_state[f"review_{run['seq']}"] = review_rows

            cov = st.session_state.get(f"cov_{run['seq']}_{track}")
            if cov:
                if cov.error:
                    st.error(cov.error)
                review_rows = st.session_state.get(f"review_{run['seq']}")
                if track == "Auto-QA" and review_rows:
                    kept = sum(1 for r in review_rows if r.get("is_valid"))
                    st.markdown(f'<span class="rolechip">Claude review</span> &nbsp; '
                                f'kept {kept} of {len(review_rows)} generated questions',
                                unsafe_allow_html=True)
                    rrows = "".join(
                        f'<tr><td class="l">{_html.escape(str(r.get("question","")))}</td>'
                        f'<td>{"✓" if r.get("is_valid") else "✗"}</td>'
                        f'<td class="l muted">{_html.escape(str(r.get("reason","")))}</td></tr>'
                        for r in review_rows)
                    st.markdown('<table class="tbl"><thead><tr><th class="l">Generated question</th>'
                                '<th>Valid</th><th class="l">Reviewer reason</th></tr></thead>'
                                f'<tbody>{rrows}</tbody></table>', unsafe_allow_html=True)
                    st.write("")
                st.markdown(pct_card(cov.coverage, f"{track} coverage · this map"), unsafe_allow_html=True)
                st.markdown(coverage_items_table(cov), unsafe_allow_html=True)
                if st.button(f"💾 Save {track} result", key=f"cov_save_{run['seq']}_{track}"):
                    st.session_state["coverage_records"].append(
                        {"seq": run["seq"], "model": run["model_short"], "language": run["language"],
                         "track": track, "coverage": cov.coverage, "n": len(cov.items)})
                    st.success("Saved.")

    # aggregates
    st.markdown("---")
    st.markdown('<div class="sec-h">Running coverage results</div>', unsafe_allow_html=True)
    crecs = st.session_state["coverage_records"]

    def track_avg(tname):
        vals = [r["coverage"] for r in crecs if r["track"] == tname]
        return round(sum(vals) / len(vals), 1) if vals else None

    a, b, c = st.columns(3)
    with a:
        st.markdown(pct_card(track_avg("Auto-QA"),
                             f"Auto-QA · {sum(1 for r in crecs if r['track']=='Auto-QA')} maps"),
                    unsafe_allow_html=True)
    with b:
        st.markdown(pct_card(track_avg("Human-QA"),
                             f"Human-QA · {sum(1 for r in crecs if r['track']=='Human-QA')} maps"),
                    unsafe_allow_html=True)
    with c:
        aq = E.REFERENCE["auto_qa_avg"]["gemini"]
        hq = E.REFERENCE["human_qa_avg"]["gemini"]
        st.markdown('<div class="statcard"><div class="bigstat" style="font-size:1.1rem">'
                    f'A {aq}% · H {hq}%<small>thesis avg (Gemini)</small></div></div>',
                    unsafe_allow_html=True)

    if crecs:
        rows = ""
        for r in crecs:
            rows += (f'<tr><td>#{r["seq"]}</td><td>{r["model"]}</td><td>{r["language"].upper()}</td>'
                     f'<td>{r["track"]}</td><td>{r["n"]}</td><td>{r["coverage"]:.1f}%</td></tr>')
        st.markdown('<table class="tbl"><thead><tr><th>Run</th><th>Model</th><th>Lang</th>'
                    '<th>Track</th><th>Q</th><th>Coverage</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        if st.button("🗑 Reset coverage records"):
            st.session_state["coverage_records"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 4 — Comprehension Time
# ---------------------------------------------------------------------------
CONDITIONS = {"qs": "⟨q, s⟩  mind map only", "qt": "⟨q, t⟩  text only",
              "qst": "⟨q, s+t⟩  mind map + text"}

with tab_time:
    st.markdown('<div class="sec-h">Comprehension-time study</div>', unsafe_allow_html=True)
    st.caption("Time how long it takes to answer a question under three context "
               "variants. The headline metric is the % reduction of mind-map-only "
               "⟨q,s⟩ versus text-only ⟨q,t⟩.")

    run = select_run("t_run", track="coverage_time")
    sample = S.get(run["sample_id"]) if run and run["sample_id"] else None
    if run and (not sample or not sample.qa):
        st.warning("Comprehension time needs a SQuAD-style sample (with gold "
                   "questions). Generate a run from a SQuAD sample in tab ①.")
    elif run:
        qopts = list(range(len(sample.qa)))
        qi = st.selectbox("Question", qopts,
                          format_func=lambda i: sample.qa[i]["question"], key="ct_q")
        cond = st.radio("Context variant", list(CONDITIONS.keys()),
                        format_func=lambda k: CONDITIONS[k], horizontal=True, key="ct_cond")

        st.markdown(f'**Question:** {sample.qa[qi]["question"]}')
        if cond in ("qs", "qst"):
            st.markdown("**Mind map (s):**")
            show_map_small(run, key="tmap")
        if cond in ("qt", "qst"):
            st.markdown("**Text (t):**")
            st.markdown(f'<div dir="{"rtl" if run["is_rtl"] else "ltr"}" '
                        f'style="white-space:pre-wrap; background:#f7f7fc; padding:.6rem; '
                        f'border-radius:8px">{_html.escape(run["input_text"])}</div>',
                        unsafe_allow_html=True)

        tk = f"ct_start_{run['seq']}_{qi}_{cond}"
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⏱ Start timer", key=f"ct_go_{run['seq']}_{qi}_{cond}", use_container_width=True):
                st.session_state[tk] = time.time()
                st.session_state.pop(f"ct_done_{run['seq']}", None)
        started = tk in st.session_state
        with c2:
            st.caption("Timer running…" if started else "Press Start, read, then answer.")

        ans = st.text_input("Your answer", key=f"ct_ans_{run['seq']}_{qi}_{cond}",
                            disabled=not started)
        if st.button("⏹ Stop & submit", key=f"ct_sub_{run['seq']}_{qi}_{cond}",
                     disabled=not started, use_container_width=True):
            elapsed = round(time.time() - st.session_state[tk], 2)
            gold = sample.qa[qi]["answer"]
            correct = E.deterministic_match(gold, ans)
            st.session_state["comp_records"].append(
                {"language": run["language"], "condition": cond,
                 "seconds": elapsed, "correct": correct})
            st.session_state.pop(tk, None)
            st.success(f"Recorded {elapsed:.2f}s · answer {'correct' if correct else 'incorrect'} "
                       f"(gold: {gold}).")

    # aggregates
    st.markdown("---")
    st.markdown('<div class="sec-h">Running time results</div>', unsafe_allow_html=True)
    trecs = st.session_state["comp_records"]

    def cond_mean(cn):
        vals = [r["seconds"] for r in trecs if r["condition"] == cn]
        return round(sum(vals) / len(vals), 2) if vals else None

    qs_m, qt_m, qst_m = cond_mean("qs"), cond_mean("qt"), cond_mean("qst")
    reduction = None
    if qs_m is not None and qt_m and qt_m > 0:
        reduction = round(100.0 * (qt_m - qs_m) / qt_m, 1)

    cols = st.columns(4)
    for col, (val, lab) in zip(cols, [
            (qs_m, "⟨q,s⟩ mean sec"), (qt_m, "⟨q,t⟩ mean sec"),
            (qst_m, "⟨q,s+t⟩ mean sec")]):
        with col:
            v = "—" if val is None else f"{val:.2f}s"
            col.markdown(f'<div class="statcard"><div class="bigstat" style="font-size:1.8rem">{v}'
                         f'<small>{lab}</small></div></div>', unsafe_allow_html=True)
    with cols[3]:
        rv = "—" if reduction is None else f"{reduction:.1f}%"
        st.markdown(f'<div class="statcard"><div class="bigstat">{rv}'
                    f'<small>faster: ⟨q,s⟩ vs ⟨q,t⟩</small></div></div>', unsafe_allow_html=True)

    ref = E.REFERENCE["comprehension_reduction"][language]
    st.caption(f"Thesis reference reduction for {LANG_NAME[language]}: {ref}% faster "
               f"(p < .01). n recorded = {len(trecs)}.")
    if trecs:
        rows = ""
        for i, r in enumerate(trecs, 1):
            rows += (f'<tr><td>{i}</td><td>{r["language"].upper()}</td>'
                     f'<td>{CONDITIONS[r["condition"]].split()[0]}</td>'
                     f'<td>{r["seconds"]:.2f}s</td>'
                     f'<td class="{ "ok" if r["correct"] else "no"}">{"✓" if r["correct"] else "✗"}</td></tr>')
        st.markdown('<table class="tbl"><thead><tr><th>#</th><th>Lang</th><th>Variant</th>'
                    '<th>Time</th><th>Correct</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
        if st.button("🗑 Reset time records"):
            st.session_state["comp_records"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# TAB 5 — Prompt library
# ---------------------------------------------------------------------------
with tab_lib:
    st.caption("Every template, reproduced verbatim from the research notebooks.")
    g, c, q, o = st.tabs(["Generation (EN·TR·AR)", "Critic prompts",
                          "QA & judge", "Other"])
    with g:
        sub = st.radio("Language", ["en", "tr", "ar"], format_func=lambda c: LANG_NAME[c],
                       horizontal=True, key="lib_lang")
        st.code(prompts.get_mindmap_system_prompt(sub), language="markdown")
    with c:
        st.markdown("**Critic 1 · Local Structure**")
        st.code(prompts.PIPELINE_PROMPTS["local_structure_critic"], language="markdown")
        st.markdown("**Critic 2 · Global Structure**")
        st.code(prompts.PIPELINE_PROMPTS["global_structure_critic"], language="markdown")
        st.markdown("**Critic 3 · Factual** (bullets → attribution → validator)")
        st.code(prompts.PIPELINE_PROMPTS["bullet_points"], language="markdown")
        st.code(prompts.PIPELINE_PROMPTS["factual_critic"], language="markdown")
        st.code(prompts.PIPELINE_PROMPTS["factual_validator"], language="markdown")
    with q:
        st.markdown("**Auto-QA — generate QA pairs**")
        st.code(prompts.AUTO_QA_PROMPT, language="markdown")
        st.markdown("**Answer using the mind map only**")
        st.code(prompts.QA_VALIDITY_PROMPT, language="markdown")
        st.markdown("**Answer equivalence**")
        st.code(prompts.EQUIVALENCE_QA_PROMPT, language="markdown")
        st.markdown("**Claude QA reviewer (Auto-QA — reviews generated questions)**")
        st.code(E.QA_REVIEWER_PROMPT, language="markdown")
        st.markdown("**Optional LLM cross-check — five criteria (auxiliary, not thesis protocol)**")
        st.code(E.JUDGE_PROMPT, language="markdown")
    with o:
        st.markdown("**JSON repair**"); st.code(prompts.PIPELINE_PROMPTS["json_repair"], language="markdown")
        st.markdown("**Path extraction**"); st.code(prompts.PIPELINE_PROMPTS["paths_extraction"], language="markdown")
        st.markdown("**Mermaid conversion**"); st.code(prompts.PIPELINE_PROMPTS["mermaid"], language="markdown")
