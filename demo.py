"""
demo.py
=======
Walkthrough ("demo") mode. When no API key is configured, the app still shows
the *complete* pipeline: the real prompts (filled with the user's text) plus
authentic example outputs taken from the research notebooks, so the Mermaid
diagram and critic verdicts still render. Every demo step is clearly labelled.

Switching to Live mode (add a key in the sidebar) runs the same steps for real.
"""

from __future__ import annotations

import json

import prompts
from llm import Provider
from pipeline import Result, Step, clean_mermaid_code, _UTILITY_SYSTEM


# Default source paragraphs (used to prefill the input box per language).
DEFAULT_TEXT = {
    "en": (
        "Stepa was an American nu metal band. The band released its sole, "
        "self-titled album on Locomotive Music on 30 July 2002. It was produced "
        "by Scott Gaines and Jay Baumgardner, with additional songwriting "
        "contributions from Mark Renk. Scott Borland, brother of Limp Bizkit "
        "guitarist Wes Borland, played additional keyboards. The singles were "
        "\"Aquarium\" and \"Spaceships And Airplanes\"."
    ),
    "tr": (
        "Halil İbrahim Şahin, 1937 yılında Denizli'nin Çal ilçesine bağlı "
        "İsabey'de doğdu. İsparta Gönen İlköğretmen Okulu ile Ankara Üniversitesi "
        "Hukuk Fakültesi'nden mezun oldu. Öğretmenlik ve serbest avukatlık yaptı; "
        "TBMM XVII. Dönem Denizli Milletvekili olarak görev aldı. Evli ve dört "
        "çocuk babasıdır."
    ),
    "ar": (
        "مونتيفيديو هي عاصمة الأوروغواي. يبلغ متوسط ارتفاعها 43 مترًا، وأعلى نقطة "
        "فيها هي قمة كيرو دي مونتيفيديو على ارتفاع 134 مترًا. من المدن القريبة "
        "منها لاس بيدراس شمالًا وسيوداد دي لا كوستا شرقًا، وتبعد كل منهما نحو 20 "
        "إلى 25 كيلومترًا عن وسط المدينة."
    ),
}

# Authentic mind-map JSON (compact) per language.
_JSON = {
    "en": {
        "root": {"id": "root", "canonical": "stepa", "label": "Stepa", "children": [
            {"label": "Album", "children": [
                {"label": "Self-titled release", "children": [
                    {"label": "Locomotive Music"},
                    {"label": "30 July 2002"}]},
                {"label": "Produced by", "children": [
                    {"label": "Scott Gaines"}, {"label": "Jay Baumgardner"}]},
                {"label": "Songwriting", "children": [{"label": "Mark Renk"}]},
                {"label": "Additional keyboards", "children": [
                    {"label": "Scott Borland"}]},
                {"label": "Singles", "children": [
                    {"label": "Aquarium"}, {"label": "Spaceships And Airplanes"}]},
            ]},
        ]}
    },
    "tr": {
        "root": {"id": "root", "canonical": "halil ibrahim sahin",
                 "label": "Halil İbrahim Şahin", "children": [
            {"label": "Kişisel Bilgiler", "children": [
                {"label": "Doğum: 1937, İsabey, Çal, Denizli"},
                {"label": "Uyruk: Türk"}]},
            {"label": "Eğitim", "children": [
                {"label": "İsparta Gönen İlköğretmen Okulu"},
                {"label": "Ankara Üniversitesi Hukuk Fakültesi"}]},
            {"label": "Meslek", "children": [
                {"label": "Öğretmen"}, {"label": "Serbest Avukat"},
                {"label": "TBMM XVII. Dönem Denizli Milletvekili"}]},
            {"label": "Aile", "children": [
                {"label": "Evli"}, {"label": "Dört çocuk babası"}]},
        ]}
    },
    "ar": {
        "root": {"id": "root", "canonical": "مونتيفيديو",
                 "label": "مونتيفيديو", "children": [
            {"label": "جغرافيا", "children": [
                {"label": "متوسط الارتفاع 43 مترا"},
                {"label": "أعلى نقطة قمة كيرو دي مونتيفيديو 134 مترا"}]},
            {"label": "المدن القريبة", "children": [
                {"label": "لاس بيدراس شمالا 20 الى 25 كم"},
                {"label": "سيوداد دي لا كوستا شرقا 20 الى 25 كم"}]},
        ]}
    },
}

_MERMAID = {
    "en": """mindmap
  root((Stepa))
    Album
      Self titled release
        Locomotive Music
        30 July 2002
      Produced by
        Scott Gaines
        Jay Baumgardner
      Songwriting
        Mark Renk
      Additional keyboards
        Scott Borland
      Singles
        Aquarium
        Spaceships And Airplanes""",
    "tr": """mindmap
  root((Halil İbrahim Şahin))
    Kişisel Bilgiler
      doğum 1937 İsabey Çal Denizli
      uyruk Türk
    Eğitim
      İsparta Gönen İlköğretmen Okulu
      Ankara Üniversitesi Hukuk Fakültesi
    Meslek
      Siyasetçi
      Serbest Avukat
      TBMM XVII Dönem Denizli Milletvekili
    Aile
      evli
      dört çocuk babası""",
    "ar": """mindmap
  root((مونتيفيديو))
    جغرافيا
      متوسط الارتفاع 43 مترا
      أعلى نقطة قمة كيرو دي مونتيفيديو 134 مترا
    المدن القريبة
      لاس بيدراس شمالا 20 الى 25 كم
      سيوداد دي لا كوستا شرقا 20 الى 25 كم""",
}

# Canned bullet points + path attribution per language for the factual critic.
_BULLETS = {
    "en": "1. Stepa was an American nu metal band.\n2. Stepa released its self-titled album on Locomotive Music on 30 July 2002.\n3. The album was produced by Scott Gaines and Jay Baumgardner.\n4. Mark Renk contributed additional songwriting.\n5. Scott Borland played additional keyboards.\n6. The singles were Aquarium and Spaceships And Airplanes.",
    "tr": "1. Halil İbrahim Şahin 1937'de İsabey, Çal, Denizli'de doğdu.\n2. İsparta Gönen İlköğretmen Okulu ve Ankara Üniversitesi Hukuk Fakültesi'nden mezun oldu.\n3. Öğretmenlik ve serbest avukatlık yaptı.\n4. TBMM XVII. Dönem Denizli Milletvekili oldu.\n5. Evli ve dört çocuk babasıdır.",
    "ar": "1. مونتيفيديو هي عاصمة الأوروغواي.\n2. يبلغ متوسط ارتفاعها 43 مترا.\n3. أعلى نقطة فيها قمة كيرو دي مونتيفيديو على ارتفاع 134 مترا.\n4. من المدن القريبة لاس بيدراس شمالا وسيوداد دي لا كوستا شرقا.",
}


def _paths_from_json(obj, prefix=""):
    """Deterministic path list used for the demo path-extraction step."""
    out = []
    root = obj.get("root", obj)

    def walk(node, trail):
        label = node.get("label") or node.get("canonical") or node.get("id") or "node"
        new_trail = trail + [label]
        children = node.get("children") or []
        if not children:
            out.append(" -> ".join(new_trail))
        for c in children:
            walk(c, new_trail)

    walk(root, [])
    return "\n".join(out)


def build_demo_result(
    *,
    input_text: str,
    language: str,
    gen_provider: Provider,
    util_provider: Provider,
    use_critics: bool,
) -> Result:
    """Construct a fully populated Result using real prompts + authentic outputs."""
    lang = language.lower()
    obj = _JSON.get(lang, _JSON["en"])
    json_str = json.dumps(obj, ensure_ascii=False, indent=2)
    paths = _paths_from_json(obj)

    res = Result(demo=True)
    note = "Demo output (no API key set). Add a key in the sidebar to run this step live."

    # 1. Generation
    res.steps.append(Step(
        number="1", key="generate", title="Mind map generation", role="Generation",
        model_label=gen_provider.label,
        system_prompt=prompts.get_mindmap_system_prompt(lang, input_text),
        user_prompt=input_text,
        output="MindMap\n" + json_str + "\nEND_THOUGHT",
        note=note,
    ))
    # 2. Repair
    res.steps.append(Step(
        number="2", key="json_repair", title="JSON repair", role="Repair",
        model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
        user_prompt=prompts.build_pipeline_prompt("json_repair", broken_json="MindMap\n" + json_str + "\nEND_THOUGHT"),
        output=json_str, note=note,
    ))
    res.mindmap_json = obj

    if use_critics:
        res.steps.append(Step(
            number="3", key="paths_extraction", title="Path extraction", role="Repair",
            model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
            user_prompt=prompts.build_pipeline_prompt("paths_extraction", json_structure=json_str),
            output=paths, note=note,
        ))
        local = Step(
            number="4", key="local_critic", title="Critic 1 · Local Structure", role="Critic",
            model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
            user_prompt=prompts.build_pipeline_prompt("local_structure_critic", json_structure=json_str),
            output="Thought: All extracted leaf values are specific names, dates and places.\nAnswer: yes",
            verdict=True, verdict_label="all leaf values specific", note=note,
        )
        res.steps.append(local)
        glob = Step(
            number="5", key="global_critic", title="Critic 2 · Global Structure", role="Critic",
            model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
            user_prompt=prompts.build_pipeline_prompt("global_structure_critic", paths=paths),
            output="Thought: The paths form sensible, informative sentences.\nUseful: yes",
            verdict=True, verdict_label="TOC is informative", note=note,
        )
        res.steps.append(glob)
        bullets = Step(
            number="6a", key="factual_bullets", title="Critic 3a · Bullet points", role="Critic",
            model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
            user_prompt=prompts.build_pipeline_prompt("bullet_points", input_text=input_text),
            output=_BULLETS.get(lang, _BULLETS["en"]), note=note,
        )
        res.steps.append(bullets)
        attribution_lines = "\n".join(f"{p} [{(i % 5) + 1}]" for i, p in enumerate(paths.split("\n")))
        attr = Step(
            number="6b", key="factual_attribution", title="Critic 3b · Path attribution", role="Critic",
            model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
            user_prompt=prompts.build_pipeline_prompt("factual_critic", bullet_points=bullets.output, paths=paths),
            output=attribution_lines, note=note,
        )
        res.steps.append(attr)
        n_paths = len(paths.split("\n"))
        validator = Step(
            number="6c", key="factual_validator", title="Critic 3c · Factual validator", role="Critic",
            model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
            user_prompt=prompts.build_pipeline_prompt("factual_validator", paths_with_citations=attribution_lines),
            output=json.dumps({
                "total_paths": n_paths, "paths_with_na": 0,
                "paths_with_valid_citations": n_paths, "factuality_score": 100,
                "decision": "ACCEPT", "reason": "Every path is supported by a source sentence."
            }, ensure_ascii=False, indent=2),
            verdict=True, verdict_label="every path grounded (zero [NA])", note=note,
        )
        res.steps.append(validator)
        res.critics_summary = {"local": True, "global": True, "factual": True}
        res.accepted = True

    # Final: Mermaid
    res.steps.append(Step(
        number="7" if use_critics else "3", key="mermaid", title="Mermaid conversion",
        role="Render", model_label=util_provider.label, system_prompt=_UTILITY_SYSTEM,
        user_prompt=prompts.build_pipeline_prompt("mermaid", json_structure=json_str),
        output=_MERMAID.get(lang, _MERMAID["en"]), note=note,
    ))
    res.mermaid_code = clean_mermaid_code(_MERMAID.get(lang, _MERMAID["en"]))
    return res
