"""
samples.py
==========
Loads the sample bank used by the lab. Routing rule (baked into each sample):

    source = "Wiki40B"  ->  track = "quality"          (Quality tab)
    source = "SQuAD"     ->  track = "coverage_time"    (Coverage + Comprehension)

The authentic bank lives in `sample_bank.json`. It is populated either by
`fetch_datasets.py` (which pulls the ORIGINAL Wiki40B / SQuAD / Arabic-SQuAD /
SQuAD-TR data on a machine with internet, caps each passage at <= 2000 words,
and samples up to 1000 per language per track) or by the authentic SQuAD
passages already harvested from the research notebooks (EN + TR).

A small CURATED set (clearly labelled, source = "Curated") is bundled so the
Quality tab and demos work before the full bank is fetched. Curated items are
the only authored passages; everything tagged Wiki40B / SQuAD is real data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_BANK_PATH = os.path.join(_HERE, "sample_bank.json")


@dataclass
class Sample:
    id: str
    language: str            # en | tr | ar
    source: str              # "Wiki40B" | "SQuAD" | "Curated"
    track: str               # "quality" | "coverage_time"
    domain: str
    title: str
    text: str
    qa: list = field(default_factory=list)   # [{question, answer}]

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def length_band(self) -> str:
        w = self.words
        if w < 80:
            return "XS (<80)"
        if w < 200:
            return "S (80–200)"
        if w < 500:
            return "M (200–500)"
        if w < 1000:
            return "L (500–1000)"
        return "XL (1000–2000)"


# --------------------------------------------------------------------------
# A few CURATED Wiki-style passages (authored) so Quality works out of the box.
# These are clearly labelled source="Curated" and are NOT presented as dataset.
# --------------------------------------------------------------------------
_CURATED = [
    Sample("cur_en_photosynthesis", "en", "Curated", "quality", "Science", "Photosynthesis",
        "Photosynthesis is the process by which green plants, algae and some "
        "bacteria convert light energy into chemical energy. In plants it takes "
        "place mainly in the chloroplasts, where the pigment chlorophyll absorbs "
        "sunlight. Using water and carbon dioxide, the process produces glucose "
        "and releases oxygen as a by-product. The reactions are commonly divided "
        "into the light-dependent reactions and the Calvin cycle."),
    Sample("cur_tr_fotosentez", "tr", "Curated", "quality", "Bilim", "Fotosentez",
        "Fotosentez, yeşil bitkilerin ve bazı bakterilerin ışık enerjisini "
        "kimyasal enerjiye dönüştürdüğü süreçtir. Bitkilerde çoğunlukla "
        "kloroplastlarda gerçekleşir ve klorofil pigmenti güneş ışığını soğurur. "
        "Su ile karbondioksit kullanılarak glikoz üretilir ve yan ürün olarak "
        "oksijen açığa çıkar."),
    Sample("cur_ar_ibnsina", "ar", "Curated", "quality", "سيرة", "ابن سينا",
        "ابن سينا طبيب وفيلسوف عاش في القرنين العاشر والحادي عشر الميلاديين. "
        "اشتهر بكتابه 'القانون في الطب' الذي ظلّ مرجعًا في الجامعات لقرون. "
        "أسهم أيضًا في الفلسفة والمنطق وعلم الفلك."),
    # one diacritised Arabic passage (tashkīl) as requested
    Sample("cur_ar_nile_diac", "ar", "Curated", "quality", "جغرافيا", "نهر النيل (مُشَكَّل)",
        "نَهْرُ النِّيلِ هُوَ أَطْوَلُ أَنْهَارِ العَالَمِ، وَيَجْرِي فِي شَمَالِ "
        "شَرْقِ إِفْرِيقْيَا. يَبْلُغُ طُولُهُ نَحْوَ سِتَّةِ آلَافٍ وَسِتِّمِائَةٍ "
        "وَخَمْسِينَ كِيلُومِتْرًا، وَيَصُبُّ فِي البَحْرِ الأَبْيَضِ المُتَوَسِّطِ. "
        "كَانَ النِّيلُ مِحْوَرًا لِقِيَامِ الحَضَارَةِ المِصْرِيَّةِ القَدِيمَةِ."),
]


def _load_bank() -> list:
    out = []
    if os.path.exists(_BANK_PATH):
        try:
            raw = json.load(open(_BANK_PATH, encoding="utf-8"))
            for r in raw:
                out.append(Sample(
                    id=r["id"], language=r["language"], source=r["source"],
                    track=r.get("track", "coverage_time" if r["source"] == "SQuAD" else "quality"),
                    domain=r.get("domain", r["source"]), title=r.get("title", r["id"]),
                    text=r["text"], qa=r.get("qa", [])))
        except Exception:
            pass
    return out


SAMPLES: list = _CURATED + _load_bank()


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------
def sources() -> list:
    seen, out = set(), []
    for s in SAMPLES:
        if s.source not in seen:
            seen.add(s.source)
            out.append(s.source)
    return out


def filter_samples(language=None, source=None, track=None,
                   min_words=0, max_words=10 ** 9) -> list:
    res = []
    for s in SAMPLES:
        if language and s.language != language:
            continue
        if source and s.source != source:
            continue
        if track and s.track != track:
            continue
        if not (min_words <= s.words <= max_words):
            continue
        res.append(s)
    return res


def counts() -> dict:
    """{language: {source: n}} summary for display."""
    out: dict = {}
    for s in SAMPLES:
        out.setdefault(s.language, {}).setdefault(s.source, 0)
        out[s.language][s.source] += 1
    return out


def get(sample_id: str):
    for s in SAMPLES:
        if s.id == sample_id:
            return s
    return None
