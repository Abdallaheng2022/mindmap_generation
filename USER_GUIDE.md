# User Guide — Multilingual Mind-Map Generator

A complete walkthrough for installing, running, presenting, and troubleshooting
the app. It accompanies the paper *“Generating Mind Maps from Textual Content:
Multilingual Text Processing and Evaluation Metrics with Large Language Models.”*

---

## 1. What the app does

You paste a paragraph, choose a model and a language, and the app turns the text
into a **mind map**, showing every step along the way. You can run it two ways:

- **Without the three critics** — the fast path: generate → repair → draw.
- **With the three critics** — adds a quality gate (Local, Global, Factual) that
  accepts or rejects the mind map.

For each step the app shows the **exact prompt** sent to the model and the
**model’s output**, including the language-specific generation prompt (English,
Turkish, Arabic) and each critic prompt on its own. The final mind map is drawn
as a Mermaid diagram. Every run is kept so you can compare settings side by side.

---

## 2. Two modes: Demo and Live

The app always works, even with no setup.

**Demo mode** (no API key): the full pipeline renders using authentic example
outputs taken from the research notebooks. The diagram, critic verdicts, and
prompts all show. This is the safe mode for a presentation — nothing can fail
mid-demo. Steps are labelled “Demo output”.

**Live mode** (a key is configured): the same steps run for real against the
models you chose.

A pill next to the title (`DEMO` / `LIVE`) and a banner in the sidebar tell you
which mode you are in. The app switches to Live automatically as soon as the
required key(s) are present.

---

## 3. Installation and first run (local)

You need Python 3.10 or newer.

```bash
# 1. unzip the project, then from inside the folder:
pip install -r requirements.txt

# 2. start the app
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. The app starts in **Demo mode**,
so you can click **Generate** immediately and see the whole flow without any key.

---

## 4. Getting API keys (for Live mode)

You only need the key(s) for the provider(s) you actually use.

| Provider | Model used | Secret name | Where to get the key |
|---|---|---|---|
| Google AI Studio | `gemini-2.0-flash` | `GEMINI_API_KEY` | https://aistudio.google.com/apikey |
| OpenRouter | `qwen/qwen-2.5-7b-instruct` (exact paper model) | `OPENROUTER_API_KEY` | https://openrouter.ai/keys |
| DeepInfra | `Qwen/Qwen2.5-7B-Instruct` (exact paper model) | `DEEPINFRA_API_KEY` | https://deepinfra.com/dash/api_keys |
| Cerebras | `qwen-3-32b` (fast, free tier — **not** 2.5-7B) | `CEREBRAS_API_KEY` | https://cloud.cerebras.ai |

**Recommended minimum to show both models:** a `GEMINI_API_KEY` (free tier) plus
one Qwen key. For the exact Qwen2.5-7B from the paper, use **OpenRouter** or
**DeepInfra**.

### Where to put the keys

You have two options.

**Option A — paste in the sidebar (quickest).** Open the sidebar, scroll to *API
keys*, and paste the key into the matching box. It is used only for that browser
session and is never written to disk.

**Option B — secrets file (persistent, recommended for deployment).** Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in the
keys you have:

```toml
GEMINI_API_KEY = "..."
OPENROUTER_API_KEY = "..."
```

The `.gitignore` already excludes `secrets.toml`, so real keys won’t be committed.

---

## 5. The controls (sidebar)

Everything you configure is in the left sidebar.

- **Language** — English, Turkish, or Arabic. Changing it reloads the matching
  example paragraph; Arabic switches the input box to right-to-left.
- **Generation model** — Gemini 2.0 Flash or Qwen2.5-7B-Instruct. This is the
  model that produces the first mind map from your text.
- **Qwen host** (only if you picked Qwen) — OpenRouter or DeepInfra serve the
  exact Qwen2.5-7B; Cerebras serves the faster Qwen3-32B instead.
- **Use the three critics** — the main toggle. On = full quality gate; off =
  generate → repair → draw only.
- **Advanced → Utility model** — which model runs the helper steps (JSON repair,
  the critics, and the Mermaid conversion). It defaults to Qwen because the paper
  runs these on Qwen; you can change it (e.g. to Gemini) if you like.
- **API keys** — status dots (🟢 found / ⚪️ missing) and paste boxes.
- **Hosting & deployment notes** — a quick reference, summarised in §9 below.

---

## 6. Running the pipeline (main panel)

1. Edit the **Input paragraph** (or keep the example). Use **↺ Reset example** to
   restore the sample for the current language.
2. Click **▶ Generate mind map**. The button label tells you whether it will run
   *with* or *without* the three critics, based on your toggle.
3. In Live mode a progress bar tracks the steps; in Demo mode the result appears
   immediately.

The result appears in the **Runs** section and stays there.

### What each run card shows

- A header with the run number and **configuration chips** (mode, model,
  language, critics on/off).
- The **quality gate**: `✓ ACCEPTED` or `✗ REJECTED` with the three critic
  verdicts (or “No critics” when the toggle is off).
- The **mind map** drawn as a Mermaid diagram, with **⬇ .mmd** and **⬇ .json**
  download buttons.
- **Detailed steps** (expander): every step with two tabs — *Prompt sent*
  (system + user) and *Model output*. This is where you show a professor the
  exact Arabic / Turkish / English generation prompt and each critic prompt.
- **Input paragraph** (expander): the exact text used for that run.
- An **✕** button to remove just that run.

---

## 7. Comparing runs side by side

Every run is kept, which is the point: you can compare configurations directly.

- Run once **with critics**, then flip the toggle off and run again **without** —
  the two appear next to each other.
- Or switch the model from Gemini to Qwen (same text, same language) and run
  again to compare models.
- Or change the language to compare behaviour across English / Turkish / Arabic.

Controls in the **Runs** toolbar:

- **Per row (1 / 2 / 3)** — how many run cards sit side by side.
- **🗑 Clear all** — remove every run and start fresh.
- When you have **two or more runs**, a compact **comparison table** appears
  above the cards, summarising for each run: mode, model, language, whether
  critics were used, the Local / Global / Factual verdicts, and the final
  Accepted / Rejected result.

> Runs live in the browser session. They persist while you work and compare, but
> a full page refresh starts a new session. Use the download buttons to keep any
> diagram or JSON you want to save.

---

## 8. Understanding the pipeline and the three critics

**Without the three critics**

1. **Generate** — the language system prompt (EN/TR/AR) plus your text go to the
   chosen model, which returns a mind map in JSON.
2. **Repair JSON** — a utility model fixes any malformed JSON.
3. **Mermaid** — the JSON is converted to a Mermaid mind-map diagram.

**With the three critics**

1. **Generate** → 2. **Repair JSON** → 3. **Extract paths** (root-to-leaf paths,
   shared by the global and factual critics), then:

4. **Critic 1 · Local Structure** — are all the leaf values *specific* (names,
   dates, places) rather than generic labels like “Background” or “Details”?
   Passes on `Answer: yes`.
5. **Critic 2 · Global Structure** — turned into a table of contents, do the
   paths read as *informative* sentences rather than vague headings? Passes on
   `Useful: yes`.
6. **Critic 3 · Factual** — three sub-steps:
   - **6a** bullet points: the source text is reduced to numbered sentences.
   - **6b** attribution: each path is linked to a supporting sentence, or marked
     `[NA]` if unsupported.
   - **6c** validator: with **zero tolerance** for `[NA]`, it returns `ACCEPT`
     or `REJECT`.
7. **AND gate** — the mind map is **accepted only if all three critics pass**.
8. **Mermaid** — the final diagram is drawn.

So a mind map shown with a green `✓ ACCEPTED` banner has passed all three checks;
a red `✗ REJECTED` means at least one critic failed (the table and chips show
which).

---

## 9. Deploying on Streamlit Community Cloud

1. Push the project folder to a GitHub repository.
2. Go to https://share.streamlit.io, sign in, and create a new app pointing at
   `app.py` in your repo.
3. Open **Settings → Secrets** and paste your keys in TOML format (the same
   content as `.streamlit/secrets.toml.example`).
4. Deploy. The public URL is what you share with your professors.

### Why the app calls hosted models instead of loading Qwen locally

Streamlit Cloud’s free tier has **no GPU**, so Qwen2.5-7B cannot be loaded
in-process the way the research notebooks do (`transformers` with
`device_map="auto"` needs CUDA). This app instead calls a **hosted,
OpenAI-compatible endpoint**, which needs no GPU and runs fine on the free tier.

- **Exact Qwen2.5-7B-Instruct** → OpenRouter or DeepInfra.
- **Cerebras** → fastest, free tier, but serves **Qwen3** (`qwen-3-32b`), not
  Qwen2.5-7B. Use it only if a hosted Qwen3 is acceptable for your comparison.
- **Gemini 2.0 Flash** → Google’s OpenAI-compatible endpoint.

---

## 10. Reading any prompt directly

At the bottom of the page, the **📚 Prompt library** expander lets you read every
template independently of a run:

- **Generation prompts** for English, Turkish, and Arabic.
- **Critic prompts** — Local, Global, and the three Factual prompts.
- **Other pipeline prompts** — JSON repair, path extraction, Mermaid conversion.

These are reproduced verbatim from the research notebooks, so they are safe to
cite or screenshot for a talk.

---

## 11. Tips for presenting to professors

- Start in **Demo mode** so the first click always works, then add a key to show
  it running live on a fresh paragraph.
- To make the three critics tangible, run the **same paragraph with and without**
  the critics and put the two cards side by side; the comparison table makes the
  effect of the quality gate obvious at a glance.
- To show the **multilingual** angle, run the same kind of text in English,
  Turkish, and Arabic and set *Per row* to 3.
- Open a step’s **Prompt sent** tab to show the exact language-specific prompt;
  open a critic step to show how acceptance is decided.
- Use **⬇ .json** / **⬇ .mmd** to save artefacts for slides, or paste the Mermaid
  source into https://mermaid.live to export an image.

---

## 12. Troubleshooting

**The app stays in Demo mode after I added a key.** Check the status dot next to
the secret name in the sidebar. The key is required for both the *generation*
model and the *utility* model; if they use different providers, both keys must be
present. Re-check for typos or trailing spaces.

**A step shows an “error”.** The pipeline never crashes — it records the error on
that step and continues. Open the step to read the message. Common causes: an
invalid or rate-limited key, or a model name not available on that provider.

**The Mermaid diagram doesn’t render.** Mermaid loads from a CDN, so the browser
needs internet access. If a model returns unusable Mermaid, the app falls back to
a deterministic JSON→Mermaid conversion and notes this on the step. You can also
open *View Mermaid source* / the `.mmd` download and paste it into mermaid.live.

**Cerebras gives a model-not-found error for Qwen2.5.** Expected — Cerebras
serves Qwen3, not Qwen2.5-7B. Pick OpenRouter or DeepInfra for the exact paper
model, or accept `qwen-3-32b` on Cerebras.

**Rate limits / quota.** Free tiers cap requests. The full critic path makes
several model calls per run, so heavy comparison can hit limits; wait, switch
provider, or reduce how many runs you fire in quick succession.

**Arabic text looks left-aligned somewhere.** The input box and the diagram are
set to right-to-left for Arabic. Inside the *Prompt sent* tabs the raw prompt is
shown as-is (mixed direction is normal there).

**My runs disappeared.** Runs are stored per browser session; a page refresh
starts fresh. Download anything you need to keep before refreshing.

**Changing the model name.** Model strings live in `llm.py` under `PROVIDERS`
(e.g. swap `gemini-2.0-flash` for a newer Gemini, or point a provider at a
different Qwen build). Edit there and restart.

---

## 13. Quick reference

| I want to… | Do this |
|---|---|
| See the whole flow with no setup | Just click **Generate** (Demo mode) |
| Run for real | Add a key in the sidebar, then **Generate** |
| Compare with vs without critics | Run, flip the toggle, run again |
| Compare two models | Run with Gemini, switch to Qwen, run again |
| Show a professor a prompt | Open a step → **Prompt sent**, or the Prompt library |
| Save a diagram | **⬇ .mmd** / **⬇ .json**, or open in mermaid.live |
| Start over | **🗑 Clear all** |

---

## 14. Evaluation Lab (tabs ② – ④)

The app is organised into tabs. Tab ① **Generate & Compare** is the generator
described above; the remaining tabs are the thesis evaluation tracks. All of
them run live when the relevant key is present and fall back to authentic demo
data otherwise, and all accumulate results into a running percentage.

### Generating for both models and from dataset samples (tab ①)

- **Generation model** in the sidebar now has a third option, **Both (side by
  side)**. With it selected, one click produces two runs — Gemini 2.0 Flash and
  Qwen — which appear next to each other for direct comparison.
- **Dataset sample picker**: choose a **Wiki40B** or **SQuAD** sample (filtered
  to the current language and labelled by domain, length band, and word count),
  press **Load sample**, then **Generate**. SQuAD samples carry gold questions,
  which tabs ③ and ④ need. You can still type your own text.

### Tab ② — Quality (five binary criteria + Claude judge)

The five criteria from the thesis: **SC** Structural Coherence, **SA** Semantic
Accuracy, **CC** Concept Centrality, **BC** Branch Completeness, **GC** Graph
Clarity. Each is binary (Good = 1 / Bad = 0).

1. Pick a run to evaluate.
2. **Your rating (human):** toggle the five criteria. A map's quality% =
   (# Good ÷ 5) × 100 updates live. Press **Save** to add it to the record.
3. **Claude judge:** press **Run Claude judge** to have a multilingual
   LLM-as-judge rate the same five criteria with reasons; **Save** to record it.
4. The **Running quality results** panel shows the mean quality% across all saved
   maps for the human track and the judge track, a per-criterion % Good table for
   each, and the thesis reference numbers for the current language.

### Tab ③ — Semantic Coverage (Auto-QA / Human-QA)

Coverage = correct answers ÷ total questions, with answers produced from the
**mind map only** and judged equivalent to the gold answer (deterministic
substring / token-overlap / numeric match, backed by an LLM equivalence judge).

- **Auto-QA**: questions are generated from the source passage, then answered
  from the mind map. Works on any text.
- **Human-QA**: uses the SQuAD-style gold questions, so the run must come from a
  SQuAD sample.

Choose the track, set how many questions to evaluate, and run. A per-question
table shows the question, gold answer, the answer derived from the mind map, the
match verdict, and which method decided it. **Save** to accumulate; the panel
shows the running Auto-QA and Human-QA coverage and the thesis reference
averages.

### Tab ④ — Comprehension Time

A timed study over the three context variants from the paper: **⟨q, s⟩** mind map
only, **⟨q, t⟩** text only, and **⟨q, s+t⟩** both. Requires a SQuAD sample.

1. Pick a question and a context variant; the relevant context is shown.
2. Press **Start timer**, read, type your answer, then **Stop & submit**. The
   elapsed seconds and answer correctness (vs the gold answer) are recorded.
3. The panel shows mean seconds per variant and the headline metric — the **%
   reduction of ⟨q,s⟩ versus ⟨q,t⟩** — alongside the thesis reference reduction.

### Notes

- Records live in the browser session; each tab has a **Reset records** button,
  and a page refresh clears them.
- The **Claude judge** uses Anthropic (`ANTHROPIC_API_KEY`, model
  `claude-sonnet-4-6` by default, editable under sidebar → Advanced). Without a
  key it runs in demo.
- All evaluation prompts — Auto-QA, answering, equivalence, and the judge
  rubric — are in the **Prompt library** tab.

---

## 15. Datasets and routing

The lab is driven by a **sample bank** of authentic passages (`sample_bank.json`),
not by text the model writes. Each passage is tagged with its dataset, and the
dataset decides which evaluation it feeds:

| Dataset | Track | Tab it feeds |
|---|---|---|
| **Wiki40B** | quality | ② Quality (five criteria + Claude judge) |
| **SQuAD** (incl. Arabic-SQuAD, SQuAD-TR) | coverage_time | ③ Semantic Coverage and ④ Comprehension Time |

So when you generate from a **Wiki40B** passage the resulting map goes to
**Quality**; when you generate from a **SQuAD** passage the map is **not** sent to
Quality — it goes through normal generation and then carries its own gold
questions to **Semantic Coverage** and **Comprehension Time**. Tab ① shows, for
the selected sample, exactly where the map will route before you generate.

In tab ① the picker filters by **dataset**, **language**, and **maximum length**
(up to 2000 words). The three evaluation tabs only list the maps that belong to
them, so a SQuAD map never appears under Quality and a Wiki40B map never appears
under Coverage or Time.

### Loading the full data (1000 per language)

The shipped bank already contains authentic SQuAD passages for English and
Turkish (harvested from the research notebooks, with their real gold questions).
One command downloads everything (no files needed) and populates the bank to the full target — Wiki40B for all three languages and
SQuAD / Arabic-SQuAD / SQuAD-TR, up to 1000 passages per language per track, each
≤ 2000 words — run the loader once on a machine with internet (and/or your local
SQuAD files):

```bash
pip install datasets
python fetch_datasets.py --online --per-lang 1000 --max-words 2000
```

The script reads the **original** datasets (nothing is generated), caps each
passage at 2000 words, samples up to 1000 per language per track, preserves the
passages already in the bank, and rewrites `sample_bank.json`. The app picks up
the new bank on its next start. Local SQuAD files use the same schema your
notebooks use (`id, context, question, answer_text, language, title`; JSON or
JSONL), **and** the official nested SQuAD JSON (`data → paragraphs → qas`) used by
Arabic-SQuAD / ARCD / SQuAD-TR. Use real absolute paths — the example paths are
placeholders; missing files are reported and skipped rather than crashing.
