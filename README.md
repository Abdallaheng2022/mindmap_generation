# Multilingual Mind-Map Generator

An interactive walkthrough of the pipeline from the paper *“Generating Mind Maps
from Textual Content: Multilingual Text Processing and Evaluation Metrics with
Large Language Models.”*

Paste a paragraph, pick a model and language, and watch it become a verified
mind map — **with** or **without** the three critics. Every step shows the exact
prompt sent to the model and the model’s output, and the final mind map is drawn
as a Mermaid diagram.

* **Models:** Gemini 2.5 Flash · Qwen2.5-7B-Instruct
* **Languages:** English · Turkish · Arabic (RTL aware)
* **Three critics:** Local Structure · Global Structure · Factual
* **Evaluation Lab:** four tabs —
  * ① Generate & Compare — one model or **both side by side**, with a
    Wiki40B / SQuAD sample picker; every run kept for comparison.
  * ② Quality — the five binary criteria (SC SA CC BC GC) rated by you and by a
    multilingual **Claude judge**, accumulating to a running quality %.
  * ③ Semantic Coverage — **Auto-QA** and **Human-QA**, answers from the mind map
    only, accumulating to a running coverage %.
  * ④ Comprehension Time — timed ⟨q,s⟩ / ⟨q,t⟩ / ⟨q,s+t⟩ study with % reduction.
* **Dataset routing:** Wiki40B → Quality; SQuAD → Semantic Coverage + Comprehension Time. Authentic sample bank (`sample_bank.json`); fill to 1000/language with one command — `python fetch_datasets.py --online` — which downloads the original Wiki40B / SQuAD / Arabic-SQuAD / SQuAD-TR directly (≤2000 words, no local files needed).

* **Modes:** *Live* (runs the models) and *Demo* (no key needed — shows the full
  flow with authentic example outputs so it never breaks during a presentation)
* **Side-by-side comparison:** every run is kept. Change the model, language or
  the critics toggle and run again — runs render next to each other with a
  compact comparison table (Local / Global / Factual / Accepted). Pick 1–3 per
  row; remove individual runs or clear all.

---

## The pipeline

**Without the three critics**

1. **Generate** — language system prompt (EN/TR/AR) + your text → chosen model
2. **Repair JSON** → valid mind-map JSON
3. **Mermaid** → render the diagram

**With the three critics**

1. Generate → 2. Repair JSON → 3. Extract paths
4. **Critic 1 · Local Structure** — are all leaf values *specific* (not generic)?
5. **Critic 2 · Global Structure** — do the paths/TOC form *informative* sentences?
6. **Critic 3 · Factual** — bullet points → attribute each path to a source
   sentence → validate (zero `[NA]` tolerance → ACCEPT/REJECT)
7. **AND gate** — accepted only if all three critics pass
8. **Mermaid** → render

The prompts in `prompts.py` are reproduced **verbatim** from the research
notebooks; only a small builder API was added.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in Demo mode immediately. To run live, add a key (see below).

## API keys

Put keys in `.streamlit/secrets.toml` (copy `.streamlit/secrets.toml.example`)
or paste them into the sidebar for the current session. You only need the key(s)
for the provider(s) you use.

| Provider | Model | Secret | Notes |
|---|---|---|---|
| Google | `gemini-2.5-flash` | `GEMINI_API_KEY` | OpenAI-compatible endpoint |
| OpenRouter | `qwen/qwen-2.5-7b-instruct:free` | `OPENROUTER_API_KEY` | **exact paper model** |
| DeepInfra | `Qwen/Qwen2.5-7B-Instruct` | `DEEPINFRA_API_KEY` | **exact paper model** |
| Cerebras | `qwen-3-32b` | `CEREBRAS_API_KEY` | fastest, free tier — **Qwen3, not 2.5-7B** |
| Anthropic | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | Claude judge for the five quality criteria |

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing
   at `app.py`.
3. Open **Settings → Secrets** and paste your keys (TOML format, same as
   `secrets.toml.example`).

### About running Qwen

Streamlit Cloud has **no GPU**, so Qwen2.5-7B cannot be loaded locally the way
the notebooks do (`transformers` + `device_map="auto"` needs CUDA). This app
calls a **hosted, OpenAI-compatible endpoint** instead, so it runs fine on the
free tier.

* For the **exact** Qwen2.5-7B-Instruct from the paper, use **OpenRouter** or
  **DeepInfra**.
* **Cerebras** is the fastest and has a free tier, but its public catalogue
  currently serves **Qwen3** (`qwen-3-32b`), not Qwen2.5-7B. Choose it only if a
  hosted Qwen3 is acceptable.

---

## Files

```
app.py            Streamlit UI
pipeline.py       Orchestration, parsing, Mermaid cleaning
llm.py            Provider registry + OpenAI-compatible client
prompts.py        Verbatim research prompts + builder API
demo.py           Demo-mode data and result builder
requirements.txt
.streamlit/config.toml          Theme
.streamlit/secrets.toml.example Where keys go
```
