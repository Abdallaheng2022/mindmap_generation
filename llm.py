"""
llm.py
======
A single OpenAI-compatible client that can talk to every supported backend by
swapping ``base_url`` / ``api_key`` / ``model``. This is what makes the app
deployable on Streamlit Community Cloud: all inference happens on a remote
host, so the app itself never needs a GPU.

Why not run Qwen-2.5-7B locally (as the notebooks do)?
  The notebooks load Qwen with `transformers` + `device_map="auto"`, which
  needs a CUDA GPU and ~16 GB VRAM. Streamlit Cloud's free tier has no GPU and
  ~1 GB RAM, so local loading is not possible there. Instead we call a hosted,
  OpenAI-compatible inference endpoint.

Provider notes (verified June 2026):
  * Gemini Flash is reachable through Google's OpenAI-compatible endpoint
    at https://generativelanguage.googleapis.com/v1beta/openai/ .
  * For the *exact* Qwen2.5-7B-Instruct used in the paper, OpenRouter
    (`qwen/qwen-2.5-7b-instruct`) and DeepInfra (`Qwen/Qwen2.5-7B-Instruct`)
    both serve it over an OpenAI-compatible API.
  * Cerebras is extremely fast and has a generous free tier, but its public
    catalogue currently serves Qwen3 (e.g. `qwen-3-32b`), NOT Qwen2.5-7B. Use
    it only if "a hosted Qwen" is acceptable rather than the exact 2.5-7B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Provider:
    key: str                # internal id
    label: str              # shown in the UI
    family: str             # "gemini" or "qwen"
    base_url: str
    model: str
    secret_name: str        # key looked up in st.secrets / env
    notes: str = ""
    exact_qwen25: bool = False  # True only for providers serving Qwen2.5-7B


# Ordered registry. The first Qwen entry serves the *exact* paper model.
PROVIDERS: dict[str, Provider] = {
    "gemini_flash": Provider(
        key="gemini_flash",
        label="Gemini 2.5 Flash  ·  Google",
        family="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash",
        secret_name="GEMINI_API_KEY",
        notes="Google AI Studio key. Free tier available. (gemini-2.0-flash was "
              "retired June 2026; 2.5-flash is the stable successor.)",
    ),
    "qwen_openrouter": Provider(
        key="qwen_openrouter",
        label="Qwen2.5-7B-Instruct  ·  OpenRouter  (free tier)",
        family="qwen",
        base_url="https://openrouter.ai/api/v1",
        model="qwen/qwen-2.5-7b-instruct:free",
        secret_name="OPENROUTER_API_KEY",
        notes="Exact Qwen2.5-7B from the paper, free variant (':free' = no credits, "
              "rate-limited ~20/min). Drop ':free' if you have OpenRouter credits.",
        exact_qwen25=True,
    ),
    "qwen_deepinfra": Provider(
        key="qwen_deepinfra",
        label="Qwen2.5-7B-Instruct  ·  DeepInfra  (exact paper model)",
        family="qwen",
        base_url="https://api.deepinfra.com/v1/openai",
        model="Qwen/Qwen2.5-7B-Instruct",
        secret_name="DEEPINFRA_API_KEY",
        notes="Serves the exact Qwen2.5-7B-Instruct from the paper.",
        exact_qwen25=True,
    ),
    "qwen_cerebras": Provider(
        key="qwen_cerebras",
        label="Qwen3-32B  ·  Cerebras  (fastest; NOT 2.5-7B)",
        family="qwen",
        base_url="https://api.cerebras.ai/v1",
        model="qwen-3-32b",
        secret_name="CEREBRAS_API_KEY",
        notes="Cerebras serves Qwen3, not Qwen2.5-7B. Very fast, free tier.",
        exact_qwen25=False,
    ),
    "qwen_hf": Provider(
        key="qwen_hf",
        label="Qwen2.5-7B-Instruct  ·  HuggingFace Inference (router)",
        family="qwen",
        base_url="https://router.huggingface.co/v1",
        model="Qwen/Qwen2.5-7B-Instruct:auto",
        secret_name="HF_TOKEN",
        notes="HF Inference Providers router (OpenAI-compatible). ':auto' lets HF "
              "route to whichever provider hosts the model. Free monthly credit, "
              "then pay-as-you-go at provider rates.",
        exact_qwen25=True,
    ),
    "qwen_custom": Provider(
        key="qwen_custom",
        label="Qwen2.5-7B  ·  Custom server (any OpenAI-compatible URL)",
        family="qwen",
        base_url="",                       # the user fills this in the sidebar
        model="Qwen/Qwen2.5-7B-Instruct",
        secret_name="CUSTOM_API_KEY",
        notes="Point at any hosted OpenAI-compatible endpoint that serves the "
              "exact Qwen2.5-7B (DeepInfra, Together, Novita, Hyperbolic, HF, etc.).",
        exact_qwen25=True,
    ),
    "claude_judge": Provider(
        key="claude_judge",
        label="Claude Sonnet 4  ·  Anthropic  (judge)",
        family="claude",
        base_url="https://api.anthropic.com/v1/",
        model="claude-sonnet-4-6",
        secret_name="ANTHROPIC_API_KEY",
        notes="Multilingual LLM-as-judge for the five quality criteria.",
    ),
}


def with_model(provider: "Provider", model: str) -> "Provider":
    """Return a copy of a provider with a different model string."""
    from dataclasses import replace
    return replace(provider, model=model)


def with_endpoint(provider: "Provider", *, base_url: str = None, model: str = None) -> "Provider":
    """Return a copy of a provider with a different base_url and/or model."""
    from dataclasses import replace
    changes = {}
    if base_url:
        changes["base_url"] = base_url
    if model:
        changes["model"] = model
    return replace(provider, **changes) if changes else provider


class LLMError(RuntimeError):
    pass


def chat(
    provider: Provider,
    api_key: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> str:
    """Single chat completion through an OpenAI-compatible endpoint."""
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise LLMError(
            "The `openai` package is required. Add it to requirements.txt."
        ) from exc

    if not api_key:
        raise LLMError(f"No API key provided for {provider.label}.")

    client = OpenAI(base_url=provider.base_url, api_key=api_key, timeout=timeout)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    try:
        resp = client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise LLMError(f"{provider.label} request failed: {exc}") from exc

    if not resp.choices:
        raise LLMError(f"{provider.label} returned no choices.")
    return resp.choices[0].message.content or ""


def resolve_api_key(provider: Provider, secrets: Optional[dict], override: str = "") -> str:
    """Pick an API key: explicit override > st.secrets > environment."""
    import os

    if override:
        return override.strip()
    if secrets:
        val = secrets.get(provider.secret_name)
        if val:
            return str(val).strip()
    return os.environ.get(provider.secret_name, "").strip()
