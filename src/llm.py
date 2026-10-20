"""Which LLM backend the generator and the RAGAS judge talk to.

We support two backends behind one interface:

  * ``openai``  (default): the hosted OpenAI API (GPT-3.5-turbo) — the generator and
    RAGAS judge agreed for this project and used in the committed results. Needs a key
    in ``.env`` (per-member; never committed).
  * ``ollama``: a local model served by Ollama. No API key, no cost, everything runs on
    the machine. Ollama exposes an OpenAI-compatible endpoint at ``/v1``, so the same
    OpenAI client code works against it unchanged — handy for a local run or a possible
    extra open-model comparison.

Pick one with ``--backend`` on the scripts, or set ``LLM_BACKEND`` in the environment.
For RAGAS we deliberately use a local sentence-transformers embedding model instead of
an OpenAI embedding, so only the judge LLM (not embeddings) depends on the backend.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Backend:
    name: str                 # "ollama" | "openai"
    base_url: str | None      # None means the real OpenAI endpoint
    api_key: str              # Ollama ignores this; any non-empty string works
    default_model: str


def get_backend(name: str | None = None) -> Backend:
    """Resolve a backend by name, falling back to $LLM_BACKEND then openai.

    Default is ``openai`` (GPT-3.5-turbo) — the generator/judge agreed for this
    project and used in the committed results. ``ollama`` remains available via
    ``--backend ollama`` / ``LLM_BACKEND=ollama`` for a local, no-cost run or a
    possible extra open-model comparison.
    """
    name = name or os.environ.get("LLM_BACKEND", "openai")
    if name == "ollama":
        return Backend("ollama", OLLAMA_BASE_URL, "ollama", "llama3.2")
    if name == "openai":
        return Backend("openai", None, os.environ.get("OPENAI_API_KEY", ""), "gpt-3.5-turbo")
    raise ValueError(f"unknown backend: {name!r} (use 'ollama' or 'openai')")


def make_client(backend: Backend):
    """An OpenAI-python client wired to the chosen backend."""
    from openai import OpenAI

    if backend.base_url:  # Ollama
        return OpenAI(base_url=backend.base_url, api_key=backend.api_key)
    return OpenAI(api_key=backend.api_key)


def make_ragas_llm(backend: Backend, model: str):
    """A LangChain chat model RAGAS can use as its judge."""
    from langchain_openai import ChatOpenAI

    kwargs = {"model": model, "temperature": 0.0}
    if backend.base_url:  # Ollama's OpenAI-compatible endpoint
        kwargs["base_url"] = backend.base_url
        kwargs["api_key"] = backend.api_key
    else:
        kwargs["api_key"] = backend.api_key
    return ChatOpenAI(**kwargs)


def make_local_embeddings():
    """Local sentence-transformers embeddings for RAGAS (no API key needed)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
