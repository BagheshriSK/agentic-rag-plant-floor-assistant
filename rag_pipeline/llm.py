"""
LLM client factory.

If OPENAI_API_KEY is set  →  uses the OpenAI-compatible API (no local models loaded).
Otherwise                 →  falls back to lightweight HuggingFace models:

  Classifier / Router : facebook/bart-large-mnli   (zero-shot classification, ~1.6 GB)
  Generator           : google/flan-t5-base         (seq2seq via text-generation, ~250 MB)
  Evaluator           : rule-based heuristics       (no model needed)

All HF pipelines are loaded LAZILY on first use — import never fails.

Override defaults via env vars:
  HF_CLASSIFIER_MODEL   (default: facebook/bart-large-mnli)
  HF_GENERATOR_MODEL    (default: google/flan-t5-base)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from rag_pipeline.config import settings

# ---------------------------------------------------------------------------
# Shared response shim (mirrors openai.chat.completions response shape)
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content

class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)

class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


# ---------------------------------------------------------------------------
# Zero-shot classifier shim  (bart-large-mnli)
# Used by: classifier node, router node
# ---------------------------------------------------------------------------

_CLASSIFIER_LABELS = [
    "safety_procedures",
    "maintenance_manuals",
    "quality_control_standards",
    "unknown",
]

_ROUTER_LABELS = [
    "safety_procedures",
    "maintenance_manuals",
    "quality_control_standards",
]


class _ZeroShotCompletions:
    """Lazy-loaded zero-shot-classification pipeline."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._pipe: Any = None  # loaded on first use

    def _load(self) -> None:
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline  # type: ignore
            self._pipe = hf_pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=-1,
            )

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 32,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> _Resp:
        self._load()
        query = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        system_text = next(
            (m["content"] for m in messages if m.get("role") == "system"), ""
        )
        candidates = _ROUTER_LABELS if "routing" in system_text.lower() else _CLASSIFIER_LABELS
        result = self._pipe(query, candidate_labels=candidates)
        best_label: str = result["labels"][0]
        return _Resp(best_label)


class _ZeroShotClient:
    def __init__(self, model_name: str) -> None:
        self._completions = _ZeroShotCompletions(model_name)
        self.chat = type("_Chat", (), {"completions": self._completions})()


# ---------------------------------------------------------------------------
# Seq2Seq / text-generation shim  (flan-t5-base)
# Used by: generator node
# flan-t5 is a seq2seq model but transformers >= 4.40 exposes it via
# "text-generation" with the T5 tokenizer — we use AutoModelForSeq2SeqLM
# directly to avoid the pipeline task name issue.
# ---------------------------------------------------------------------------

class _Seq2SeqCompletions:
    """Lazy-loaded flan-t5 generator."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is None:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> _Resp:
        self._load()
        prompt = self._build_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
        raw: str = self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return _Resp(raw)

    @staticmethod
    def _build_prompt(messages: list[dict]) -> str:
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return f"{system}\n\n{user}" if system else user


class _Seq2SeqClient:
    def __init__(self, model_name: str) -> None:
        self._completions = _Seq2SeqCompletions(model_name)
        self.chat = type("_Chat", (), {"completions": self._completions})()


# ---------------------------------------------------------------------------
# Rule-based judge shim  (no model)
# Used by: evaluator node
# ---------------------------------------------------------------------------

class _RuleBasedCompletions:
    """Heuristic eval scores — no model required."""

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
        response_format: dict | None = None,
        **kwargs: Any,
    ) -> _Resp:
        user_text = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        question, context, answer = self._parse(user_text)
        return _Resp(json.dumps(self._score(question, context, answer)))

    @staticmethod
    def _parse(text: str) -> tuple[str, str, str]:
        q = re.search(r"Question:\s*(.*?)(?:\n|Context:)", text, re.DOTALL)
        c = re.search(r"Context:\s*(.*?)(?:\n\nAnswer:)", text, re.DOTALL)
        a = re.search(r"Answer:\s*(.*?)$", text, re.DOTALL)
        return (
            q.group(1).strip() if q else "",
            c.group(1).strip() if c else "",
            a.group(1).strip() if a else "",
        )

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        ta, tb = set(a.lower().split()), set(b.lower().split())
        return len(ta & tb) / max(len(ta | tb), 1)

    def _score(self, question: str, context: str, answer: str) -> dict:
        ar = min(1.0, self._overlap(answer, question) * 3)
        cr = min(1.0, self._overlap(context, question) * 2)
        gr = min(1.0, self._overlap(answer, context) * 2)
        cp = min(1.0, len(answer.split()) / 20)
        return {
            "answer_relevance":  round(max(ar, 0.4), 2),
            "context_relevance": round(max(cr, 0.4), 2),
            "groundedness":      round(max(gr, 0.4), 2),
            "completeness":      round(max(cp, 0.3), 2),
        }


class _RuleBasedClient:
    def __init__(self) -> None:
        self.chat = type("_Chat", (), {"completions": _RuleBasedCompletions()})()


# ---------------------------------------------------------------------------
# Public factories — return a shim or real OpenAI client
# ---------------------------------------------------------------------------

_DEFAULT_CLASSIFIER = "facebook/bart-large-mnli"
_DEFAULT_GENERATOR  = "google/flan-t5-base"


def _openai_client() -> Any:
    """Return an OpenAI-compatible client: Groq > OpenAI > None."""
    from openai import OpenAI
    if settings.groq_api_key:
        return OpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
    if settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    return None


def get_chat_client() -> Any:
    """Generator node client."""
    client = _openai_client()
    if client:
        return client
    model = os.getenv("HF_GENERATOR_MODEL", _DEFAULT_GENERATOR)
    return _Seq2SeqClient(model)


def get_classifier_client() -> Any:
    """Classifier and router node client."""
    client = _openai_client()
    if client:
        return client
    model = os.getenv("HF_CLASSIFIER_MODEL", _DEFAULT_CLASSIFIER)
    return _ZeroShotClient(model)


def get_judge_client() -> Any:
    """Evaluator node client."""
    client = _openai_client()
    if client:
        return client
    return _RuleBasedClient()
