import os
import threading
from urllib.parse import urlparse
from typing import Any, Optional

import httpx
import ollama
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .retrieval import (
    upsert_image_dataset,
    upsert_text_dataset,
    hybrid_search_image_query,
)
from .retrieval.utils import (
    mask_numbers,
    strip_markdown,
    build_query_text_from_auto,
    build_rag_context,
)
from .prompts import (
    PROMPT_DRAFT_FROM_IMAGE,
    PROMPT_DOMAIN_REWRITE,
    AUTO_TEXT_COMPRESS_NOTE,
    RAG_TEMPLATE,
)

load_dotenv()

# ── Module-level caches ──────────────────────────────────────────────
_LLM_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATASET_JSON = os.getenv(
    "LLM_DATASET_JSON", os.path.join(_LLM_ROOT, "dataset.json"),
)
_runtime_cache: dict = {}
_runtime_locks: dict[str, threading.Lock] = {}
_runtime_locks_guard = threading.Lock()
_PROVIDER_ALIASES = {
    "chatgpt": "openai",
    "gemini": "google_genai",
    "google": "google_genai",
    "google_genai": "google_genai",
    "google-genai": "google_genai",
}
_DEFAULT_PROVIDER = "ollama"
_CLOUD_PROVIDERS = {"openai", "google_genai"}
_DEFAULT_MODELS = {
    "ollama": "ministral-3:14b",
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-2.0-flash",
}
_PROVIDER_PACKAGES = {
    "openai": "langchain-openai",
    "google_genai": "langchain-google-genai",
}


# ── Errors ───────────────────────────────────────────────────────────

class VisionLLMConfigError(RuntimeError):
    """Raised when LLM configuration is invalid."""


class VisionLLMServiceUnavailableError(RuntimeError):
    """Raised when the LLM service cannot be reached."""


def _env_float(name: str, default: float, *, minimum: float) -> float:
    raw = str(os.getenv(name, default) or "").strip()
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(float(minimum), value)


def _clean_env_text(raw_value: Optional[str]) -> str:
    return str(raw_value or "").strip().strip('"').strip("'")


def _normalize_base_url(
    raw_value: Optional[str],
    *,
    default: str = "",
    env_name: str = "LLM_BASE_URL",
) -> str:
    """Normalize and validate a base URL from environment."""
    base_url = _clean_env_text(raw_value)
    if not base_url:
        base_url = default

    # Common typo recovery: "http:/host:port" -> "http://host:port"
    if base_url.startswith("http:/") and not base_url.startswith("http://"):
        base_url = "http://" + base_url[len("http:/"):].lstrip("/")
    elif base_url.startswith("https:/") and not base_url.startswith("https://"):
        base_url = "https://" + base_url[len("https:/"):].lstrip("/")

    if not base_url:
        return ""

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VisionLLMConfigError(
            f"Invalid {env_name}={raw_value!r}. Expected format: http://host:port"
        )
    return base_url.rstrip("/")


def _normalize_model_provider(raw_value: Optional[str]) -> str:
    provider = _clean_env_text(raw_value).lower().replace("-", "_")
    if not provider:
        return _DEFAULT_PROVIDER
    return _PROVIDER_ALIASES.get(provider, provider)


def _resolve_api_key(provider: str) -> str:
    if provider == "openai":
        return _clean_env_text(os.getenv("OPENAI_API_KEY"))
    if provider == "google_genai":
        return _clean_env_text(os.getenv("GOOGLE_API_KEY"))
    return ""


def _resolve_model_name(active_provider: str, requested_provider: str) -> str:
    provider_model_env = {
        "ollama": "LLM_OLLAMA_MODEL",
        "openai": "LLM_OPENAI_MODEL",
        "google_genai": "LLM_GOOGLE_MODEL",
    }.get(active_provider)
    provider_model = _clean_env_text(os.getenv(provider_model_env)) if provider_model_env else ""
    if provider_model:
        return provider_model

    if active_provider == "google_genai":
        alt_google_model = _clean_env_text(os.getenv("LLM_GOOGLE_GENAI_MODEL"))
        if alt_google_model:
            return alt_google_model

    return _DEFAULT_MODELS[active_provider]


def _resolve_model_settings(
    connect_timeout_sec: float,
    request_timeout_sec: float,
) -> tuple[dict[str, Any], str, str]:
    requested_provider = _normalize_model_provider(os.getenv("LLM_MODEL_PROVIDER"))
    if requested_provider not in _CLOUD_PROVIDERS | {"ollama"}:
        raise VisionLLMConfigError(
            "Unsupported LLM_MODEL_PROVIDER="
            f"{requested_provider!r}. Supported providers: ollama, openai, google_genai."
        )

    api_key = _resolve_api_key(requested_provider)
    active_provider = requested_provider
    if requested_provider in _CLOUD_PROVIDERS and not api_key:
        active_provider = "ollama"

    model_name = _resolve_model_name(active_provider, requested_provider)
    kwargs: dict[str, Any] = {
        "model": model_name,
        "model_provider": active_provider,
    }

    if active_provider == "ollama":
        base_url = _normalize_base_url(
            os.getenv("LLM_BASE_URL"),
            default="http://localhost:11434",
            env_name="LLM_BASE_URL",
        )
        timeout = httpx.Timeout(
            connect=connect_timeout_sec,
            read=request_timeout_sec,
            write=request_timeout_sec,
            pool=connect_timeout_sec,
        )
        kwargs["base_url"] = base_url
        kwargs["sync_client_kwargs"] = {"timeout": timeout}
        kwargs["async_client_kwargs"] = {"timeout": timeout}
        return kwargs, active_provider, base_url

    kwargs["api_key"] = api_key
    if active_provider == "openai":
        base_url = _normalize_base_url(
            os.getenv("LLM_OPENAI_BASE_URL"),
            env_name="LLM_OPENAI_BASE_URL",
        )
        kwargs["timeout"] = httpx.Timeout(
            connect=connect_timeout_sec,
            read=request_timeout_sec,
            write=request_timeout_sec,
            pool=connect_timeout_sec,
        )
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs, active_provider, base_url or "OpenAI API"

    kwargs["timeout"] = request_timeout_sec
    return kwargs, active_provider, "Google Gemini API"


# ── VisionLLMClient ──────────────────────────────────────────────────

class VisionLLMClient:
    """Thin wrapper around a LangChain chat model with vision support."""

    def __init__(self) -> None:
        load_dotenv()
        self.connect_timeout_sec = _env_float("LLM_CONNECT_TIMEOUT_SEC", 10.0, minimum=0.5)
        self.request_timeout_sec = _env_float("LLM_REQUEST_TIMEOUT_SEC", 300.0, minimum=1.0)
        model_kwargs, self.provider, self.service_target = _resolve_model_settings(
            self.connect_timeout_sec,
            self.request_timeout_sec,
        )
        try:
            self.llm = init_chat_model(**model_kwargs)
        except ImportError as exc:
            package_name = _PROVIDER_PACKAGES.get(self.provider)
            extra = f" Install `{package_name}`." if package_name else ""
            raise VisionLLMConfigError(
                f"Missing dependency for LLM provider {self.provider!r}.{extra}"
            ) from exc
        except ValueError as exc:
            raise VisionLLMConfigError(str(exc)) from exc

    def invoke(self, text: str, image_base64: str) -> str:
        """Send a text + image prompt and return the cleaned response."""
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_base64}",
                    },
                ]
            )
        ]
        try:
            response = self.llm.invoke(messages)
        except httpx.ConnectTimeout as exc:
            raise VisionLLMServiceUnavailableError(
                f"Timed out connecting to {self.provider} service at {self.service_target} "
                f"after {self.connect_timeout_sec:.1f}s."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise VisionLLMServiceUnavailableError(
                f"Timed out waiting for LLM response from {self.provider} service at "
                f"{self.service_target} "
                f"after {self.request_timeout_sec:.1f}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise VisionLLMServiceUnavailableError(
                f"Cannot connect to {self.provider} service at {self.service_target}. "
                "Ensure the provider is reachable and LLM_BASE_URL is correct when used."
            ) from exc
        except httpx.RequestError as exc:
            raise VisionLLMServiceUnavailableError(
                f"LLM request failed for {self.provider} at {self.service_target}: {exc}"
            ) from exc
        except ollama.RequestError as exc:
            raise VisionLLMServiceUnavailableError(
                f"Ollama request failed at {self.service_target}: {exc}"
            ) from exc
        except ollama.ResponseError as exc:
            raise VisionLLMServiceUnavailableError(
                f"Ollama returned an error at {self.service_target}: {exc}"
            ) from exc
        return strip_markdown(response.content)


# ── Runtime (lazy-init databases + LLM) ─────────────────────────────

def _normalize_dataset_json(dataset_json: Optional[str]) -> str:
    path = (dataset_json or _DEFAULT_DATASET_JSON).strip() or _DEFAULT_DATASET_JSON
    if not os.path.isabs(path):
        path = os.path.join(_LLM_ROOT, path)
    return os.path.abspath(path)


def _get_runtime_lock(dataset_path: str) -> threading.Lock:
    with _runtime_locks_guard:
        lock = _runtime_locks.get(dataset_path)
        if lock is None:
            lock = threading.Lock()
            _runtime_locks[dataset_path] = lock
        return lock


def get_runtime(dataset_json: Optional[str] = None) -> dict:
    """Return (or lazily create) the runtime dict with chart_db, text_db, and vision_llm."""
    dataset_path = _normalize_dataset_json(dataset_json)
    runtime = _runtime_cache.get(dataset_path)
    if runtime is not None:
        return runtime

    with _get_runtime_lock(dataset_path):
        runtime = _runtime_cache.get(dataset_path)
        if runtime is not None:
            return runtime

        chart_db = upsert_image_dataset(dataset_path)
        text_db = upsert_text_dataset(dataset_path)
        vision_llm = VisionLLMClient()

        runtime = {
            "dataset_json": dataset_path,
            "chart_db": chart_db,
            "text_db": text_db,
            "vision_llm": vision_llm,
        }
        _runtime_cache[dataset_path] = runtime
        return runtime


# ── RAG pipeline ─────────────────────────────────────────────────────

def run_rag_pipeline(
    chart_db,
    text_db,
    vision_llm: VisionLLMClient,
    dataset_json: str,
    base64_image: str,
) -> str:
    """Run the full 3-step RAG pipeline and return the final analysis text.

    Steps
    -----
    1. Draft — LLM describes the chart image.
    2. Domain rewrite — LLM rewrites using domain terminology.
    3. Retrieval + final answer — hybrid search → RAG context → LLM answer.
    """
    # Step 1: Draft from image
    draft_clean = vision_llm.invoke(PROMPT_DRAFT_FROM_IMAGE, base64_image)

    # Step 2: Domain rewrite
    ex_docs = text_db.similarity_search(draft_clean, k=6)
    domain_examples = "\n\n---\n\n".join(
        mask_numbers(d.page_content)
        for d in ex_docs
        if getattr(d, "page_content", None)
    )
    if not domain_examples:
        domain_examples = (
            "ไม่มีตัวอย่าง (fallback): ให้ใช้สำนวนเทคนิคแบบนักเทรดไทย "
            "เน้น PA logic และคำค้นที่ชัดเจน"
        )

    rewrite_prompt = (
        f"DRAFT:\n{draft_clean}\n\n"
        f"DOMAIN EXAMPLES (จาก dataset เดิม):\n{domain_examples}\n\n"
        f"{PROMPT_DOMAIN_REWRITE}\n\n"
        f"{AUTO_TEXT_COMPRESS_NOTE}"
    )
    auto_text = vision_llm.invoke(rewrite_prompt, base64_image)
    query_text = build_query_text_from_auto(auto_text)

    # Step 3: Hybrid retrieval → final answer
    results = hybrid_search_image_query(
        chart_db=chart_db,
        text_db=text_db,
        dataset_json=dataset_json,
        base64_image=base64_image,
        auto_text=query_text,
        k_img=10,
        k_t=10,
        final_k=5,
        w_img=0.85,
        w_t=0.15,
        rerank=True,
        rerank_top_m=20,
        w_rerank=0.45,
    )

    rag_context = build_rag_context(results, max_chars=1500)
    formatted_prompt = RAG_TEMPLATE.format(context=rag_context)
    return vision_llm.invoke(formatted_prompt, base64_image)
