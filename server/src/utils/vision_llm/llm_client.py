import os
import threading
from urllib.parse import urlparse
from typing import Optional

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


def _normalize_llm_base_url(raw_value: Optional[str]) -> str:
    """Normalize and validate LLM base URL from environment."""
    base_url = str(raw_value or "").strip().strip('"').strip("'")
    if not base_url:
        base_url = "http://localhost:11434"

    # Common typo recovery: "http:/host:port" -> "http://host:port"
    if base_url.startswith("http:/") and not base_url.startswith("http://"):
        base_url = "http://" + base_url[len("http:/"):].lstrip("/")
    elif base_url.startswith("https:/") and not base_url.startswith("https://"):
        base_url = "https://" + base_url[len("https:/"):].lstrip("/")

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise VisionLLMConfigError(
            f"Invalid LLM_BASE_URL={raw_value!r}. Expected format: http://host:port"
        )
    return base_url.rstrip("/")


# ── VisionLLMClient ──────────────────────────────────────────────────

class VisionLLMClient:
    """Thin wrapper around a LangChain chat model with vision support."""

    def __init__(self) -> None:
        load_dotenv()
        self.base_url = _normalize_llm_base_url(
            os.getenv("LLM_BASE_URL", "http://localhost:11434")
        )
        self.connect_timeout_sec = _env_float("LLM_CONNECT_TIMEOUT_SEC", 10.0, minimum=0.5)
        self.request_timeout_sec = _env_float("LLM_REQUEST_TIMEOUT_SEC", 300.0, minimum=1.0)
        timeout = httpx.Timeout(
            connect=self.connect_timeout_sec,
            read=self.request_timeout_sec,
            write=self.request_timeout_sec,
            pool=self.connect_timeout_sec,
        )
        self.llm = init_chat_model(
            model=os.getenv("LLM_MODEL", "ministral-3:14b"),
            model_provider=os.getenv("LLM_MODEL_PROVIDER", "ollama"),
            api_key=os.getenv("LLM_API_KEY",""),
            base_url=self.base_url,
            sync_client_kwargs={"timeout": timeout},
            async_client_kwargs={"timeout": timeout},
        )

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
                f"Timed out connecting to LLM service at {self.base_url} "
                f"after {self.connect_timeout_sec:.1f}s."
            ) from exc
        except httpx.ReadTimeout as exc:
            raise VisionLLMServiceUnavailableError(
                f"Timed out waiting for LLM response from {self.base_url} "
                f"after {self.request_timeout_sec:.1f}s."
            ) from exc
        except httpx.ConnectError as exc:
            raise VisionLLMServiceUnavailableError(
                f"Cannot connect to LLM service at {self.base_url}. "
                "Ensure Ollama is running and LLM_BASE_URL is correct."
            ) from exc
        except httpx.RequestError as exc:
            raise VisionLLMServiceUnavailableError(
                f"LLM request failed at {self.base_url}: {exc}"
            ) from exc
        except ollama.RequestError as exc:
            raise VisionLLMServiceUnavailableError(
                f"Ollama request failed at {self.base_url}: {exc}"
            ) from exc
        except ollama.ResponseError as exc:
            raise VisionLLMServiceUnavailableError(
                f"Ollama returned an error at {self.base_url}: {exc}"
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
