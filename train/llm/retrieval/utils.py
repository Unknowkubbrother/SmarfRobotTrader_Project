import os
import json
import re
import base64
from typing import Dict, List, Tuple, Any

from datetime import datetime


def norm_path(p: str) -> str:
    p = (p or "").strip()
    p2 = os.path.normpath(p).replace("\\", "/")
    # windows: make ids case-insensitive to avoid duplicates
    if os.name == "nt":
        p2 = p2.lower()
    return p2


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dataset_unique_paths(dataset_json: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    raw = load_dataset(dataset_json)
    paths = [norm_path(it["image"]) for it in raw if "image" in it and it["image"]]
    uniq = sorted(set(paths))
    return raw, uniq


def build_text_lookup(dataset_json: str) -> Dict[str, str]:
    raw = load_dataset(dataset_json)
    out: Dict[str, str] = {}
    for it in raw:
        p0 = it.get("image")
        if not p0:
            continue
        out[norm_path(p0)] = (it.get("data", "") or "")
    return out


def mask_numbers(s: str) -> str:
    return re.sub(r"\d[\d,.\s-]*", "<NUM>", s)


def strip_markdown(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"_(.+?)_", r"\1", s)
    s = re.sub(r"#+\s*", "", s)
    s = re.sub(r"^[\-\*•]\s*", "", s, flags=re.MULTILINE)
    return s.strip()


def build_query_text_from_auto(auto_text: str) -> str:
    if not auto_text:
        return ""

    lines = [l.strip() for l in auto_text.splitlines() if l.strip()]
    if not lines:
        return ""

    kw_line = ""
    for l in reversed(lines):
        if l.lower().startswith("keywords:"):
            kw_line = l.replace("**", "").strip()
            break

    summary_parts = []
    for l in lines:
        low = l.lower()
        if low.startswith("keywords:"):
            continue
        if l.startswith(("-", "*", "•")) or l.startswith("#"):
            continue
        l = l.replace("**", "").strip()
        if not l:
            continue
        summary_parts.append(l)
        if len(summary_parts) >= 2:
            break

    summary = " ".join(summary_parts).strip()
    if not summary:
        summary = lines[0].replace("**", "").strip()

    words = summary.split()
    if len(words) > 80:
        summary = " ".join(words[:80])

    if kw_line:
        return f"{summary} {kw_line}".strip()
    return summary


def print_results(title: str, results):
    print(f"\n🏁 {title}")
    for i, r in enumerate(results, 1):
        snippet = (r.get("data") or "").replace("\n", " ").strip()
        if len(snippet) > 180:
            snippet = snippet[:180] + "..."

        ranks = []
        for k in ["img_rank", "t_rank"]:
            if r.get(k) is not None:
                ranks.append(f"{k}={r[k]}")

        extra = []
        if r.get("final_score") is not None:
            extra.append(f"final={float(r['final_score']):.4f}")
        if r.get("rerank_text_score") is not None:
            extra.append(f"rerank={float(r['rerank_text_score']):.4f}")

        print(f"{i}. {r['image']}")
        print(f"   rrf={r['rrf']:.6f}"
              + (f" | {' | '.join(ranks)}" if ranks else "")
              + (f" | {' | '.join(extra)}" if extra else ""))
        print(f"   {snippet}\n")


def build_rag_context(results, max_chars: int = 1500) -> str:
    chunks = []
    for r in results:
        txt = (r.get("data") or "").strip()
        if txt:
            chunks.append(txt)
    ctx = "\n\n---\n\n".join(chunks)
    return ctx[:max_chars]


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def parse_dt_from_filename(name: str):
    try:
        base = os.path.splitext(name)[0]
        dt_str = base.split("_", 1)[1]
        return datetime.strptime(dt_str, "%Y.%m.%d %H.%M")
    except Exception:
        return None


def list_fileDate_folder(folder: str) -> List[str]:
    items = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        dt = parse_dt_from_filename(name)
        if dt is None:
            continue
        items.append((dt, name))
    items.sort(key=lambda x: x[0])
    return items