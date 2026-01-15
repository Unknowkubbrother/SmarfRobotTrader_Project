import os
import json
from typing import Dict, List, Tuple, Any


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
