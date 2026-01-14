import os
import json
from typing import Dict, List, Tuple, Any

def norm_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")

def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def dataset_unique_paths(dataset_json: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    raw = load_dataset(dataset_json)
    paths = [norm_path(it["image"]) for it in raw if "image" in it]
    uniq = sorted(set(paths))
    return raw, uniq

def build_text_lookup(dataset_json: str) -> Dict[str, str]:
    raw = load_dataset(dataset_json)
    return {norm_path(it["image"]): (it.get("data", "") or "") for it in raw if "image" in it}
