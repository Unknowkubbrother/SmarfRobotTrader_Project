import os
import hashlib
from typing import Dict, List, Any, Optional

import torch
from FlagEmbedding import BGEM3FlagModel
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .utils import norm_path, load_dataset


TEXT_MODEL_NAME = "BAAI/bge-m3"
TEXT_PERSIST_BASE = "chroma_store_text_bge"
TEXT_COLLECTION_BASE = "chart_text_bge_m3"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

_text_embedder: Optional[BGEM3FlagModel] = None
_TEXT_DIM: Optional[int] = None
_text_splitter: Optional[RecursiveCharacterTextSplitter] = None


def get_text_embedder() -> BGEM3FlagModel:
    global _text_embedder
    if _text_embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _text_embedder = BGEM3FlagModel(
            TEXT_MODEL_NAME,
            use_fp16=True if device == "cuda" else False,
            device=device
        )
    return _text_embedder


def get_text_dim() -> int:
    global _TEXT_DIM
    if _TEXT_DIM is None:
        _TEXT_DIM = 1024
    return _TEXT_DIM


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""]
        )
    return _text_splitter


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def chunk_text(text: str) -> List[str]:
    splitter = get_text_splitter()
    chunks = splitter.split_text(text)
    return chunks if chunks else [text]


class BGEM3Embeddings(Embeddings):
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model = get_text_embedder()
        output = model.encode(
            texts,
            batch_size=12,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output['dense_vecs'].tolist()

    def embed_query(self, text: str) -> List[float]:
        model = get_text_embedder()
        output = model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output['dense_vecs'][0].tolist()


def open_text_db():
    dim = get_text_dim()
    persist_dir = f"{TEXT_PERSIST_BASE}_{dim}"
    collection = f"{TEXT_COLLECTION_BASE}_{dim}"

    return Chroma(
        collection_name=collection,
        embedding_function=BGEM3Embeddings(),
        persist_directory=persist_dir,
    )


def upsert_text_dataset(dataset_json: str):
    raw = load_dataset(dataset_json)

    by_path: Dict[str, str] = {}
    for it in raw:
        p0 = it.get("image")
        if not p0:
            continue
        p = norm_path(p0)
        txt = (it.get("data") or "").strip()
        if not txt:
            continue
        by_path[p] = txt

    paths = sorted(by_path.keys())
    if not paths:
        raise ValueError("dataset.json has no valid 'data' text")

    db = open_text_db()

    existing_sha: Dict[str, str] = {}
    for i in range(0, len(paths), 1000):
        chunk = paths[i:i + 1000]
        got = db._collection.get(ids=chunk, include=["metadatas"])
        ids = (got or {}).get("ids") or []
        metas = (got or {}).get("metadatas") or []
        for _id, meta in zip(ids, metas):
            if meta and meta.get("text_sha1"):
                existing_sha[_id] = meta["text_sha1"]

    up_ids: List[str] = []
    up_docs: List[str] = []
    up_metas: List[Dict[str, Any]] = []

    for p in paths:
        txt = by_path[p]
        h = sha1_text(txt)
        if existing_sha.get(p) == h:
            continue
        up_ids.append(p)
        up_docs.append(txt)
        up_metas.append({"image": p, "text_sha1": h})

    if not up_ids:
        print(f"✅ Text DB: no new/changed texts. count={db._collection.count()}")
        return db

    embeddings = BGEM3Embeddings().embed_documents(up_docs)

    db._collection.upsert(
        ids=up_ids,
        documents=up_docs,
        metadatas=up_metas,
        embeddings=embeddings,
    )

    if hasattr(db, "persist"):
        try:
            db.persist()
        except Exception:
            pass

    print(f"✅ Text DB: upserted {len(up_ids)} texts. count={db._collection.count()}")
    return db
