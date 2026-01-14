import hashlib
from typing import Dict

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings

from .dataset_utils import norm_path, load_dataset


TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# base names (จะถูกเติม _{dim} อัตโนมัติ)
TEXT_PERSIST_BASE = "chroma_store_text"
TEXT_COLLECTION_BASE = "chart_text_multilingual"

_text_embedder = None
_TEXT_DIM = None


def get_text_embedder() -> SentenceTransformer:
    global _text_embedder
    if _text_embedder is None:
        _text_embedder = SentenceTransformer(TEXT_MODEL_NAME)
    return _text_embedder


def get_text_dim() -> int:
    global _TEXT_DIM
    if _TEXT_DIM is None:
        _TEXT_DIM = int(get_text_embedder().get_sentence_embedding_dimension())
    return _TEXT_DIM


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class MultilingualTextEmbeddings(Embeddings):
    def embed_documents(self, texts):
        emb = get_text_embedder().encode(texts, normalize_embeddings=True)
        return emb.tolist()

    def embed_query(self, text):
        emb = get_text_embedder().encode([text], normalize_embeddings=True)
        return emb[0].tolist()


def open_text_db():
    dim = get_text_dim()
    persist_dir = f"{TEXT_PERSIST_BASE}_{dim}"
    collection = f"{TEXT_COLLECTION_BASE}_{dim}"

    return Chroma(
        collection_name=collection,
        embedding_function=MultilingualTextEmbeddings(),
        persist_directory=persist_dir,
    )


def upsert_text_dataset(dataset_json: str):
    """
    - id = normalized image path
    - upsert เฉพาะใหม่/เปลี่ยน
    - IMPORTANT: ส่ง embeddings= เข้า chroma เพื่อกันปัญหา dim mismatch และให้ update จริง
    """
    raw = load_dataset(dataset_json)

    by_path: Dict[str, str] = {}
    for it in raw:
        if "image" not in it:
            continue
        p = norm_path(it["image"])
        txt = (it.get("data") or "").strip()
        if not txt:
            continue
        by_path[p] = txt

    paths = sorted(by_path.keys())
    if not paths:
        raise ValueError("dataset.json has no valid 'data' text")

    db = open_text_db()

    # read existing sha1
    existing_sha: Dict[str, str] = {}
    for i in range(0, len(paths), 1000):
        chunk = paths[i:i + 1000]
        got = db._collection.get(ids=chunk, include=["metadatas"])
        ids = (got or {}).get("ids") or []
        metas = (got or {}).get("metadatas") or []
        for _id, meta in zip(ids, metas):
            if meta and "text_sha1" in meta:
                existing_sha[_id] = meta["text_sha1"]

    up_ids, up_docs, up_metas = [], [], []
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

    # compute embeddings explicitly (512-dim for this model)
    embeddings = MultilingualTextEmbeddings().embed_documents(up_docs)

    # true upsert with embeddings
    db._collection.upsert(
        ids=up_ids,
        documents=up_docs,
        metadatas=up_metas,
        embeddings=embeddings,
    )

    print(f"✅ Text DB: upserted {len(up_ids)} texts. count={db._collection.count()}")
    return db
