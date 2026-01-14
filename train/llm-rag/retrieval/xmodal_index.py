import os
from typing import List, Tuple

from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document

from .dataset_utils import norm_path, dataset_unique_paths


# ============================================================
# CONFIG
# ============================================================
# IMPORTANT:
# - Use clip-ViT-B-32 for IMAGE embeddings
# - Use clip-ViT-B-32-multilingual-v1 for TEXT embeddings (aligned to the image space)
IMG_MODEL_NAME = "clip-ViT-B-32"
TXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

XMODAL_PERSIST_DIR = "chroma_store_xmodal"
XMODAL_COLLECTION = "chart_xmodal_images"

_img_model = None
_txt_model = None


def get_img_model() -> SentenceTransformer:
    global _img_model
    if _img_model is None:
        _img_model = SentenceTransformer(IMG_MODEL_NAME)
    return _img_model


def get_txt_model() -> SentenceTransformer:
    global _txt_model
    if _txt_model is None:
        _txt_model = SentenceTransformer(TXT_MODEL_NAME)
    return _txt_model


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class XModalClipEmbeddings(Embeddings):
    """
    Documents: image paths -> embed with IMG model (clip-ViT-B-32)
    Query: text (TH/EN)    -> embed with multilingual TEXT model
    """
    def embed_documents(self, image_paths: List[str]):
        imgs = [_load_rgb(p) for p in image_paths]
        vecs = get_img_model().encode(imgs, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str):
        vec = get_txt_model().encode([text], normalize_embeddings=True)
        return vec[0].tolist()


def open_xmodal_db():
    return Chroma(
        collection_name=XMODAL_COLLECTION,
        embedding_function=XModalClipEmbeddings(),
        persist_directory=XMODAL_PERSIST_DIR,
    )


def _get_existing_ids_batched(db: Chroma, ids: List[str], batch=1000):
    existing = set()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        got = db._collection.get(ids=chunk, include=[])
        if got and "ids" in got and got["ids"]:
            existing.update(got["ids"])
    return existing


def upsert_xmodal_image_dataset(dataset_json: str):
    """
    Store images in x-modal DB with id = normalized path (dedupe).
    """
    db = open_xmodal_db()
    _, uniq_paths = dataset_unique_paths(dataset_json)
    if not uniq_paths:
        raise ValueError("dataset.json contains no image paths")

    existing = _get_existing_ids_batched(db, uniq_paths, batch=1000)
    new_paths = [p for p in uniq_paths if p not in existing]

    if not new_paths:
        print(f"✅ XModal Image DB: no new images. count={db._collection.count()}")
        return db

    docs = [Document(page_content=p, metadata={"image": p}) for p in new_paths]
    db.add_documents(docs, ids=new_paths[:])
    print(f"✅ XModal Image DB: added {len(new_paths)} new images. count={db._collection.count()}")
    return db
