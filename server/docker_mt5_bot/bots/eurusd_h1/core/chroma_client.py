import os
import chromadb
from datetime import datetime
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any, Optional

class ChromaDBClient:
    def __init__(self, persist_path: str = "chroma_db"):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)

        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.model = None

    def _ensure_model(self):
        if self.model is None:
            self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_dense(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        return [v.tolist() for v in dense]

    def insert_document(self, symbol: str, symbol_datetime: datetime, timeframe: str, content: str):
        try:
            emb = self.embed_dense([content])[0]

            doc_id = f"{symbol}_{symbol_datetime.strftime('%Y%m%d_%H%M%S')}"

            metadata = {
                "symbol": symbol,
                "symbol_datetime": symbol_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "timeframe": timeframe
            }

            self.collection.add(
                ids=[doc_id],
                embeddings=[emb],
                metadatas=[metadata],
                documents=[content]
            )
            return emb

        except Exception as e:
            print(f" Failed to insert document into ChromaDB: {e}")
            return None

    def get_document(self, doc_id: str):
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["metadatas", "documents", "embeddings"]
            )

            if results['ids']:
                return {
                    "id": results['ids'][0],
                    "content": results['documents'][0],
                    "metadata": results['metadatas'][0],
                    "embedding": results['embeddings'][0]
                }
            return None
        except Exception as e:
            print(f" Failed to get document: {e}")
            return None

    def search_similar(self, query_text: str, k: int = 5):
        try:
            q_emb = self.embed_dense([query_text])[0]

            results = self.collection.query(
                query_embeddings=[q_emb],
                n_results=k
            )

            formatted_results = []

            if results['ids']:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i] if 'distances' in results else 0
                    })

            return formatted_results

        except Exception as e:
            print(f" ChromaDB Search Error: {e}")
            return []

    def get_latest_document_datetime(self) -> Optional[datetime]:
        try:
            results = self.collection.get(include=["metadatas"])

            if not results or not results['metadatas']:
                return None

            latest_dt = None

            for meta in results['metadatas']:
                if not meta or 'symbol_datetime' not in meta:
                    continue

                try:
                    dt_str = meta['symbol_datetime']
                    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

                    if latest_dt is None or dt > latest_dt:
                        latest_dt = dt
                except Exception:
                    continue

            return latest_dt

        except Exception as e:
            print(f" Error getting latest datetime: {e}")
            return None
