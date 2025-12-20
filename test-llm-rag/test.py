import torch
import clip
import json
import numpy as np
from PIL import Image

from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# ===================== 1) LOAD CLIP =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ===================== 2) CLIP TEXT EMBEDDINGS =====================
class CLIPEmbeddings(Embeddings):
    def embed_documents(self, texts):
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(device)
            vecs = model.encode_text(tokens)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedder = CLIPEmbeddings()

# ===================== 3) IMAGE → VECTOR =====================
def image_to_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]

# ===================== 4) LOAD DATASET (JSON) =====================
def load_docs_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for item in raw:
        docs.append(
            Document(
                page_content=item["data"],   # คำอธิบายกราฟ
                metadata={"path": item["image"]}
            )
        )
    return docs

docs = load_docs_from_json("dataset.json")

db = Chroma.from_documents(
    docs,
    embedding=embedder,
    persist_directory="./chroma"
)

print("✅ Chroma DB ready")

# ===================== 5) IMAGE → RAG (NO LLM) =====================
NEW_IMAGE = "datasets/chart5.png"

img_vec = image_to_embedding(NEW_IMAGE)

rag_docs = db.similarity_search_by_vector(
    img_vec.tolist(),
    k=2
)

rag_context = "\n".join(
    d.page_content for d in rag_docs
)

print("\n🔎 Retrieved RAG context:")
print(rag_context)

# ===================== 6) FINAL ANALYSIS (VISION LLM ONLY HERE) =====================
vision_llm = OllamaLLM(
    model="ministral-3:14b",
    temperature=0
)

final_prompt = f"""
คุณเป็นนักวิเคราะห์กราฟราคามืออาชีพ

มีกราฟราคาใหม่ให้วิเคราะห์
ด้านล่างคือรูปแบบกราฟในอดีตที่คล้ายกัน:

{rag_context}

ช่วยอธิบายแนวโน้มของกราฟใหม่
เป็นภาษาคนธรรมชาติ เหมือนอธิบายให้เพื่อนฟัง

ข้อกำหนด:
- ข้อความล้วน
- ไม่ใช้ Markdown
- ไม่ใช้สัญลักษณ์พิเศษ
- ไม่ต้องตั้งคำถาม
"""

final_answer = vision_llm.invoke(
    final_prompt,
    images=[NEW_IMAGE]
)

print("\n🧠 Final Analysis:")
print(final_answer)
