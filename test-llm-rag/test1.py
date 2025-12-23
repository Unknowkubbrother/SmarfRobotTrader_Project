import torch
import clip
import json
from PIL import Image
import numpy as np

from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# 1) SETUP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Multilingual text embedding (รองรับภาษาไทยดี)
text_embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Multilingual Cross-Encoder สำหรับ re-ranking
reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)

# ============================================================
# 2) TEXT EMBEDDING (Multilingual)
# ============================================================
class MultilingualTextEmbeddings(Embeddings):
    def embed_documents(self, texts):
        vecs = text_embedder.encode(texts, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text):
        vec = text_embedder.encode([text], normalize_embeddings=True)
        return vec[0].tolist()

# ============================================================
# 3) IMAGE EMBEDDING (CLIP)
# ============================================================


class CLIPImageEmbeddings(Embeddings):
    def embed_documents(self, image_paths):
        vectors = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = clip_model.encode_image(img)
                vec = vec / vec.norm(dim=-1, keepdim=True)
            vectors.append(vec.cpu().numpy()[0].tolist())
        return vectors

    def embed_query(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = clip_model.encode_image(img)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy()[0].tolist()

# helper สำหรับ by_vector
def embed_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = clip_model.encode_image(img)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0].tolist()

# ============================================================
# 4) LOAD DATASET
# ============================================================
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def remove_duplicates(hits):
    seen = set()
    unique = []
    for h in hits:
        img = h.metadata.get("image")
        if img not in seen:
            seen.add(img)
            unique.append(h)
    return unique

raw = load_dataset("dataset.json")

# ============================================================
# 5) BUILD TEXT RAG
# ============================================================
text_docs = [
    Document(
        page_content=item["data"],
        metadata={"image": item["image"]}
    )
    for item in raw
]

text_db = Chroma.from_documents(
    text_docs,
    embedding=MultilingualTextEmbeddings(),
    collection_name="text_multilingual"
)

print("✅ Text RAG ready")

# ============================================================
# 6) BUILD IMAGE RAG
# ============================================================
image_docs = [
    Document(
        page_content=item["image"], 
        metadata={"image": item["image"]}
    )
    for item in raw
]


image_db = Chroma.from_documents(
    image_docs,
    embedding=CLIPImageEmbeddings(),
    collection_name="image_clip"
)

print("✅ Image RAG ready")

# ============================================================
# 7) LOAD VISION LLM
# ============================================================
vision_llm = OllamaLLM(
    model="ministral-3:14b",
    temperature=0
)

NEW_IMAGE = "datasets/new_chart3.png"

# ============================================================
# 8) STEP A — IMAGE → AUTO TEXT (Pattern Focus)
# ============================================================
auto_text = vision_llm.invoke(
    """
วิเคราะห์พฤติกรรมราคาและโครงสร้างของกราฟ (PA Logic) เพื่อใช้ค้นหากราฟที่มีทรงคล้ายกันในอดีต

- ลักษณะการเคลื่อนไหวโดยรวม
- การยก High/Low หรือการถูกปฏิเสธราคา
- พฤติกรรมแท่งเทียนสำคัญ
- อยู่ในช่วงสะสม ไล่ราคา หรือกระจายของ
- เหตุผลเชิงพฤติกรรมตลาด

ห้ามระบุตัวเลขราคา
เขียนเชิงอธิบายเชิงเทคนิค
ประมาณ 80–120 คำ
""",
    images=[NEW_IMAGE]
)

print("\n📝 Auto-text:")
print(auto_text)

# ============================================================
# 9) STEP B — TEXT RAG (MMR)
# ============================================================
text_hits_raw = text_db.max_marginal_relevance_search(
    auto_text,
    k=3,
    fetch_k=15,
    lambda_mult=0.5
)

text_hits = remove_duplicates(text_hits_raw)

# ============================================================
# 10) STEP C — IMAGE RAG (by_vector)
# ============================================================
query_vec = embed_image(NEW_IMAGE)

image_hits_raw = image_db.similarity_search_by_vector(
    query_vec,
    k=3
)

image_hits = remove_duplicates(image_hits_raw)

# ============================================================
# 11) MERGE CANDIDATES
# ============================================================
candidates = []

for hit in text_hits:
    candidates.append({
        "text": hit.page_content,
        "image": hit.metadata["image"],
        "source": "text_rag"
    })
    print(f"\nImage: {hit.metadata['image']} by Text RAG")

for hit in image_hits:
    for item in raw:
        if item["image"] == hit.metadata["image"]:
            if not any(c["image"] == item["image"] for c in candidates):
                candidates.append({
                    "text": item["data"],
                    "image": item["image"],
                    "source": "image_rag"
                })
                print(f"\nImage: {item['image']} by Image RAG")

print(f"\nCandidates before re-rank: {len(candidates)}")

# ============================================================
# 12) RE-RANK (Cross-Encoder)
# ============================================================
pairs = [[auto_text, c["text"]] for c in candidates]
scores = reranker.predict(pairs)

ranked = sorted(
    zip(candidates, scores),
    key=lambda x: x[1],
    reverse=True
)

top_candidates = [c for c, _ in ranked[:5]]

# ============================================================
# 13) BUILD FINAL CONTEXT
# ============================================================
rag_context = "\n\n---\n\n".join([c["text"] for c in top_candidates])
rag_context = rag_context[:1500]

print("\n🔎 RAG Context:")
print(rag_context)

# ============================================================
# 14) FINAL ANALYSIS
# ============================================================
final_prompt = f"""
คุณคือนักเทรดผู้เชี่ยวชาญ

1. วิเคราะห์กราฟปัจจุบันจากภาพ โดยอ้างอิงระดับราคาที่เห็นชัดเจนจากกราฟเท่านั้น (ไม่เดา ไม่เทียบกับบริบท)
2. นำแนวคิดการเข้าออกจากกราฟในอดีตที่คล้ายกันมาประยุกต์ใช้ โดยดูที่ Logic การเล่น ไม่ใช่ตัวเลขราคา

บริบทจากกราฟเก่า (Pattern คล้ายกัน):
{rag_context}

กติกา:
- เขียนเป็นย่อหน้าเดียว
- ไม่เกิน 3 ประโยค
- ไม่อธิบายเชิงทฤษฎี
- ไม่ใช้ bullet
- ไม่พูดคำว่า แนวโน้ม โมเมนตัม โครงสร้างราคา
- ข้อความล้วน
- ไม่มี Markdown
- ไม่มีสัญลักษณ์พิเศษ

รูปแบบ:
1. เปิดด้วย "PA อยู่ในช่วง..." หรือ "PA เป็นของฝั่ง..."
2. ระบุแนวรับและแนวต้านจากภาพปัจจุบัน พร้อมบอกว่าใครเสียเปรียบ
3. ปิดด้วยกลยุทธ์สั้น ๆ ว่าควรรอเล่นตรงไหน
"""

final_answer = vision_llm.invoke(
    final_prompt,
    images=[NEW_IMAGE]
)

print("\n🧠 Final Analysis:")
print(final_answer)