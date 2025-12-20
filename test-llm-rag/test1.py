import torch
import clip
import json
from PIL import Image

from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

# ===================== SETUP =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ===================== TEXT EMBEDDING =====================
class CLIPTextEmbeddings(Embeddings):
    def embed_documents(self, texts):
        with torch.no_grad():
            tokens = clip.tokenize(texts, truncate=True).to(device)
            vecs = model.encode_text(tokens)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy().tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# ===================== IMAGE EMBEDDING =====================
class CLIPImageEmbeddings(Embeddings):
    def embed_documents(self, image_paths):
        vectors = []
        for path in image_paths:
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model.encode_image(image)
                vec = vec / vec.norm(dim=-1, keepdim=True)
            vectors.append(vec.cpu().numpy()[0].tolist())
        return vectors

    def embed_query(self, image_path):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(image)
            vec = vec / vec.norm(dim=-1, keepdim=True)
        return vec.cpu().numpy()[0].tolist()

# ===================== MANUAL IMAGE EMBED (FOR by_vector) =====================
def embed_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0].tolist()

# ===================== LOAD DATASET =====================
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

raw = load_dataset("dataset.json")

# ===================== BUILD TEXT RAG =====================
text_docs = [
    Document(
        page_content=item["data"],
        metadata={"image": item["image"]}
    )
    for item in raw
]

text_db = Chroma.from_documents(
    text_docs,
    embedding=CLIPTextEmbeddings(),
    persist_directory="./chroma_text"
)

print("✅ Text RAG ready")

# ===================== BUILD IMAGE RAG =====================
image_docs = [
    Document(
        page_content=item["image"],   # path เป็น identifier
        metadata={"image": item["image"]}
    )
    for item in raw
]

image_db = Chroma.from_documents(
    image_docs,
    embedding=CLIPImageEmbeddings(),
    persist_directory="./chroma_image"
)

print("✅ Image RAG ready")

# ===================== LOAD VISION LLM =====================
vision_llm = OllamaLLM(
    model="ministral-3:14b",
    temperature=0
)

NEW_IMAGE = "datasets/chart2.png"

# ===================== STEP A: IMAGE → AUTO TEXT =====================
auto_text = vision_llm.invoke(
    """
อธิบายกราฟจากภาพนี้แบบสั้น
บอกว่าเป็น ขาขึ้น ขาลง หรือ แกว่ง
บอกสภาพแรงซื้อแรงขายคร่าว ๆ
ไม่เกิน 50 tokens
""",
    images=[NEW_IMAGE]
)

print("\n📝 Auto-text from image:")
print(auto_text)

# ===================== STEP B: TEXT RAG =====================
text_hits = text_db.similarity_search(auto_text, k=2)

print("\n📝 Text-based hits:")
for hit in text_hits:
    print(hit.metadata["image"])

# ===================== STEP C: IMAGE RAG (by_vector) =====================
query_vec = embed_image(NEW_IMAGE)

image_hits = image_db.similarity_search_by_vector(
    query_vec,
    k=2
)

print("\n🖼️ Image-based hits (by_vector):")
for hit in image_hits:
    print(hit.metadata["image"])

# ===================== MERGE HYBRID CONTEXT =====================
contexts = set()

for d in text_hits:
    contexts.add(d.page_content)

for d in image_hits:
    for item in raw:
        if item["image"] == d.metadata["image"]:
            contexts.add(item["data"])

rag_context = "\n".join(contexts)

print("\n🔎 Hybrid RAG context:")
print(rag_context)

# ===================== FINAL ANALYSIS =====================
final_prompt = f"""
คุณคือนักเทรดที่เขียนไอเดียเทรดแบบสั้น คม และอ่านรู้เรื่องทันที

บริบทจากกราฟเก่าที่คล้ายกัน:
{rag_context}

ภาพรวมกราฟปัจจุบัน:
{auto_text}

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
2. บอกแนวรับ แนวต้าน และใครเสียเปรียบ
3. ปิดด้วยกลยุทธ์ว่าควรรอเล่นตรงไหน
"""

final_answer = vision_llm.invoke(
    final_prompt,
    images=[NEW_IMAGE]
)

print("\n🧠 Final Analysis:")
print(final_answer)
