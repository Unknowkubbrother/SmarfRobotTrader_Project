import json
import torch
import numpy as np
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM

import chromadb
from chromadb.config import Settings

# ============================================================
# 1) SETUP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device).eval()

clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

text_embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vision_llm = OllamaLLM(
    model="ministral-3:14b",
    temperature=0
)

# ============================================================
# 2) LOAD DATASET
# ============================================================
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"📦 Loaded dataset: {len(dataset)} samples")

# ============================================================
# 3) SAFE CLIP TEXT EMBEDDING (77 TOKENS FIX)
# ============================================================
def get_clip_text_vector(text: str) -> np.ndarray:
    inputs = clip_processor(
        text=[text],
        truncation=True,
        max_length=77,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        vec = clip_model.get_text_features(**inputs)
        vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.cpu().numpy()[0]

# ============================================================
# 4) HYBRID VECTOR BUILDER (1281 dim)
# ============================================================
def build_hybrid_vector(image_path: str, text: str) -> np.ndarray:
    # ----- image embedding -----
    img = Image.open(image_path).convert("RGB")
    img_inputs = clip_processor(
        images=img,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        img_vec = clip_model.get_image_features(**img_inputs)
        img_vec = img_vec / img_vec.norm(dim=-1, keepdim=True)

    img_vec = img_vec.cpu().numpy()[0]          # 512

    # ----- semantic text embedding -----
    text_vec = text_embedder.encode(
        text,
        normalize_embeddings=True
    )                                           # 768

    # ----- image-text alignment -----
    clip_text_vec = get_clip_text_vector(text)  # 512
    clip_similarity = float(np.dot(img_vec, clip_text_vec))  # 1

    # ----- hybrid -----
    hybrid_vec = np.concatenate([
        img_vec * 0.4,
        text_vec * 0.5,
        np.array([clip_similarity])
    ])                                          # 1281

    return hybrid_vec

# ============================================================
# 5) CREATE CHROMA COLLECTION (CUSTOM VECTOR)
# ============================================================
client = chromadb.Client(
    Settings(anonymized_telemetry=False)
)

# ลบ collection เก่าถ้ามี (สำคัญ)
try:
    client.delete_collection("hybrid_chart_dataset")
except:
    pass

collection = client.create_collection(
    name="hybrid_chart_dataset",
    embedding_function=None   # 🔥 custom embedding
)

# ============================================================
# 6) INDEX DATASET
# ============================================================
for idx, item in enumerate(dataset):
    vec = build_hybrid_vector(item["image"], item["data"])

    collection.add(
        ids=[str(idx)],
        documents=[item["data"]],
        metadatas=[{"image": item["image"]}],
        embeddings=[vec.tolist()]
    )

print("✅ Hybrid dataset indexed correctly")

# ============================================================
# 7) QUERY WITH NEW IMAGE
# ============================================================
NEW_IMAGE = "datasets1/new_chart1.png"

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

print("\n📝 Auto text:\n", auto_text)

query_vec = build_hybrid_vector(
    NEW_IMAGE,
    auto_text
)

results = collection.query(
    query_embeddings=[query_vec.tolist()],
    n_results=3
)

# ============================================================
# 8) SHOW RESULTS
# ============================================================
print("\n🔍 Similar historical patterns:")
for i in range(len(results["documents"][0])):
    print(f"\n#{i+1}")
    print("Image:", results["metadatas"][0][i]["image"])
    print("Text :", results["documents"][0][i][:160], "...")
