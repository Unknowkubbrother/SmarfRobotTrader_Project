import os
import json
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "openai/clip-vit-large-patch14"

ROI_CUT = dict(left=0.02, top=0.06, right=0.10, bottom=0.10)

LAYERS = (-6, -3)
TOPK_RATIO = 0.12
PRIOR_STRENGTH = 1.6
DENSITY_POWER = 2.0
DENSITY_MIN_CLIP = 0.12

HLINE_MIN_RUN = 3
HLINE_PENALTY = 0.15

CC_MIN_AREA = 2
CC_ASPECT_THRESH = 0.45

W_REAL = 0.55
W_STRUCT = 0.45

PROMPTS = [
    "candlestick bodies and wicks",
    "bullish and bearish candle shapes",
    "candlestick rejection wicks",
    "price action candlesticks",
]

PERSIST_DIR = "chroma_store_images"
COLLECTION = "chart_clip"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def crop_chart_roi(pil_img: Image.Image, left=0.02, top=0.06, right=0.10, bottom=0.10) -> Image.Image:
    w, h = pil_img.size
    x1 = int(w * left)
    y1 = int(h * top)
    x2 = int(w * (1 - right))
    y2 = int(h * (1 - bottom))
    return pil_img.crop((x1, y1, x2, y2))

def letterbox(pil_img: Image.Image, out_size=224) -> Image.Image:
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = out_size / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), resample=Image.BICUBIC)

    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    x0 = (out_size - nw) // 2
    y0 = (out_size - nh) // 2
    canvas.paste(img, (x0, y0))
    return canvas

def build_views(pil_img: Image.Image):
    roi = crop_chart_roi(pil_img, **ROI_CUT)
    w, h = roi.size
    zoom = roi.crop((0, 0, w, int(h * 0.55)))
    return letterbox(roi, 224), letterbox(zoom, 224)

def make_structure_view(pil_224: Image.Image) -> Image.Image:
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

@torch.no_grad()
def build_text_feature(prompts):
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    text_feat = clip_model.get_text_features(**text_inputs)
    text_feat = F.normalize(text_feat, dim=-1).mean(dim=0, keepdim=True)
    return F.normalize(text_feat, dim=-1)

def spatial_prior(grid_size: int, strength=1.6) -> np.ndarray:
    g = grid_size
    ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
    cy, cx = (g - 1) / 2.0, (g - 1) / 2.0
    dy = (ys - cy) / (cy + 1e-9)
    dx = (xs - cx) / (cx + 1e-9)
    r2 = dx * dx + dy * dy
    prior = np.exp(-strength * r2)
    prior = prior / (prior.max() + 1e-9)
    return prior

def patch_density_mask(pil_224: Image.Image, grid_size: int) -> np.ndarray:
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag_x = np.abs(gx)
    mag_y = np.abs(gy)

    H, W = gray.shape
    ph, pw = H // grid_size, W // grid_size

    score = np.zeros((grid_size, grid_size), dtype=np.float32)
    for y in range(grid_size):
        for x in range(grid_size):
            px = mag_x[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            py = mag_y[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            v_like = float(px.mean())
            h_like = float(py.mean())
            score[y, x] = v_like / (h_like + 1e-6)

    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    score = np.clip(score, DENSITY_MIN_CLIP, 1.0)
    return score

def suppress_long_horizontal_runs(density_grid: np.ndarray, min_run: int = 3, penalty: float = 0.15) -> np.ndarray:
    g = density_grid.shape[0]
    out = density_grid.copy()

    for y in range(g):
        run_start = None
        for x in range(g):
            if out[y, x] > 0:
                if run_start is None:
                    run_start = x
            else:
                if run_start is not None:
                    run_len = x - run_start
                    if run_len >= min_run:
                        out[y, run_start:x] *= penalty
                    run_start = None

        if run_start is not None:
            run_len = g - run_start
            if run_len >= min_run:
                out[y, run_start:g] *= penalty

    return np.clip(out, 0.0, 1.0)

def prune_by_connectivity(keep_mask: np.ndarray, min_area: int = 2, aspect_ratio_thresh: float = 0.45) -> np.ndarray:
    m = (keep_mask.astype(np.uint8) * 255)
    num, labels = cv2.connectedComponents(m, connectivity=8)

    out = np.zeros_like(keep_mask, dtype=bool)

    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        area = len(xs)
        if area < min_area:
            continue

        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        aspect = min(w, h) / (max(w, h) + 1e-6)

        if aspect < aspect_ratio_thresh:
            out[ys, xs] = True

    return out

@torch.no_grad()
def patch_pooled_embedding(
    pil_224: Image.Image,
    text_feat: torch.Tensor,
    layers=LAYERS,
    topk_ratio=TOPK_RATIO,
    prior_strength=PRIOR_STRENGTH,
    density_power=DENSITY_POWER,
):
    inputs = processor(images=pil_224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    pooled_layers = []

    for ly in layers:
        hs = vision_out.hidden_states[ly] if ly != -1 else vision_out.last_hidden_state
        patch_tokens = hs[:, 1:, :]

        patch_emb = clip_model.visual_projection(patch_tokens)
        patch_emb = F.normalize(patch_emb, dim=-1)

        sim = (patch_emb @ text_feat.transpose(-1, -2)).squeeze(0).squeeze(-1)
        sim_np = sim.detach().cpu().numpy()

        P = sim_np.shape[0]
        g = int(round(math.sqrt(P)))
        if g * g != P:
            raise ValueError(f"Patch count {P} not square.")

        sim_grid = sim_np.reshape(g, g)

        prior = spatial_prior(g, strength=prior_strength)
        sim_grid = sim_grid * prior

        dens = patch_density_mask(pil_224, g)
        dens = suppress_long_horizontal_runs(dens, min_run=HLINE_MIN_RUN, penalty=HLINE_PENALTY)

        sim_grid = sim_grid * (dens ** density_power)

        flat = sim_grid.flatten()
        k = max(1, int(len(flat) * float(topk_ratio)))
        thr = np.sort(flat)[-k]
        keep0 = (sim_grid >= thr)

        keep = prune_by_connectivity(keep0, min_area=CC_MIN_AREA, aspect_ratio_thresh=CC_ASPECT_THRESH)
        if not keep.any():
            keep = keep0

        keep_flat = keep.flatten()
        sim2 = torch.tensor(sim_grid.flatten(), device=DEVICE, dtype=sim.dtype)

        mask = torch.tensor(keep_flat, device=DEVICE)
        selected = patch_emb[0][mask]
        selected_sim = sim2[mask]

        weights = F.softmax(selected_sim, dim=0).unsqueeze(-1)
        pooled = (selected * weights).sum(dim=0)
        pooled = F.normalize(pooled, dim=-1)
        pooled_layers.append(pooled)

    out = torch.stack(pooled_layers, dim=0).mean(dim=0)
    out = F.normalize(out, dim=-1)
    return out.detach().cpu().numpy()


def embed_chart_image(image_path: str) -> list:
    pil = load_rgb(image_path)
    wide224, zoom224 = build_views(pil)
    text_feat = build_text_feature(PROMPTS)

    v_real = (patch_pooled_embedding(wide224, text_feat) + patch_pooled_embedding(zoom224, text_feat)) / 2.0
    v_real = v_real / (np.linalg.norm(v_real) + 1e-9)

    wide_struct = make_structure_view(wide224)
    zoom_struct = make_structure_view(zoom224)

    v_struct = (patch_pooled_embedding(wide_struct, text_feat) + patch_pooled_embedding(zoom_struct, text_feat)) / 2.0
    v_struct = v_struct / (np.linalg.norm(v_struct) + 1e-9)

    out = (W_REAL * v_real) + (W_STRUCT * v_struct)
    out = out / (np.linalg.norm(out) + 1e-9)
    return out.tolist()

class ProductionCLIPChartEmbeddings(Embeddings):
    def embed_documents(self, image_paths):
        return [embed_chart_image(p) for p in image_paths]

    def embed_query(self, image_path):
        return embed_chart_image(image_path)

def load_dataset(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_image_db(dataset_json: str, collection_name=COLLECTION, persist_directory=PERSIST_DIR, rebuild=False):
    """
    rebuild=False: ถ้ามี DB อยู่แล้ว -> เปิดใช้เลย (ไม่ add ซ้ำ)
    rebuild=True : ลบทิ้งแล้วสร้างใหม่ (index ใหม่ทั้งหมด)
    """
    if rebuild and os.path.exists(persist_directory):
        # ลบทิ้งทั้งโฟลเดอร์เพื่อกันข้อมูลเก่าปน (ชัวร์สุด)
        import shutil
        shutil.rmtree(persist_directory)

    # ถ้าเคยสร้างไว้แล้ว -> OPEN (ไม่สร้างซ้ำ)
    if os.path.exists(persist_directory) and not rebuild:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=ProductionCLIPChartEmbeddings(),  # ต้องเหมือนเดิม
            persist_directory=persist_directory,
        )
        return db

    # ไม่เจอของเดิม -> BUILD
    raw = load_dataset(dataset_json)

    docs = []
    for item in raw:
        p = os.path.normpath(item["image"])
        docs.append(Document(page_content=p, metadata={"image": p}))

    db = Chroma.from_documents(
        docs,
        embedding=ProductionCLIPChartEmbeddings(),
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    # ไม่ต้อง db.persist() แล้ว (Chroma ใหม่ auto persist)
    return db

def search_similar(db: Chroma, query_image: str, k=10):
    hits = db.similarity_search(query_image, k=k)
    seen = set()
    out = []
    for h in hits:
        img = h.metadata.get("image")
        # if img not in seen:
        seen.add(img)
        out.append(h)
    return out

def _percentile_norm(x: np.ndarray, lo_p=20, hi_p=95) -> np.ndarray:
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

@torch.no_grad()
def visualize_debug(
    image_path: str,
    lo_p: int = 20,
    hi_p: int = 95,
    heat_alpha: float = 0.55,
    dim_factor: float = 0.25,
):
    pil = load_rgb(image_path)
    wide224, _ = build_views(pil)
    text_feat = build_text_feature(PROMPTS)

    ly = LAYERS[-1]

    inputs = processor(images=wide224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    hs = vision_out.hidden_states[ly]
    patch_tokens = hs[:, 1:, :]
    patch_emb = clip_model.visual_projection(patch_tokens)
    patch_emb = F.normalize(patch_emb, dim=-1)

    sim = (patch_emb @ text_feat.transpose(-1, -2)).squeeze(0).squeeze(-1).detach().cpu().numpy()
    P = sim.shape[0]
    g = int(round(math.sqrt(P)))
    sim_grid = sim.reshape(g, g)

    prior = spatial_prior(g, strength=PRIOR_STRENGTH)
    dens = patch_density_mask(wide224, g)
    dens = suppress_long_horizontal_runs(dens, min_run=HLINE_MIN_RUN, penalty=HLINE_PENALTY)

    sim2 = sim_grid * prior * (dens ** DENSITY_POWER)

    flat = sim2.flatten()
    k = max(1, int(len(flat) * float(TOPK_RATIO)))
    thr = np.sort(flat)[-k]
    keep0 = (sim2 >= thr)
    keep = prune_by_connectivity(keep0, min_area=CC_MIN_AREA, aspect_ratio_thresh=CC_ASPECT_THRESH)
    if not keep.any():
        keep = keep0

    sim_norm = _percentile_norm(sim2, lo_p=lo_p, hi_p=hi_p)

    img_np = np.array(wide224)
    H, W = img_np.shape[:2]

    heat = np.kron(sim_norm, np.ones((H // g, W // g)))[:H, :W]
    keep_map = np.kron(keep.astype(np.float32), np.ones((H // g, W // g)))[:H, :W]

    masked = img_np.astype(np.float32)
    masked[keep_map < 0.5] *= float(dim_factor)
    masked = np.clip(masked, 0, 255).astype(np.uint8)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.title("Wide View (ROI+letterbox)")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title(f"Heatmap (layer {ly}) [percentile norm {lo_p}-{hi_p}]")
    plt.imshow(img_np)
    plt.imshow(heat, alpha=float(heat_alpha))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Top-{int(TOPK_RATIO*100)}% kept (pruned) [dim={dim_factor}]")
    plt.imshow(masked)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATASET_JSON = "dataset.json"
    QUERY_IMAGE = "datasets1/new_chart3.png"

    # visualize_debug(QUERY_IMAGE)
    text_data = load_dataset(DATASET_JSON)

    db = build_image_db(DATASET_JSON, rebuild=True)
    print("db count:", db._collection.count())
    hits = search_similar(db, QUERY_IMAGE, k=5)
    print("\nTop matches:")
    for i, h in enumerate(hits, 1):
        for item in text_data:
            if item['image'] == h.metadata['image']:
                print(f"{i}. {h.metadata['image']} : {item['data']}")
            