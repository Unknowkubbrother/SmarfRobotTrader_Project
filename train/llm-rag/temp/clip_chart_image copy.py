# clip_chart_prod_v4.py
# ------------------------------------------------------------
# Production-grade Chart Retrieval (CLIP hidden-layer) - V4
# - No rebuild unless you want (rebuild=False by default)
# - Upsert only new images (no duplicates) with ID = normalized path
# - Accuracy upgrades:
#   (1) Prompt ensemble MAX per patch (instead of avg prompt feature)
#   (2) STRUCT view removes horizontal lines via morphology
#   (3) Auto focus crop by vertical-edge energy (wide + zoom + focus)
#
# Install:
#   pip install opencv-python chromadb langchain-community transformers torch pillow matplotlib
# ------------------------------------------------------------

import os
import json
import math
import shutil
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


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-large-patch14"

# ROI crop to remove UI (tune right/bottom for MT5 if needed)
ROI_CUT = dict(left=0.02, top=0.06, right=0.14, bottom=0.12)

LAYERS = (-6, -3)
TOPK_RATIO = 0.12
PRIOR_STRENGTH = 1.6
DENSITY_POWER = 2.3
DENSITY_MIN_CLIP = 0.12

HLINE_MIN_RUN = 3
HLINE_PENALTY = 0.10

CC_MIN_AREA = 2
CC_ASPECT_THRESH = 0.45

W_REAL = 0.55
W_STRUCT = 0.45

ANCHOR_PROMPTS = [
    "TradingView candlestick chart screenshot",
    "MetaTrader 5 candlestick chart screenshot",
]

CANDLE_PROMPTS = [
    "OHLC candlesticks body and wicks close-up on trading chart",
    "red and green candlesticks with wicks close-up",
    "pin bar long upper wick rejection at resistance on candlestick chart",
    "pin bar long lower wick rejection at support on candlestick chart",
    "long wick rejection candle at swing high",
    "long wick rejection candle at swing low",
    "liquidity sweep wick then reversal on candlestick chart",
    "fakeout breakout wick then reversal on candlestick chart",
]


PERSIST_DIR = "chroma_store_images1"
COLLECTION = "chart_clip_images1"


# ============================================================
# MODEL
# ============================================================
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)


# ============================================================
# PATH + DATASET
# ============================================================
def norm_path(p: str) -> str:
    return os.path.normpath(p).replace("\\", "/")

def load_dataset(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def dataset_unique_paths(dataset_json: str):
    raw = load_dataset(dataset_json)
    paths = [norm_path(it["image"]) for it in raw if "image" in it]
    uniq = sorted(set(paths))
    return raw, uniq


# ============================================================
# IMAGE PREP
# ============================================================
def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def crop_chart_roi(pil_img: Image.Image, left=0.02, top=0.06, right=0.14, bottom=0.12) -> Image.Image:
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

def _auto_focus_crop(roi: Image.Image, win_ratio=0.72) -> Image.Image:
    """
    หาโซนแท่งเทียนด้วยพลังงานขอบแนวตั้ง (Sobel-x)
    แล้ว crop window กว้าง ~70% ที่มีพลังงานมากสุด
    """
    img = np.array(roi)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    energy = np.abs(gx).mean(axis=0)  # per-column

    w = gray.shape[1]
    win = max(32, int(w * win_ratio))
    # moving sum
    cs = np.cumsum(energy, dtype=np.float64)
    cs = np.concatenate([[0.0], cs])
    best_s, best_v = 0, -1e18
    for s in range(0, w - win + 1, max(1, win // 40)):
        v = cs[s + win] - cs[s]
        if v > best_v:
            best_v, best_s = v, s
    x1, x2 = best_s, best_s + win
    return roi.crop((x1, 0, x2, roi.size[1]))

def build_views(pil_img: Image.Image):
    """
    wide: ROI เต็ม
    zoom: ROI ส่วนบน
    focus: auto crop โซนแท่งเทียน (ช่วย MT5/TV ที่มี UI/scale ต่างกัน)
    """
    roi = crop_chart_roi(pil_img, **ROI_CUT)
    w, h = roi.size
    zoom = roi.crop((0, 0, w, int(h * 0.55)))
    focus = _auto_focus_crop(roi, win_ratio=0.72)
    return (
        letterbox(roi, 224),
        letterbox(zoom, 224),
        letterbox(focus, 224),
    )

def make_structure_view(pil_224: Image.Image) -> Image.Image:
    """
    STRUCT view: edges + remove horizontal lines (grid/price line)
    """
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 40, 120)

    # find horizontal lines and subtract them
    # (stronger than just penalizing runs)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

    # slight dilate to connect candle strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


# ============================================================
# TEXT FEAT (PROMPT ENSEMBLE MAX)
# ============================================================
@torch.no_grad()
def build_text_features(prompts: list[str]):
    """
    Return: [T,E] normalized text features (NO averaging)
    We'll take max similarity per patch across prompts.
    """
    t_in = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    t_feat = clip_model.get_text_features(**t_in)     # [T,E]
    t_feat = F.normalize(t_feat, dim=-1)
    return t_feat


# ============================================================
# PRIORS & MASKS
# ============================================================
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

def suppress_long_horizontal_runs(density_grid: np.ndarray, min_run: int = 3, penalty: float = 0.10) -> np.ndarray:
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


# ============================================================
# CORE: PATCH POOLING (PROMPT MAX)
# ============================================================
@torch.no_grad()
def patch_pooled_embedding(
    pil_224: Image.Image,
    text_feats: torch.Tensor,   # [T,E]
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
        patch_tokens = hs[:, 1:, :]  # [1,P,D]

        patch_emb = clip_model.visual_projection(patch_tokens)  # [1,P,E]
        patch_emb = F.normalize(patch_emb, dim=-1)

        # sim per prompt -> take MAX (key improvement)
        sim_all = patch_emb @ text_feats.T  # [1,P,T]
        sim = sim_all.max(dim=-1).values.squeeze(0)  # [P]
        sim_np = sim.detach().cpu().numpy()

        P = sim_np.shape[0]
        g = int(round(math.sqrt(P)))
        if g * g != P:
            raise ValueError(f"Patch count {P} not square (got {P}).")
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


# ============================================================
# FINAL EMBEDDING (REAL+STRUCT, WIDE+ZOOM+FOCUS)
# ============================================================
def embed_chart_image(image_path: str) -> list:
    pil = load_rgb(image_path)
    wide224, zoom224, focus224 = build_views(pil)

    text_anchor = build_text_features(ANCHOR_PROMPTS)
    text_candle = build_text_features(CANDLE_PROMPTS)

    # REAL: average 3 views
    v_real = (
        patch_pooled_embedding(wide224, text_anchor) +
        patch_pooled_embedding(zoom224, text_candle) +
        patch_pooled_embedding(focus224, text_candle)
    ) / 3.0
    v_real = v_real / (np.linalg.norm(v_real) + 1e-9)

    # STRUCT: average 3 views
    wide_s = make_structure_view(wide224)
    zoom_s = make_structure_view(zoom224)
    focus_s = make_structure_view(focus224)

    v_struct = (
        patch_pooled_embedding(wide_s, text_anchor) +
        patch_pooled_embedding(zoom_s, text_candle) +
        patch_pooled_embedding(focus_s, text_candle)
    ) / 3.0
    v_struct = v_struct / (np.linalg.norm(v_struct) + 1e-9)

    out = (W_REAL * v_real) + (W_STRUCT * v_struct)
    out = out / (np.linalg.norm(out) + 1e-9)
    return out.tolist()


# ============================================================
# LANGCHAIN EMBEDDINGS
# ============================================================
class ProductionCLIPChartEmbeddings(Embeddings):
    def embed_documents(self, image_paths):
        return [embed_chart_image(p) for p in image_paths]

    def embed_query(self, image_path):
        return embed_chart_image(image_path)


# ============================================================
# CHROMA: OPEN / UPSERT / REBUILD
# ============================================================
def open_db():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=ProductionCLIPChartEmbeddings(),
        persist_directory=PERSIST_DIR,
    )

def _get_existing_ids_batched(db: Chroma, ids, batch=1000):
    """
    chroma get(ids=...) can be heavy for very large lists. batch it.
    """
    existing = set()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        got = db._collection.get(ids=chunk, include=[])
        if got and "ids" in got and got["ids"]:
            existing.update(got["ids"])
    return existing

def upsert_dataset(dataset_json: str):
    """
    Production default:
    - dedupe dataset.json by path
    - add only missing docs using ID = path
    - never rebuild and never duplicates
    """
    db = open_db()
    raw, uniq_paths = dataset_unique_paths(dataset_json)
    if not uniq_paths:
        raise ValueError("dataset.json contains no image paths")

    existing = _get_existing_ids_batched(db, uniq_paths, batch=1000)
    new_paths = [p for p in uniq_paths if p not in existing]

    if not new_paths:
        print(f"✅ No new images. count={db._collection.count()}")
        return db

    docs = [Document(page_content=p, metadata={"image": p}) for p in new_paths]
    ids = new_paths[:]  # stable unique ids

    db.add_documents(docs, ids=ids)
    print(f"✅ Added {len(new_paths)} new images. count={db._collection.count()}")
    return db

def rebuild_db(dataset_json: str):
    """
    Use only when:
    - you changed embedding pipeline significantly, or
    - you want a clean rebuild.
    """
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    db = open_db()
    raw, uniq_paths = dataset_unique_paths(dataset_json)

    docs = [Document(page_content=p, metadata={"image": p}) for p in uniq_paths]
    ids = uniq_paths[:]
    db.add_documents(docs, ids=ids)

    print(f"✅ Rebuilt DB. unique_docs={len(uniq_paths)} count={db._collection.count()}")
    return db


# ============================================================
# SEARCH
# ============================================================
def search_similar(db: Chroma, query_image: str, k=5):
    return db.similarity_search(query_image, k=k)


# ============================================================
# DEBUG
# ============================================================
def _percentile_norm(x: np.ndarray, lo_p=20, hi_p=95) -> np.ndarray:
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

@torch.no_grad()
def visualize_debug(image_path: str, lo_p=20, hi_p=95, heat_alpha=0.55, dim_factor=0.25):
    pil = load_rgb(image_path)
    wide224, _, _ = build_views(pil)
    anchor_feats = build_text_features(ANCHOR_PROMPTS)
    candle_feats = build_text_features(CANDLE_PROMPTS)
    
    ly = LAYERS[-1]

    inputs = processor(images=wide224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    hs = vision_out.hidden_states[ly]
    patch_tokens = hs[:, 1:, :]
    patch_emb = clip_model.visual_projection(patch_tokens)
    patch_emb = F.normalize(patch_emb, dim=-1)

    sim_anchor = patch_emb @ anchor_feats.T
    sim_candle = patch_emb @ candle_feats.T
    sim_anchor = sim_anchor.max(dim=-1).values.squeeze(0).detach().cpu().numpy()
    sim_candle = sim_candle.max(dim=-1).values.squeeze(0).detach().cpu().numpy()
    sim = sim_candle + 0.15 *sim_anchor

    P = sim.shape[0]
    g = int(round(math.sqrt(P)))
    sim_grid = sim.reshape(g, g)

    prior = spatial_prior(g, strength=PRIOR_STRENGTH)
    dens = patch_density_mask(wide224, g)
    dens = suppress_long_horizontal_runs(dens, min_run=HLINE_MIN_RUN, penalty=HLINE_PENALTY)
    sim2 = sim_grid * prior * (dens ** DENSITY_POWER)

    # top-k + prune
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

    overlay = img_np.copy()
    overlay[keep_map >= 0.5] = (255, 0, 0)  # แดงทับพื้นที่ที่ keep

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1); plt.title("Wide (ROI+letterbox)"); plt.imshow(img_np); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title(f"Heatmap MAX-prompt (layer {ly})"); plt.imshow(img_np); plt.imshow(heat, alpha=float(heat_alpha)); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Masked"); plt.imshow(img_np); plt.imshow(overlay, alpha=0.25); plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DATASET_JSON = "dataset.json"
    QUERY_IMAGE = "datasets1/new_chart3.png"

    # Production default: upsert only new, no duplicates, no rebuild
    # db = upsert_dataset(DATASET_JSON)

    # print("db count:", db._collection.count())

    # Optional debug
    visualize_debug(QUERY_IMAGE)

    # hits = search_similar(db, QUERY_IMAGE, k=5)

    # raw = load_dataset(DATASET_JSON)
    # lookup = {norm_path(it["image"]): it.get("data", "") for it in raw if "image" in it}

    # print("\nTop matches:")
    # for i, h in enumerate(hits, 1):
    #     p = norm_path(h.metadata["image"])
    #     print(f"{i}. {p} : {lookup.get(p, '')}")
