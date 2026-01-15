# clip_chart_prod_v4_noprompt.py
# ------------------------------------------------------------
# Production-grade Chart Retrieval (CLIP hidden-layer) - V4 (NO PROMPTS)
# - No rebuild unless you want
# - Upsert only new images (no duplicates) with ID = normalized path
# - Patch pooling WITHOUT text prompts:
#   use edge/gradient saliency (dens + edge gate + priors) to pick patches
# - Multi-layer weighted pooling + multi-view (wide/zoom/focus) + STRUCT view
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

# ROI crop to remove UI (tune for TV/MT5)
ROI_CUT = dict(left=0.05, top=0.06, right=0.14, bottom=0.15)

# layers: allow negative indexing and positive (hidden_states index)
LAYERS = (-6, -3, 24)
LAYER_WEIGHTS = (0.25, 0.25, 0.50)  # must match len(LAYERS)

# patch selection
TOPK_RATIO = 0.12
KEEP_DILATE_ITERS = 1      # grow keep a bit
CC_MIN_AREA = 1            # IMPORTANT: allow single-cell keep
HORIZ_W_OVER_H = 2.2       # remove long horizontal blobs

# priors / saliency
PRIOR_STRENGTH = 1.6       # center prior
RIGHT_PRIOR_POWER = 0.7    # prefer right side (recent candles)
RIGHT_PRIOR_BASE = 0.55

DENSITY_POWER = 1.3        # lower than before (more coverage)
MAG_FLOOR = 0.12           # edge magnitude gate (raise to 0.15 if background leaks)
MAG_GAMMA = 2.0            # gate^gamma (stronger suppression for flat bg)
EDGE_THR = 0.02            # patch edge occupancy threshold
USE_EDGE_GATE = True

# remove horizontal grid lines
HLINE_MIN_RUN = 3
HLINE_PENALTY = 0.05
HLINE_THR = 0.35           # treat dens > thr as "horizontal-ish" to penalize

# pooling temperature (higher -> focus on top patches)
POOL_TEMP = 6.0

# REAL/STRUCT blend
W_REAL = 0.60
W_STRUCT = 0.40

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

def crop_chart_roi(pil_img: Image.Image, left=0.05, top=0.06, right=0.14, bottom=0.15) -> Image.Image:
    w, h = pil_img.size
    x1 = int(w * left)
    y1 = int(h * top)
    x2 = int(w * (1 - right))
    y2 = int(h * (1 - bottom))
    return pil_img.crop((x1, y1, x2, y2))

def content_crop_by_edges(pil_img: Image.Image, pad_frac=0.04) -> Image.Image:
    """Crop to edge-bounding-box (helps remove large black areas after ROI)."""
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    ys, xs = np.where(edges > 0)
    if len(xs) < 50:
        return pil_img  # fallback

    h, w = gray.shape
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x1 = max(0, x1 - pad_x)
    x2 = min(w - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)

    return pil_img.crop((x1, y1, x2 + 1, y2 + 1))

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
    """Find candle zone by vertical-edge energy and crop a wide window."""
    img = np.array(roi)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    energy = np.abs(gx).mean(axis=0)

    w = gray.shape[1]
    win = max(32, int(w * win_ratio))
    cs = np.cumsum(energy, dtype=np.float64)
    cs = np.concatenate([[0.0], cs])

    best_s, best_v = 0, -1e18
    step = max(1, win // 40)
    for s in range(0, w - win + 1, step):
        v = cs[s + win] - cs[s]
        if v > best_v:
            best_v, best_s = v, s

    x1, x2 = best_s, best_s + win
    return roi.crop((x1, 0, x2, roi.size[1]))

def build_views(pil_img: Image.Image):
    roi = crop_chart_roi(pil_img, **ROI_CUT)
    roi = content_crop_by_edges(roi, pad_frac=0.04)

    w, h = roi.size
    zoom = roi.crop((0, 0, w, int(h * 0.60)))
    focus = _auto_focus_crop(roi, win_ratio=0.72)

    return (
        letterbox(roi, 224),
        letterbox(zoom, 224),
        letterbox(focus, 224),
    )

def make_structure_view(pil_224: Image.Image) -> Image.Image:
    """STRUCT: edges + remove horizontal lines."""
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 40, 120)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


# ============================================================
# PRIORS & MASKS (NO PROMPTS)
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

def right_prior(grid_size: int, power=0.7, base=0.55) -> np.ndarray:
    """Prefer right side (recent candles). base keeps left side non-zero."""
    g = grid_size
    xs = np.linspace(0.0, 1.0, g, dtype=np.float32)
    w = base + (1.0 - base) * (xs ** power)
    return np.tile(w[None, :], (g, 1))

def patch_density_mask(pil_224: Image.Image, grid_size: int):
    """
    dens: (vertical-ish / horizontal-ish) * edge-magnitude gate
    valid: gate > eps (remove flat background)
    """
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag_x = np.abs(gx)
    mag_y = np.abs(gy)
    mag = mag_x + mag_y

    H, W = gray.shape
    ph, pw = H // grid_size, W // grid_size

    ratio = np.zeros((grid_size, grid_size), dtype=np.float32)
    mgrid = np.zeros((grid_size, grid_size), dtype=np.float32)

    for y in range(grid_size):
        for x in range(grid_size):
            px = mag_x[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            py = mag_y[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            mm = mag[y*ph:(y+1)*ph, x*pw:(x+1)*pw]

            v_like = float(px.mean())
            h_like = float(py.mean())
            ratio[y, x] = v_like / (h_like + 1e-6)
            mgrid[y, x] = float(mm.mean())

    # magnitude gate (0..1), no baseline
    mg = mgrid / (float(mgrid.max()) + 1e-9)
    gate = np.clip((mg - float(MAG_FLOOR)) / (1.0 - float(MAG_FLOOR) + 1e-9), 0.0, 1.0)
    gate = gate ** float(MAG_GAMMA)
    valid = gate > 1e-3

    # normalize ratio inside valid
    if valid.any():
        rmin = float(ratio[valid].min()); rmax = float(ratio[valid].max())
    else:
        rmin = float(ratio.min()); rmax = float(ratio.max())
    ratio_norm = (ratio - rmin) / (rmax - rmin + 1e-9)
    ratio_norm = np.clip(ratio_norm, 0.0, 1.0)

    dens = ratio_norm * gate
    dens = np.clip(dens, 0.0, 1.0)
    return dens, valid

def patch_edge_mask(pil_224: Image.Image, grid_size: int, thr=0.02) -> np.ndarray:
    """Boolean mask: patch has enough edge pixels."""
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)

    H, W = gray.shape
    ph, pw = H // grid_size, W // grid_size

    out = np.zeros((grid_size, grid_size), dtype=bool)
    for y in range(grid_size):
        for x in range(grid_size):
            e = edges[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            out[y, x] = (float((e > 0).mean()) >= float(thr))
    return out

def suppress_long_horizontal_runs(density_grid: np.ndarray, min_run: int = 3, penalty: float = 0.05, thr: float = 0.35) -> np.ndarray:
    """
    Penalize long horizontal runs of high density (often grid/price lines).
    Only penalize cells above thr.
    """
    g = density_grid.shape[0]
    out = density_grid.copy()

    for y in range(g):
        run_start = None
        for x in range(g):
            if out[y, x] > thr:
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

def prune_by_connectivity(keep_mask: np.ndarray, min_area: int = 1, horiz_w_over_h: float = 2.2) -> np.ndarray:
    """
    Remove blobs that are too horizontal (grid lines).
    Keep all with area >= min_area AND not too horizontal.
    """
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
        if h > 0 and (w / float(h)) >= float(horiz_w_over_h):
            continue
        out[ys, xs] = True

    return out


# ============================================================
# CORE: PATCH POOLING (NO PROMPTS)
# ============================================================
def _get_hs_for_layer(vision_out, ly: int):
    # allow positive index in hidden_states (0..N), negative python indexing
    if ly == -1:
        return vision_out.last_hidden_state
    return vision_out.hidden_states[ly]

@torch.no_grad()
def patch_pooled_embedding(
    pil_224: Image.Image,
    layers=LAYERS,
    layer_weights=LAYER_WEIGHTS,
    topk_ratio=TOPK_RATIO,
):
    assert len(layers) == len(layer_weights), "LAYER_WEIGHTS must match LAYERS"

    inputs = processor(images=pil_224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    pooled_list = []
    wsum = float(sum(layer_weights)) + 1e-9

    for ly, w in zip(layers, layer_weights):
        hs = _get_hs_for_layer(vision_out, int(ly))
        patch_tokens = hs[:, 1:, :]  # [1,P,D]

        patch_emb = clip_model.visual_projection(patch_tokens)  # [1,P,E]
        patch_emb = F.normalize(patch_emb, dim=-1)

        P = patch_emb.shape[1]
        g = int(round(math.sqrt(P)))
        if g * g != P:
            raise ValueError(f"Patch count {P} not square (got {P}).")

        # saliency (no prompt)
        prior = spatial_prior(g, strength=float(PRIOR_STRENGTH))
        rprior = right_prior(g, power=float(RIGHT_PRIOR_POWER), base=float(RIGHT_PRIOR_BASE))

        dens, valid = patch_density_mask(pil_224, g)
        dens = suppress_long_horizontal_runs(dens, min_run=int(HLINE_MIN_RUN), penalty=float(HLINE_PENALTY), thr=float(HLINE_THR))

        if USE_EDGE_GATE:
            em = patch_edge_mask(pil_224, g, thr=float(EDGE_THR))
            valid = valid & em

        sal = dens * prior * rprior
        sal = np.clip(sal, 0.0, 1.0)

        sal_cut = sal.copy()
        sal_cut[~valid] = 0.0

        flat = sal_cut[valid].flatten()
        if flat.size == 0:
            # fallback: allow all
            valid = np.ones_like(valid, dtype=bool)
            sal_cut = sal.copy()
            flat = sal_cut.flatten()

        k = max(1, int(len(flat) * float(topk_ratio)))
        thr = np.sort(flat)[-k]
        keep0 = (sal_cut >= thr) & valid

        # dilate to connect thin strokes
        if KEEP_DILATE_ITERS > 0:
            k3 = np.ones((3, 3), np.uint8)
            keep0 = cv2.dilate(keep0.astype(np.uint8), k3, iterations=int(KEEP_DILATE_ITERS)).astype(bool) & valid

        keep = prune_by_connectivity(keep0, min_area=int(CC_MIN_AREA), horiz_w_over_h=float(HORIZ_W_OVER_H))
        if not keep.any():
            keep = keep0

        mask = torch.tensor(keep.flatten(), device=DEVICE)
        selected = patch_emb[0][mask]

        # weights from saliency (temperature)
        sal_t = torch.tensor(sal.flatten(), device=DEVICE, dtype=torch.float32)
        sel_sal = sal_t[mask]
        weights = F.softmax(sel_sal * float(POOL_TEMP), dim=0).unsqueeze(-1)

        pooled = (selected * weights).sum(dim=0)
        pooled = F.normalize(pooled, dim=-1)

        pooled_list.append(pooled * (float(w) / wsum))

    out = torch.stack(pooled_list, dim=0).sum(dim=0)
    out = F.normalize(out, dim=-1)
    return out.detach().cpu().numpy()


# ============================================================
# FINAL EMBEDDING (REAL+STRUCT, WIDE+ZOOM+FOCUS)
# ============================================================
def embed_chart_image(image_path: str) -> list:
    pil = load_rgb(image_path)
    wide224, zoom224, focus224 = build_views(pil)

    v_real = (
        patch_pooled_embedding(wide224) +
        patch_pooled_embedding(zoom224) +
        patch_pooled_embedding(focus224)
    ) / 3.0
    v_real = v_real / (np.linalg.norm(v_real) + 1e-9)

    wide_s = make_structure_view(wide224)
    zoom_s = make_structure_view(zoom224)
    focus_s = make_structure_view(focus224)

    v_struct = (
        patch_pooled_embedding(wide_s) +
        patch_pooled_embedding(zoom_s) +
        patch_pooled_embedding(focus_s)
    ) / 3.0
    v_struct = v_struct / (np.linalg.norm(v_struct) + 1e-9)

    out = (float(W_REAL) * v_real) + (float(W_STRUCT) * v_struct)
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
    existing = set()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        got = db._collection.get(ids=chunk, include=[])
        if got and "ids" in got and got["ids"]:
            existing.update(got["ids"])
    return existing

def upsert_dataset(dataset_json: str):
    db = open_db()
    _, uniq_paths = dataset_unique_paths(dataset_json)
    if not uniq_paths:
        raise ValueError("dataset.json contains no image paths")

    existing = _get_existing_ids_batched(db, uniq_paths, batch=1000)
    new_paths = [p for p in uniq_paths if p not in existing]

    if not new_paths:
        print(f"✅ No new images. count={db._collection.count()}")
        return db

    docs = [Document(page_content=p, metadata={"image": p}) for p in new_paths]
    db.add_documents(docs, ids=new_paths[:])
    print(f"✅ Added {len(new_paths)} new images. count={db._collection.count()}")
    return db

def rebuild_db(dataset_json: str):
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    db = open_db()
    _, uniq_paths = dataset_unique_paths(dataset_json)
    docs = [Document(page_content=p, metadata={"image": p}) for p in uniq_paths]
    db.add_documents(docs, ids=uniq_paths[:])
    print(f"✅ Rebuilt DB. unique_docs={len(uniq_paths)} count={db._collection.count()}")
    return db


# ============================================================
# SEARCH
# ============================================================
def search_similar(db: Chroma, query_image: str, k=5):
    return db.similarity_search(query_image, k=k)


# ============================================================
# DEBUG (COMBINED HEAT + KEEP)
# ============================================================
def _percentile_norm(x: np.ndarray, lo_p=20, hi_p=95) -> np.ndarray:
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

def content_mask_from_edges(pil_224: Image.Image, g: int, thr=0.01):
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)

    H, W = gray.shape
    ph, pw = H // g, W // g
    m = np.zeros((g, g), dtype=bool)
    for y in range(g):
        for x in range(g):
            e = edges[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            m[y, x] = (float((e > 0).mean()) >= float(thr))
    return m


@torch.no_grad()
def visualize_debug(image_path: str, debug_layer="combined", lo_p=20, hi_p=95, heat_alpha=0.55):
    pil = load_rgb(image_path)
    wide224, _, _ = build_views(pil)
    img_np = np.array(wide224)

    inputs = processor(images=wide224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    def sim_sal_keep_for_layer(ly: int):
        hs = _get_hs_for_layer(vision_out, int(ly))
        patch_tokens = hs[:, 1:, :]
        patch_emb = F.normalize(clip_model.visual_projection(patch_tokens), dim=-1)

        P = patch_emb.shape[1]
        g = int(round(math.sqrt(P)))

        prior = spatial_prior(g, strength=float(PRIOR_STRENGTH))
        rprior = right_prior(g, power=float(RIGHT_PRIOR_POWER), base=float(RIGHT_PRIOR_BASE))

        dens, valid = patch_density_mask(wide224, g)
        dens = suppress_long_horizontal_runs(dens, min_run=int(HLINE_MIN_RUN), penalty=float(HLINE_PENALTY), thr=float(HLINE_THR))

        if USE_EDGE_GATE:
            em = patch_edge_mask(wide224, g, thr=float(EDGE_THR))
            valid = valid & em

        sal = np.clip(dens * prior * rprior, 0.0, 1.0)

        sal_cut = sal.copy()
        sal_cut[~valid] = 0.0
        flat = sal_cut[valid].flatten()
        if flat.size == 0:
            valid = np.ones_like(valid, dtype=bool)
            sal_cut = sal.copy()
            flat = sal_cut.flatten()

        k = max(1, int(len(flat) * float(TOPK_RATIO)))
        thr = np.sort(flat)[-k]
        keep0 = (sal_cut >= thr) & valid

        if KEEP_DILATE_ITERS > 0:
            keep0 = cv2.dilate(keep0.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(KEEP_DILATE_ITERS)).astype(bool) & valid

        keep = prune_by_connectivity(keep0, min_area=int(CC_MIN_AREA), horiz_w_over_h=float(HORIZ_W_OVER_H))
        if not keep.any():
            keep = keep0

        return sal, keep, g

    # ---------- single layer ----------
    if debug_layer != "combined":
        sal, keep, g = sim_sal_keep_for_layer(int(debug_layer))
        sal_norm = _percentile_norm(sal, lo_p=lo_p, hi_p=hi_p)

        H, W = img_np.shape[:2]
        heat = np.kron(sal_norm, np.ones((H // g, W // g)))[:H, :W]
        keep_map = np.kron(keep.astype(np.float32), np.ones((H // g, W // g)))[:H, :W]

        overlay = img_np.copy()
        overlay[keep_map >= 0.5] = (255, 0, 0)
        overlay2 = content_mask_from_edges(wide224, g)
        overlay2 = np.kron(overlay2.astype(np.float32), np.ones((H // g, W // g)))[:H, :W]
        overlay2 = overlay2 * 255

        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1); plt.title("Wide (ROI+content crop+letterbox)"); plt.imshow(img_np); plt.axis("off")
        plt.subplot(1, 3, 2); plt.title(f"Heatmap (layer {debug_layer}) [NO PROMPT]"); plt.imshow(img_np); plt.imshow(heat, alpha=float(heat_alpha)); plt.axis("off")
        plt.subplot(1, 3, 3); plt.title("Keep mask overlay"); plt.imshow(img_np); plt.imshow(overlay, alpha=0.25); plt.imshow(overlay2, alpha=0.25); plt.axis("off")
        plt.tight_layout()
        plt.show()
        return

    # ---------- combined ----------
    wsum = float(sum(LAYER_WEIGHTS)) + 1e-9
    heat_sum = None
    keep_union = None
    g0 = None

    for ly, w in zip(LAYERS, LAYER_WEIGHTS):
        sal, keep, g = sim_sal_keep_for_layer(int(ly))
        sal_norm = _percentile_norm(sal, lo_p=lo_p, hi_p=hi_p)

        if heat_sum is None:
            heat_sum = (float(w) / wsum) * sal_norm
            keep_union = keep.copy()
            g0 = g
        else:
            heat_sum += (float(w) / wsum) * sal_norm
            keep_union |= keep

    H, W = img_np.shape[:2]
    heat = np.kron(heat_sum, np.ones((H // g0, W // g0)))[:H, :W]
    keep_map = np.kron(keep_union.astype(np.float32), np.ones((H // g0, W // g0)))[:H, :W]

    overlay = img_np.copy()
    overlay[keep_map >= 0.5] = (255, 0, 0)

    overlay2 = content_mask_from_edges(wide224, g0)
    overlay2 = np.kron(overlay2.astype(np.float32), np.ones((H // g0, W // g0)))[:H, :W]
    overlay2 = overlay2 * 255

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1); plt.title("Wide (ROI+content crop+letterbox)"); plt.imshow(img_np); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title(f"Heatmap COMBINED {LAYERS} w={LAYER_WEIGHTS} [NO PROMPT]"); plt.imshow(img_np); plt.imshow(heat, alpha=float(heat_alpha)); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Keep mask overlay (UNION)"); plt.imshow(img_np); plt.imshow(overlay, alpha=0.25); plt.imshow(overlay2, alpha=0.25); plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    DATASET_JSON = "dataset.json"
    QUERY_IMAGE = "datasets1/chart29.jpg"

    # Production default: upsert only new, no duplicates, no rebuild
    # db = upsert_dataset(DATASET_JSON)
    # print("db count:", db._collection.count())

    # Debug visualize
    visualize_debug(QUERY_IMAGE, debug_layer="combined")

    # Example search:
    # db = open_db()
    # hits = search_similar(db, QUERY_IMAGE, k=5)
    # for i, h in enumerate(hits, 1):
    #     print(i, h.metadata["image"])
