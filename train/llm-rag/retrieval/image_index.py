# image_index.py
# ------------------------------------------------------------
# Production-grade Chart Retrieval (CLIP hidden-layer) - V6 (NO PROMPT)
# MASTER SOURCE: clip_chart_prod_v6_noprompt.py (user provided)
#
# Fixes:
# ✅ content-global (from edge-rich patches) instead of global image embedding
# ✅ edge mask removes horizontal lines BEFORE scoring/valid-mask
# ✅ edge-boost + background-subtract to suppress background bands
# ✅ keep mask never empty (adaptive + min_keep + fallback)
# ✅ single pipeline = USED for VectorDB (no global/local confusion)
#
# Port notes for module:
# - lazy-load CLIP model/processor to avoid double-load on import
# - dataset_unique_paths imported from .dataset_utils
# - dedupe/upsert by normalized path ids
# ------------------------------------------------------------

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import math
import shutil
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document

from .dataset_utils import dataset_unique_paths


# ============================================================
# CONFIG (match clip_chart_prod_v6_noprompt.py)
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-large-patch14"

# You can override via env without changing code
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_store_images1")
COLLECTION = os.getenv("CHROMA_COLLECTION", "chart_clip_images1")

# Crop UI (tune for MT5/TV)
ROI_CUT = dict(left=0.02, top=0.06, right=0.14, bottom=0.12)

# Remove big black bars inside ROI
CONTENT_ROW_THR = 10
CONTENT_MARGIN = 6

# Multi-layer pooling (USED for VectorDB)
LAYERS = (-6, -3, -1)
LAYER_WEIGHTS = (0.15, 0.20, 0.65)

# Patch selection
TOPK_RATIO = 0.16
MIN_KEEP_PATCHES = 12
CC_MIN_AREA = 2
CC_ASPECT_THRESH = 0.45

# Spatial + density priors
PRIOR_STRENGTH = 1.35
DENSITY_POWER = 2.0
DENSITY_MIN_CLIP = 0.10
HLINE_MIN_RUN = 4
HLINE_PENALTY = 0.08

# Edge valid-mask (adaptive)
EDGE_BASE_THR = 0.0035
EDGE_PCT = 75.0

# Suppress background
EDGE_BOOST_POWER = 1.6
BG_SUBTRACT = 0.45
CONTENT_GLOBAL_EDGE_GAMMA = 1.5

# Views (REAL + STRUCT)
W_REAL = 0.60
W_STRUCT = 0.40

# Pooling
SOFTMAX_TEMP = 1.2


# ============================================================
# LAZY MODEL LOADER (avoid double-load on import)
# ============================================================
_clip_model: Optional[CLIPModel] = None
_processor: Optional[CLIPProcessor] = None


def get_clip() -> Tuple[CLIPModel, CLIPProcessor]:
    global _clip_model, _processor
    if _clip_model is None or _processor is None:
        _clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME, use_fast=True)
    return _clip_model, _processor


# ============================================================
# PATH NORMALIZATION (dedupe ids across slash/case differences)
# ============================================================
def norm_path(p: str) -> str:
    p2 = os.path.normpath(p).replace("\\", "/")
    if os.name == "nt":
        p2 = p2.lower()
    return p2


# ============================================================
# IMAGE PREP (match latest: wide-only)
# ============================================================
def load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def crop_chart_roi(
    pil_img: Image.Image,
    left=0.02,
    top=0.06,
    right=0.14,
    bottom=0.12,
) -> Image.Image:
    w, h = pil_img.size
    x1 = int(w * left)
    y1 = int(h * top)
    x2 = int(w * (1 - right))
    y2 = int(h * (1 - bottom))
    return pil_img.crop((x1, y1, x2, y2))


def content_crop_rows(pil_img: Image.Image, thr=10, margin=6) -> Image.Image:
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    row_mean = gray.mean(axis=1)
    ys = np.where(row_mean > float(thr))[0]
    if ys.size < 10:
        return pil_img
    y1 = max(0, int(ys.min()) - int(margin))
    y2 = min(gray.shape[0], int(ys.max()) + int(margin) + 1)
    if y2 - y1 < 20:
        return pil_img
    return pil_img.crop((0, y1, pil_img.size[0], y2))


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


def build_views(pil_img: Image.Image) -> Image.Image:
    roi = crop_chart_roi(pil_img, **ROI_CUT)
    roi = content_crop_rows(roi, thr=CONTENT_ROW_THR, margin=CONTENT_MARGIN)
    return letterbox(roi, 224)


def make_structure_view(pil_224: Image.Image) -> Image.Image:
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 40, 120)

    # remove horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

    # thicken
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)

    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


# ============================================================
# PRIORS / MASKS
# ============================================================
def spatial_prior(grid_size: int, strength=1.35) -> np.ndarray:
    g = grid_size
    ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
    cy, cx = (g - 1) / 2.0, (g - 1) / 2.0
    dy = (ys - cy) / (cy + 1e-9)
    dx = (xs - cx) / (cx + 1e-9)
    r2 = dx * dx + dy * dy
    prior = np.exp(-float(strength) * r2)
    return prior / (prior.max() + 1e-9)


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
            px = mag_x[y * ph : (y + 1) * ph, x * pw : (x + 1) * pw]
            py = mag_y[y * ph : (y + 1) * ph, x * pw : (x + 1) * pw]
            v_like = float(px.mean())
            h_like = float(py.mean())
            score[y, x] = v_like / (h_like + 1e-6)

    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    score = np.clip(score, float(DENSITY_MIN_CLIP), 1.0)
    return score


def suppress_long_horizontal_runs(
    density_grid: np.ndarray,
    min_run: int = 4,
    penalty: float = 0.08,
) -> np.ndarray:
    g = density_grid.shape[0]
    out = density_grid.copy()

    z = (out <= float(DENSITY_MIN_CLIP) + 1e-6)
    for y in range(g):
        run_start = None
        for x in range(g):
            if z[y, x]:
                if run_start is None:
                    run_start = x
            else:
                if run_start is not None:
                    run_len = x - run_start
                    if run_len >= int(min_run):
                        out[y, run_start:x] *= float(penalty)
                    run_start = None
        if run_start is not None:
            run_len = g - run_start
            if run_len >= int(min_run):
                out[y, run_start:g] *= float(penalty)

    return np.clip(out, 0.0, 1.0)


def prune_by_connectivity(
    keep_mask: np.ndarray,
    min_area: int = 2,
    aspect_ratio_thresh: float = 0.45,
) -> np.ndarray:
    m = (keep_mask.astype(np.uint8) * 255)
    num, labels = cv2.connectedComponents(m, connectivity=8)
    out = np.zeros_like(keep_mask, dtype=bool)

    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        area = len(xs)
        if area < int(min_area):
            continue
        w = xs.max() - xs.min() + 1
        h = ys.max() - ys.min() + 1
        aspect = min(w, h) / (max(w, h) + 1e-6)
        if aspect < float(aspect_ratio_thresh):
            out[ys, xs] = True
    return out


def _topk_mask_from_grid(grid: np.ndarray, ratio: float) -> np.ndarray:
    flat = grid.flatten()
    k = max(1, int(len(flat) * float(ratio)))
    idx = np.argpartition(flat, -k)[-k:]
    m = np.zeros_like(flat, dtype=bool)
    m[idx] = True
    return m.reshape(grid.shape)


def edge_density_grid_no_hline(pil_224: Image.Image, g: int) -> np.ndarray:
    """Edge density per patch, with horizontal lines removed first."""
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)

    # remove horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

    H, W = gray.shape
    ph, pw = H // g, W // g
    er = np.zeros((g, g), dtype=np.float32)
    for y in range(g):
        for x in range(g):
            e = edges[y * ph : (y + 1) * ph, x * pw : (x + 1) * pw]
            er[y, x] = float((e > 0).mean())
    return er


def edge_valid_mask_adaptive(
    pil_224: Image.Image,
    g: int,
    base_thr: float = 0.0035,
    pct: float = 75.0,
    min_keep: int = 12,
):
    er = edge_density_grid_no_hline(pil_224, g)
    thr = max(float(base_thr), float(np.percentile(er, float(pct))) * 0.6)
    m = (er >= thr)

    if int(m.sum()) < int(min_keep):
        flat = er.flatten()
        kk = min(int(min_keep), flat.size)
        idx = np.argpartition(flat, -kk)[-kk:]
        m2 = np.zeros_like(flat, dtype=bool)
        m2[idx] = True
        m = m2.reshape(er.shape)

    return m, er


# ============================================================
# POOLING CORE (NO PROMPT) - content-global
# ============================================================
def _softmax_np(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = x.astype(np.float32) / float(max(1e-6, temp))
    x = x - float(np.max(x))
    e = np.exp(x)
    return e / (float(e.sum()) + 1e-9)


@torch.no_grad()
def pooled_embedding(
    pil_224: Image.Image,
    layers=LAYERS,
    layer_weights=LAYER_WEIGHTS,
):
    clip_model, processor = get_clip()
    assert len(layers) == len(layer_weights), "LAYERS and LAYER_WEIGHTS must have same length"

    inputs = processor(images=pil_224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    # grid size
    first_hs = vision_out.hidden_states[layers[0]] if layers[0] != -1 else vision_out.last_hidden_state
    P = first_hs.shape[1] - 1
    g = int(round(math.sqrt(P)))
    if g * g != P:
        raise ValueError(f"Patch count {P} not square, got g={g}")

    prior = spatial_prior(g, strength=PRIOR_STRENGTH)
    dens = patch_density_mask(pil_224, g)
    dens = suppress_long_horizontal_runs(dens, min_run=HLINE_MIN_RUN, penalty=HLINE_PENALTY)

    # normalize layer weights
    lw = np.array(layer_weights, dtype=np.float32)
    lw = lw / (lw.sum() + 1e-9)

    score_comb = np.zeros((g, g), dtype=np.float32)
    keep_union = np.zeros((g, g), dtype=bool)
    wgrid_comb = np.zeros((g, g), dtype=np.float32)
    pooled_layers = []

    for ly, wly in zip(layers, lw):
        hs = vision_out.hidden_states[ly] if ly != -1 else vision_out.last_hidden_state
        patch_tokens = hs[:, 1:, :]  # [1,P,D]

        patch_emb = clip_model.visual_projection(patch_tokens)  # [1,P,E]
        patch_emb = F.normalize(patch_emb, dim=-1)[0]           # [P,E]

        # valid mask + edge density (no-hline)
        valid, er = edge_valid_mask_adaptive(
            pil_224, g,
            base_thr=EDGE_BASE_THR,
            pct=EDGE_PCT,
            min_keep=max(MIN_KEEP_PATCHES, int(TOPK_RATIO * g * g))
        )
        er_flat = er.flatten()
        ern = er_flat / (float(er_flat.max()) + 1e-9)
        ern = np.power(ern, float(CONTENT_GLOBAL_EDGE_GAMMA)).astype(np.float32)

        # build content-global from valid patches (weighted by edge density)
        mask_t = torch.tensor(valid.flatten(), device=DEVICE)
        if int(mask_t.sum().item()) <= 0:
            gfeat = patch_emb.mean(dim=0)
        else:
            w_t = torch.tensor(ern, device=DEVICE)
            w_sel = w_t[mask_t].clamp(min=1e-6)
            w_sel = w_sel / (w_sel.sum() + 1e-9)
            gfeat = (patch_emb[mask_t] * w_sel.unsqueeze(-1)).sum(dim=0)
        gfeat = F.normalize(gfeat, dim=-1)  # [E]

        # score per patch = cosine(patch, content-global)
        sim = (patch_emb @ gfeat).detach().float().cpu().numpy()  # [P]
        score_grid = sim.reshape(g, g).astype(np.float32)

        # background subtract (use invalid area mean)
        inv = (~valid)
        if inv.any():
            bg_mean = float(score_grid[inv].mean())
            score_grid = score_grid - float(BG_SUBTRACT) * bg_mean

        # priors
        score_grid = score_grid * prior
        score_grid = score_grid * (dens ** float(DENSITY_POWER))

        # edge boost (favor real strokes)
        er_grid = er.reshape(g, g)
        er_norm = er_grid / (float(er_grid.max()) + 1e-9)
        score_grid = score_grid * np.power(er_norm + 1e-6, float(EDGE_BOOST_POWER))

        # keep selection: top-k within valid
        keep0 = _topk_mask_from_grid(score_grid, TOPK_RATIO)
        keep = keep0 & valid
        if not keep.any():
            keep = keep0.copy()

        keep_pruned = prune_by_connectivity(keep, min_area=CC_MIN_AREA, aspect_ratio_thresh=CC_ASPECT_THRESH)
        if keep_pruned.any():
            keep = keep_pruned

        keep_union |= keep

        # pooling weights inside this layer (only keep)
        flat_scores = score_grid.flatten()
        sel_scores = flat_scores[keep.flatten()].astype(np.float32)
        ww = _softmax_np(sel_scores, temp=SOFTMAX_TEMP)

        # pooled vector for this layer
        keep_t = torch.tensor(keep.flatten(), device=DEVICE)
        sel_emb = patch_emb[keep_t]
        ww_t = torch.tensor(ww, device=DEVICE).unsqueeze(-1)
        pooled = (sel_emb * ww_t).sum(dim=0)
        pooled = F.normalize(pooled, dim=-1)
        pooled_layers.append(pooled)

        # debug combine (kept for potential future debug)
        score_comb += float(wly) * score_grid
        w_grid = np.zeros((g, g), dtype=np.float32)
        w_grid.flatten()[keep.flatten()] = ww
        wgrid_comb += float(wly) * w_grid

    # combine pooled vectors across layers
    pooled_stack = torch.stack(pooled_layers, dim=0)  # [L,E]
    lw_t = torch.tensor(lw, device=DEVICE).unsqueeze(-1)
    vec = (pooled_stack * lw_t).sum(dim=0)
    vec = F.normalize(vec, dim=-1).detach().cpu().numpy()

    # normalize combined weight grid to sum=1
    s = float(wgrid_comb.sum())
    if s > 1e-9:
        wgrid_comb = wgrid_comb / s

    return vec, score_comb, keep_union, wgrid_comb


# ============================================================
# FINAL EMBEDDING (REAL + STRUCT) (wide-only)
# ============================================================
def embed_chart_image(image_path: str) -> List[float]:
    # IMPORTANT: image_path here should be actual file path (not normalized id only)
    pil = load_rgb(image_path)

    # wide only (match latest)
    wide224 = build_views(pil)

    # REAL (wide only)
    v_real, _, _, _ = pooled_embedding(wide224)
    v_real = v_real / (np.linalg.norm(v_real) + 1e-9)

    # STRUCT (wide only)
    wide_struct = make_structure_view(wide224)
    v_struct, _, _, _ = pooled_embedding(wide_struct)
    v_struct = v_struct / (np.linalg.norm(v_struct) + 1e-9)

    out = float(W_REAL) * v_real + float(W_STRUCT) * v_struct
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
# CHROMA: OPEN / UPSERT / REBUILD / SEARCH
# ============================================================
def open_image_db():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=ProductionCLIPChartEmbeddings(),
        persist_directory=PERSIST_DIR,
    )


def _get_existing_ids_batched(db: Chroma, ids: List[str], batch=1000):
    existing = set()
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        got = db._collection.get(ids=chunk, include=[])
        if got and "ids" in got and got["ids"]:
            existing.update(got["ids"])
    return existing


def upsert_image_dataset(dataset_json: str):
    db = open_image_db()
    _, uniq_paths = dataset_unique_paths(dataset_json)
    if not uniq_paths:
        raise ValueError("dataset.json contains no image paths")

    # normalize + dedupe ids (store ids normalized to prevent duplicates)
    uniq_norm: List[str] = []
    seen = set()
    for p in uniq_paths:
        pid = norm_path(p)
        if pid not in seen:
            seen.add(pid)
            uniq_norm.append(pid)

    existing = _get_existing_ids_batched(db, uniq_norm, batch=1000)
    new_ids = [p for p in uniq_norm if p not in existing]

    if not new_ids:
        print(f"✅ Image DB: no new images. count={db._collection.count()}")
        return db

    docs = [Document(page_content=p, metadata={"image": p}) for p in new_ids]
    db.add_documents(docs, ids=new_ids[:])
    print(f"✅ Image DB: added {len(new_ids)} new images. count={db._collection.count()}")
    return db


def rebuild_image_db(dataset_json: str):
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    db = open_image_db()
    _, uniq_paths = dataset_unique_paths(dataset_json)
    if not uniq_paths:
        raise ValueError("dataset.json contains no image paths")

    uniq_norm: List[str] = []
    seen = set()
    for p in uniq_paths:
        pid = norm_path(p)
        if pid not in seen:
            seen.add(pid)
            uniq_norm.append(pid)

    docs = [Document(page_content=p, metadata={"image": p}) for p in uniq_norm]
    db.add_documents(docs, ids=uniq_norm[:])
    print(f"✅ Image DB: rebuilt. unique_docs={len(uniq_norm)} count={db._collection.count()}")
    return db


def search_image(db: Chroma, query_image: str, k=10):
    # IMPORTANT:
    # - similarity_search() will call embed_query(query_image) which expects a real file path
    # - so DO NOT norm_path(query_image) here (norm_path is for ids in DB)
    return db.similarity_search(query_image, k=k)
