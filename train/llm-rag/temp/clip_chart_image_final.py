# clip_chart_prod_v6_noprompt.py
# ------------------------------------------------------------
# Production-grade Chart Retrieval (CLIP hidden-layer) - V6 (NO PROMPT)
# Fixes:
# ✅ content-global (from edge-rich patches) instead of global image embedding
# ✅ edge mask removes horizontal lines BEFORE scoring/valid-mask
# ✅ edge-boost + background-subtract to suppress background bands
# ✅ keep mask never empty (adaptive + min_keep + fallback)
# ✅ single pipeline = USED for VectorDB (no global/local confusion)
# ------------------------------------------------------------

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
# MODEL_NAME = "openai/clip-vit-base-patch32"

PERSIST_DIR = "chroma_store_images1"
COLLECTION = "chart_clip_images1"

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

# New: suppress background
EDGE_BOOST_POWER = 1.6     # emphasize edge-rich patches
BG_SUBTRACT = 0.45        # subtract mean score on invalid/background-ish patches
CONTENT_GLOBAL_EDGE_GAMMA = 1.5  # weights for building content-global from edge density

# Views
W_REAL = 0.60
W_STRUCT = 0.40

# Pooling
SOFTMAX_TEMP = 1.2


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
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

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
            px = mag_x[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            py = mag_y[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            v_like = float(px.mean())
            h_like = float(py.mean())
            score[y, x] = v_like / (h_like + 1e-6)

    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    score = np.clip(score, float(DENSITY_MIN_CLIP), 1.0)
    return score

def suppress_long_horizontal_runs(density_grid: np.ndarray, min_run: int = 4, penalty: float = 0.08) -> np.ndarray:
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

def prune_by_connectivity(keep_mask: np.ndarray, min_area: int = 2, aspect_ratio_thresh: float = 0.45) -> np.ndarray:
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
            e = edges[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            er[y, x] = float((e > 0).mean())
    return er

def edge_valid_mask_adaptive(pil_224: Image.Image, g: int,
                             base_thr: float = 0.0035,
                             pct: float = 75.0,
                             min_keep: int = 12):
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

        # debug combine
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
# FINAL EMBEDDING (REAL + STRUCT) (wide + zoom + focus)
# ============================================================
def embed_chart_image(image_path: str) -> list:
    pil = load_rgb(image_path)

    # wide only
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
# DEBUG VIS
# ============================================================
def _percentile_norm(x: np.ndarray, lo_p=20, hi_p=95) -> np.ndarray:
    lo = np.percentile(x, float(lo_p))
    hi = np.percentile(x, float(hi_p))
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    
@torch.no_grad()
def visualize_used_for_vectordb(image_path: str, lo_p=20, hi_p=95,
                                figsize=(13.0, 7.6), dpi=120, title_fs=9, suptitle_fs=11,
                                pad=0.6, w_pad=0.3, h_pad=0.6):
    """
    Single-page (3x4) compact layout for MacBook 13":
    - REAL / STRUCT / FINAL in one figure
    - uses wide224 only
    """
    pil = load_rgb(image_path)
    wide224 = build_views(pil)  # wide-only build_views()

    def _prep(proc_img: Image.Image):
        vec, score, keep, wgrid = pooled_embedding(proc_img)
        return vec, score.astype(np.float32), keep.astype(bool), wgrid.astype(np.float32)

    def _upsample(xg: np.ndarray, g: int, H: int, W: int) -> np.ndarray:
        return np.kron(xg, np.ones((H // g, W // g), dtype=np.float32))[:H, :W]

    def _make_panels(bg_img: Image.Image, score_g: np.ndarray, keep_g: np.ndarray, wgrid_g: np.ndarray):
        img_np = np.array(bg_img)
        H, W = img_np.shape[:2]
        g = score_g.shape[0]

        score_norm = _percentile_norm(score_g, lo_p=lo_p, hi_p=hi_p)
        score_up = _upsample(score_norm, g, H, W)

        keep_up = _upsample(keep_g.astype(np.float32), g, H, W)

        w_up = _upsample(wgrid_g.astype(np.float32), g, H, W)
        w_mx = float(w_up.max())
        w_vis = np.power(w_up / (w_mx + 1e-9), 0.35) if w_mx > 0 else w_up

        used_only = img_np.copy().astype(np.float32)
        used_only[keep_up < 0.5] *= 0.07
        used_only = np.clip(used_only, 0, 255).astype(np.uint8)

        return img_np, score_up, keep_up, used_only, w_vis

    # ---------- REAL (wide) ----------
    v_real, score_real, keep_real, wgrid_real = _prep(wide224)

    # ---------- STRUCT (structure(wide)) ----------
    wide_struct = make_structure_view(wide224)
    v_struct, score_struct, keep_struct, wgrid_struct = _prep(wide_struct)

    # ---------- FINAL (match embed_chart_image wide-only) ----------
    out_vec = float(W_REAL) * v_real + float(W_STRUCT) * v_struct
    out_vec = out_vec / (np.linalg.norm(out_vec) + 1e-9)

    score_final = float(W_REAL) * score_real + float(W_STRUCT) * score_struct
    keep_final = (float(W_REAL) * keep_real.astype(np.float32) + float(W_STRUCT) * keep_struct.astype(np.float32)) > 0.5

    wgrid_final = float(W_REAL) * wgrid_real + float(W_STRUCT) * wgrid_struct
    s = float(wgrid_final.sum())
    if s > 1e-9:
        wgrid_final = wgrid_final / s

    g = score_real.shape[0]
    print(f"[REAL]   USED patches: {int(keep_real.sum())}/{g*g} = {int(keep_real.sum())/(g*g):.2%} | vec_dim={len(v_real)}")
    print(f"[STRUCT] USED patches: {int(keep_struct.sum())}/{g*g} = {int(keep_struct.sum())/(g*g):.2%} | vec_dim={len(v_struct)}")
    print(f"[FINAL]  USED patches: {int(keep_final.sum())}/{g*g} = {int(keep_final.sum())/(g*g):.2%} | vec_dim={len(out_vec)} | W_REAL={W_REAL} W_STRUCT={W_STRUCT}")

    # panels
    r_img, r_score, r_keep, r_used, r_wvis = _make_panels(wide224,     score_real,   keep_real,   wgrid_real)
    s_img, s_score, s_keep, s_used, s_wvis = _make_panels(wide_struct, score_struct, keep_struct, wgrid_struct)
    f_img, f_score, f_keep, f_used, f_wvis = _make_panels(wide224,     score_final,  keep_final,  wgrid_final)

    # ---------- PLOT (single page, compact) ----------
    fig, axs = plt.subplots(3, 4, figsize=figsize, dpi=dpi)
    fig.suptitle("DEBUG CHART EIEI", fontsize=suptitle_fs)

    rows = [
        ("REAL",   r_img, r_score, r_keep, r_used, r_wvis),
        ("STRUCT", s_img, s_score, s_keep, s_used, s_wvis),
        ("FINAL (REAL + STRUCT (UNION))",  f_img, f_score, f_keep, f_used, f_wvis),
    ]

    for ri, (name, img_np, score_up, keep_up, used_only, w_vis) in enumerate(rows):
        # col 0
        axs[ri, 0].set_title(f"{name}: Base", fontsize=title_fs)
        axs[ri, 0].imshow(img_np); axs[ri, 0].axis("off")

        # col 1
        axs[ri, 1].set_title("Heat map Score", fontsize=title_fs)
        axs[ri, 1].imshow(img_np)
        axs[ri, 1].imshow(score_up, alpha=0.55)
        axs[ri, 1].axis("off")

        # col 2
        axs[ri, 2].set_title("Keep", fontsize=title_fs)
        axs[ri, 2].imshow(img_np)
        axs[ri, 2].imshow(keep_up, alpha=0.35)
        axs[ri, 2].axis("off")

        # col 3
        axs[ri, 3].set_title("Weights", fontsize=title_fs)
        axs[ri, 3].imshow(used_only)
        # axs[ri, 3].imshow(w_vis, alpha=0.30)
        axs[ri, 3].axis("off")

    # tighter spacing for 13" screens
    plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    plt.show()



# ============================================================
# MAIN (example)
# ============================================================
if __name__ == "__main__":
    DATASET_JSON = "dataset.json"
    QUERY_IMAGE = "datasets1/image.png"

    visualize_used_for_vectordb(QUERY_IMAGE, figsize=(12, 7), dpi=120, title_fs=8, h_pad=0.3, w_pad=0.1)



    # db = upsert_dataset(DATASET_JSON)
    # hits = search_similar(db, QUERY_IMAGE, k=5)
    # for i, h in enumerate(hits, 1):
    #     print(i, h.metadata.get("image"))
