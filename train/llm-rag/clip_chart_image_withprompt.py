# clip_chart_prod_v4_fixed4_weighted_layers.py
# ------------------------------------------------------------
# FIXED4:
# - รองรับ LAYERS แบบผสม (-6, -3, 24) และ "ถ่วงน้ำหนัก" ต่อเลเยอร์
# - visualize_debug() เลือกดู heatmap ของเลเยอร์ที่ต้องการได้ (debug_layer)
# - KEEP_MODE="edge" เพื่อให้ keep mask ครอบกราฟมากขึ้น
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

ROI_CUT = dict(left=0.06, top=0.06, right=0.14, bottom=0.16)

# ✅ เลเยอร์ที่ใช้ + น้ำหนัก (ให้ 24 หนักกว่าเพราะมันครอบทรงกราฟดี)
LAYERS = (-6, -3, 24)
LAYER_WEIGHTS = (0.25, 0.25, 0.50)  # ต้องยาวเท่ากันกับ LAYERS

# --- selection mode ---
KEEP_MODE = "edge"  # "topk" | "edge" | "valid"

TOPK_RATIO = 0.10
PRIOR_STRENGTH = 1.4
DENSITY_POWER = 1.2

MAG_FLOOR = 0.12

HLINE_MIN_RUN = 3
HLINE_PENALTY = 0.05
HLINE_THR = 0.35

CC_MIN_AREA = 1
HORIZ_W_OVER_H = 2.2

EXCLUDE_LEFT_FRAC = 0.08
RIGHT_PRIOR_POWER = 1.6
RIGHT_PRIOR_BASE = 0.35

ANCHOR_ALPHA = 0.06

# --- edge mask controls ---
EDGE_CANNY1 = 35
EDGE_CANNY2 = 110
EDGE_PATCH_THR = 0.03     # ลดลง = ครอบคลุมมากขึ้น (ลอง 0.02 ถ้ายังไม่เต็ม)
EDGE_DILATE_ITERS = 1     # เพิ่มเป็น 2 ถ้าอยากให้เชื่อมมากขึ้น

W_REAL = 0.55
W_STRUCT = 0.45

ANCHOR_PROMPTS = [
    "TradingView candlestick chart screenshot",
    "MetaTrader 5 candlestick chart screenshot",
]

CANDLE_PROMPTS = [
    "OHLC candlesticks body and wicks close-up on trading chart",
    "red and green candlesticks with wicks close-up",
    "single candlestick body and wick close-up",
    "multiple candlesticks body and wicks close-up",
    "pin bar long upper wick rejection at resistance on candlestick chart",
    "pin bar long lower wick rejection at support on candlestick chart",
    "long wick rejection candle at swing high",
    "long wick rejection candle at swing low",
    "liquidity sweep wick then reversal on candlestick chart",
    "fakeout breakout wick then reversal on candlestick chart",
    "break structure then pullback on candlestick chart",
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

def crop_chart_roi(pil_img: Image.Image, left=0.06, top=0.06, right=0.14, bottom=0.16) -> Image.Image:
    w, h = pil_img.size
    x1 = int(w * left)
    y1 = int(h * top)
    x2 = int(w * (1 - right))
    y2 = int(h * (1 - bottom))
    return pil_img.crop((x1, y1, x2, y2))

def content_bbox_crop(pil_img: Image.Image, pad=8, min_edges=80) -> Image.Image:
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)

    ys, xs = np.where(edges > 0)
    if xs.size < min_edges:
        return pil_img

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    H, W = gray.shape
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)

    if (x2 - x1) < W * 0.35 or (y2 - y1) < H * 0.20:
        return pil_img

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
    roi = content_bbox_crop(roi, pad=8, min_edges=80)
    w, h = roi.size
    zoom = roi.crop((0, 0, w, int(h * 0.65)))
    focus = _auto_focus_crop(roi, win_ratio=0.72)
    return (
        letterbox(roi, 224),
        letterbox(zoom, 224),
        letterbox(focus, 224),
    )

def make_structure_view(pil_224: Image.Image) -> Image.Image:
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 40, 120)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


# ============================================================
# TEXT FEATS
# ============================================================
@torch.no_grad()
def build_text_features(prompts):
    t_in = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    t_feat = clip_model.get_text_features(**t_in)
    return F.normalize(t_feat, dim=-1)


# ============================================================
# PRIORS & MASKS
# ============================================================
def spatial_prior(grid_size: int, strength=1.4) -> np.ndarray:
    g = grid_size
    ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
    cy, cx = (g - 1) / 2.0, (g - 1) / 2.0
    dy = (ys - cy) / (cy + 1e-9)
    dx = (xs - cx) / (cx + 1e-9)
    r2 = dx * dx + dy * dy
    prior = np.exp(-strength * r2)
    return prior / (prior.max() + 1e-9)

def right_prior(grid_size: int, power=1.6, base=0.35) -> np.ndarray:
    g = grid_size
    xs = np.arange(g, dtype=np.float32) / (g - 1 + 1e-9)
    p = (base + (1.0 - base) * (xs ** power)).reshape(1, g)
    return np.repeat(p, g, axis=0)

def patch_density_mask(pil_224: Image.Image, grid_size: int):
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
            ratio[y, x] = float(px.mean()) / (float(py.mean()) + 1e-6)
            mgrid[y, x] = float(mm.mean())

    mg = mgrid / (float(mgrid.max()) + 1e-9)
    gate = np.clip((mg - float(MAG_FLOOR)) / (1.0 - float(MAG_FLOOR) + 1e-9), 0.0, 1.0)
    valid = gate > 1e-3
    gate = gate ** 2.0

    if valid.any():
        rmin = float(ratio[valid].min()); rmax = float(ratio[valid].max())
    else:
        rmin = float(ratio.min()); rmax = float(ratio.max())

    ratio_norm = (ratio - rmin) / (rmax - rmin + 1e-9)
    ratio_norm = np.clip(ratio_norm, 0.0, 1.0)

    dens = np.clip(ratio_norm * gate, 0.0, 1.0)
    return dens, valid

def suppress_long_horizontal_runs(density_grid: np.ndarray, min_run: int = 3, penalty: float = 0.05, thr: float = 0.35) -> np.ndarray:
    g = density_grid.shape[0]
    out = density_grid.copy()
    for y in range(g):
        run_start = None
        for x in range(g):
            active = out[y, x] >= thr
            if active:
                if run_start is None:
                    run_start = x
            else:
                if run_start is not None:
                    if (x - run_start) >= min_run:
                        out[y, run_start:x] *= penalty
                    run_start = None
        if run_start is not None and (g - run_start) >= min_run:
            out[y, run_start:g] *= penalty
    return np.clip(out, 0.0, 1.0)

def prune_by_connectivity(keep_mask: np.ndarray, min_area: int = 1, horiz_w_over_h: float = 2.2) -> np.ndarray:
    m = (keep_mask.astype(np.uint8) * 255)
    num, labels = cv2.connectedComponents(m, connectivity=8)
    out = np.zeros_like(keep_mask, dtype=bool)
    for lab in range(1, num):
        ys, xs = np.where(labels == lab)
        area = int(xs.size)
        if area < int(min_area):
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if (w / (h + 1e-6)) >= float(horiz_w_over_h):
            continue
        out[ys, xs] = True
    return out

def patch_edge_mask(pil_224: Image.Image, grid_size: int):
    img = np.array(pil_224)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, int(EDGE_CANNY1), int(EDGE_CANNY2))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    edges = cv2.subtract(edges, h_lines)

    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)

    H, W = edges.shape
    g = grid_size
    ph, pw = H // g, W // g
    m = np.zeros((g, g), dtype=np.float32)
    e = (edges > 0).astype(np.float32)

    for y in range(g):
        for x in range(g):
            block = e[y*ph:(y+1)*ph, x*pw:(x+1)*pw]
            m[y, x] = float(block.mean())

    keep = m >= float(EDGE_PATCH_THR)

    if int(EDGE_DILATE_ITERS) > 0:
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(EDGE_DILATE_ITERS)).astype(bool)

    return keep


# ============================================================
# CORE: PATCH POOLING (✅ weighted layers)
# ============================================================
@torch.no_grad()
def patch_pooled_embedding(pil_224: Image.Image, anchor_feats: torch.Tensor, candle_feats: torch.Tensor):
    inputs = processor(images=pil_224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)

    pooled_list = []
    weight_list = []

    for ly, w in zip(LAYERS, LAYER_WEIGHTS):
        hs = vision_out.hidden_states[ly] if ly != -1 else vision_out.last_hidden_state
        patch_tokens = hs[:, 1:, :]
        patch_emb = F.normalize(clip_model.visual_projection(patch_tokens), dim=-1)

        sim_anchor = (patch_emb @ anchor_feats.T).max(dim=-1).values.squeeze(0)
        sim_candle = (patch_emb @ candle_feats.T).max(dim=-1).values.squeeze(0)
        sim = sim_candle + float(ANCHOR_ALPHA) * sim_anchor
        sim_np = sim.detach().cpu().numpy()

        P = sim_np.shape[0]
        g = int(round(math.sqrt(P)))
        sim_grid = sim_np.reshape(g, g)

        prior = spatial_prior(g, strength=float(PRIOR_STRENGTH))
        rprior = right_prior(g, power=float(RIGHT_PRIOR_POWER), base=float(RIGHT_PRIOR_BASE))

        dens, valid = patch_density_mask(pil_224, g)
        dens = suppress_long_horizontal_runs(dens, min_run=int(HLINE_MIN_RUN), penalty=float(HLINE_PENALTY), thr=float(HLINE_THR))

        left_cols = int(round(g * float(EXCLUDE_LEFT_FRAC)))
        if left_cols > 0:
            valid[:, :left_cols] = False

        sim2 = sim_grid * prior * rprior * (dens ** float(DENSITY_POWER))

        # keep mask
        if KEEP_MODE == "valid":
            keep0 = valid.copy()
        elif KEEP_MODE == "edge":
            ekeep = patch_edge_mask(pil_224, g)
            keep0 = valid & ekeep
            if not keep0.any():
                keep0 = valid.copy()
        else:
            sim2_cut = sim2.copy()
            sim2_cut[~valid] = -1e9
            flat = sim2_cut[valid].flatten()
            if flat.size == 0:
                valid = np.ones_like(valid, dtype=bool)
                sim2_cut = sim2.copy()
                flat = sim2_cut.flatten()
            k = max(1, int(len(flat) * float(TOPK_RATIO)))
            thr = np.sort(flat)[-k]
            keep0 = (sim2_cut >= thr) & valid

        keep = prune_by_connectivity(keep0, min_area=int(CC_MIN_AREA), horiz_w_over_h=float(HORIZ_W_OVER_H))
        if not keep.any():
            keep = keep0

        mask = torch.tensor(keep.flatten(), device=DEVICE)
        selected = patch_emb[0][mask]
        selected_sim = torch.tensor(sim2.flatten(), device=DEVICE, dtype=sim.dtype)[mask]

        weights = F.softmax(selected_sim, dim=0).unsqueeze(-1)
        pooled = (selected * weights).sum(dim=0)
        pooled = F.normalize(pooled, dim=-1)

        pooled_list.append(pooled)
        weight_list.append(float(w))

    # ✅ weighted average across layers
    wsum = float(sum(weight_list)) + 1e-9
    out = (sum(p * (w / wsum) for p, w in zip(pooled_list, weight_list)))
    out = F.normalize(out, dim=-1)
    return out.detach().cpu().numpy()


# ============================================================
# FINAL EMBEDDING
# ============================================================
def embed_chart_image(image_path: str) -> list:
    pil = load_rgb(image_path)
    wide224, zoom224, focus224 = build_views(pil)

    anchor_feats = build_text_features(ANCHOR_PROMPTS)
    candle_feats = build_text_features(CANDLE_PROMPTS)

    v_real = (
        patch_pooled_embedding(wide224,  anchor_feats, candle_feats) +
        patch_pooled_embedding(zoom224,  anchor_feats, candle_feats) +
        patch_pooled_embedding(focus224, anchor_feats, candle_feats)
    ) / 3.0
    v_real = v_real / (np.linalg.norm(v_real) + 1e-9)

    wide_s = make_structure_view(wide224)
    zoom_s = make_structure_view(zoom224)
    focus_s = make_structure_view(focus224)

    v_struct = (
        patch_pooled_embedding(wide_s,  anchor_feats, candle_feats) +
        patch_pooled_embedding(zoom_s,  anchor_feats, candle_feats) +
        patch_pooled_embedding(focus_s, anchor_feats, candle_feats)
    ) / 3.0
    v_struct = v_struct / (np.linalg.norm(v_struct) + 1e-9)

    out = (W_REAL * v_real) + (W_STRUCT * v_struct)
    out = out / (np.linalg.norm(out) + 1e-9)
    return out.tolist()


# ============================================================
# DEBUG VIS
# ============================================================
def _percentile_norm(x: np.ndarray, lo_p=20, hi_p=95) -> np.ndarray:
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

@torch.no_grad()
def visualize_debug(image_path: str, debug_layer=None, lo_p=20, hi_p=95, heat_alpha=0.55):
    pil = load_rgb(image_path)
    wide224, _, _ = build_views(pil)

    anchor_feats = build_text_features(ANCHOR_PROMPTS)
    candle_feats = build_text_features(CANDLE_PROMPTS)

    # ✅ เลือกเลเยอร์ที่จะดู heatmap
    ly = debug_layer if debug_layer is not None else LAYERS[-1]

    inputs = processor(images=wide224, return_tensors="pt").to(DEVICE)
    vision_out = clip_model.vision_model(**inputs, output_hidden_states=True)
    hs = vision_out.hidden_states[ly] if ly != -1 else vision_out.last_hidden_state

    patch_tokens = hs[:, 1:, :]
    patch_emb = F.normalize(clip_model.visual_projection(patch_tokens), dim=-1)

    sim_anchor = (patch_emb @ anchor_feats.T).max(dim=-1).values.squeeze(0).detach().cpu().numpy()
    sim_candle = (patch_emb @ candle_feats.T).max(dim=-1).values.squeeze(0).detach().cpu().numpy()
    sim = sim_candle + float(ANCHOR_ALPHA) * sim_anchor

    P = sim.shape[0]
    g = int(round(math.sqrt(P)))
    sim_grid = sim.reshape(g, g)

    prior = spatial_prior(g, strength=float(PRIOR_STRENGTH))
    rprior = right_prior(g, power=float(RIGHT_PRIOR_POWER), base=float(RIGHT_PRIOR_BASE))
    dens, valid = patch_density_mask(wide224, g)
    dens = suppress_long_horizontal_runs(dens, min_run=int(HLINE_MIN_RUN), penalty=float(HLINE_PENALTY), thr=float(HLINE_THR))

    left_cols = int(round(g * float(EXCLUDE_LEFT_FRAC)))
    if left_cols > 0:
        valid[:, :left_cols] = False

    sim2 = sim_grid * prior * rprior * (dens ** float(DENSITY_POWER))

    # keep
    if KEEP_MODE == "valid":
        keep = valid.copy()
    elif KEEP_MODE == "edge":
        keep = (valid & patch_edge_mask(wide224, g))
        if not keep.any():
            keep = valid.copy()
    else:
        sim2_cut = sim2.copy()
        sim2_cut[~valid] = -1e9
        flat = sim2_cut[valid].flatten()
        k = max(1, int(len(flat) * float(TOPK_RATIO)))
        thr = np.sort(flat)[-k]
        keep = (sim2_cut >= thr) & valid

    keep = prune_by_connectivity(keep, min_area=int(CC_MIN_AREA), horiz_w_over_h=float(HORIZ_W_OVER_H))
    if not keep.any():
        keep = valid.copy()

    sim_norm = _percentile_norm(sim2, lo_p=lo_p, hi_p=hi_p)

    img_np = np.array(wide224)
    H, W = img_np.shape[:2]
    heat = np.kron(sim_norm, np.ones((H // g, W // g)))[:H, :W]
    keep_map = np.kron(keep.astype(np.float32), np.ones((H // g, W // g)))[:H, :W]

    overlay = img_np.copy()
    overlay[keep_map >= 0.5] = (255, 0, 0)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1); plt.title("Wide (ROI+content crop+letterbox)"); plt.imshow(img_np); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title(f"Heatmap MAX (layer {ly})"); plt.imshow(img_np); plt.imshow(heat, alpha=float(heat_alpha)); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title(f"Keep mask overlay (mode={KEEP_MODE})"); plt.imshow(img_np); plt.imshow(overlay, alpha=0.25); plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    QUERY_IMAGE = "datasets1/new_chart3.png"

    # ดู heatmap เลเยอร์ 24
    # visualize_debug(QUERY_IMAGE, debug_layer=-3)

    # หรือดู -3
    # visualize_debug(QUERY_IMAGE, debug_layer=-3)

    # หรือดู -6
    # visualize_debug(QUERY_IMAGE, debug_layer=-6)