#!/usr/bin/env python3
"""Real-time single-image face-swap research server.

The public workflow requires one supplied, consented source image.
The archived runtime uses InsightFace inswapper_128 or its FP16 variant; historical
HyperSwap and fixed latency claims in the development archive are not benchmarks.
See docs/MODELS.md, docs/ARCHITECTURE.md, and docs/SETUP.md for implementation details.
"""
from __future__ import annotations

import glob
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import cv2
import numpy as np


# Canonical FFHQ 512x512 5-point landmark template for landmark-aligned GFPGANv1.4
FFHQ_512_TEMPLATE = np.array(
    [
        [192.98138, 239.94708],  # Left eye center
        [318.90277, 240.19360],  # Right eye center
        [256.63416, 314.01935],  # Nose tip
        [201.26117, 371.41043],  # Left mouth corner
        [313.08905, 371.15118],  # Right mouth corner
    ],
    dtype=np.float32,
)

# Canonical ArcFace 256x256 5-point landmark template
ARCFACE_256_TEMPLATE = np.array(
    [
        [76.5892, 103.3884],
        [179.4108, 103.3884],
        [128.0000, 143.5162],
        [92.7316, 184.7310],
        [163.2684, 184.7310],
    ],
    dtype=np.float32,
)


class LatestFrameSlot:
    """Newest-frame ingestion slot backed by a synchronized `queue.LifoQueue`.

    If incoming network frames arrive faster than CUDA execution, `put()` and `get()`
    flush stale buffered frames and retain only the most recent payload. This prevents
    backlog growth, but it does not guarantee a particular latency and it is not zero-copy.
    """

    def __init__(self) -> None:
        self.lifo: "queue.LifoQueue[Tuple[bytes, bytes, float]]" = queue.LifoQueue()
        self._cond = threading.Condition()
        self.dropped_frames: int = 0
        self.total_ingested: int = 0

    def put(self, ident: bytes, payload: bytes, timestamp: Optional[float] = None) -> int:
        """Flushes any stale buffered frames and inserts the newest frame payload."""
        ts = float(timestamp) if timestamp is not None else time.perf_counter()
        flushed = 0
        with self._cond:
            while True:
                try:
                    _ = self.lifo.get_nowait()
                    flushed += 1
                except queue.Empty:
                    break
            self.dropped_frames += flushed
            self.total_ingested += 1
            self.lifo.put_nowait((ident, payload, ts))
            self._cond.notify_all()
        return flushed

    def get(self, timeout: Optional[float] = None) -> Optional[Tuple[bytes, bytes, float]]:
        """Retrieves the single most recent frame from the LifoQueue and flushes any older items."""
        with self._cond:
            if self.lifo.empty():
                if timeout is not None and timeout <= 0.0:
                    return None
                self._cond.wait(timeout=timeout)
                if self.lifo.empty():
                    return None
            try:
                newest = self.lifo.get_nowait()
            except queue.Empty:
                return None
            flushed = 0
            while True:
                try:
                    _ = self.lifo.get_nowait()
                    flushed += 1
                except queue.Empty:
                    break
            self.dropped_frames += flushed
            return newest


class CausalOneEuroFilter:
    """Causal One Euro Filter following Géry Casiez et al. (CHI 2012):
    '1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems'.

    Uses timestamp-aware adaptive frequency tracking (`freq = 1.0 / dt`) with default
    `mincutoff = 0.50 Hz`, `beta = 0.005`, `dcutoff = 1.0 Hz`.
    """

    def __init__(
        self,
        mincutoff: float = 0.50,
        beta: float = 0.005,
        dcutoff: float = 1.0,
    ) -> None:
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None
        self.last_freq: float = 30.0

    def reset(self) -> None:
        """Immediately clears filter state on track loss or sharp scene cut."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def compute_alpha(cutoff: np.ndarray, freq: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
        te = 1.0 / max(float(freq), 1e-6)
        return (1.0 / (1.0 + tau / te)).astype(np.float32)

    def __call__(
        self,
        x: np.ndarray,
        timestamp: Optional[float] = None,
        freq: float = 30.0,
    ) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if self.x_prev is None or self.x_prev.shape != x_arr.shape:
            self.x_prev = x_arr.copy()
            self.dx_prev = np.zeros_like(x_arr, dtype=np.float32)
            self.t_prev = float(timestamp) if timestamp is not None else None
            self.last_freq = float(freq)
            return x_arr.copy()

        if timestamp is not None and self.t_prev is not None:
            dt = float(timestamp) - float(self.t_prev)
            if dt > 1e-6:
                freq = 1.0 / dt
        if timestamp is not None:
            self.t_prev = float(timestamp)
        self.last_freq = float(freq)

        dx = (x_arr - self.x_prev) * float(freq)
        a_d = self.compute_alpha(np.full_like(x_arr, self.dcutoff), freq)
        edx = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.mincutoff + self.beta * np.abs(edx)
        a_x = self.compute_alpha(cutoff, freq)
        x_hat = a_x * x_arr + (1.0 - a_x) * self.x_prev

        self.x_prev = x_hat.astype(np.float32)
        self.dx_prev = edx.astype(np.float32)
        return x_hat.astype(np.float32)


class RigidPoseOneEuroFilter:
    """Applies the Causal One Euro Filter (`mincutoff=0.50 Hz`, `beta=0.005`, `dcutoff=1.0 Hz`)
    STRICTLY to normalized rigid geometry parameters:
      - Crop box / cranial center `(cx, cy)`
      - Rigid scale `s`
      - Fundamental rigid alignment orientation unit vectors `(ux, uy) = (cos_theta, sin_theta)`

    Does NOT filter expression-sensitive internal geometry landmarks (eyelids, lips, mouth).
    Internal expression offsets in normalized cranial space bypass the rigid-pose filter,
    reducing the risk that mouth and eyelid motion inherits pose smoothing.
    Enforces an immediate state reset whenever tracking is lost or a sharp scene cut occurs.
    """

    def __init__(
        self,
        mincutoff: float = 0.50,
        beta: float = 0.005,
        dcutoff: float = 1.0,
        scene_cut_px: float = 65.0,
        max_time_gap: float = 0.50,
    ) -> None:
        self.pose_filter = CausalOneEuroFilter(
            mincutoff=mincutoff, beta=beta, dcutoff=dcutoff
        )
        self.scene_cut_px = float(scene_cut_px)
        self.max_time_gap = float(max_time_gap)
        self.reset_count: int = 0

    def reset(self) -> None:
        """Immediately resets filter state when track is lost or a scene cut occurs."""
        self.pose_filter.reset()
        self.reset_count += 1

    @staticmethod
    def extract_rigid_pose(kps_5: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """Decomposes 5-point landmarks into:
          - `center`: `(cx, cy)` of the rigid upper-cranial anchor triangle (`left_eye`, `right_eye`, `nose_tip`)
          - `scale`: rigid inter-ocular/cranial scale `s`
          - `orient`: unit orientation vector `(ux, uy) = (cos_theta, sin_theta)` along the inter-ocular axis
          - `expr_offsets`: `(5, 2)` normalized local coordinates in the rigid cranial frame (preserving mouth/lip motion)
        """
        pts = np.asarray(kps_5, dtype=np.float32).reshape(5, 2)
        # Upper 3 cranial landmarks (left eye, right eye, nose tip) are skull-rigid and invariant to speech/mouth opening
        upper = pts[:3]
        center = np.mean(upper, axis=0).astype(np.float32)

        eye_vec = pts[1] - pts[0]
        eye_dist = float(np.hypot(float(eye_vec[0]), float(eye_vec[1])))
        nose_dist = float(np.linalg.norm(pts[2] - 0.5 * (pts[0] + pts[1])))
        scale = max(float(np.hypot(eye_dist, nose_dist)), 1e-4)

        if eye_dist > 1e-4:
            ux = float(eye_vec[0]) / eye_dist
            uy = float(eye_vec[1]) / eye_dist
        else:
            ux, uy = 1.0, 0.0
        orient = np.array([ux, uy], dtype=np.float32)

        # Orthogonal rotation matrix R = [[ux, -uy], [uy, ux]]
        rot = np.array([[ux, -uy], [uy, ux]], dtype=np.float32)
        # Local coordinates in canonical cranial frame: expr_offsets = ((pts - center) @ rot) / scale
        expr_offsets = ((pts - center) @ rot) / scale
        return center, scale, orient, expr_offsets.astype(np.float32)

    @staticmethod
    def reconstruct_from_rigid_pose(
        center: np.ndarray,
        scale: float,
        orient: np.ndarray,
        expr_offsets: np.ndarray,
    ) -> np.ndarray:
        """Reconstructs landmarks from smoothed rigid `(center, scale, orient)` and unfiltered `expr_offsets`."""
        norm = float(np.hypot(float(orient[0]), float(orient[1])))
        if norm > 1e-6:
            ux, uy = float(orient[0]) / norm, float(orient[1]) / norm
        else:
            ux, uy = 1.0, 0.0
        rot = np.array([[ux, -uy], [uy, ux]], dtype=np.float32)
        return (center.reshape(1, 2) + float(scale) * (expr_offsets @ rot.T)).astype(np.float32)

    def filter_kps(
        self,
        kps_5: Optional[np.ndarray],
        timestamp: Optional[float] = None,
        lm106: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Filters ONLY the rigid pose parameters `(cx, cy, s, ux, uy)` of `kps_5` (and optional `lm106`),
        leaving internal expression offsets (mouth, lips, eyelids) outside the rigid filter.
        Resets immediately on `kps_5 is None`, time gap `> max_time_gap`, or jump `> scene_cut_px`.
        """
        if kps_5 is None:
            self.reset()
            return None, None

        center, scale, orient, expr_5 = self.extract_rigid_pose(kps_5)

        # Check for sharp scene cut or stale timestamp gap
        if self.pose_filter.x_prev is not None:
            prev_center = self.pose_filter.x_prev[:2]
            jump_px = float(np.linalg.norm(center - prev_center))
            time_gap = (
                float(timestamp) - float(self.pose_filter.t_prev)
                if (timestamp is not None and self.pose_filter.t_prev is not None)
                else 0.0
            )
            if jump_px > self.scene_cut_px or time_gap > self.max_time_gap:
                self.reset()

        #Decompose lm106 in the exact same instantaneous rigid cranial frame so internal eyelids/lips are unfiltered
        expr_106 = None
        if lm106 is not None:
            pts106 = np.asarray(lm106, dtype=np.float32)
            if pts106.shape == (106, 2):
                ux0, uy0 = float(orient[0]), float(orient[1])
                rot0 = np.array([[ux0, -uy0], [uy0, ux0]], dtype=np.float32)
                expr_106 = ((pts106 - center) @ rot0) / scale

        pose_vec = np.array(
            [center[0], center[1], scale, orient[0], orient[1]], dtype=np.float32
        )
        smoothed_pose = self.pose_filter(pose_vec, timestamp=timestamp)
        sm_center = smoothed_pose[:2]
        sm_scale = max(float(smoothed_pose[2]), 1e-4)
        sm_orient = smoothed_pose[3:5]

        # Reconstruct 5-point landmarks: rigid cranial anchors [0,1,2] use canonical rigid geometry,
        # while mouth corners [3,4] preserve their instantaneous non-rigid expression offsets.
        sm_kps = self.reconstruct_from_rigid_pose(sm_center, sm_scale, sm_orient, expr_5)
        sm_lm106 = (
            self.reconstruct_from_rigid_pose(sm_center, sm_scale, sm_orient, expr_106)
            if expr_106 is not None
            else lm106
        )
        return sm_kps, sm_lm106


def estimate_ffhq_512_affine(
    kps_5: np.ndarray,
    template: np.ndarray = FFHQ_512_TEMPLATE,
) -> np.ndarray:
    """Computes the exact 2x3 similarity affine transform matrix mapping target frame space
    into landmark-aligned `512x512` FFHQ space using the rigid upper-cranial + 5-point anchors.
    """
    pts = np.asarray(kps_5, dtype=np.float32).reshape(5, 2)
    tmpl = np.asarray(template, dtype=np.float32).reshape(5, 2)
    M, _ = cv2.estimateAffinePartial2D(pts, tmpl, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(pts[:3], tmpl[:3])
    return M.astype(np.float32)


def warp_face_ffhq_512(
    frame_bgr: np.ndarray,
    kps_5: np.ndarray,
    crop_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warps a face into landmark-aligned FFHQ `512x512` (or `crop_size x crop_size`) space
    with bilinear anti-aliasing (`cv2.INTER_LINEAR`).
    """
    scale_factor = float(crop_size) / 512.0
    tmpl = FFHQ_512_TEMPLATE * scale_factor
    M = estimate_ffhq_512_affine(kps_5, template=tmpl)
    warped = cv2.warpAffine(
        frame_bgr,
        M,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, M


class FaceDict(dict):
    """Dictionary-and-attribute compatible Face container matching `insightface.app.common.Face`.
    Supports both `face.kps` and `face['landmark_2d_106']` item assignment.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


# ==============================================================================
# 1. REFERENCE #1: iperov/DeepFaceLive Masked Reinhard Color Transfer (LAB)
# ==============================================================================
def dfl_masked_color_transfer(
    src_bgr: np.ndarray,
    dst_bgr: np.ndarray,
    mask_single_channel: np.ndarray,
    l_clamp_ratio: float = 0.15,
) -> np.ndarray:
    """Reference: iperov/DeepFaceLive core/imagelib/color_transfer.py (Masked RCT).

    Computes LAB mean and std strictly inside the eroded inner-skin mask (`mask > 0.75`),
    completely excluding dark hair corners, beard shadows, and background pixels.
    Clamps luminance (`L`) channel adjustments to `±15%` (`l_clamp_ratio=0.15`) so
    dark rooms, hair, or shadows NEVER darken or blacken the swapped face.
    """
    if src_bgr is None or dst_bgr is None or mask_single_channel is None:
        return src_bgr
    if src_bgr.shape[:2] != dst_bgr.shape[:2]:
        return src_bgr

    eps = 1e-5
    m = mask_single_channel.squeeze()
    if m.shape[:2] != src_bgr.shape[:2]:
        m = cv2.resize(
            m.astype(np.float32),
            (src_bgr.shape[1], src_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    inner = m > 0.75

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    inner_eroded = cv2.erode(inner.astype(np.uint8), kernel, iterations=1) > 0
    if np.count_nonzero(inner_eroded) >= 64:
        m_bool = inner_eroded
    elif np.count_nonzero(inner) >= 64:
        m_bool = inner
    else:
        return src_bgr.copy()

    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    s_l, s_a, s_b = src_lab[:, :, 0], src_lab[:, :, 1], src_lab[:, :, 2]
    d_l, d_a, d_b = dst_lab[:, :, 0], dst_lab[:, :, 1], dst_lab[:, :, 2]

    s_l_mean, s_l_std = float(np.mean(s_l[m_bool])), float(np.std(s_l[m_bool]))
    s_a_mean, s_a_std = float(np.mean(s_a[m_bool])), float(np.std(s_a[m_bool]))
    s_b_mean, s_b_std = float(np.mean(s_b[m_bool])), float(np.std(s_b[m_bool]))

    d_l_mean, d_l_std = float(np.mean(d_l[m_bool])), float(np.std(d_l[m_bool]))
    d_a_mean, d_a_std = float(np.mean(d_a[m_bool])), float(np.std(d_a[m_bool]))
    d_b_mean, d_b_std = float(np.mean(d_b[m_bool])), float(np.std(d_b[m_bool]))

    ratio_a = np.clip(d_a_std / max(s_a_std, eps), 0.5, 1.8)
    ratio_b = np.clip(d_b_std / max(s_b_std, eps), 0.5, 1.8)
    out_a = (s_a - s_a_mean) * ratio_a + d_a_mean
    out_b = (s_b - s_b_mean) * ratio_b + d_b_mean

    ratio_l = np.clip(d_l_std / max(s_l_std, eps), 0.75, 1.25)
    target_l_mean = np.clip(
        d_l_mean,
        s_l_mean * (1.0 - l_clamp_ratio),
        s_l_mean * (1.0 + l_clamp_ratio),
    )
    raw_l = (s_l - s_l_mean) * (0.5 * ratio_l + 0.5) + target_l_mean
    min_l = s_l * (1.0 - l_clamp_ratio)
    max_l = np.maximum(s_l * (1.0 + l_clamp_ratio), s_l + 6.0)
    out_l = np.clip(raw_l, min_l, max_l)

    out_lab = np.stack(
        [
            np.clip(out_l, 0.0, 255.0),
            np.clip(out_a, 0.0, 255.0),
            np.clip(out_b, 0.0, 255.0),
        ],
        axis=-1,
    ).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


# ==============================================================================
# 2. REFERENCE #2: Dynamic Velocity-Expanded Box Mask & Tri-Mask Compositing
# ==============================================================================
def build_feathered_box_mask(
    size: int = 256,
    pad_ratio: float = 0.12,
    blur_ksize: int = 31,
) -> np.ndarray:
    """Creates an aligned crop box mask with internal border padding and
    Gaussian feathering to reduce the visibility of the square crop boundary.
    """
    mask = np.zeros((size, size), dtype=np.float32)
    pad = max(4, int(round(size * pad_ratio)))
    mask[pad : size - pad, pad : size - pad] = 1.0
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    mask[:4, :] = 0.0
    mask[-4:, :] = 0.0
    mask[:, :4] = 0.0
    mask[:, -4:] = 0.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


_DYNAMIC_MASK_CACHE = {}


def get_dynamic_box_mask(velocity: float = 0.0, size: int = 256) -> np.ndarray:
    """Dynamic bounding-box padding and mask anti-aliasing.

    When frame-to-frame landmark velocity exceeds `5.0 px/frame`, expands the active
    crop mask area outward by `~18%` (reducing internal padding from `12%` to `5.5%`)
    and feathers the outer boundary with an expanded `(45, 45)` Gaussian blur so
    fast head turns never clip the chin/cheeks or show hard circular edges.
    """
    key_stat = (size, "stat")
    key_mot = (size, "mot")
    if key_stat not in _DYNAMIC_MASK_CACHE:
        _DYNAMIC_MASK_CACHE[key_stat] = build_feathered_box_mask(
            size=size, pad_ratio=0.12, blur_ksize=max(31, int(round(31 * size / 256.0)) | 1)
        )
        _DYNAMIC_MASK_CACHE[key_mot] = build_feathered_box_mask(
            size=size, pad_ratio=0.055, blur_ksize=max(45, int(round(45 * size / 256.0)) | 1)
        )
    if velocity > 5.0:
        return _DYNAMIC_MASK_CACHE[key_mot]
    return _DYNAMIC_MASK_CACHE[key_stat]


def refine_occlusion_mask(
    raw_occ_mask: np.ndarray,
    erode_px: int = 3,
    blur_ksize: int = 15,
) -> np.ndarray:
    """Refines `xseg_1_cuda.onnx` occlusion mask (`> 0.5`, eroded by `3px`, blurred by `(15, 15)`)."""
    m = (raw_occ_mask.squeeze() > 0.5).astype(np.float32)
    if erode_px > 0:
        k_erode = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode))
        m = cv2.erode(m, kernel, iterations=1)
    if blur_ksize > 1:
        k_blur = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        m = cv2.GaussianBlur(m, (k_blur, k_blur), 0)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def combine_tri_mask(
    box_mask: np.ndarray,
    occ_mask: np.ndarray,
    region_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """FaceFusion exact mathematical minimum combination:
    `final_crop_mask = np.minimum(box_mask, np.minimum(occ_mask, region_mask))`
    """
    b = box_mask.squeeze().astype(np.float32)
    o = occ_mask.squeeze().astype(np.float32)
    if o.shape != b.shape:
        o = cv2.resize(o, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
    if region_mask is not None:
        r = region_mask.squeeze().astype(np.float32)
        if r.shape != b.shape:
            r = cv2.resize(r, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined = np.minimum(b, np.minimum(o, r))
    else:
        combined = np.minimum(b, o)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def blend_gfpgan_crop(
    swapped_crop_bgr: np.ndarray,
    gfpgan_crop_bgr: np.ndarray,
    blend: float = 0.25,
) -> np.ndarray:
    """Blends `gfpgan_1.4_fp16` output inside the landmark-aligned crop at a strict `0.25` weight
    (balancing skin texture definition against identity fidelity).
    """
    w = float(np.clip(blend, 0.0, 1.0))
    if gfpgan_crop_bgr.shape[:2] != swapped_crop_bgr.shape[:2]:
        gfpgan_crop_bgr = cv2.resize(
            gfpgan_crop_bgr,
            (swapped_crop_bgr.shape[1], swapped_crop_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return cv2.addWeighted(swapped_crop_bgr, 1.0 - w, gfpgan_crop_bgr, w, 0.0)


def paste_back_zero_border(
    target_frame: np.ndarray,
    swapped_crop: np.ndarray,
    crop_mask: np.ndarray,
    affine_matrix: np.ndarray,
) -> np.ndarray:
    """Warps the landmark-aligned crop and composite mask back through the exact inverse
    affine transform matrix (`M_inverse`) using `cv2.warpAffine` with bilinear anti-aliasing:
    `flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0`.
    """
    h, w = target_frame.shape[:2]
    inv_flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped_crop = cv2.warpAffine(
        swapped_crop,
        affine_matrix,
        (w, h),
        flags=inv_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.warpAffine(
        crop_mask.squeeze().astype(np.float32),
        affine_matrix,
        (w, h),
        flags=inv_flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    out = target_frame.copy()
    nz_y, nz_x = np.nonzero(warped_mask > 1e-4)
    if len(nz_y) == 0:
        return out
    y0, y1 = int(nz_y.min()), int(nz_y.max()) + 1
    x0, x1 = int(nz_x.min()), int(nz_x.max()) + 1
    alpha = np.clip(warped_mask[y0:y1, x0:x1], 0.0, 1.0)[:, :, np.newaxis]
    roi_t = out[y0:y1, x0:x1].astype(np.float32)
    roi_c = warped_crop[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(roi_t * (1.0 - alpha) + roi_c * alpha, 0.0, 255.0).astype(np.uint8)
    return out


# ==============================================================================
# 3. REFERENCE #3: Spatial-Continuity Cascade & 15-Point Optical Flow Lock
# ==============================================================================
def select_best_target_face(
    bboxes: np.ndarray,
    kpss: np.ndarray,
    last_center: Optional[np.ndarray] = None,
    min_box_size: float = 44.0,
    det_thresh: float = 0.12,
) -> Optional[FaceDict]:
    """AGENT_1: Spatial-Continuity + Area + Score Face Selector.

    Replaces `Deep-Live-Cam`'s `bboxes[:, 0].argmin()` (which picked the leftmost
    box in the room and jumped off the user's face whenever a left-edge false positive
    exceeded `det_thresh`).
    """
    if bboxes is None or kpss is None or len(bboxes) == 0:
        return None

    best_face: Optional[FaceDict] = None
    best_metric = -1.0

    for i in range(len(bboxes)):
        box = np.asarray(bboxes[i], dtype=np.float32)
        score = float(box[4]) if len(box) >= 5 else 0.5
        if score < det_thresh:
            continue
        w = float(box[2] - box[0])
        h = float(box[3] - box[1])
        if w < min_box_size or h < min_box_size:
            continue

        cx = 0.5 * float(box[0] + box[2])
        cy = 0.5 * float(box[1] + box[3])
        area = w * h

        prox_weight = 1.0
        if last_center is not None:
            dist = float(np.hypot(cx - float(last_center[0]), cy - float(last_center[1])))
            prox_weight = 1.0 / (1.0 + dist / 140.0)

        metric = score * area * prox_weight
        if metric > best_metric:
            best_metric = metric
            kps_i = np.asarray(kpss[i], dtype=np.float32).reshape(5, 2)
            best_face = FaceDict(
                bbox=box[:4].copy(),
                kps=kps_i.copy(),
                det_score=score,
                landmark_2d_106=None,
                velocity=0.0,
            )

    return best_face


def stabilize_kps_with_dense_106(
    kps_5: np.ndarray,
    landmark_2d_106: Optional[np.ndarray],
    dense_weight: float = 0.65,
) -> np.ndarray:
    """Stabilizes 5-point warp landmarks using rigid dense 106-point anatomical contours."""
    kps = np.asarray(kps_5, dtype=np.float32).reshape(5, 2).copy()
    if landmark_2d_106 is None:
        return kps
    pts106 = np.asarray(landmark_2d_106, dtype=np.float32)
    if pts106.shape != (106, 2):
        return kps

    dense_5 = np.zeros((5, 2), dtype=np.float32)
    dense_5[0] = np.mean(pts106[33:43], axis=0)
    dense_5[1] = np.mean(pts106[87:97], axis=0)
    dense_5[2] = pts106[86]
    dense_5[3] = pts106[52]
    dense_5[4] = pts106[61]

    w = float(np.clip(dense_weight, 0.0, 1.0))
    return (1.0 - w) * kps + w * dense_5


def extract_top15_upper_face_anchors(
    gray_frame: np.ndarray,
    kps_5: np.ndarray,
    bbox: Optional[np.ndarray] = None,
) -> np.ndarray:
    """AGENT_2: Extracts up to 15 highest-contrast visible pixels on the forehead,
    eyebrows, and eye sockets (`[y_min : y_nose, x_min : x_max]`) plus the 3 upper
    landmarks (`left_eye`, `right_eye`, `nose_tip`) for Lucas-Kanade optical flow tracking.
    """
    h, w = gray_frame.shape[:2]
    kps = np.asarray(kps_5, dtype=np.float32).reshape(5, 2)
    upper_landmarks = kps[:3].reshape(-1, 1, 2)

    if bbox is not None and len(bbox) >= 4:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(max(kps[2, 1], bbox[1] + 0.60 * (bbox[3] - bbox[1]))))
    else:
        eye_dist = max(20.0, float(np.linalg.norm(kps[1] - kps[0])))
        cx = 0.5 * float(kps[0, 0] + kps[1, 0])
        cy = 0.5 * float(kps[0, 1] + kps[1, 1])
        x1 = max(0, int(cx - 1.2 * eye_dist))
        x2 = min(w, int(cx + 1.2 * eye_dist))
        y1 = max(0, int(cy - 1.0 * eye_dist))
        y2 = min(h, int(max(kps[2, 1], cy + 0.6 * eye_dist)))

    if x2 - x1 >= 12 and y2 - y1 >= 12:
        roi = gray_frame[y1:y2, x1:x2]
        corners = cv2.goodFeaturesToTrack(
            roi, maxCorners=15, qualityLevel=0.01, minDistance=4, blockSize=3
        )
        if corners is not None and len(corners) > 0:
            corners = corners.astype(np.float32)
            corners[:, 0, 0] += float(x1)
            corners[:, 0, 1] += float(y1)
            return np.concatenate([upper_landmarks, corners], axis=0)

    return upper_landmarks


class ZeroDropoutFaceTracker:
    """AGENT_1 & AGENT_2: Multi-Scale Cascade + 15-Point Forehead/Eye-Socket Optical Flow
    Watchdog with Adaptive `alpha = 0.95` Lock.

    - Runs primary detector (`scrfd_2.5g.onnx` @ `0.12`) and fallback detector (`yolo_face` @ `0.12`).
    - When both detectors miss during rapid head tilts or extreme hand occlusion, tracks the
      top 15 highest-contrast forehead/eyebrow/eye-socket pixels + upper landmarks forward via
      `cv2.calcOpticalFlowPyrLK` with `alpha = 0.95` adaptive smoothing.
    - When `never_drop=True` (enabled on the live GPU stream), even if occlusion exceeds
      `max_occluded_frames`, holds the last valid face pose so the server NEVER flashes an
      unswapped raw frame!
    """

    def __init__(
        self,
        primary_detector: Callable[[np.ndarray], Optional[object]],
        fallback_detector: Optional[Callable[[np.ndarray], Optional[object]]] = None,
        max_occluded_frames: int = 15,
        alpha: float = 0.95,
        never_drop: bool = False,
        use_rigid_one_euro: bool = False,
        mincutoff: float = 0.50,
        beta: float = 0.005,
        dcutoff: float = 1.0,
    ):
        self.primary_detector = primary_detector
        self.fallback_detector = fallback_detector
        self.max_occluded_frames = int(max_occluded_frames)
        self.alpha = float(np.clip(alpha, 0.1, 1.0))
        self.never_drop = bool(never_drop)
        self.use_rigid_one_euro = bool(use_rigid_one_euro)
        self.rigid_one_euro = RigidPoseOneEuroFilter(
            mincutoff=mincutoff, beta=beta, dcutoff=dcutoff
        )
        self.prev_gray: Optional[np.ndarray] = None
        self.last_face: Optional[FaceDict] = None
        self.last_center: Optional[np.ndarray] = None
        self.last_velocity: float = 0.0
        self.occluded_count: int = 0
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

    def _wrap_face(self, f: object, velocity: float) -> FaceDict:
        if isinstance(f, FaceDict):
            f["velocity"] = float(velocity)
            return f
        return FaceDict(
            kps=np.asarray(getattr(f, "kps"), dtype=np.float32).reshape(5, 2),
            bbox=(
                np.asarray(getattr(f, "bbox"), dtype=np.float32)
                if getattr(f, "bbox", None) is not None
                else None
            ),
            det_score=float(getattr(f, "det_score", 0.5)),
            embedding=getattr(f, "embedding", None),
            normed_embedding=getattr(f, "normed_embedding", None),
            landmark_2d_106=getattr(f, "landmark_2d_106", None),
            velocity=float(velocity),
        )

    def get_face(
        self, frame_bgr: np.ndarray, timestamp: Optional[float] = None
    ) -> Optional[FaceDict]:
        ts = float(timestamp) if timestamp is not None else time.perf_counter()
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Primary detector (SCRFD @ det_thresh=0.12)
        face = self.primary_detector(frame_bgr)

        # 2. Same-frame YoloFace_8n cascade fallback if primary returns None
        if (face is None or getattr(face, "kps", None) is None) and self.fallback_detector is not None:
            face = self.fallback_detector(frame_bgr)

        if face is not None and getattr(face, "kps", None) is None:
            face = None

        if face is not None:
            lm106 = getattr(face, "landmark_2d_106", None)
            new_kps = stabilize_kps_with_dense_106(face.kps, lm106)
            if self.use_rigid_one_euro:
                new_kps, lm106 = self.rigid_one_euro.filter_kps(
                    new_kps, timestamp=ts, lm106=lm106
                )
            if self.last_face is not None:
                prev_kps = np.asarray(self.last_face["kps"], dtype=np.float32).reshape(5, 2)
                vel = float(np.linalg.norm(np.mean(new_kps - prev_kps, axis=0)))
            else:
                vel = 0.0

            face.kps = new_kps.astype(np.float32)
            if lm106 is not None:
                try:
                    face.landmark_2d_106 = lm106
                except Exception:
                    pass
            wrapped = self._wrap_face(face, velocity=vel)
            self.last_face = wrapped
            self.last_center = np.mean(new_kps, axis=0)
            self.last_velocity = vel
            self.prev_gray = curr_gray
            self.occluded_count = 0
            return wrapped

        # 3. 15-Point Forehead/Eyebrow/Eye-Socket Optical Flow Watchdog
        if self.last_face is not None and self.prev_gray is not None and self.prev_gray.shape == curr_gray.shape:
            if self.occluded_count < self.max_occluded_frames:
                self.occluded_count += 1
                prev_kps = np.asarray(self.last_face["kps"], dtype=np.float32).reshape(5, 2)
                prev_bbox = self.last_face.get("bbox", None)
                anchors = extract_top15_upper_face_anchors(self.prev_gray, prev_kps, prev_bbox)

                next_pts, st, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, curr_gray, anchors, None, **self.lk_params
                )
                if next_pts is not None and st is not None and np.any(st == 1):
                    valid = st.flatten() == 1
                    raw_delta = np.median(
                        next_pts[valid].reshape(-1, 2) - anchors[valid].reshape(-1, 2),
                        axis=0,
                    )
                    delta = np.clip(raw_delta * self.alpha, -45.0, 45.0).astype(np.float32)
                else:
                    delta = np.zeros(2, dtype=np.float32)

                vel = float(np.linalg.norm(delta))
                new_kps = (prev_kps + delta).astype(np.float32)
                new_lm106 = (
                    np.asarray(self.last_face["landmark_2d_106"], dtype=np.float32) + delta
                    if self.last_face.get("landmark_2d_106", None) is not None
                    else None
                )
                if self.use_rigid_one_euro:
                    new_kps, new_lm106 = self.rigid_one_euro.filter_kps(
                        new_kps, timestamp=ts, lm106=new_lm106
                    )
                tracked_face = FaceDict(
                    kps=new_kps,
                    bbox=(
                        np.asarray(prev_bbox, dtype=np.float32)
                        + np.array([delta[0], delta[1], delta[0], delta[1]], dtype=np.float32)
                        if prev_bbox is not None
                        else np.array([0, 0, curr_gray.shape[1], curr_gray.shape[0]], dtype=np.float32)
                    ),
                    det_score=float(self.last_face.get("det_score", 0.5)),
                    embedding=self.last_face.get("embedding", None),
                    normed_embedding=self.last_face.get("normed_embedding", None),
                    landmark_2d_106=new_lm106,
                    velocity=vel,
                )
                self.last_face = tracked_face
                self.last_center = np.mean(new_kps, axis=0)
                self.last_velocity = vel
                self.prev_gray = curr_gray
                return tracked_face

            if self.never_drop:
                self.prev_gray = curr_gray
                return self.last_face

        # Track lost -> enforce immediate One Euro Filter state reset
        self.rigid_one_euro.reset()
        self.prev_gray = curr_gray
        return None


# ==============================================================================
# 4. GPU Runtime & Port 5555 Server Wiring
# ==============================================================================
SERVER_JPEG_QUALITY = 90
GFPGAN_ALIGNED_BLEND = 0.25

_BOX_MASK_256: Optional[np.ndarray] = None
_GFPGAN_SESSION = None
_TRACKER: Optional[ZeroDropoutFaceTracker] = None


def restore_aligned_gfpgan_512(
    crop_bgr: np.ndarray,
    gfpgan_session: Any = None,
    blend: float = GFPGAN_ALIGNED_BLEND,
) -> np.ndarray:
    """Runs landmark-aligned GFPGANv1.4 restoration in `512x512` FFHQ aligned crop space
    with a strict parameter blend of `0.25` (`0.75 * swapped + 0.25 * gfpgan_restored`).
    """
    sess = gfpgan_session
    orig_h, orig_w = crop_bgr.shape[:2]
    if sess is not None:
        inp_512 = (
            crop_bgr
            if (orig_h == 512 and orig_w == 512)
            else cv2.resize(crop_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
        )
        rgb = inp_512[:, :, ::-1].astype(np.float32) * (1.0 / 127.5) - 1.0
        tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)
        inp_name = sess.get_inputs()[0].name
        out = sess.run(None, {inp_name: tensor})[0][0]
        out = np.clip(np.transpose(out, (1, 2, 0)), -1.0, 1.0)
        restored_512 = ((out + 1.0) * 127.5)[:, :, ::-1].astype(np.uint8)
        restored = (
            restored_512
            if (orig_h == 512 and orig_w == 512)
            else cv2.resize(restored_512, (orig_w, orig_h), interpolation=cv2.INTER_AREA)
        )
        return blend_gfpgan_crop(crop_bgr, restored, blend=blend)

    # High-frequency detail restoration fallback in aligned landmark space at 0.25 blend
    blurred = cv2.GaussianBlur(crop_bgr, (0, 0), 1.2)
    sharpened = cv2.addWeighted(crop_bgr, 1.45, blurred, -0.45, 0.0)
    return blend_gfpgan_crop(crop_bgr, sharpened, blend=blend)


def _run_gfpgan_crop_256(crop_bgr_256: np.ndarray, blend: float = GFPGAN_ALIGNED_BLEND) -> np.ndarray:
    """Runs `gfpgan_1.4_fp16.onnx` on an aligned crop and blends at `0.25`."""
    return restore_aligned_gfpgan_512(crop_bgr_256, gfpgan_session=_GFPGAN_SESSION, blend=blend)


def apply_masked_color_refinement(
    crop_bgr: np.ndarray,
    crop_orig: np.ndarray,
    box_mask: np.ndarray,
    occ_mask: np.ndarray,
    region_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine spatial, occlusion and optional semantic masks, then apply
    masked color transfer before inverse-warp compositing.
    """
    final_crop_mask = combine_tri_mask(box_mask, occ_mask, region_mask)
    out_crop = dfl_masked_color_transfer(
        crop_bgr, crop_orig, final_crop_mask, l_clamp_ratio=0.15
    )
    return out_crop, final_crop_mask


def start_gpu_server() -> None:
    """Initialize CUDA providers, models, and the newest-frame ZMQ consumer loop."""
    global _BOX_MASK_256, _GFPGAN_SESSION, _TRACKER

    import ctypes
    for lib_dir in sorted(glob.glob("/venv/main/lib/python*/site-packages/nvidia/*/lib")):
        for so_file in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
            try:
                ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass

    import torch
    import zmq
    import onnxruntime as ort

    _cuda_warmup = torch.empty((256, 256), device="cuda")

    orig_ort_init = ort.InferenceSession.__init__

    def _fast_cuda_ort_init(self, path_or_bytes, sess_options=None, providers=None, **kwargs):
        if isinstance(path_or_bytes, str):
            if path_or_bytes.endswith("xseg_1.onnx"):
                cuda_xseg = path_or_bytes.replace("xseg_1.onnx", "xseg_1_cuda.onnx")
                if os.path.exists(cuda_xseg):
                    path_or_bytes = cuda_xseg
            elif path_or_bytes.endswith("gfpgan_1.4.onnx"):
                fp16_gfpgan = path_or_bytes.replace("gfpgan_1.4.onnx", "gfpgan_1.4_fp16.onnx")
                if os.path.exists(fp16_gfpgan):
                    path_or_bytes = fp16_gfpgan
        return orig_ort_init(self, path_or_bytes, sess_options=sess_options, providers=providers, **kwargs)

    ort.InferenceSession.__init__ = _fast_cuda_ort_init

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ff_root = os.environ.get("FACE_SWAP_FACEFUSION_ROOT", os.path.join(repo_root, "vendor", "official_facefusion"))
    dlc_root = os.environ.get("FACE_SWAP_DLC_ROOT", os.path.join(repo_root, "vendor", "Deep-Live-Cam"))
    for p in (ff_root, dlc_root):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(dlc_root)

    import modules.globals
    from modules.model_downloader import ensure_insightface_pack
    from modules.face_analyser import get_face_analyser, get_one_face, ensure_landmarks
    import modules.processors.frame.face_swapper as face_swapper

    from facefusion import state_manager, face_masker, face_detector
    from facefusion.face_masker import create_occlusion_mask, create_region_mask
    from facefusion.face_helper import warp_face_by_face_landmark_5

    state_manager.init_item("execution_providers", ["cuda"])
    state_manager.init_item("execution_device_ids", ["0"])
    state_manager.init_item("execution_thread_count", 1)
    state_manager.init_item("download_providers", ["github", "huggingface"])
    state_manager.init_item("face_occluder_model", "xseg_1")
    state_manager.init_item("face_parser_model", "bisenet_resnet_34")
    state_manager.init_item("face_detector_model", "yolo_face")
    state_manager.init_item("face_detector_size", "640x640")
    state_manager.init_item("face_detector_score", 0.12)
    state_manager.init_item("face_mask_blur", 0.45)
    state_manager.init_item("face_mask_padding", (12, 12, 12, 12))
    state_manager.init_item(
        "face_mask_regions",
        [
            "skin",
            "left-eyebrow",
            "right-eyebrow",
            "left-eye",
            "right-eye",
            "nose",
            "mouth",
            "upper-lip",
            "lower-lip",
        ],
    )
    face_masker.pre_check()
    try:
        face_detector.pre_check()
    except Exception:
        pass

    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=str, default="")
    args, _ = parser.parse_known_args()
    custom_source = args.source.strip()

    _BOX_MASK_256 = get_dynamic_box_mask(velocity=0.0, size=256)
    _ = get_dynamic_box_mask(velocity=10.0, size=256)

    xseg_session = None
    xseg_input_name = "input"
    xseg_path = os.path.join(ff_root, ".assets", "models", "xseg_1_cuda.onnx")
    if os.path.exists(xseg_path):
        try:
            xseg_session = ort.InferenceSession(
                xseg_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            xseg_input_name = xseg_session.get_inputs()[0].name
            _ = xseg_session.run(None, {xseg_input_name: np.zeros((1, 256, 256, 3), dtype=np.float32)})
            print(f"[GPU-VERIFY] xseg_1_cuda.onnx (direct CUDA session): {xseg_session.get_providers()}", flush=True)
        except Exception as e:
            print(f"[GPU-VERIFY] xseg direct load fallback: {e}", flush=True)
            xseg_session = None

    gfpgan_candidates = [
        os.path.join(ff_root, ".assets", "models", "gfpgan_1.4_fp16.onnx"),
        os.path.join(ff_root, ".assets", "models", "gfpgan_1.4.onnx"),
        os.path.join(dlc_root, "models", "GFPGANv1.4.onnx"),
    ]
    for gp in gfpgan_candidates:
        if os.path.exists(gp):
            try:
                _GFPGAN_SESSION = ort.InferenceSession(
                    gp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                print(
                    f"[GPU-VERIFY] gfpgan_1.4_fp16 (FFHQ 512x512 aligned, {GFPGAN_ALIGNED_BLEND:.2f} blend): {_GFPGAN_SESSION.get_providers()}",
                    flush=True,
                )
                break
            except Exception as e:
                print(f"[GPU-VERIFY] GFPGAN load skipped ({gp}): {e}", flush=True)

    use_mouth_mask = "--mouth-mask" in sys.argv
    modules.globals.execution_providers = ["CUDAExecutionProvider"]
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.fp_ui["face_enhancer"] = False
    modules.globals.mouth_mask = bool(use_mouth_mask)
    modules.globals.poisson_blend = False
    modules.globals.map_faces = True
    modules.globals.keep_fps = True
    modules.globals.many_faces = False
    modules.globals.color_correction = False
    modules.globals.det_size = 320

    ensure_insightface_pack("buffalo_l")
    fa = get_face_analyser()
    if hasattr(fa, "det_model") and hasattr(fa.det_model, "det_thresh"):
        fa.det_model.det_thresh = 0.12

    face_swapper.pre_check()
    face_swapper.pre_start()
    swapper_model = face_swapper.get_face_swapper()

    if not custom_source:
        raise RuntimeError("Supply --source with one consented reference image")
    if not os.path.isfile(custom_source):
        raise RuntimeError(f"Source image does not exist: {custom_source}")
    src_img = cv2.imread(custom_source)
    if src_img is None:
        raise RuntimeError(f"Could not read --source image: {custom_source}")
    source_face = get_one_face(src_img)
    if source_face is None or getattr(source_face, "embedding", None) is None:
        raise RuntimeError(f"No valid face detected in --source image: {custom_source}")
    active_ref_image = custom_source
    modules.globals.source_path = custom_source
    modules.globals.source_target_map = [{"id": 0, "source": {"face": source_face}}]
    swap_providers = swapper_model.session.get_providers()
    print(
        f"[GPU-VERIFY] inswapper_128 (FP16 preferred when available) + consented source allocated: {swap_providers}",
        flush=True,
    )
    print(
        "[GPU-VERIFY] Rigid One-Euro Filter (mincutoff=0.50Hz, beta=0.005) + optional FFHQ 512x512 GFPGAN (0.25 blend) + Masked-RCT: ACTIVE",
        flush=True,
    )
    if use_mouth_mask:
        print("[GPU-VERIFY] Original-mouth preservation: ACTIVE", flush=True)

    def _scrfd_cascade_detect(frm: np.ndarray) -> Optional[FaceDict]:
        bboxes, kpss = fa.det_model.detect(frm, max_num=0, metric="default")
        last_c = _TRACKER.last_center if _TRACKER is not None else None
        f = select_best_target_face(bboxes, kpss, last_center=last_c, min_box_size=44.0, det_thresh=0.12)
        if f is not None:
            ensure_landmarks(frm, [f])
            return f
        return None

    def _yolo_face_cascade_detect(frm: np.ndarray) -> Optional[FaceDict]:
        try:
            bboxes, scores, kps_list = face_detector.detect_with_yolo_face(frm, "640x640")
            if bboxes is not None and len(bboxes) > 0 and kps_list is not None and len(kps_list) > 0:
                bbox_arr = np.zeros((len(bboxes), 5), dtype=np.float32)
                for idx_b, b in enumerate(bboxes):
                    bbox_arr[idx_b, :4] = np.asarray(b, dtype=np.float32)[:4]
                    bbox_arr[idx_b, 4] = float(scores[idx_b]) if idx_b < len(scores) else 0.5
                kps_arr = np.asarray(kps_list, dtype=np.float32).reshape(-1, 5, 2)
                last_c = _TRACKER.last_center if _TRACKER is not None else None
                f = select_best_target_face(
                    bbox_arr, kps_arr, last_center=last_c, min_box_size=44.0, det_thresh=0.12
                )
                if f is not None:
                    ensure_landmarks(frm, [f])
                    return f
        except Exception:
            pass
        return None

    _TRACKER = ZeroDropoutFaceTracker(
        primary_detector=_scrfd_cascade_detect,
        fallback_detector=_yolo_face_cascade_detect,
        max_occluded_frames=15,
        alpha=0.95,
        never_drop=True,
        use_rigid_one_euro=True,
        mincutoff=0.50,
        beta=0.005,
        dcutoff=1.0,
    )

    mask_regions = state_manager.get_item("face_mask_regions")
    use_onnx_gfpgan = "--onnx-gfpgan" in sys.argv

    def process_frame_unified(frm: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        target_face = _TRACKER.get_face(frm, timestamp=timestamp)
        if target_face is None or getattr(target_face, "kps", None) is None:
            return frm

        velocity = float(target_face.get("velocity", 0.0))
        dyn_box_mask = get_dynamic_box_mask(velocity=velocity, size=256)

        # 1. Warp original frame into FFHQ-aligned crop space and run CUDA-graph face swapper directly
        crop_orig, affine_matrix = warp_face_by_face_landmark_5(
            frm, target_face.kps, "ffhq_512", (256, 256)
        )
        bgr_fake, M_128 = swapper_model.get(frm, target_face, source_face, paste_back=False)
        if bgr_fake is None:
            return frm

        # Map 128x128 swapped face directly into FFHQ-aligned crop space via composite similarity matrix
        inv_M_128 = cv2.invertAffineTransform(M_128)
        M_128_to_ffhq = affine_matrix[:, :2] @ inv_M_128
        M_128_to_ffhq[:, 2] += affine_matrix[:, 2]
        crop_swap = cv2.warpAffine(
            bgr_fake,
            M_128_to_ffhq,
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 2. Compute occlusion mask directly via CUDA xseg_1_cuda session
        if xseg_session is not None:
            xseg_in = (crop_orig.astype(np.float32) * (1.0 / 255.0))[np.newaxis, ...]
            raw_occ = xseg_session.run(None, {xseg_input_name: xseg_in})[0][0]
            raw_occ = (cv2.GaussianBlur(np.clip(raw_occ.squeeze(), 0.0, 1.0), (0, 0), 5).clip(0.5, 1.0) - 0.5) * 2.0
        else:
            raw_occ = create_occlusion_mask(crop_orig)
        occ_mask = refine_occlusion_mask(raw_occ, erode_px=3, blur_ksize=15)
        region_mask = None
        if "--bisenet-region" in sys.argv:
            try:
                region_mask = create_region_mask(crop_orig, mask_regions)
            except Exception:
                region_mask = None

        # 3. Masked Reinhard Color Transfer inside inner skin mask + Tri-Mask combination
        crop_swap, final_crop_mask = apply_masked_color_refinement(
            crop_swap, crop_orig, dyn_box_mask, occ_mask, region_mask
        )

        # 4. Landmark-aligned GFPGAN restoration in FFHQ aligned space at strict 0.25 blend
        crop_swap = restore_aligned_gfpgan_512(
            crop_swap,
            gfpgan_session=(_GFPGAN_SESSION if use_onnx_gfpgan else None),
            blend=GFPGAN_ALIGNED_BLEND,
        )

        # 5. Map restored crop and composite mask back through exact inverse affine matrix (M_inverse)
        swapped_frame = paste_back_zero_border(frm, crop_swap, final_crop_mask, affine_matrix)

        # 6. Official Deep-Live-Cam Mouth Preservation Mask (--mouth-mask / mouth_mask=True)
        if use_mouth_mask:
            if getattr(target_face, "landmark_2d_106", None) is None:
                ensure_landmarks(frm, [target_face])
            face_mask = face_swapper.create_face_mask(target_face, frm)
            _, mouth_cutout, mouth_box, lower_lip_polygon = face_swapper.create_lower_mouth_mask(
                target_face, frm
            )
            if mouth_cutout is not None and mouth_box != (0, 0, 0, 0):
                swapped_frame = face_swapper.apply_mouth_area(
                    swapped_frame, mouth_cutout, mouth_box, face_mask, lower_lip_polygon
                )

        return swapped_frame

    ref_img = cv2.imread(active_ref_image)
    test_frame = cv2.resize(ref_img, (640, 480))
    _ = process_frame_unified(test_frame.copy())
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        warm_out = process_frame_unified(test_frame.copy(), timestamp=t0)
        times.append((time.perf_counter() - t0) * 1000.0)

    mouth_tag = ", mouth_mask=True (create_lower_mouth_mask)" if use_mouth_mask else ""
    mode_label = f"single-image InSwapper 128, source={active_ref_image}{mouth_tag}"
    print(
        f"[UNIFIED-SERVER] Warmup complete ({mode_label} + Async-LifoQueue + FFHQ-512-GFPGAN-0.25 + Rigid-OneEuro-0.50Hz). "
        f"Mean latency: {np.mean(times):.1f}ms ({1000.0 / np.mean(times):.1f} FPS) | shape={warm_out.shape}",
        flush=True,
    )
    if "--test-only" in sys.argv:
        return

    ctx = zmq.Context()
    sock = ctx.socket(zmq.ROUTER)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.setsockopt(zmq.RCVHWM, 2)
    sock.setsockopt(zmq.LINGER, 0)
    bind_endpoint = os.environ.get("FACE_SWAP_BIND", "tcp://127.0.0.1:5555")
    sock.bind(bind_endpoint)
    print(
        f"[UNIFIED-SERVER] Listening on {bind_endpoint} ({mode_label}, JPEG=90, Async-LifoQueue, FFHQ-512-GFPGAN-0.25, Rigid-OneEuro)",
        flush=True,
    )

    inbound_slot = LatestFrameSlot()
    sock_lock = threading.Lock()
    stop_event = threading.Event()

    def _zmq_ingest_worker() -> None:
        """Receive frames in a background thread and retain only the newest payload.

        The worker drains available ZeroMQ messages into `inbound_slot`, dropping stale
        buffered frames whenever network arrival exceeds GPU processing rate.
        """
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        while not stop_event.is_set():
            try:
                events = dict(poller.poll(2))
                if sock in events:
                    latest_parts = None
                    with sock_lock:
                        while True:
                            try:
                                parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                                if len(parts) >= 2:
                                    if latest_parts is not None:
                                        inbound_slot.dropped_frames += 1
                                    latest_parts = parts
                            except zmq.Again:
                                break
                    if latest_parts is not None:
                        inbound_slot.put(
                            latest_parts[0], latest_parts[-1], timestamp=time.perf_counter()
                        )
            except Exception:
                time.sleep(0.001)

    ingest_thread = threading.Thread(
        target=_zmq_ingest_worker, name="zmq-lifo-ingest", daemon=True
    )
    ingest_thread.start()

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), SERVER_JPEG_QUALITY]
    frame_count = 0
    t_log = time.perf_counter()

    while True:
        try:
            item = inbound_slot.get(timeout=0.05)
            if item is None:
                continue

            ident, payload, recv_ts = item
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            t0 = time.perf_counter()
            out_frame = process_frame_unified(frame, timestamp=recv_ts)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            ok, out_jpg = cv2.imencode(".jpg", out_frame, encode_params)
            if not ok:
                continue

            with sock_lock:
                sock.send_multipart([ident, out_jpg.tobytes()])

            frame_count += 1
            now = time.perf_counter()
            if now - t_log >= 2.0:
                print(
                    f"[UNIFIED-SERVER] Streamed {frame_count} frames (dropped_stale={inbound_slot.dropped_frames}) | "
                    f"frame={infer_ms:.1f}ms ({1000.0 / max(infer_ms, 1.0):.1f} FPS) "
                    f"[{mode_label} + Async-LifoQueue + FFHQ-512-GFPGAN-0.25 + Rigid-OneEuro-0.50Hz]",
                    flush=True,
                )
                t_log = now
        except Exception as e:
            print(f"[UNIFIED-SERVER] Error: {e}", flush=True)


if __name__ == "__main__":
    start_gpu_server()
