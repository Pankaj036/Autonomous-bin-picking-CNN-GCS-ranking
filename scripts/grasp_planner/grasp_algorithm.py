"""
Grasp Planning Algorithm
Adapted from https://github.com/praj441/Bin-Picking-CAS-GCS/blob/main/grasp_planning/commons/custom_grasp_planning_algorithm_dense_cas.py

Full pipeline:
  1. Receive CNN output: masks (H×W label image), depth (H×W float32 metres),
     GCS scores per object
  2. For each object mask → generate grasp pivot + N candidate orientations
  3. Evaluate each candidate with GDI2 (parallel if joblib available)
  4. Rank by FLS + CRS + GCS
  5. Return best grasp: (centroid_px, centroid_py, depth_z, angle_rad, gripper_opening_m)

Hardware:  OAK-D Pro Wide wrist camera  +  Robotiq 3F gripper
"""

import numpy as np
import cv2
from math import atan2
from typing import Optional, List, Tuple, Dict

from .grasp_utils import Parameters, get_pivot_points_from_masks, select_best_grasps
from .grasp_evaluation import calculate_GDI2

# Optional parallel processing
try:
    from joblib import Parallel, delayed
    _PARALLEL = True
except ImportError:
    _PARALLEL = False

N_JOBS      = 4    # parallel workers
N_ANGLES    = 4    # grasp orientations per pivot


# ── single pose evaluation (used as joblib worker) ────────────────────────────

def _eval_one(inputs: dict, rect: np.ndarray, angle: float) -> tuple:
    """Evaluate one grasp rectangle. Returns (fls, crs, gdi2, rect, angle)."""
    fls, crs, gdi2 = calculate_GDI2(inputs, rect, angle)
    return fls, crs, gdi2, rect, angle


# ── main algorithm entry point ────────────────────────────────────────────────

def run_grasp_planning(
    rgb_img:     np.ndarray,
    depth_img:   np.ndarray,
    masks:       np.ndarray,
    gcs_scores:  np.ndarray,
    param:       Parameters,
    top_k:       int = 5,
) -> Optional[Dict]:
    """
    Run full grasp planning on one frame.

    Args:
        rgb_img    : H×W×3 uint8
        depth_img  : H×W uint16 (millimetres from OAK-D stereo) OR
                     H×W float32 (metres from CNN depth estimate)
        masks      : N×H×W bool/uint8  — one binary mask per detected object
        gcs_scores : (N,) float         — GCS score per object from CNN
        param      : Parameters instance (camera intrinsics already set)
        top_k      : return this many grasp candidates

    Returns:
        dict with keys:
            cx, cy          : grasp centroid in image pixels
            depth_m         : object depth in metres
            angle_rad       : grasp rotation angle
            gripper_open_m  : suggested gripper opening (metres)
            gcs             : GCS score of selected object
            fls             : FLS score
            crs             : CRS score
            score           : combined score
            rank            : 0 = best
        or None if no valid grasp found.
    """
    # ── 1. Normalise depth to metres ─────────────────────────────────────────
    if depth_img.dtype == np.uint16:
        depth_m = depth_img.astype(np.float32) / 1000.0
    else:
        depth_m = depth_img.astype(np.float32)
    depth_m = np.clip(depth_m, 0.05, 2.0)

    # ── 2. Build segmentation label image (0=background, 1..N=objects) ──────
    h, w = depth_m.shape
    seg_mask = np.zeros((h, w), dtype=np.int32)
    n_obj = masks.shape[0] if masks.ndim == 3 else 0
    for i in range(n_obj):
        seg_mask[masks[i].astype(bool)] = i + 1

    # ── 3. Get pivot points from masks ───────────────────────────────────────
    pivots = get_pivot_points_from_masks(masks, param)
    if not pivots:
        return None

    # ── 4. Build all candidate (rect, angle, gcs) triplets ───────────────────
    all_rects  = []
    all_angles = []
    all_gcs    = []

    for (centroid, obj_id) in pivots:
        gcs = float(gcs_scores[obj_id - 1]) if obj_id - 1 < len(gcs_scores) else 0.5
        param.target = obj_id

        # Get depth at centroid
        cx_px, cy_px = centroid
        z = float(depth_m[cy_px, cx_px]) if (0 <= cy_px < h and 0 <= cx_px < w) else param.datum_z
        if z < 0.02:
            z = param.datum_z

        rects, angles = param.draw_rect_fixed_angles(centroid, directions=N_ANGLES)
        for rect, angle in zip(rects, angles):
            all_rects.append((rect, obj_id, centroid, z))
            all_angles.append(angle)
            all_gcs.append(gcs)

    if not all_rects:
        return None

    # ── 5. Evaluate all candidates ────────────────────────────────────────────
    inputs_base = {
        'darray':       depth_m,
        'seg_mask':     seg_mask,
        'param':        param,
        'run_with_dbcc': param.DBCC_enable,
    }

    results = []
    if _PARALLEL and len(all_rects) > N_JOBS:
        raw = Parallel(n_jobs=N_JOBS)(
            delayed(_eval_one)(
                {**inputs_base, 'param': _clone_param_for_obj(param, obj_id)},
                rect, angle
            )
            for (rect, obj_id, centroid, z), angle in zip(all_rects, all_angles)
        )
    else:
        raw = []
        for (rect, obj_id, centroid, z), angle in zip(all_rects, all_angles):
            p = _clone_param_for_obj(param, obj_id)
            raw.append(_eval_one({**inputs_base, 'param': p}, rect, angle))

    # ── 6. Collect valid results ──────────────────────────────────────────────
    rect_list, fls_list, crs_list, gcs_list = [], [], [], []
    meta_list = []  # (centroid, z)

    for i, (fls, crs, gdi2, rect, angle) in enumerate(raw):
        (_, obj_id, centroid, z) = all_rects[i]
        gcs = all_gcs[i]
        rect_list.append(rect)
        fls_list.append(fls)
        crs_list.append(crs)
        gcs_list.append(gcs)
        meta_list.append({
            'idx':      i,
            'centroid': centroid,
            'z':        z,
            'angle':    angle,
            'gdi2':     gdi2,
            'obj_id':   obj_id,
        })

    # ── 7. Rank grasps ────────────────────────────────────────────────────────
    ranked = select_best_grasps(rect_list, fls_list, crs_list, gcs_list, top_n=top_k)
    if not ranked:
        return None

    best = ranked[0]
    idx  = best['_src_idx']   # stored by select_best_grasps — avoids numpy array comparison
    meta = meta_list[idx]
    centroid = meta['centroid']

    # Gripper opening in metres
    gdi2_obj = meta['gdi2']
    if gdi2_obj is not None and gdi2_obj.object_width > 0:
        gripper_open_m = min(
            float(gdi2_obj.object_width) + 0.02,   # object width + margin
            param.gripper_finger_space_max
        )
    else:
        gripper_open_m = 0.06   # default 6 cm

    return {
        'cx':            centroid[0],
        'cy':            centroid[1],
        'depth_m':       meta['z'],
        'angle_rad':     meta['angle'],
        'gripper_open_m': gripper_open_m,
        'gcs':           best['gcs'],
        'fls':           best['fls'],
        'crs':           best['crs'],
        'score':         best['score'],
        'rank':          best['rank'],
        'all_ranked':    ranked,
    }


def _clone_param_for_obj(param: Parameters, obj_id: int) -> Parameters:
    """Shallow-clone Parameters and set target object label."""
    import copy
    p = copy.copy(param)
    p.target = obj_id
    return p


# ── Convenience: pixel centroid → 3-D camera-frame point ─────────────────────

def centroid_to_3d(cx_px: int, cy_px: int, depth_m: np.ndarray,
                   param: Parameters) -> Tuple[float, float, float]:
    """
    Back-project image centroid to 3-D camera-frame coordinates.

    Args:
        cx_px, cy_px : pixel coordinates of grasp centroid
        depth_m      : H×W depth image in metres
        param        : Parameters (contains fx, fy, cx_img, cy_img)

    Returns:
        (X, Y, Z) in camera frame (metres)
    """
    h, w = depth_m.shape
    cx_px = int(np.clip(cx_px, 0, w - 1))
    cy_px = int(np.clip(cy_px, 0, h - 1))
    Z = float(depth_m[cy_px, cx_px])
    if Z < 0.02:
        # use median depth in small region around centroid
        r0 = max(cy_px - 5, 0)
        r1 = min(cy_px + 5, h)
        c0 = max(cx_px - 5, 0)
        c1 = min(cx_px + 5, w)
        region = depth_m[r0:r1, c0:c1]
        valid  = region[region > 0.02]
        Z = float(np.median(valid)) if len(valid) else param.datum_z
    X, Y = param.pixel_to_xyz(cx_px, cy_px, Z)
    return X, Y, Z
