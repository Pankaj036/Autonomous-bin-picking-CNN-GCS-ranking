"""
functions_for_length_wise_cuts.py
==================================
Geometric utilities for binary mask analysis used by utils_gs_cas.py.
These operate on 2D binary numpy arrays (H×W, dtype uint8 or bool).

Functions
---------
major_axis_length(mask)       -> (theta, major_len, minor_len)
axis_angle(mask)              -> dict with "angle", "centroid", axis endpoint lists
calculate_center(mask)        -> (row, col)  centroid of nonzero pixels
rotate_boolean_array(mask, theta, center) -> rotated binary mask
crop_binary_mask(mask, col_split)  -> (left_half, right_half)  after vertical split
"""

from math import atan2, cos, sin, radians
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
def _nonzero_points(mask: np.ndarray) -> np.ndarray:
    """
    Extract nonzero pixel coordinates from a binary mask.
    Returns Nx2 float array of (x, y) = (col, row) pairs.
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return np.zeros((1, 2), dtype=np.float64)
    return np.column_stack([cols.astype(np.float64),
                            rows.astype(np.float64)])


# ─────────────────────────────────────────────────────────────────────────────
def major_axis_length(mask: np.ndarray):
    """
    Compute the major and minor axis lengths of a binary mask using PCA.

    Parameters
    ----------
    mask : np.ndarray  (H, W)  binary

    Returns
    -------
    (theta, major_length, minor_length)
        theta       : angle of major axis in radians
        major_length: length of major axis in pixels
        minor_length: length of minor axis in pixels
    """
    points = _nonzero_points(mask)
    X = points[:, 0]
    Y = points[:, 1]
    x = X - np.mean(X)
    y = Y - np.mean(Y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]
    theta = -np.arctan2(y_v1, x_v1)

    rotation_mat = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
    transformed = rotation_mat @ np.vstack([X, Y])
    x_t, y_t = transformed
    major = float(np.max(x_t) - np.min(x_t))
    minor = float(np.max(y_t) - np.min(y_t))
    return theta, major, minor


# ─────────────────────────────────────────────────────────────────────────────
def axis_angle(mask: np.ndarray) -> dict:
    """
    Compute the orientation axes of a binary mask.

    Parameters
    ----------
    mask : np.ndarray  (H, W)  binary

    Returns
    -------
    dict with keys:
        "angle"             : float  major axis angle in radians
        "centroid"          : (cx, cy) float pixel coordinates
        "major_axis_points" : [(x1,y1), (x2,y2)]
        "minor_axis_points" : [(x1,y1), (x2,y2)]
    """
    points = _nonzero_points(mask)
    cx = float(np.mean(points[:, 0]))
    cy = float(np.mean(points[:, 1]))

    modi_x = points[:, 0] - cx
    modi_y = points[:, 1] - cy
    num = float(np.sum(modi_x * modi_y))
    den = float(np.sum(modi_x ** 2 - modi_y ** 2))
    angle = 0.5 * atan2(2.0 * num, den)

    # compute axis lengths for endpoint visualisation
    _, major_len, minor_len = major_axis_length(mask)
    half_maj = major_len / 2.0
    half_min = minor_len / 2.0

    x1_ma = int(cx + half_maj * cos(angle))
    y1_ma = int(cy + half_maj * sin(angle))
    x2_ma = int(cx - half_maj * cos(angle))
    y2_ma = int(cy - half_maj * sin(angle))
    x1_mi = int(cx + half_min * cos(angle + radians(90)))
    y1_mi = int(cy + half_min * sin(angle + radians(90)))
    x2_mi = int(cx - half_min * cos(angle + radians(90)))
    y2_mi = int(cy - half_min * sin(angle + radians(90)))

    return {
        "angle"            : angle,
        "centroid"         : (cx, cy),
        "major_axis_points": [(x1_ma, y1_ma), (x2_ma, y2_ma)],
        "minor_axis_points": [(x1_mi, y1_mi), (x2_mi, y2_mi)],
    }


# ─────────────────────────────────────────────────────────────────────────────
def calculate_center(mask: np.ndarray):
    """
    Compute the centroid of a binary mask.

    Parameters
    ----------
    mask : np.ndarray  (H, W)  binary

    Returns
    -------
    (row, col) : tuple of floats  — centroid in (row, col) order
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return (mask.shape[0] / 2.0, mask.shape[1] / 2.0)
    return (float(np.mean(rows)), float(np.mean(cols)))


# ─────────────────────────────────────────────────────────────────────────────
def rotate_boolean_array(mask: np.ndarray, theta: float,
                         center) -> np.ndarray:
    """
    Rotate a binary mask by angle theta (radians) around a center point.

    Parameters
    ----------
    mask   : np.ndarray  (H, W)  binary uint8 or bool
    theta  : float  rotation angle in radians (positive = counter-clockwise)
    center : (row, col) or (x, y)  — pivot point (row-major)

    Returns
    -------
    rotated_mask : np.ndarray  (H, W)  binary uint8
    """
    h, w = mask.shape
    # center expected as (row, col) — convert to (cx, cy) for OpenCV
    if hasattr(center, '__len__') and len(center) == 2:
        cy_px, cx_px = float(center[0]), float(center[1])
    else:
        cy_px, cx_px = h / 2.0, w / 2.0

    deg = float(np.degrees(theta))
    M = cv2.getRotationMatrix2D((cx_px, cy_px), deg, 1.0)
    rotated = cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                             flags=cv2.INTER_NEAREST)
    return rotated


# ─────────────────────────────────────────────────────────────────────────────
def cut_long_masks_along_major_axis(mask: np.ndarray,
                                    cut_length: float = 60.0):
    """
    Divide a binary mask into segments of roughly `cut_length` pixels along
    the major axis and return the centroid of each segment.

    Used by custom_grasp_planning_algorithm_dense_cas.py to generate
    multiple grasp pivot points from a single long object mask.

    Parameters
    ----------
    mask       : np.ndarray  (H, W)  binary (may be transposed before call)
    cut_length : float  desired segment length in pixels (default 60)

    Returns
    -------
    list of (cx, cy) tuples — pixel centroids of each segment
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return [(int(mask.shape[1] / 2), int(mask.shape[0] / 2))]

    # PCA to find major axis angle
    points = np.column_stack([cols.astype(np.float64), rows.astype(np.float64)])
    cx_all = float(np.mean(points[:, 0]))
    cy_all = float(np.mean(points[:, 1]))

    centered = points - np.array([cx_all, cy_all])
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    major_vec = evecs[:, np.argmax(evals)]

    # Project each pixel onto the major axis
    proj = centered @ major_vec   # 1-D coordinate along major axis
    proj_min, proj_max = float(proj.min()), float(proj.max())
    total_len = proj_max - proj_min

    if total_len <= cut_length:
        # short object — single pivot at centroid
        return [(int(round(cx_all)), int(round(cy_all)))]

    # compute segment edges and centroids
    n_segs = max(1, int(round(total_len / cut_length)))
    edges = np.linspace(proj_min, proj_max, n_segs + 1)

    pivots = []
    for i in range(n_segs):
        lo, hi = edges[i], edges[i + 1]
        seg_mask = (proj >= lo) & (proj <= hi)
        if seg_mask.sum() == 0:
            continue
        seg_pts = points[seg_mask]
        pcx = float(np.mean(seg_pts[:, 0]))
        pcy = float(np.mean(seg_pts[:, 1]))
        pivots.append((int(round(pcx)), int(round(pcy))))

    return pivots if pivots else [(int(round(cx_all)), int(round(cy_all)))]


def crop_binary_mask(mask: np.ndarray, col_split: float):
    """
    Split a binary mask vertically at a given column position.

    Parameters
    ----------
    mask      : np.ndarray  (H, W)  binary
    col_split : float  column index at which to split

    Returns
    -------
    (left_half, right_half) : two (H, W) binary arrays
    """
    col = int(np.clip(round(col_split), 0, mask.shape[1] - 1))
    left  = mask.copy()
    right = mask.copy()
    left[:, col:]  = 0
    right[:, :col] = 0
    return left, right
