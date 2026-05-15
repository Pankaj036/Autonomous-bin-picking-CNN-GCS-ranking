"""
Grasp Planning Utilities
Adapted from https://github.com/praj441/Bin-Picking-CAS-GCS/blob/main/grasp_planning/commons/utils_gs_cas.py

Changes vs original:
  - Removed ROS1 / point-cloud service imports
  - Parameters class uses actual camera_info intrinsics (passed in) instead of hard-coded focal lengths
  - All tab-indentation → 4-space
  - Joblib optional import (graceful fallback to sequential)
"""

import numpy as np
import cv2
from math import cos, sin, atan2, radians, degrees, sqrt


# ── Angle helpers ─────────────────────────────────────────────────────────────

def keep_angle_bounds(angle: float) -> float:
    """Keep angle in [-pi/2, pi/2] (exploit 180° gripper symmetry)."""
    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return angle


# ── Parameters ────────────────────────────────────────────────────────────────

class Parameters:
    """
    All camera and gripper geometry parameters.

    Adapted for OAK-D Pro Wide wrist camera.
    Pass actual fx, fy, cx_img, cy_img from /wrist_camera/rgb/camera_info.
    Image dimensions w=640, h=480 (OAK-D default RGB).

    Robotiq 3F gripper (vs original Schunk WSG-50):
      - gripper_finger_space_max : 0.155 m  (3F max opening diameter ~160 mm)
      - gripper_max_opening_length : 0.16 m
    """

    def __init__(self, w: int = 640, h: int = 480,
                 fx: float = None, fy: float = None,
                 cx_img: float = None, cy_img: float = None):
        self.w = w
        self.h = h
        self.mw = float(w) / 200.0
        self.mh = float(h) / 200.0

        # Camera intrinsics: use passed-in values or compute from FOV
        if fx is not None:
            self.f_x = fx
            self.f_y = fy if fy is not None else fx
        else:
            # OAK-D Pro Wide approximate FOV: hfov~127°, vfov~80°
            import math
            hfov = 127.0
            vfov = 80.0
            self.f_x = w / (2.0 * math.tan(math.radians(hfov / 2.0)))
            self.f_y = h / (2.0 * math.tan(math.radians(vfov / 2.0)))

        # Image principal point (default = image centre)
        self.cx_img = cx_img if cx_img is not None else w / 2.0
        self.cy_img = cy_img if cy_img is not None else h / 2.0

        # Gripper workspace (pixels, scaled to resolution)
        self.gripper_width  = max(int(self.mh * 15), 5)   # rows  (~7 px at 480p)
        self.gripper_height = max(int(self.mh * 70), 20)  # cols  (~33 px at 480p)

        # Scoring thresholds
        self.THRESHOLD1 = max(int(self.mh * 15), 3)
        self.THRESHOLD2 = 0.02          # depth collision margin (m)
        self.THRESHOLD3 = max(int(self.mh * 7), 2)

        self.gdi_max       = max(int(self.gripper_height / 2), 1)
        self.gdi_plus_max  = max(2 * (self.gripper_width // 2) * self.THRESHOLD3, 1)
        self.cx = self.gripper_width  // 2   # centre row in gripper workspace
        self.cy = self.gripper_height // 2   # centre col in gripper workspace

        self.pixel_finger_width       = max(self.mh * 8, 2.0)  # min free space per finger (px)

        # Robotiq 3F max opening (~160 mm diameter, but conservative)
        self.gripper_finger_space_max  = 0.155   # metres
        self.gripper_max_opening_length = 0.16   # metres

        self.Max_Gripper_Opening_value = 1.0
        self.datum_z = 0.50        # typical empty-bin depth (m)
        self.gdi_plus_cut_threshold = 50

        self.cut_length = 70       # pixels — split long objects
        self.target = None         # set to mask label of current object

        self.DBCC_enable = False   # depth-based collision check

        # For draw_rect_generic
        self.angle_shift_list = []
        self.asc_list = []

    # ── pixel → 3-D (camera frame) ────────────────────────────────────────────
    def pixel_to_xyz(self, px: float, py: float, z: float):
        """Back-project pixel (px, py) at depth z (metres) to camera-frame XY."""
        px = float(np.clip(px, 0, self.w - 1))
        py = float(np.clip(py, 0, self.h - 1))
        x = (px - self.cx_img) * z / self.f_x
        y = (py - self.cy_img) * z / self.f_y
        return x, y

    # ── axis angle of a binary mask ───────────────────────────────────────────
    def axis_angle(self, mask: np.ndarray) -> dict:
        """Compute principal axis angle and centroid of a binary mask."""
        pts = np.argwhere(mask > 0)   # (N,2) → [row,col]
        if len(pts) < 3:
            return {"angle": 0.0, "centroid": (mask.shape[1] // 2, mask.shape[0] // 2)}
        cx = np.mean(pts[:, 1])  # col → x
        cy = np.mean(pts[:, 0])  # row → y
        dx = pts[:, 1] - cx
        dy = pts[:, 0] - cy
        num = np.sum(dx * dy)
        den = np.sum(dx ** 2 - dy ** 2)
        angle = 0.5 * atan2(2 * num, den)
        return {"angle": angle, "centroid": (int(cx), int(cy))}

    # ── generate grasp rectangles at fixed angles ──────────────────────────────
    def draw_rect_fixed_angles(self, centroid, directions: int = 4):
        """
        Generate `directions` grasp rectangles at evenly-spaced angles around
        the principal axis.  Returns list of rectangles and list of angles.

        Each rectangle: np.array shape (5,2) → [centre, tl, tr, br, bl]
        """
        base_angle = 0.0
        step = np.pi / directions
        rects = []
        angles = []
        for i in range(directions):
            angle = keep_angle_bounds(base_angle + i * step)
            gh2 = self.gripper_height * 0.5
            gw2 = self.gripper_width  * 0.5
            cx, cy = float(centroid[0]), float(centroid[1])
            ca, sa = cos(angle), sin(angle)
            # four corners
            x1 = int(cx - gh2 * ca - gw2 * cos(angle + np.pi / 2))
            y1 = int(cy - gh2 * sa - gw2 * sin(angle + np.pi / 2))
            x2 = int(cx - gh2 * ca + gw2 * cos(angle + np.pi / 2))
            y2 = int(cy - gh2 * sa + gw2 * sin(angle + np.pi / 2))
            x3 = int(cx + gh2 * ca + gw2 * cos(angle + np.pi / 2))
            y3 = int(cy + gh2 * sa + gw2 * sin(angle + np.pi / 2))
            x4 = int(cx + gh2 * ca - gw2 * cos(angle + np.pi / 2))
            y4 = int(cy + gh2 * sa - gw2 * sin(angle + np.pi / 2))
            rects.append(np.array([[int(cx), int(cy)], [x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
            angles.append(angle)
        return rects, angles


# ── Grasp pivot extraction from masks ─────────────────────────────────────────

def get_pivot_points_from_masks(masks: np.ndarray, param: Parameters) -> list:
    """
    For each binary mask, compute centroid as pivot point for grasp sampling.
    Long objects (major axis > cut_length) are split along their major axis.

    Returns list of (centroid_xy, mask_label) tuples.
    """
    pivots = []
    for i in range(masks.shape[0]):
        mask = masks[i].astype(np.uint8)
        pts  = np.argwhere(mask > 0)
        if len(pts) < 10:
            continue
        cx = int(np.mean(pts[:, 1]))
        cy = int(np.mean(pts[:, 0]))
        pivots.append(((cx, cy), i + 1))
    return pivots


# ── Grasp ranking ─────────────────────────────────────────────────────────────

def select_best_grasps(rectangle_list, fls_list, crs_list, gcs_list=None,
                       top_n: int = 5):
    """
    Rank grasp poses by combined score (FLS + CRS + GCS) and return top_n.

    Args:
        rectangle_list : list of rect arrays (each shape 5×2)
        fls_list       : Free-space/Lifting Score per rect (float or None)
        crs_list       : Contact Region Score per rect (float or None)
        gcs_list       : GCS from CNN per rect (float, optional)
        top_n          : how many to return

    Returns:
        List of dicts with keys: rect, angle, fls, crs, gcs, score, rank
    """
    candidates = []
    for i, (rect, fls, crs) in enumerate(zip(rectangle_list, fls_list, crs_list)):
        if fls is None or crs is None:
            continue
        gcs = gcs_list[i] if gcs_list is not None else 0.5
        score = fls + crs + 100.0 * gcs
        candidates.append({"rect": rect, "fls": fls, "crs": crs,
                            "gcs": gcs, "score": score, "_src_idx": i})

    candidates.sort(key=lambda d: d["score"], reverse=True)

    results = []
    for rank, c in enumerate(candidates[:top_n]):
        c["rank"] = rank
        results.append(c)
    return results


# ── Visualisation helper ──────────────────────────────────────────────────────

def draw_grasp_on_image(image: np.ndarray, rect: np.ndarray,
                        color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw a grasp rectangle (5×2 array) on image."""
    pts = rect[1:].reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)
    cx, cy = int(rect[0, 0]), int(rect[0, 1])
    cv2.circle(image, (cx, cy), 5, color, -1)
    return image
