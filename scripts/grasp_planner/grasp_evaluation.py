"""
GDI2 Grasp Evaluation
Adapted from https://github.com/praj441/Bin-Picking-CAS-GCS/blob/main/grasp_planning/commons/grasp_evaluation_cas.py

GDI2 evaluates a candidate grasp rectangle by:
  1. Projecting a gripper workspace (gw × gh pixels) onto the scene
  2. Segmenting the workspace into contact / collision / free regions
  3. Scoring: FLS (free-space lifting score) + CRS (contact region score)

Changes vs original:
  - 4-space indentation
  - Removed all ROS1 / rospy references
  - Device-agnostic (CPU NumPy only)
"""

import numpy as np
import cv2
from math import cos, sin


class GDI2:
    """Grasp Descriptor Index v2 — evaluates one grasp rectangle."""

    def __init__(self, rotation_point, angle, darray, param):
        self.rotation_point = rotation_point
        x, y = rotation_point[0], rotation_point[1]
        t = angle
        # Rotation matrix for workspace → image mapping
        self.rotation_matrix = np.array([
            [cos(t), -sin(t), -x * cos(t) + y * sin(t) + x],
            [sin(t),  cos(t), -x * sin(t) - y * cos(t) + y],
            [0, 0, 1]
        ])
        self.tx = x - param.gripper_height // 2
        self.ty = y - param.gripper_width  // 2

        self.darray = darray
        self.param   = param

        self.dmap = None
        self.smap = None

        self.new_centroid       = np.array([param.cx, param.cy])
        self.gripper_opening    = param.gripper_height
        self.gripper_opening_meter = 0.1
        self.object_width       = 0.05

        self.FLS_score  = None
        self.CRS_score  = None
        self.final_center = None
        self.invalid_reason = 'NA'
        self.target = None

    # ── coordinate mapping ────────────────────────────────────────────────────

    def map_the_point_vectorized(self, I):
        """Map gripper-workspace grid indices → image pixel coordinates."""
        I[0] += self.tx
        I[1] += self.ty
        I_h = np.vstack([I, np.ones((1, I.shape[1]))])
        O = (self.rotation_matrix @ I_h).astype(np.int32)[:2, :]
        return O

    # ── width in metres ───────────────────────────────────────────────────────

    def calculate_width_in_meter(self, col_left: int, col_right: int) -> float:
        cx_ws = self.param.cx
        x1, y1 = self.map_point(col_left,  cx_ws)
        x2, y2 = self.map_point(col_right, cx_ws)
        z = self.param.datum_z
        X1, Y1 = self.param.pixel_to_xyz(x1, y1, z)
        X2, Y2 = self.param.pixel_to_xyz(x2, y2, z)
        return float(np.sqrt((X1 - X2) ** 2 + (Y1 - Y2) ** 2))

    def map_point(self, col, row):
        """Map single (col, row) workspace point → image (x, y)."""
        xp = col + self.tx
        yp = row + self.ty
        pt = np.array([[xp], [yp], [1.0]])
        op = (self.rotation_matrix @ pt).astype(int)
        return int(op[0]), int(op[1])

    # ── centroid of a binary map ──────────────────────────────────────────────

    @staticmethod
    def _calc_centre(arr: np.ndarray):
        total = arr.sum()
        if total == 0:
            return arr.shape[0] // 2, arr.shape[1] // 2
        r = int(round((arr.sum(axis=1) @ np.arange(arr.shape[0])) / total))
        c = int(round((arr.sum(axis=0) @ np.arange(arr.shape[1])) / total))
        return r, c

    # ── main evaluation ───────────────────────────────────────────────────────

    def pose_refinement(self, param, DBCC_enable: bool = False):
        """
        Evaluate grasp pose.  Returns FLS score (float) or None if invalid.
        Also sets self.CRS_score on success.
        """
        dmap = self.dmap
        smap = self.smap
        gw   = param.gripper_width
        gh   = param.gripper_height
        cx   = param.cx
        cy   = param.cy

        # which label is at the gripper centre?
        target = smap[cx, cy]
        if target <= 0 or (param.target is not None and param.target != target):
            self.invalid_reason = 'occluded or no object at centre'
            return None
        self.target = target

        # ── region masks ─────────────────────────────────────────────────────
        contact_mask   = (smap == target)
        compr_depth    = dmap[cx, cy]

        other_obj = (smap != target) & (smap > 0)
        if DBCC_enable and compr_depth > 0:
            diff_map       = dmap - compr_depth
            collision_mask = other_obj & (diff_map < param.THRESHOLD2)
        else:
            collision_mask = other_obj

        free_mask = ~contact_mask & ~collision_mask

        # split free region into left / right halves
        left_half = np.zeros_like(free_mask)
        left_half[:, :cy] = free_mask[:, :cy]
        right_half = np.zeros_like(free_mask)
        right_half[:, cy:] = free_mask[:, cy:]

        # ── validity check ───────────────────────────────────────────────────
        max_obj_width = contact_mask.sum(axis=1).max()
        self.object_width = self.calculate_width_in_meter(0, int(max_obj_width))

        min_fsl = left_half.sum(axis=1).min()
        min_fsr = right_half.sum(axis=1).min()

        if (min_fsl > param.pixel_finger_width and
                min_fsr > param.pixel_finger_width and
                self.object_width < param.gripper_finger_space_max):
            valid = True
        else:
            self.invalid_reason = 'large object or insufficient free space'
            return None

        # ── scores ───────────────────────────────────────────────────────────
        if valid:
            _, cy_new = self._calc_centre(contact_mask)
            self.new_centroid = np.array([cx, int(cy_new)])

            _, c_fsl = self._calc_centre(left_half)
            _, c_fsr = self._calc_centre(right_half)
            self.gripper_opening = max(int(c_fsr - c_fsl), 1)

            free_score = min(left_half.sum(), right_half.sum())
            self.FLS_score = 100.0 * float(free_score) / max(0.5 * gw * gh, 1)

            crs_raw = contact_mask[:, max(cy - gw // 2, 0): cy + gw // 2].sum()
            self.CRS_score = 100.0 * float(crs_raw) / max(gw * gw, 1)

            return self.FLS_score
        return None

    # ── visualisation ─────────────────────────────────────────────────────────

    def draw_refined_pose(self, image: np.ndarray, scale: int = 1,
                          thickness: int = 2):
        cy_new = self.new_centroid[1]
        xmin, xmax = 0, self.param.gripper_width - 1
        ymin = cy_new - self.gripper_opening // 2
        ymax = cy_new + self.gripper_opening // 2

        def mp(c, r):
            x, y = self.map_point(c, r)
            return (x * scale, y * scale)

        p0 = mp(cy_new, self.new_centroid[0])
        p1 = mp(ymax, xmax)
        p2 = mp(ymin, xmax)
        p3 = mp(ymin, xmin)
        p4 = mp(ymax, xmin)

        color_long  = (255, 200, 0)
        color_short = (0, 200, 255)
        cv2.line(image, p1, p2, color_long,  thickness)
        cv2.line(image, p2, p3, color_short, thickness)
        cv2.line(image, p3, p4, color_long,  thickness)
        cv2.line(image, p4, p1, color_short, thickness)
        cv2.circle(image, p0, thickness + 1, color_long, -1)

        self.final_center = np.array(p0) / scale
        return self.final_center, self.gripper_opening, self.object_width


# ── Top-level evaluation function ─────────────────────────────────────────────

def calculate_GDI2(inputs: dict, rectangle: np.ndarray, angle: float):
    """
    Build a GDI2 evaluator for one grasp rectangle and run it.

    inputs dict keys:
        darray      : depth image  (H×W float32, metres)
        seg_mask    : segmentation (H×W int32, 0=background, 1..N=objects)
        param       : Parameters instance
        run_with_dbcc : bool
    rectangle: shape (5,2) — [centre, tl, tr, br, bl]
    angle    : grasp angle (radians)

    Returns: (fls_score, crs_score, gdi2_obj)
             fls_score / crs_score are None if pose is invalid.
    """
    darray   = inputs['darray']
    param    = inputs['param']
    seg_mask = inputs['seg_mask']
    dbcc     = inputs.get('run_with_dbcc', False)

    gdi2 = GDI2(rectangle[0], angle, darray, param)

    gw = param.gripper_width
    gh = param.gripper_height
    w  = param.w
    h  = param.h

    # Build grid over gripper workspace and map to image
    Imap = np.mgrid[0:gh:1, 0:gw:1].reshape(2, -1)   # 2×(gh*gw)
    Omap = gdi2.map_the_point_vectorized(Imap)         # 2×(gh*gw)

    # Clamp out-of-bounds pixels
    oob = ((Omap[0] < 0) | (Omap[0] > w - 1) |
           (Omap[1] < 0) | (Omap[1] > h - 1))
    Omap[0] = np.where(oob, 0, Omap[0])
    Omap[1] = np.where(oob, 0, Omap[1])

    dmap = np.where(oob, 0.0, darray[Omap[1], Omap[0]]).reshape(gh, gw).T
    smap = np.where(oob, 0,   seg_mask[Omap[1], Omap[0]]).reshape(gh, gw).T

    gdi2.dmap = dmap
    gdi2.smap = smap

    fls = gdi2.pose_refinement(param, DBCC_enable=dbcc)
    crs = gdi2.CRS_score

    return fls, crs, gdi2
