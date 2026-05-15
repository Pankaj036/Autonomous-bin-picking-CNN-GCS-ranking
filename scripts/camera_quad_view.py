#!/usr/bin/env python3
"""
Camera View — single window, 1×2 grid (wrist camera only)
══════════════════════════════════════════════════════════
┌──────────────────┬──────────────────┐
│  WRIST — RGB     │  WRIST — Depth   │
└──────────────────┴──────────────────┘

Usage:
  ros2 run ur_yt_sim camera_quad_view
"""

import threading
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image


# ── cv_bridge-free image decoder ───────────────────────────────────────────────
# cv_bridge is compiled against NumPy 1.x and segfaults with NumPy 2.x when
# a callback actually fires.  Decode ROS Image messages directly instead.
def _imgmsg_to_cv2(msg: Image, desired_encoding: str) -> np.ndarray:
    """Minimal ROS Image → numpy decoder, no cv_bridge dependency."""
    dtype_map = {
        "rgb8":   (np.uint8,   3),
        "bgr8":   (np.uint8,   3),
        "rgba8":  (np.uint8,   4),
        "bgra8":  (np.uint8,   4),
        "mono8":  (np.uint8,   1),
        "16uc1":  (np.uint16,  1),
        "32fc1":  (np.float32, 1),
    }
    enc = msg.encoding.lower().replace(" ", "")  # normalise to lowercase
    if enc not in dtype_map:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    dtype, channels = dtype_map[enc]
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    if channels == 1:
        arr = arr.reshape((msg.height, msg.width))
    else:
        arr = arr.reshape((msg.height, msg.width, channels))

    if desired_encoding == "bgr8":
        if enc == "rgb8":
            arr = arr[:, :, ::-1].copy()
        elif enc in ("rgba8",):
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif enc in ("bgra8",):
            arr = arr[:, :, :3].copy()
        elif enc == "mono8":
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif desired_encoding == "passthrough":
        pass   # return as-is

    return arr

# ── Layout config ─────────────────────────────────────────────────────────────
CELL_W = 640
CELL_H = 480
WIN_TITLE = "Camera Views  |  WRIST RGB · WRIST Depth"

LABELS = {
    "wrist_rgb":  "WRIST — RGB",
    "wrist_depth":"WRIST — Depth",
}

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICK = 2
LABEL_CLR  = (255, 255, 255)
LABEL_BG   = (0, 0, 0)
BORDER_CLR = (80, 80, 80)
BORDER_T   = 2


def _colorise_depth(depth_np: np.ndarray) -> np.ndarray:
    """Convert uint16 depth (mm) or float depth to a BGR colour image."""
    if depth_np.dtype == np.uint16:
        d = depth_np.astype(np.float32)
        valid = d > 0
        if valid.any():
            d[~valid] = 0
            d_norm = np.zeros_like(d, dtype=np.uint8)
            dmax = d[valid].max()
            if dmax > 0:
                d_norm[valid] = (255.0 * d[valid] / dmax).astype(np.uint8)
        else:
            d_norm = np.zeros(depth_np.shape, dtype=np.uint8)
        colour = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
    else:
        # float or other — normalise to 0-255
        d = depth_np.astype(np.float32)
        mn, mx = d.min(), d.max()
        if mx > mn:
            d = ((d - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            d = np.zeros(d.shape, dtype=np.uint8)
        if d.ndim == 2:
            colour = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
        else:
            colour = cv2.applyColorMap(d[:, :, 0], cv2.COLORMAP_TURBO)
    return colour


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _add_label(cell: np.ndarray, text: str) -> np.ndarray:
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    x, y = 10, th + 10
    cv2.rectangle(cell, (x - 4, y - th - 4), (x + tw + 4, y + 4), LABEL_BG, -1)
    cv2.putText(cell, text, (x, y), FONT, FONT_SCALE, LABEL_CLR, FONT_THICK, cv2.LINE_AA)
    return cell


def _placeholder(label: str) -> np.ndarray:
    cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
    (tw, _), _ = cv2.getTextSize("Waiting…", FONT, FONT_SCALE, FONT_THICK)
    cx = (CELL_W - tw) // 2
    cv2.putText(cell, "Waiting…", (cx, CELL_H // 2),
                FONT, FONT_SCALE, (100, 100, 100), FONT_THICK, cv2.LINE_AA)
    return _add_label(cell, label)


class CameraQuadView(Node):

    def __init__(self):
        super().__init__("camera_quad_view")

        self._lock   = threading.Lock()
        self._frames = {k: None for k in LABELS}

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, "/wrist_camera/rgb/image_raw",
                                 lambda m: self._cb(m, "wrist_rgb",  False), qos)
        self.create_subscription(Image, "/wrist_camera/stereo/image_raw",
                                 lambda m: self._cb(m, "wrist_depth",True),  qos)

        # display timer at ~20 Hz
        self.create_timer(0.05, self._display)
        cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_TITLE, CELL_W * 2, CELL_H)
        self.get_logger().info("Camera quad view ready.")

    # ── image callbacks ───────────────────────────────────────────────────────
    def _cb(self, msg: Image, key: str, is_depth: bool):
        try:
            if is_depth:
                raw = _imgmsg_to_cv2(msg, desired_encoding="passthrough")
                bgr = _colorise_depth(raw)
            else:
                bgr = _imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"{key} decode: {e}", throttle_duration_sec=5.0)
            return
        cell = _resize(bgr, CELL_W, CELL_H)
        _add_label(cell, LABELS[key])
        with self._lock:
            self._frames[key] = cell

    # ── display timer ─────────────────────────────────────────────────────────
    def _display(self):
        with self._lock:
            left  = self._frames["wrist_rgb"]   if self._frames["wrist_rgb"]   is not None else _placeholder(LABELS["wrist_rgb"])
            right = self._frames["wrist_depth"] if self._frames["wrist_depth"] is not None else _placeholder(LABELS["wrist_depth"])

        # draw borders
        for cell in (left, right):
            cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), BORDER_CLR, BORDER_T)

        grid = np.hstack([left, right])

        cv2.imshow(WIN_TITLE, grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:   # q or Esc → close
            cv2.destroyAllWindows()
            rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = CameraQuadView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
