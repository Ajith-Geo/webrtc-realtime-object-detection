"""YOLOv8 frame-by-frame inference helpers for aiortc server.

Extracted / adapted from standalone detect.py to work on in-memory frames
coming from a WebRTC VideoStreamTrack. Provides normalized bbox results
mirroring the JSON shape used previously (label, score, xmin..ymax).
"""
from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import torch
try:
    import onnxruntime as ort  # only used for CPU path
except ImportError:  # graceful if not installed yet
    ort = None

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo11n.pt")
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.getenv("YOLO_IOU", "0.45"))
CPU_AUTO_TUNE = int(os.getenv("CPU_AUTO_TUNE", "1"))  # enable adaptive imgsz on CPU (torch path only)
CPU_TARGET_MS = float(os.getenv("CPU_TARGET_MS", "55"))  # target per-frame inference time (torch path)
CPU_MIN_IMGSZ = int(os.getenv("CPU_MIN_IMGSZ", "320"))
CPU_MAX_IMGSZ = int(os.getenv("CPU_MAX_IMGSZ", str(YOLO_IMGSZ)))
CPU_STEP = int(os.getenv("CPU_STEP", "64"))
CPU_WARM_FRAMES = int(os.getenv("CPU_WARM_FRAMES", "4"))
CPU_THREADS = os.getenv("CPU_THREADS")  # optional explicit torch threads
USE_ONNX_CPU = int(os.getenv("USE_ONNX_CPU", "1"))  # prefer ONNX Runtime when no CUDA
ONNX_WEIGHTS = os.getenv("YOLO_ONNX", "yolov8n.onnx")  # expected exported model name
ONNX_CONF = float(os.getenv("ONNX_CONF", str(YOLO_CONF)))
ONNX_IOU = float(os.getenv("ONNX_IOU", str(YOLO_IOU)))
VERBOSE_DETECTOR = int(os.getenv("DETECTOR_VERBOSE", "0"))

def load_model() -> YOLO:
    """Load a YOLO model (cached globally)."""
    model = YOLO(YOLO_WEIGHTS)  # ultralytics downloads if missing
    return model

def normalize_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    return x1 / W, y1 / H, x2 / W, y2 / H

class YoloDetector:
    """Wrapper that prefers GPU + half precision if available, falls back to CPU.

    Adds a small warm-up on GPU to avoid first-frame latency spikes and returns
    per-inference timing metadata.
    """

    def __init__(self):
        self.use_onnx = False
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_shape = (YOLO_IMGSZ, YOLO_IMGSZ)
        # Device selection
        cuda_ok = torch.cuda.is_available()
        self.device = 'cuda' if cuda_ok else 'cpu'
        self.use_half = cuda_ok  # only attempt half on CUDA
        self.current_imgsz = YOLO_IMGSZ
        self.frames_seen = 0
        if self.device == 'cuda':
            # Standard Ultralytics path
            self.model = load_model()
            try:
                self.model.to(self.device)
                if self.use_half:
                    try:
                        self.model.model.half()
                    except Exception:
                        self.use_half = False
                # Warm-up
                try:
                    dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
                    _ = self.model.predict(dummy, imgsz=YOLO_IMGSZ, device=self.device, half=self.use_half, verbose=False)
                    torch.cuda.synchronize()
                except Exception as e:
                    print('[detector] Warm-up failed, continuing:', e)
            except Exception as e:
                print('[detector] CUDA init failure; falling back to CPU torch path:', e)
                self.device = 'cpu'
                self.use_half = False
        else:
            # CPU path preference: ONNX Runtime if available and allowed
            if USE_ONNX_CPU and ort is not None:
                if not os.path.exists(ONNX_WEIGHTS):
                    # Attempt export if .pt present
                    pt_candidate = YOLO_WEIGHTS
                    if os.path.exists(pt_candidate):
                        try:
                            if VERBOSE_DETECTOR:
                                print('[detector] ONNX weights missing; exporting from', pt_candidate)
                            YOLO(pt_candidate).export(format='onnx', opset=12, dynamic=False)
                        except Exception as e:
                            if VERBOSE_DETECTOR:
                                print('[detector] Export to ONNX failed, will fallback to torch CPU:', e)
                    else:
                        if VERBOSE_DETECTOR:
                            print('[detector] ONNX weights not found and no .pt to export; fallback torch CPU')
                if os.path.exists(ONNX_WEIGHTS):
                    so = ort.SessionOptions()
                    if CPU_THREADS:
                        try:
                            so.intra_op_num_threads = int(CPU_THREADS)
                            so.inter_op_num_threads = int(CPU_THREADS)
                        except Exception:
                            pass
                    providers = ['CPUExecutionProvider']
                    try:
                        self.onnx_session = ort.InferenceSession(ONNX_WEIGHTS, so, providers=providers)
                        self.onnx_input_name = self.onnx_session.get_inputs()[0].name
                        self.use_onnx = True
                        if VERBOSE_DETECTOR:
                            print(f'[detector] ONNX Runtime session loaded: {ONNX_WEIGHTS}')
                    except Exception as e:
                        print('[detector] Failed to init ONNX session:', e)
            if not self.use_onnx:
                # Fallback to torch CPU Ultralytics
                self.model = load_model()
                if CPU_THREADS:
                    try:
                        torch.set_num_threads(int(CPU_THREADS))
                        if VERBOSE_DETECTOR:
                            print(f"[detector] Set torch threads={CPU_THREADS}")
                    except Exception as e:
                        print('[detector] Failed to set torch threads:', e)

        if VERBOSE_DETECTOR:
            print(f"[detector] Using backend={'onnxruntime' if self.use_onnx else 'ultralytics'} device={self.device} half={self.use_half}")

    def infer(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        H, W = frame_bgr.shape[:2]
        t0 = time.time()
        dets: List[Dict[str, Any]] = []
        if self.use_onnx:
            # Preprocess: letterbox resize to fixed ONNX shape (assumed square)
            target = self.onnx_shape[0]
            scale = min(target / W, target / H)
            new_w, new_h = int(W * scale), int(H * scale)
            resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((target, target, 3), dtype=np.uint8)
            pad_x = (target - new_w) // 2
            pad_y = (target - new_h) // 2
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
            # BGR->RGB, normalize
            img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            inp = img_rgb.astype(np.float32) / 255.0
            inp = np.transpose(inp, (2, 0, 1))[None, ...]  # 1x3xHxW
            ort_out = self.onnx_session.run(None, {self.onnx_input_name: inp})[0]  # shape (1,84,8400)
            data = ort_out[0]  # 84 x 8400
            num_props = data.shape[1]
            stride_cls = num_props
            confidence_threshold = ONNX_CONF
            iou_threshold = ONNX_IOU
            raw_boxes = []
            for i in range(num_props):
                # Find best class
                cls_slice = data[4:, i]
                best_idx = np.argmax(cls_slice)
                best_score = cls_slice[best_idx]
                if best_score < confidence_threshold:
                    continue
                xc, yc, bw, bh = data[0, i], data[1, i], data[2, i], data[3, i]
                x1 = (xc - bw / 2 - pad_x) / scale
                y1 = (yc - bh / 2 - pad_y) / scale
                x2 = (xc + bw / 2 - pad_x) / scale
                y2 = (yc + bh / 2 - pad_y) / scale
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                raw_boxes.append((x1, y1, x2, y2, float(best_score), int(best_idx)))
            # NMS
            def iou(a, b):
                inter_x1 = max(a[0], b[0]); inter_y1 = max(a[1], b[1])
                inter_x2 = min(a[2], b[2]); inter_y2 = min(a[3], b[3])
                iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
                inter = iw * ih
                if inter <= 0:
                    return 0.0
                area_a = (a[2] - a[0]) * (a[3] - a[1])
                area_b = (b[2] - b[0]) * (b[3] - b[1])
                return inter / (area_a + area_b - inter)
            raw_boxes.sort(key=lambda b: b[4], reverse=True)
            final_boxes = []
            for box in raw_boxes:
                if all(iou(box, fb) <= iou_threshold for fb in final_boxes):
                    final_boxes.append(box)
            # Map to detections normalized
            for (x1, y1, x2, y2, score, cls_id) in final_boxes:
                nx1, ny1, nx2, ny2 = normalize_xyxy(x1, y1, x2, y2, W, H)
                dets.append({
                    "label": str(cls_id),  # class labels mapping not embedded w/out Ultralytics names
                    "score": score,
                    "xmin": float(nx1),
                    "ymin": float(ny1),
                    "xmax": float(nx2),
                    "ymax": float(ny2),
                })
            infer_done = time.time()
            infer_ms = (infer_done - t0) * 1000.0
        else:
            # Ultralytics (torch) path
            res = self.model.predict(
                frame_bgr,
                imgsz=self.current_imgsz,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                device=self.device,
                half=self.use_half,
                verbose=False,
            )[0]
            if self.device == 'cuda':
                torch.cuda.synchronize()
            infer_done = time.time()
            infer_ms = (infer_done - t0) * 1000.0
            if self.device == 'cpu' and CPU_AUTO_TUNE and self.frames_seen >= CPU_WARM_FRAMES:
                if (self.frames_seen % 3) == 0:
                    if infer_ms > CPU_TARGET_MS * 1.25 and self.current_imgsz > CPU_MIN_IMGSZ:
                        new_size = max(CPU_MIN_IMGSZ, self.current_imgsz - CPU_STEP)
                        if new_size != self.current_imgsz:
                            if VERBOSE_DETECTOR:
                                print(f"[detector][auto] Downscaling imgsz {self.current_imgsz}->{new_size} (infer {infer_ms:.1f}ms)")
                            self.current_imgsz = new_size
                    elif infer_ms < CPU_TARGET_MS * 0.65 and self.current_imgsz < CPU_MAX_IMGSZ:
                        new_size = min(CPU_MAX_IMGSZ, self.current_imgsz + CPU_STEP)
                        if new_size != self.current_imgsz:
                            if VERBOSE_DETECTOR:
                                print(f"[detector][auto] Upscaling imgsz {self.current_imgsz}->{new_size} (infer {infer_ms:.1f}ms)")
            if res.boxes is not None and len(res.boxes) > 0:
                arr = res.boxes.data.detach().cpu().numpy()
                for (x1, y1, x2, y2, conf, cls) in arr:
                    nx1, ny1, nx2, ny2 = normalize_xyxy(x1, y1, x2, y2, W, H)
                    dets.append({
                        "label": self.model.names[int(cls)],
                        "score": float(conf),
                        "xmin": float(nx1),
                        "ymin": float(ny1),
                        "xmax": float(nx2),
                        "ymax": float(ny2),
                    })
        self.frames_seen += 1
        payload = {
            "capture_ts": t0,
            "inference_ts": infer_done,
            "latency_ms": infer_ms,
            "inference_ms": infer_ms,
            "device": self.device,
            "half": self.use_half,
            "detections": dets,
            "img": {"w": W, "h": H},
            "effective_imgsz": self.current_imgsz if not self.use_onnx else self.onnx_shape[0],
            "backend": 'onnx' if self.use_onnx else 'torch',
        }
        return payload
