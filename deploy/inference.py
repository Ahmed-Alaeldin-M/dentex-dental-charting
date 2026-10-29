"""Pure ONNX Runtime inference for the DENTEX dental charting models (no torch, no ultralytics).

Pipeline per X-ray:
  CLAHE (same as training) -> letterbox to 1024 -> teeth model + diagnosis model
  -> decode YOLOv8 output -> per-class NMS -> map each finding to its FDI tooth.
"""
import ast

import cv2
import numpy as np
import onnxruntime as ort

FDI = [f"{q}{t}" for q in range(1, 5) for t in range(1, 9)]  # 11..48
DIAG_FALLBACK = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class OnnxDetector:
    def __init__(self, path, fallback_names, threads=2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape  # [1, 3, H, W]
        self.imgsz = int(shape[2]) if isinstance(shape[2], int) else 1024
        meta = self.sess.get_modelmeta().custom_metadata_map
        try:
            names = ast.literal_eval(meta["names"])
            self.names = [str(names[i]) for i in sorted(names)]
        except Exception:
            self.names = list(fallback_names)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        out[top:top + nh, left:left + nw] = resized
        return out, r, left, top

    def __call__(self, bgr, conf=0.25, iou=0.7, max_det=300, agnostic=False):
        h0, w0 = bgr.shape[:2]
        lb, r, left, top = self._letterbox(bgr)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, CHW
        pred = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0].T  # (N, 4+nc)
        scores_all = pred[:, 4:]
        cls = scores_all.argmax(1)
        scores = scores_all[np.arange(len(cls)), cls]
        keep = scores > conf
        boxes, scores, cls = pred[keep, :4], scores[keep], cls[keep]
        if len(scores) == 0:
            return []
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        # per-class NMS via class offset (same trick ultralytics uses); agnostic = one label per box
        offset = 0.0 if agnostic else cls[:, None].astype(np.float32) * 7680.0
        nms_boxes = xyxy + offset
        wh = nms_boxes[:, 2:] - nms_boxes[:, :2]
        idx = cv2.dnn.NMSBoxes(np.concatenate([nms_boxes[:, :2], wh], 1).tolist(), scores.tolist(), conf, iou)
        idx = np.array(idx).reshape(-1)[:max_det]
        xyxy, scores, cls = xyxy[idx], scores[idx], cls[idx]
        # undo letterbox
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - left) / r).clip(0, w0)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - top) / r).clip(0, h0)
        order = np.argsort(-scores)
        return [dict(label=self.names[int(cls[i])], confidence=round(float(scores[i]), 3),
                     box=[round(float(v), 1) for v in xyxy[i]]) for i in order]


def preprocess(image_bytes):
    """Decode any image, apply the training CLAHE, return 3-channel BGR."""
    arr = np.frombuffer(image_bytes, np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode image")
    return cv2.cvtColor(CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)


def assign_tooth(box, teeth):
    """Tooth whose box covers the largest share of the finding box."""
    x1, y1, x2, y2 = box
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    best, score = None, 0.0
    for t in teeth:
        a1, b1, a2, b2 = t["box"]
        inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
        if inter / area > score:
            best, score = t["label"], inter / area
    return best, round(score, 3)


def unique_teeth(teeth):
    """A chart has each FDI number at most once: keep the most confident box per number."""
    seen, out = set(), []
    for t in sorted(teeth, key=lambda t: -t["confidence"]):
        if t["label"] not in seen:
            seen.add(t["label"])
            out.append(t)
    return out


def chart(teeth_model, diag_model, image_bytes, conf=0.25, iou=0.7):
    img = preprocess(image_bytes)
    # teeth: class-agnostic NMS (one number per tooth box), then one box per number
    teeth = unique_teeth(teeth_model(img, conf=conf, iou=iou, agnostic=True))
    teeth.sort(key=lambda t: t["label"])
    # findings: class-agnostic too, so one lesion isn't reported as both caries and deep caries
    findings = diag_model(img, conf=conf, iou=iou, agnostic=True)
    for f in findings:
        f["tooth"], f["overlap"] = assign_tooth(f["box"], teeth)
        f["diagnosis"] = f.pop("label")
    findings.sort(key=lambda f: f["tooth"] or "99")
    return img, teeth, findings


COLORS = {"Impacted": (0, 0, 255), "Caries": (0, 165, 255), "Deep Caries": (0, 255, 255),
          "Periapical Lesion": (255, 0, 255)}


def draw(img, teeth, findings):
    out = img.copy()
    for t in teeth:
        x1, y1, x2, y2 = map(int, t["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (90, 200, 90), 2)
        cv2.putText(out, t["label"], (x1 + 3, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 200, 90), 2)
    for f in findings:
        x1, y1, x2, y2 = map(int, f["box"])
        col = COLORS.get(f["diagnosis"], (255, 128, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 4)
        cv2.putText(out, f"{f['tooth']} {f['diagnosis']} {f['confidence']:.2f}", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
    return out
