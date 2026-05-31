import numpy as np
import onnxruntime as ort
from PIL import Image


class CatDetector:
    def __init__(self, onnx_path, imgsz=640, conf=0.05, class_names=("cat",)):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = class_names
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img, target_size):
        orig_w, orig_h = img.size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        pad_w = (target_size - new_w) / 2.0
        pad_h = (target_size - new_h) / 2.0
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        padded = Image.new("RGB", (target_size, target_size), (114, 114, 114))
        padded.paste(resized, (int(round(pad_w)), int(round(pad_h))))
        return padded, scale, (pad_w, pad_h)

    def predict(self, image_path: str) -> list:
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        x, scale, (pad_x, pad_y) = self._letterbox(img, self.imgsz)
        x = (np.array(x, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

        # YOLO26 e2e output: (1, 300, 6) -> [x1, y1, x2, y2, score, class]
        out = self.session.run(None, {self.input_name: x})[0][0]

        results = []
        for x1, y1, x2, y2, score, cls in out:
            if score < self.conf:
                continue
            x1 = max(0.0, min(orig_w, (x1 - pad_x) / scale))
            y1 = max(0.0, min(orig_h, (y1 - pad_y) / scale))
            x2 = max(0.0, min(orig_w, (x2 - pad_x) / scale))
            y2 = max(0.0, min(orig_h, (y2 - pad_y) / scale))
            results.append({
                "xmin": float(x1), "ymin": float(y1),
                "xmax": float(x2), "ymax": float(y2),
                "confidence": float(score),
                "class": self.class_names[int(cls)],
            })
        return results
