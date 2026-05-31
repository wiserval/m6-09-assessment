import sys
import os
import json
import csv
from pathlib import Path

# Ensure the directory containing this script is on the path.
# Required so Docker can resolve `from detector import CatDetector`
# when the ENTRYPOINT is `python /app/app/cli.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detector import CatDetector


def cmd_info():
    student_file = Path("/app/STUDENT.json")
    if student_file.exists():
        print(student_file.read_text())
    else:
        print(json.dumps({"error": "STUDENT.json not found"}))
        sys.exit(1)


def cmd_predict():
    input_dir  = Path("/data/input")
    output_csv = Path("/data/output/predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    detector   = CatDetector("/app/models/best.onnx", imgsz=640, conf=0.05)
    valid_exts = {".jpg", ".jpeg", ".png", ".webp",
                  ".JPG", ".JPEG", ".PNG", ".WEBP"}

    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "xmin", "ymin", "xmax", "ymax",
            "confidence", "class"
        ])
        for img_path in sorted(input_dir.rglob("*")):
            if not (img_path.is_file() and
                    img_path.suffix in valid_exts):
                continue
            rel_path = img_path.relative_to(input_dir).as_posix()
            try:
                preds = detector.predict(str(img_path))
                if not preds:
                    writer.writerow([rel_path, "", "", "", "", "", ""])
                else:
                    for p in preds:
                        writer.writerow([
                            rel_path,
                            round(p["xmin"], 4),
                            round(p["ymin"], 4),
                            round(p["xmax"], 4),
                            round(p["ymax"], 4),
                            round(p["confidence"], 4),
                            p["class"],
                        ])
            except Exception:
                writer.writerow([rel_path, "", "", "", "", "", ""])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    cmd = sys.argv[1].lower()
    if cmd == "info":
        cmd_info()
    elif cmd == "predict":
        cmd_predict()
    else:
        sys.exit(1)
