# Cat Detection v2 — Improve, Export, Containerise

## Image for leaderboard

```text
docker pull aliguluali/cat-detector:final
```

**Image:** `aliguluali/cat-detector:final`  
**Student:** Ali Aligulu

## Quick start

```bash
docker run --rm aliguluali/cat-detector:final info

docker run --rm \
  -v /absolute/path/to/images:/data/input:ro \
  -v /absolute/path/to/results:/data/output \
  alialigulu/cat-detector:final predict
```

## Output schema

`predictions.csv` columns: `image_path, xmin, ymin, xmax, ymax, confidence, class`

One row per detection. Images with no detections produce a single row with empty coordinate fields.

## Model

- Architecture: YOLOv26m (medium)
- Training: 150 epochs, AdamW, cosine LR, mosaic + mixup augmentation
- Dataset: 3,327 photorealistic cat images + 600 COCO hard negatives
- Val mAP@0.5: 0.887