# resize_dataset.py
import os
from PIL import Image

SRC = "artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
DST = "dataset"
SIZE = (224, 224)

for cls in ["Cyst", "Normal", "Stone", "Tumor"]:
    os.makedirs(os.path.join(DST, cls), exist_ok=True)
    src_dir = os.path.join(SRC, cls)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB").resize(SIZE)
            img.save(os.path.join(DST, cls, fname), quality=90)