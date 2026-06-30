import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def extract_image_features(image_dir):
    """Extracts simple statistical features (brightness, contrast, size) from images."""
    records = []
    for path in glob.glob(os.path.join(image_dir, "*.*")):
        try:
            img = Image.open(path).convert("L")  # grayscale
            arr = np.array(img)
            records.append({
                "mean_brightness": float(np.mean(arr)),
                "std_brightness": float(np.std(arr)),
                "width": img.width,
                "height": img.height,
                "min_pixel": float(np.min(arr)),
                "max_pixel": float(np.max(arr)),
            })
        except Exception:
            continue
    return pd.DataFrame(records)


def run_drift_check(reference_dir="artifacts/drift_data/reference",
                     current_dir="artifacts/drift_data/current",
                     output_path="reports/drift_report.html"):
    reference_df = extract_image_features(reference_dir)
    current_df = extract_image_features(current_dir)

    if reference_df.empty or current_df.empty:
        print("Insufficient images found in reference or current directory.")
        return None

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset drift detected: {drift_detected}")
    return drift_detected


if __name__ == "__main__":
    run_drift_check()