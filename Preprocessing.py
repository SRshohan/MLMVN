import os
from pathlib import Path
import cv2
import imutils
import numpy as np
from tqdm import tqdm

IMG_SIZE = 256
ROOT_DIR = Path("Tif_images")
CLEANED_DIR = Path("cleaned")

def crop_img(img):
    """Crop to the bounding box of the largest contour."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return img  # no contour found, return original

    c = max(cnts, key=cv2.contourArea)
    x_left, y_top = c[:, :, 0].min(), c[:, :, 1].min()
    x_right, y_bot = c[:, :, 0].max(), c[:, :, 1].max()

    # clamp to image bounds
    y_top, y_bot = max(y_top, 0), min(y_bot, img.shape[0])
    x_left, x_right = max(x_left, 0), min(x_right, img.shape[1])

    return img[y_top:y_bot, x_left:x_right]

def process_split(split: str):
    src_root = ROOT_DIR / split
    dst_root = CLEANED_DIR / split
    for class_dir in sorted(src_root.iterdir()):
        if not class_dir.is_dir(): continue

        dst_class = dst_root / class_dir.name
        dst_class.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(class_dir.glob("*.tif")),
                             desc=f"{split}/{class_dir.name}", unit="img"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            cropped = crop_img(img)
            resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

            out_path = dst_class / img_path.name
            cv2.imwrite(str(out_path), resized)

if __name__ == "__main__":
    for split in ["Training", "Testing"]:
        process_split(split)
