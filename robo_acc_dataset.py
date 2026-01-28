
import json, random, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd 

import torch
from torch.utils.data import Dataset
from PIL import Image

# implementing the accident dataset. 
# accident images are cropped using annotations from the roboflow dataset 
# non accident images are obtained from the bdd100k image dataset. 



def square_crop_around_bbox(
    img: Image.Image,
    bbox_xywh,
    out_size=(256, 256),
    margin=0.0,          # e.g. 0.15 adds 15% context
) -> Image.Image:
    """
    COCO bbox format: [x, y, w, h] in pixels.
    Returns a square crop centered on bbox center, resized to out_size.
    """
    x, y, w, h = map(float, bbox_xywh)
    W, H = img.size

    # bbox center
    cx = x + w / 2.0
    cy = y + h / 2.0

    # square side length (+ optional margin)
    side = max(256.0, max(w, h)) * (1.0 + margin)
    side = max(1.0, side)  # avoid zero

    half = side / 2.0

    # raw crop coords
    left   = cx - half
    top    = cy - half
    right  = cx + half
    bottom = cy + half

    # clamp to image bounds (keeps the crop inside the image)
    left   = max(0.0, left)
    top    = max(0.0, top)
    right  = min(float(W), right)
    bottom = min(float(H), bottom)

    # If clamp made it non-square (happens near edges), force square by shrinking to min side
    cw = right - left
    ch = bottom - top
    side2 = min(cw, ch)
    # recenter within the clamped window
    cx2 = (left + right) / 2.0
    cy2 = (top + bottom) / 2.0
    left   = cx2 - side2 / 2.0
    right  = cx2 + side2 / 2.0
    top    = cy2 - side2 / 2.0
    bottom = cy2 + side2 / 2.0

    # final clamp (just in case float rounding)
    left   = max(0.0, left)
    top    = max(0.0, top)
    right  = min(float(W), right)
    bottom = min(float(H), bottom)

    crop = img.crop((int(round(left)), int(round(top)), int(round(right)), int(round(bottom))))
    crop = crop.resize(out_size, resample=Image.BILINEAR)
    return crop

    
# unsupervised dataset for AnyAttack. annotations are bbox annotations that we use to crop
# otherwise, there are no labels 

class AccidentTargetDataset(Dataset):
    """
    loads roboflow acc dataset for one split 
    """
    def __init__(self, acc_root, split="train", transform=None, out_size=(256,256), margin=0.15,
                 accident_category_id=1):
        if split not in ['train', 'test', 'valid']:
            raise ValueError("Split must be either 'train', 'test', 'valid'")
        self.acc_root = Path(acc_root).expanduser() / split
        self.transform = transform
        self.out_size = out_size
        self.margin = float(margin)
        self.accident_category_id = int(accident_category_id)

        ann_path = self.acc_root / "_annotations.coco.json"
        with ann_path.open("r") as f:
            coco = json.load(f)

        df_images = pd.DataFrame(coco.get("images", []))
        df_ann = pd.DataFrame(coco.get("annotations", []))

        df = df_images.merge(df_ann, how="left", left_on="id", right_on="image_id")
        df = df[df["category_id"] == self.accident_category_id].copy()
        df = df.dropna(subset=["bbox", "file_name"]).reset_index(drop=True)

        self.samples = []
        for row in df.itertuples(index=False):
            img_path = self.acc_root / row.file_name
            self.samples.append((img_path, row.bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = square_crop_around_bbox(img, bbox, out_size=self.out_size, margin=self.margin)
        if self.transform is not None:
            img = self.transform(img)
        meta = {"acc_path": str(img_path), "bbox": bbox}
        return img, meta

        
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# generic bdd dataset to store images 
class BddBaseDataset(Dataset): 
    def __init__(self, bdd_root, split='train', transform=None, out_size=(256,256)):
        self.root = Path(bdd_root).expanduser() / split 
        self.transform = transform 
        self.out_size = out_size 

        self.paths = [p for p in self.root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if len(self.paths) == 0: 
            raise ValueError(f'no images found in {self.root}')
        
    def __len__(self): 
        return len(self.paths)

    
    def __getitem__(self, idx): 
        p = self.paths[idx]
        img = Image.open(p).convert('RGB').resize(self.out_size, resample=Image.BILINEAR)

        if self.transform is not None: 
            img = self.transform(img)
        meta = {"bdd_path": str(p)}

        return img, meta 
