import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class VisDroneYOLODataset(Dataset):
    def __init__(self, root_dir, size=640):
        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))]
        self.size = size

    def __len__(self): return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.img_names[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Letterbox Padding
        scale = self.size / max(w, h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        
        new_img = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        new_img.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        img_tensor = TF.to_tensor(new_img)
        
        # Load Labels for Loss Mask
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())
                    # 패딩된 좌표로 보정
                    xc = (xc * nw + (self.size - nw) / 2) / self.size
                    yc = (yc * nh + (self.size - nh) / 2) / self.size
                    bw = (bw * nw) / self.size
                    bh = (bh * nh) / self.size
                    boxes.append([xc, yc, bw, bh])

        return img_tensor, torch.tensor(boxes) if boxes else torch.zeros((0, 4))
