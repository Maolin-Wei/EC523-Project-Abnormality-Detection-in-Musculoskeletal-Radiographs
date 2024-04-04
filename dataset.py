import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
import numpy as np


class MURADataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.transform = transform
        self.data_list = []

        for body_part, data in data_dict.items():
            for _, row in data.iterrows():
                img_dir = os.path.dirname(row['Image_Path'])
                for img_file in os.listdir(img_dir):
                    if img_file.startswith('.'):
                        continue
                    img_path = os.path.join(img_dir, img_file)
                    if os.path.isfile(img_path):
                        # Include body part information in the data_list
                        self.data_list.append((img_path, row['Label'], body_part))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label, body_part = self.data_list[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, body_part