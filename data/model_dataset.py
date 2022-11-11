import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, image_root, image_files, labels, transform, scale = 1):
        self.image_root = image_root
        self.image_files = image_files
        self.labels = labels
        self.scale = scale
        self.transform = transform
       
    def __len__(self):
        return int(len(self.labels) * self.scale)

    def __getitem__(self, index):
        index = index % len(self.labels)
        img = Image.open(os.path.join(self.image_root, self.image_files[index])).convert('RGB')
        return self.transform(img), self.labels[index]

class Preprocessing():
    def __init__(self, transform, augmenter = None):
        self.transform = transform
        self.augmenter = augmenter

    def transform(self, pil_image, do_aug = False):
        t_img = self.augmenter(pil_image)  if do_aug and self.augmenter != None else pil_image
        return self.transform(t_img)
