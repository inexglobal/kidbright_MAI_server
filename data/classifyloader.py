import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class ImageClassifyData(data.Dataset):
    def __init__(self, root,
                labels
        ):
        self.root = root
        self.labels = labels
        self.ids = list()
        for i, label in enumerate(labels):
            path = os.path.join(root, label)
            for name in os.listdir(path):
                if not os.path.splitext(name.lower())[1] in [".jpg", ".jpeg", ".png"]:
                    continue
                self.ids.append((path, name, i))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = cv2.imread(os.path.join(img_id[0], img_id[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, img_id[2]

    def __len__(self):
        return len(self.ids)

    def reset_transform(self, transform):
        self.transform = transform

    def pull_item(self, index):
        img_id = self.ids[index]
        img = cv2.imread(os.path.join(img_id[0], img_id[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, img_id[2]
