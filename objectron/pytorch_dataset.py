
from __future__ import annotations
import os
import re
import torch
import pickle
import numpy as np
import skimage
from torch.utils.data import Dataset, DataLoader

import objectron.constants as constants


class ObjectronImageDataset(Dataset):

    def __init__(self, classes=None, transform=None):
        self.classes = classes
        if self.classes is None:
            self.classes = constants.CLASSES.copy()

        self.transform = transform
        self.root_dir_path = constants.DATA_DIR_PATH
        
        self.image_paths = []
        for class_ in self.classes:
            dir_path = constants.IMAGES_DIR_PATH.format(class_=class_)
            for filename in os.listdir(dir_path):
                self.image_paths.append(os.path.join(dir_path, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_paths[idx]
        image = skimage.io.imread(image_path)
        image_name = re.search(r'/([^/]+)\.jpg$', image_path).group(1)
        class_ = re.search(r'^(.+)_batch', image_name).group(1)
        annotation_path = os.path.join(constants.ANNOTATIONS_DIR_PATH.format(class_=class_), f'{image_name}.pickle')
        annotation = None
        with open(annotation_path, 'rb') as f:
            annotation = pickle.load(f)

        if self.transform:
            image = self.transform(image)

        sample = annotation
        sample['image'] = image

        return sample
