# -*- coding: utf-8 -*-
"""Custom datasets and data loaders for Cornell dataset."""

from collections import defaultdict
import random

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .base_data_loader import BaseDataset


class CornellDataset(BaseDataset):
    """Dataset utilities for Cornell dataset."""

    def __init__(self, annotations, config, features):
        """
        Initialize dataset.

        Inputs:
            - annotations: list of annotations per utterance
            - config: config class, see config.py
            - features: set of str, features to use in train/test
        """
        super().__init__(annotations, config, features)

    def get_depth_image(self, anno):
        """Get depth images."""
        img = Image.open(self._orig_data_path + anno['id'] + '_depth.tiff')
        _, _, left, top = self._get_grasps_crops(anno)
        top_left = (top, left)
        bottom_right = (
            min(480, top + self._im_size), min(640, left + self._im_size)
        )
        img = np.array(img)
        img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        img = transforms.functional.to_tensor(img).float()
        return torch.clamp(img - img.mean(), -1, 1)

    def get_grasps(self, anno, for_target=False):
        """Get ground-truth grasping points."""
        grasps, _, _, _ = self._get_grasps_crops(anno)
        return grasps

    def get_image(self, anno):
        """Get RGB images."""
        img = Image.open(self._orig_data_path + anno['id'] + 'r.png')
        _, _, left, top = self._get_grasps_crops(anno)
        top_left = (top, left)
        bottom_right = (
            min(480, top + self._im_size), min(640, left + self._im_size)
        )
        img = np.array(img)
        img = img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        img = transforms.functional.to_tensor(img).float()
        return (img - img.mean()) / 255

    def _get_grasps_crops(self, anno):
        grasps = []
        with open(self._orig_data_path + anno['id'] + 'cpos.txt') as fid:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                corner_0 = fid.readline()
                if not corner_0:
                    break  # EOF
                corner_1 = fid.readline()
                corner_2 = fid.readline()
                corner_3 = fid.readline()
                try:
                    grasps.append(np.array([
                        self._corner2points(corner_0),
                        self._corner2points(corner_1),
                        self._corner2points(corner_2),
                        self._corner2points(corner_3)
                    ]))
                except ValueError:  # some files contain weird values.
                    continue
        grasps = np.array(grasps)
        center = np.mean(np.vstack(grasps), axis=0).astype(np.int)
        left = max(0, min(center[1] - self._im_size // 2, 640 - self._im_size))
        top = max(0, min(center[0] - self._im_size // 2, 480 - self._im_size))
        grasps = grasps + np.array((-top, -left)).reshape((1, 2))
        return grasps, center, left, top

    @staticmethod
    def _corner2points(corner):
        c_x, c_y = corner.split()
        return [int(round(float(c_y))), int(round(float(c_x)))]
