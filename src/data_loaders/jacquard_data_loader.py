# -*- coding: utf-8 -*-
"""Custom datasets and data loaders for Jacquard dataset."""

from collections import defaultdict
import random

from imageio import imread
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .base_data_loader import BaseDataset


class JacquardDataset(BaseDataset):
    """Dataset utilities for Jacquard dataset."""

    def __init__(self, annotations, config, features):
        """
        Initialize dataset.

        Inputs:
            - annotations: list of annotations per utterance
            - config: config class, see config.py
            - features: set of str, features to use in train/test
        """
        for anno in annotations:
            anno['id'] = '_'.join(anno['id'].split('_')[1:]) + '/' + anno['id']
        super().__init__(annotations, config, features)

    def get_depth_image(self, anno):
        """Get depth images."""
        img = Image.open(
            self._orig_data_path + anno['id']
            + '_perfect_depth' + self._suffix + '.tiff'
        )
        if img.size[0] != self._im_size:
            img = img.resize((self._im_size, self._im_size))
        img = transforms.functional.to_tensor(img).float()
        return torch.clamp(img - img.mean(), -1, 1)

    def get_grasps(self, anno, for_target=False):
        """Get ground-truth grasping points."""
        txts = '_grasps' + self._suffix
        with open(self._orig_data_path + anno['id'] + txts + '.txt') as fid:
            txt_annos = np.array([  # x, y, theta, w, h
                [float(v) for v in line.strip('\n').split(';')] for line in fid
            ])
        if not txt_annos.any():
            return txt_annos
        if not self._handle_as_ggcnn and (self._training or for_target):
            # pick a single jaw size
            diff_grasps = defaultdict(list)
            if self._jaw_size_policy != 'all':
                for txt_anno in txt_annos:
                    diff_grasps[tuple(txt_anno[:4])].append(txt_anno[4])
                for grasp in diff_grasps:
                    if self._jaw_size_policy == 'min':
                        diff_grasps[grasp] = min(diff_grasps[grasp])
                    elif self._jaw_size_policy == 'max':
                        diff_grasps[grasp] = max(diff_grasps[grasp])
                    elif self._jaw_size_policy == 'random':
                        rand = random.randint(0, len(diff_grasps[grasp]) - 1)
                        diff_grasps[grasp] = diff_grasps[grasp][rand]
                txt_annos = np.array([
                    list(grasp) + [diff_grasps[grasp]] for grasp in diff_grasps
                ])
        center = txt_annos[:, (1, 0)]
        angle = -txt_annos[:, 2] / 180.0 * np.pi
        length = txt_annos[:, 3]
        width = txt_annos[:, 4]
        cos = np.cos(angle)
        sin = np.sin(angle)
        y_1 = center[:, 0] + length / 2 * sin
        x_1 = center[:, 1] - length / 2 * cos
        y_2 = center[:, 0] - length / 2 * sin
        x_2 = center[:, 1] + length / 2 * cos
        grasps = np.stack([
            np.stack([y_1 - width / 2 * cos, x_1 - width / 2 * sin], -1),
            np.stack([y_2 - width / 2 * cos, x_2 - width / 2 * sin], -1),
            np.stack([y_2 + width / 2 * cos, x_2 + width / 2 * sin], -1),
            np.stack([y_1 + width / 2 * cos, x_1 + width / 2 * sin], -1)
        ], axis=1)
        return grasps * self._im_size / 1024.0

    def get_image(self, anno):
        """Get RGB images."""
        img = Image.open(
            self._orig_data_path + anno['id'] + '_RGB' + self._suffix + '.png'
        )
        if img.size[0] != self._im_size and self._suffix != '':
            img = img.resize((self._im_size, self._im_size))
        img = transforms.functional.to_tensor(img).float()
        return (img - img.mean()) / 255

    def get_object_mask(self, anno):
        """Get object segmentation mask."""
        mask = np.load(
            self._orig_data_path + anno['id']
            + '_mask' + self._suffix + '.npy'
        )
        mask[mask > 0] = 1.0
        return mask.astype(np.float32)
