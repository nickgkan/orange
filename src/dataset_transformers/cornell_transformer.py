# -*- coding: utf-8 -*-
"""Class to transform annotations into standard format."""

import glob
import json
import os

import cv2
from imageio import imsave
import numpy as np
from tqdm import tqdm


class CornellTransformer:
    """A class to transform a dataset's annotations."""

    def __init__(self, config, crop=False):
        """Initialize transformer."""
        self._orig_images_path = config.orig_data_path
        self._anno_json = config.paths['json_path'] + 'cornell.json'

    def transform(self):
        """Run transform pipeline."""
        if not os.path.exists(self._anno_json):
            with open(self._anno_json, 'w') as fid:
                json.dump(self.handle_annotations(), fid)
        self.create_images()

    def handle_annotations(self):
        """Create json annotations for the dataset."""
        annos = []
        txts = sorted(glob.glob(
            os.path.join(self._orig_images_path, '*', '*pcd*cpos.txt')
        ))
        assert any(txts)  # throw an error if dataset path is wrong
        split_ids = np.zeros(len(txts))
        split_ids[int(len(txts) * 0.9):] = 2
        for txt, split_id in tqdm(zip(txts, split_ids)):
            folder_id = txt.split('/')[-2]
            img_id = txt.split('/')[-1].replace('cpos.txt', '')
            annos.append({
                'id': folder_id + '/' + img_id,
                'split_id': int(split_id)
            })
        return annos

    def create_images(self):
        """Create depth images for training/testing."""
        pcds = sorted(glob.glob(
            os.path.join(self._orig_images_path, '*', '*pcd*[0-9].txt')
        ))
        for pcd in tqdm(pcds):
            depth_img = self._depth_from_pcd(pcd)
            imsave(
                pcd.replace('.txt', '_depth.tiff'),
                depth_img.astype(np.float32)
            )

    @staticmethod
    def _depth_from_pcd(pcd_filename):
        """Create a depth image from an unstructured PCD file."""
        img = np.zeros((480, 640))
        with open(pcd_filename) as fid:
            for line in fid.readlines():
                split_line = line.split()
                if len(split_line) != 5:
                    continue  # Not a point line in the file.
                try:
                    float(split_line[0])
                except ValueError:  # Not a number, continue.
                    continue
                row = int(split_line[4]) // 640
                col = int(split_line[4]) % 640
                img[row, col] = np.sqrt(
                    float(split_line[0]) ** 2  # x
                    + float(split_line[1]) ** 2  # y
                    + float(split_line[2]) ** 2  # z
                )
        img = img / 1000.0

        # Inpaint missing values
        img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
        mask = (img == 0).astype(np.uint8)
        scale = np.abs(img).max()  # # scale to keep as float in -1:1
        img = img.astype(np.float32) / scale  # has to be float32.
        img = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
        img = img[1:-1, 1:-1]  # back to original size...
        img = img * scale  # ...and value range
        return img
