# -*- coding: utf-8 -*-
"""Class to transform annotations into standard format."""

import glob
import json
import os

import cv2
from imageio import imread
import numpy as np
from skimage.transform import resize
from tqdm import tqdm


class JacquardTransformer:
    """A class to transform a dataset's annotations."""

    def __init__(self, config, crop=False):
        """Initialize transformer."""
        self._crop_images = crop
        self._orig_images_path = config.orig_data_path
        self._anno_json = config.paths['json_path'] + 'jacquard.json'

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
            os.path.join(self._orig_images_path, '*', '*_grasps.txt')
        ))
        assert any(txts)  # throw an error if dataset path is wrong
        split_ids = np.zeros(len(txts))
        split_ids[int(len(txts) * 0.9):] = 2
        for txt, split_id in tqdm(zip(txts, split_ids)):
            img_id = txt.split('/')[-1].replace('_grasps.txt', '')
            annos.append({
                'id': img_id,
                'object_label': '_'.join(img_id.split('_')[1:]),
                'split_id': int(split_id)
            })
        return annos

    def create_images(self):
        """Create images for training/testing."""
        with open(self._anno_json) as fid:
            annos = json.load(fid)
        resize_dim = 600
        start = (1024 - resize_dim) // 2
        end = (1024 + resize_dim) // 2

        for anno in tqdm(annos):
            # Read
            mask = imread(
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_mask.png'
            )
            img = imread(
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_RGB.png'
            )
            depth_img = imread(
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_perfect_depth.tiff'
            )
            txt = (
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_grasps.txt'
            )
            with open(txt) as fid:
                txt_annos = np.array([
                    [float(num) for num in line.strip().split(';')]
                    for line in fid.readlines()
                ])
            # Resize/crop
            if not self._crop_images:
                # Resize
                img = resize(
                    img, (resize_dim, resize_dim), preserve_range=True
                ).astype(img.dtype)
                depth_img = resize(
                    depth_img, (resize_dim, resize_dim), preserve_range=True
                ).astype(depth_img.dtype)
                mask = np.array(resize(
                    mask, (resize_dim, resize_dim), preserve_range=True
                ).astype(mask.dtype))
                mask[mask < 100] = 0
                mask[mask >= 100] = 1
                suffix = str(resize_dim)
            else:
                # Crop
                img = img[start:end, start:end].astype(img.dtype)
                depth_type = depth_img.dtype
                depth_img = depth_img[start:end, start:end].astype(depth_type)
                mask = np.array(mask[start:end, start:end].astype(mask.dtype))
                mask[mask < 100] = 0
                mask[mask >= 100] = 1
                cols = txt_annos[:, :2]
                to_keep = (end > cols) & (start <= cols)
                txt_annos = txt_annos[to_keep.all(1)]
                txt_annos[:, :2] -= start
                suffix = 'cropped'
            # Write
            cv2.imwrite(
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_RGB_' + suffix + '.png',
                img
            )
            cv2.imwrite(
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_perfect_depth_' + suffix + '.tiff',
                depth_img
            )
            np.save(
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_mask_' + suffix + '.npy',
                mask
            )
            txt = (
                self._orig_images_path + anno['id'].split('_')[1] + '/'
                + anno['id'] + '_grasps_' + suffix + '.txt'
            )
            with open(txt, 'w') as fid:
                fid.write('\n'.join([
                    ';'.join([str(num) for num in anno])
                    for anno in txt_annos.tolist()
                ]))
