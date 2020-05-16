# -*- coding: utf-8 -*-
"""Custom datasets and data loaders for grasping datasets."""

import json

import numpy as np
from skimage.draw import polygon
import torch


class BaseDataset(torch.utils.data.Dataset):
    """Dataset utilities for grasping datasets."""

    def __init__(self, annotations, config, features):
        """
        Initialize dataset.

        Inputs:
            - annotations: list of annotations per utterance
            - config: config class, see config.py
            - features: set of str, features to use in train/test
        """
        self._annotations = annotations
        self._config = config
        self.features = features
        self._set_init()
        self._set_methods()

    def __getitem__(self, index):
        """Get image's data (used by loader to later form a batch)."""
        anno = self._annotations[index]
        return_dict = {
            feature: self._methods[feature](anno)
            for feature in self.features if feature in self._methods
        }
        return_dict['ids'] = anno['id']
        return return_dict

    def __len__(self):
        """Override __len__ method, return dataset's size."""
        return len(self._annotations)

    def _set_init(self):
        """Set dataset variables."""
        # Load from config
        self._dataset = self._config.dataset
        self._handle_as_ggcnn = self._config.handle_as_ggcnn
        self._im_size = self._config.im_size
        self._jaw_size_policy = self._config.jaw_size_policy
        self._json_path = self._config.paths['json_path']
        self._num_of_bins = self._config.num_of_bins
        self._orig_data_path = self._config.orig_data_path
        self._use_binary_map = self._config.use_binary_map
        if self._config.use_rgbd_img:
            self.features.add('images')
        self._suffix = '_320'

        # Transform annotations
        split = self._annotations[0]['split_id']  # 0/1/2 for train/val/test
        self._training = split == 0
        self._config.logger.debug(
            'Set up dataset of %d files' % len(self._annotations))

    def _set_methods(self):
        """Correspond a method to each feature type."""
        self._methods = {
            'depth_images': self.get_depth_image,
            'grasps': self.get_grasps,
            'grasp_targets': self.get_grasp_targets,
            'ids': self.get_id,
            'images': self.get_image,
            'object_masks': self.get_object_mask
        }

    @staticmethod
    def get_depth_image(anno):
        """Get depth images."""
        return []

    @staticmethod
    def get_grasps(anno, for_target=False):
        """Get ground-truth grasping points."""
        return []

    def get_grasp_targets(self, anno):
        """Get drawed ground-truth grasping points."""
        if self._handle_as_ggcnn:
            draw_func = self.draw_as_ggcnn
        else:
            draw_func = self.draw
        pos_img, ang_img, width_img, grasp_img = draw_func(
            list(self.get_grasps(anno, True)),
            (self._im_size, self._im_size))
        width_img = np.clip(width_img, 0.0, 150.0) / 150.0
        return (
            pos_img,
            np.cos(2 * ang_img),
            np.sin(2 * ang_img),
            width_img, grasp_img
        )

    @staticmethod
    def get_id(anno):
        """Get image id."""
        return anno['id']

    @staticmethod
    def get_image(anno):
        """Get RGB images."""
        return []

    @staticmethod
    def get_object_mask(anno):
        """Get object segmentation mask."""
        return np.array([])

    @staticmethod
    def draw_as_ggcnn(grasps, shape):
        """Plot all solid rectangles in an array."""
        pos_out = np.zeros(shape)
        ang_out = np.zeros(shape)
        width_out = np.zeros(shape)

        for grasp in grasps:
            rows, cols = compact_polygon_coords(grasp.astype(np.int), shape)
            angle = np.arctan2(
                -grasp[1, 0] + grasp[0, 0],
                grasp[1, 1] - grasp[0, 1]
            )
            angle = (angle + np.pi / 2) % np.pi - np.pi / 2
            length = np.sqrt(
                (grasp[1, 1] - grasp[0, 1]) ** 2
                + (grasp[1, 0] - grasp[0, 0]) ** 2
            )

            pos_out[rows, cols] = 1.0
            ang_out[rows, cols] = angle
            width_out[rows, cols] = length

        return pos_out, ang_out, width_out, pos_out

    def draw(self, grasps, shape):
        """Plot solid rectangles of interest in an array."""
        bins = self._num_of_bins
        pos_out = np.zeros((bins, shape[0], shape[1]))
        ang_out = np.zeros((bins, shape[0], shape[1]))
        width_out = np.zeros((bins, shape[0], shape[1]))
        grasp_out = np.zeros(shape)

        for grasp in grasps:
            # Compute polygon and values to fill
            rows, cols = compact_polygon_coords(grasp, shape)
            if not rows.tolist() or not cols.tolist():
                continue
            b_map = np.zeros(shape)  # auxiliary binary map
            b_map[rows, cols] = 1
            angle = np.arctan2(
                -grasp[1, 0] + grasp[0, 0],
                grasp[1, 1] - grasp[0, 1]
            )
            angle = (angle + np.pi / 2) % np.pi
            ang_bin = int(np.floor(bins * angle / np.pi))
            angle -= np.pi / 2
            width = np.sqrt(
                (grasp[1, 1] - grasp[0, 1]) ** 2
                + (grasp[1, 0] - grasp[0, 0]) ** 2
            )

            # Fill grasp map
            grasp_out[rows, cols] = 1.0

            # Fill quality map
            if not self._use_binary_map:
                grid_x, grid_y = np.meshgrid(
                    np.linspace(-1, 1, max(cols) - min(cols) + 1),
                    np.linspace(-1, 1, max(rows) - min(rows) + 1)
                )
                gauss_grid = np.exp(-(grid_x ** 2 + grid_y ** 2) / 2)
                gauss_grid[(gauss_grid > 0) & (gauss_grid < 0.9)] = 0.9
                gauss_map = np.zeros(b_map.shape)
                gauss_map[
                    min(rows):max(rows) + 1,
                    min(cols):max(cols) + 1
                ] = gauss_grid
                gauss_map = gauss_map * b_map
                pos_out[ang_bin] = np.maximum(pos_out[ang_bin], gauss_map)
            else:
                pos_out[ang_bin] = np.maximum(pos_out[ang_bin], b_map)

            # Fill angle map
            ang_out[ang_bin][(b_map > 0) & (ang_out[ang_bin] == 0)] = angle
            ang_out[ang_bin][b_map * ang_out[ang_bin] != 0] = np.minimum(
                ang_out[ang_bin][b_map * ang_out[ang_bin] != 0], angle
            )

            # Fill width map
            width_out[ang_bin][(b_map > 0) & (width_out[ang_bin] == 0)] = width
            width_out[ang_bin] = np.maximum(b_map * width, width_out[ang_bin])
        return (
            pos_out.squeeze(), ang_out.squeeze(),
            width_out.squeeze(), grasp_out
        )


def grasp_collate_fn(batch_data, features):
    """Collate function for custom data loading."""
    return_batch = {'ids': [item['ids'] for item in batch_data]}
    tensor_features = {
        'depth_images', 'images'
    }
    to_tensor_features = {
        'object_masks'
    }
    for feature in features:
        if feature in tensor_features:
            return_batch[feature] = torch.stack([
                item[feature].float() for item in batch_data
            ])
        elif feature in to_tensor_features:
            return_batch[feature] = torch.stack([
                torch.from_numpy(item[feature]).float() for item in batch_data
            ])
        elif feature == 'grasp_targets':
            pos_imgs = torch.stack([
                torch.from_numpy(item[feature][0]).float()
                for item in batch_data])
            cos_imgs = torch.stack([
                torch.from_numpy(item[feature][1]).float()
                for item in batch_data])
            sin_imgs = torch.stack([
                torch.from_numpy(item[feature][2]).float()
                for item in batch_data])
            width_imgs = torch.stack([
                torch.from_numpy(item[feature][3]).float()
                for item in batch_data])
            grasp_imgs = torch.stack([
                torch.from_numpy(item[feature][4]).float()
                for item in batch_data])
            return_batch[feature] = (
                pos_imgs, cos_imgs, sin_imgs, width_imgs, grasp_imgs
            )
        else:  # list of numpy arrays
            return_batch[feature] = [item[feature] for item in batch_data]
    return return_batch


class GraspDataLoader(torch.utils.data.DataLoader):
    """Custom data loader for Jacquard dataset."""

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=2,
                 drop_last=False, use_cuda=False):
        """Initialize loader for given dataset and annotations."""
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last,
            collate_fn=lambda data: grasp_collate_fn(data, dataset.features)
        )
        self._use_cuda = use_cuda

    def get(self, feature, batch):
        """Get specific feature from a given batch."""
        not_tensors = {'grasps', 'ids', 'grasp_targets'}
        if feature == 'grasp_targets' and self._use_cuda:
            return [feat.cuda() for feat in batch[feature]]
        if feature in not_tensors or not self._use_cuda:
            return batch[feature]
        return batch[feature].cuda()


def compact_polygon_coords(grasp, shape=None):
    """Return pixels within the centre third of the grasp rectangle."""
    center = grasp.mean(axis=0)  # .astype(np.int)
    angle = np.arctan2(-grasp[1, 0] + grasp[0, 0], grasp[1, 1] - grasp[0, 1])
    angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    length = np.sqrt(
        (grasp[1, 1] - grasp[0, 1]) ** 2 + (grasp[1, 0] - grasp[0, 0]) ** 2)
    width = np.sqrt(
        (grasp[2, 0] - grasp[1, 0]) ** 2 + (grasp[2, 1] - grasp[1, 1]) ** 2)
    length = length / 3  # center third

    # Create points
    x_0, y_0 = (np.cos(angle), np.sin(angle))
    y_1 = center[0] + length / 2 * y_0
    x_1 = center[1] - length / 2 * x_0
    y_2 = center[0] - length / 2 * y_0
    x_2 = center[1] + length / 2 * x_0
    points = np.array([
        [y_1 - width / 2 * x_0, x_1 - width / 2 * y_0],
        [y_2 - width / 2 * x_0, x_2 - width / 2 * y_0],
        [y_2 + width / 2 * x_0, x_2 + width / 2 * y_0],
        [y_1 + width / 2 * x_0, x_1 + width / 2 * y_0],
    ]).astype(np.float)
    return polygon(points[:, 0], points[:, 1], shape)
