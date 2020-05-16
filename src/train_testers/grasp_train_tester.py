# -*- coding: utf-8 -*-
"""Functions for training and testing a network."""

from copy import deepcopy
import json

import numpy as np
from skimage.filters import gaussian
import torch
from torch.nn import functional as F
from tqdm import tqdm

from src.data_loaders import GraspDataLoader, JacquardDataset, CornellDataset
from src.evaluators import IoUEvaluator
from .base_train_tester import BaseTrainTester


class GraspTrainTester(BaseTrainTester):
    """
    Train and test utilities for grasping point detection.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to use in train/test
    """

    def __init__(self, net, config, features):
        """Initiliaze train/test instance."""
        super().__init__(net, config)
        self.features = features

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info("Testing %s on %s" % self._net_name)
        self.net.eval()
        self.net.mode = 'test'
        if self._use_cuda:
            self.net.cuda()
        self._set_data_loaders({'test': 2})
        self.data_loader = self._data_loaders['test']
        self.evaluator = IoUEvaluator(self.config)

        # Forward pass on test set
        results = {}
        for batch in tqdm(self.data_loader):
            pos, cos, sin, width, graspness, bins = self._net_outputs(batch)
            if self._use_graspness_loss:
                if pos.shape != graspness.shape:
                    graspness = graspness.unsqueeze(1)
                pos = pos * torch.sigmoid(graspness)
            if self._use_bin_loss:
                pos = pos * torch.sigmoid(bins)
            grasps = self.data_loader.get('grasps', batch)
            ids = self.data_loader.get('ids', batch)
            for cnt, (grasp_list, _id) in enumerate(zip(grasps, ids)):
                q_img, ang_img, width_img = post_process_output(
                    pos[cnt], cos[cnt],
                    sin[cnt], width[cnt],
                    self._handle_as_ggcnn,
                    self._use_binary_map
                )
                results[_id] = self.evaluator.step(
                    q_img, ang_img, width_img, grasp_list
                )
        self.evaluator.print_stats(self._net_name)
        with open(self._results_path + self._net_name + '.json', 'w') as fid:
            json.dump(results, fid)

    def _compute_loss(self, batch):
        """Compute loss for current batch."""
        pos, cos, sin, width, graspness, bins = self._net_forward(batch)
        pos_imgs, cos_imgs, sin_imgs, width_imgs, grasp_imgs = \
            self.data_loader.get('grasp_targets', batch)
        return self._loss_function(
            batch, pos, cos, sin, width, graspness, bins,
            pos_imgs, cos_imgs, sin_imgs, width_imgs, grasp_imgs
        )

    def _loss_function(self, batch, pos, cos, sin, width, graspness, bins,
                       pos_imgs, cos_imgs, sin_imgs, width_imgs, grasp_imgs):
        """Implement loss function for most networks."""
        # Compute loss
        pos_loss = F.mse_loss(pos, pos_imgs)
        cos_loss = F.mse_loss(cos, cos_imgs)
        sin_loss = F.mse_loss(sin, sin_imgs)
        width_loss = F.mse_loss(width, width_imgs)
        angle_loss = (
            float(self._use_angle_loss)
            * F.mse_loss(cos ** 2, 1 - sin ** 2)
        )
        grasp_loss = (
            float(self._use_graspness_loss)
            * F.binary_cross_entropy_with_logits(graspness, grasp_imgs)
        )
        bin_loss = (
            float(self._use_bin_loss)
            * F.binary_cross_entropy_with_logits(bins, (pos_imgs > 0).float())
        )
        bin_att_loss = (
            float(self._use_bin_attention_loss)
            * F.mse_loss(pos * torch.sigmoid(bins), pos_imgs)
        )

        # Store and return losses
        ret_losses = {
            'pos': pos_loss, 'cos': cos_loss,
            'sin': sin_loss, 'width': width_loss
        }
        if self._use_angle_loss:
            ret_losses['angle'] = angle_loss
        if self._use_graspness_loss:
            ret_losses['graspness'] = grasp_loss
        if self._use_bin_loss:
            ret_losses['bins'] = bin_loss
        if self._use_bin_attention_loss:
            ret_losses['bins*pos'] = bin_att_loss
        return (
            self._num_of_bins * (
                pos_loss + cos_loss + sin_loss + width_loss
                + 0.5 * angle_loss + bin_att_loss
            )
            + grasp_loss + bin_loss,
            ret_losses
        )

    def _net_forward(self, batch):
        """Forward pass of a single batch."""
        if self._use_rgbd_img:
            inputs = torch.cat([
                self.data_loader.get('depth_images', batch),
                self.data_loader.get('images', batch)
            ], dim=1)
        else:
            inputs = self.data_loader.get('depth_images', batch)
        return self.net(inputs)

    def _net_outputs(self, batch):
        """Forward pass of an inference batch."""
        return self._net_forward(batch)

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        # Annotations
        with open(self._json_path + self._dataset + '.json') as fid:
            annotations = json.load(fid)
        val_annos = []
        for anno in annotations:
            if anno['split_id'] == 2:
                anno = deepcopy(anno)
                anno['split_id'] = 1
                val_annos.append(anno)
        annotations += val_annos
        annotations = np.array(annotations)
        # Datasets
        split_ids = np.array([anno['split_id'] for anno in annotations])
        if self._dataset == 'jacquard':
            dataset = JacquardDataset
        else:
            dataset = CornellDataset
        datasets = {
            split: dataset(
                deepcopy(annotations[split_ids == split_id].tolist()),
                self.config, self.features)
            for split, split_id in mode_ids.items()
        }
        # Data loaders
        self._data_loaders = {
            split: GraspDataLoader(
                datasets[split], batch_size=self._batch_size,
                shuffle=split == 'train', num_workers=self._num_workers,
                drop_last=split in {'train', 'val'},
                use_cuda=self._use_cuda)
            for split in mode_ids
        }


def post_process_output(q_img, cos_img, sin_img, width_img, handle_as_ggnn,
                        use_binary_map):
    """Post-process the raw output of the GG-CNN."""
    q_img = q_img.cpu().numpy()
    ang_img = torch.atan2(sin_img, cos_img).cpu().numpy() / 2.0
    width_img = width_img.cpu().numpy() * 150.0

    if handle_as_ggnn:
        q_img = gaussian(q_img, 2.0, preserve_range=True)
        ang_img = gaussian(ang_img, 2.0, preserve_range=True)
        width_img = gaussian(width_img, 1.0, preserve_range=True)
    elif use_binary_map:
        q_img = np.stack([
            gaussian(img, 2.0, preserve_range=True) for img in q_img
        ])
    return q_img, ang_img, width_img
