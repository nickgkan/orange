# -*- coding: utf-8 -*-
"""Functions for training and testing a network."""

from collections import defaultdict
import json
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from src.tools.early_stopping_scheduler import EarlyStopping


class BaseTrainTester:
    """
    Train and test utilities for multiple datasets and tasks.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
    """

    def __init__(self, net, config):
        """Initiliaze train/test instance."""
        self.net = net
        self.config = config
        self._set_from_config(config)

    def _set_from_config(self, config):
        """Load config variables."""
        self._batch_size = config.batch_size
        self._dataset = config.dataset
        self._handle_as_ggcnn = config.handle_as_ggcnn
        self._im_size = config.im_size
        self._jaw_size = config.jaw_size
        self._json_path = config.paths['json_path']
        self._learning_rate = config.learning_rate
        self._loss_path = config.paths['loss_path']
        self._models_path = config.paths['models_path']
        self._net_name = config.net_name
        self._num_of_bins = config.num_of_bins
        self._num_workers = config.num_workers
        self._patience = config.patience
        self._restore_on_plateau = config.restore_on_plateau
        self._results_path = config.paths['results_path']
        self._use_angle_loss = config.use_angle_loss
        self._use_binary_map = config.use_binary_map
        self._use_bin_loss = config.use_bin_loss
        self._use_bin_attention_loss = config.use_bin_attention_loss
        self._use_cuda = config.use_cuda
        self._use_early_stopping = config.use_early_stopping
        self._use_graspness_loss = config.use_graspness_loss
        self._use_rgbd_img = config.use_rgbd_img
        self._weight_decay = config.weight_decay
        self.logger = config.logger

    def train_test(self):
        """Train and test a net, general handler."""
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            self.net.parameters(), self._learning_rate,
            weight_decay=self._weight_decay)
        if self._use_early_stopping:
            scheduler = EarlyStopping(
                optimizer, patience=self._patience, factor=0.3)
        else:
            scheduler = MultiStepLR(optimizer, [4, 8], gamma=0.3)

        # Train and test pipeline
        self.train(
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=50 if self._use_early_stopping else 10)
        self.test()
        self.logger.info('Test complete')

    def train(self, optimizer, scheduler=None, epochs=1):
        """Train a neural network if it does not already exist."""
        self.logger.info("Performing training for " + self._net_name)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Check if the model is already trained
        model_path_name = self._models_path + self._net_name + '.pt'
        if self._check_for_existent_model(model_path_name):
            return self.net

        # Check for existent checkpoint of previous epochs
        checkpoints, loss_history = self._check_for_checkpoint(model_path_name)
        epochs = list(range(epochs))[max(checkpoints):]

        # Settings and loading
        self.net.train()
        if self._use_cuda:
            self.net.cuda()
        self._set_data_loaders({'train': 0, 'val': 1})
        self.data_loader = self._data_loaders['train']
        self.logger.debug("Batch size is " + str(self._batch_size))

        # Main training procedure
        self.training = True
        for epoch in epochs:
            loss_history, keep_training = self._train_epoch(
                epoch, model_path_name, loss_history)
            if not keep_training:
                self.logger.info('Model converged, exit training')
                break

        # Training is complete, save model
        self._save_model(model_path_name)
        self.logger.info('Finished Training')
        return self.net

    def _train_epoch(self, epoch, model_path_name, loss_history):
        """Train the network for one epoch."""
        keep_training = True

        # Adjust learning rate
        if self.scheduler is not None and not self._use_early_stopping:
            self.scheduler.step()
        for param_group in self.optimizer.param_groups:
            param_group['base_lr'] = param_group['lr']
        curr_lr = max(p['lr'] for p in self.optimizer.param_groups)
        self.logger.debug("Learning rate is now " + str(curr_lr))

        # Main epoch pipeline
        accum_loss = 0
        for batch in tqdm(self.data_loader):
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + Backward + Optimize on batch data
            loss, _ = self._compute_train_loss(batch)
            loss.backward()
            self.optimizer.step()
            accum_loss += loss.item()

        # After each epoch: reset lr (necessary for dynamic lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['base_lr']

        # After each epoch: check validation loss and convergence
        val_loss, print_loss = self._compute_validation_loss()
        if self._use_early_stopping and self.scheduler is not None:
            ret_epoch, keep_training = self.scheduler.step(val_loss)
            if ret_epoch < epoch:
                if self._restore_on_plateau or not keep_training:
                    self._load_model(model_path_name, epoch=ret_epoch,
                                     restore={'net', 'optimizer'})
                    self.scheduler.reduce_lr()

        # Print training statistics
        accum_loss /= len(self.data_loader)
        loss_history.append((accum_loss, val_loss, curr_lr))
        val_string = ', '.join([
            key + ': ' + '%.5f' % print_loss[key]
            for key in sorted(list(print_loss.keys()))
        ])
        self.logger.info((
            '[Epoch %d] loss: %.4f, validation loss: %.4f ('
            % (epoch, accum_loss, val_loss)
        ) + val_string + ')')
        self._save_model(model_path_name, epoch=epoch)
        with open(self._loss_path + self._net_name + '.json', 'w') as fid:
            json.dump(loss_history, fid)
        return loss_history, keep_training

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        self.logger.info("Testing %s" % self._net_name)

    def _check_for_existent_model(self, model_path_name):
        """Check if a trained model is existent."""
        if os.path.exists(model_path_name):
            self._load_model(model_path_name, restore={'net'})
            self.logger.debug("Found existing trained model.")
            return True
        return False

    def _check_for_checkpoint(self, model_path_name):
        """Check if an intermediate checkpoint exists."""
        epochs_to_resume = [
            int(name[(len(self._net_name) + 6): -3]) + 1
            for name in os.listdir(self._models_path)
            if name.startswith(self._net_name + '_epoch')
            and name[(len(self._net_name) + 6): -3].isdigit()
        ] + [0]
        loss_history = []
        if any(epochs_to_resume):
            self._load_model(model_path_name, epoch=max(epochs_to_resume) - 1)
            self.logger.debug(
                'Found checkpoint for epoch: %d' % (max(epochs_to_resume) - 1))
            self.net.train()
            if os.path.exists(self._loss_path + self._net_name + '.json'):
                with open(self._loss_path + self._net_name + '.json') as fid:
                    loss_history = json.load(fid)
            loss_history = loss_history[:max(epochs_to_resume)]
        return epochs_to_resume, loss_history

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 2, 'test': 2}):
        self._data_loaders = {
            split: None
            for split in mode_ids
        }

    def _compute_train_loss(self, batch):
        """Compute train loss."""
        return self._compute_loss(batch)

    @torch.no_grad()
    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.training = False
        self.net.eval()
        self.data_loader = self._data_loaders['val']
        loss = 0
        print_loss = defaultdict(int)
        for batch in self.data_loader:
            loss_val, loss_dict = self._compute_loss(batch)
            loss += loss_val.item()
            for key, val in loss_dict.items():
                print_loss[key] += val.item()
        loss /= len(self.data_loader)
        for key in print_loss:
            print_loss[key] /= len(self.data_loader)
        self.net.train()
        self.data_loader = self._data_loaders['train']
        self.training = True
        return loss, print_loss

    def _load_model(self, model_path_name, epoch=None,
                    restore={'net', 'optimizer', 'scheduler'}):
        """Load a checkpoint, possibly referring to specific epoch."""
        if epoch is not None:
            checkpoint = torch.load(
                model_path_name[:-3] + '_epoch' + str(epoch) + '.pt')
        else:
            checkpoint = torch.load(model_path_name)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if self._use_cuda:
            self.net.cuda()
        else:
            self.net.cpu()
        if 'optimizer' in restore:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler' in restore:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def _save_model(self, model_path_name, epoch=None):
        """Save a checkpoint, possibly referring to specific epoch."""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if epoch is not None:
            torch.save(
                checkpoint,
                model_path_name[:-3] + '_epoch' + str(epoch) + '.pt')
        else:
            torch.save(checkpoint, model_path_name)

    def _compute_loss(self, batch):
        """Compute loss for current batch."""
        return self.net()
