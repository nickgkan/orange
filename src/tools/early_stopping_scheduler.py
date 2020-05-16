# -*- coding: utf-8 -*-
"""A custom scheduler for handling early stopping."""


class EarlyStopping:
    """
    ReduceLROnPlateau with Early Stopping and model restoration.

    Inputs (see PyTorch docs on ReduceLROnPlateau for more info):
        - optimizer: PyTorch Optimizer
        - mode: str in 'min', 'max'. Whether min or max value of metric
            is best
        - factor: float, factor to multiply the learning rate
        - patience: int, number of epochs with no improvement after
            which learning rate will be reduced.
        - threshold: float, threshold for measuring the new optimum
        - threshold_mode: str in 'rel', 'abs'
        - max_decays: int, max number of lr decays allowed
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=1,
                 threshold=1e-4, threshold_mode='rel', max_decays=3):
        """Initialize scheduler."""
        assert (
            factor < 1.0
            and mode in ('min', 'max')
            and threshold_mode in ('rel', 'abs')
        )
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.max_decays = max_decays
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = -1
        self.decay_times = 0

    def step(self, metrics):
        """Scheduler step."""
        self.last_epoch += 1

        if self.best is None or self._is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.decay_times += 1
            self.reduce_lr()
            self.num_bad_epochs = 0
            return (
                self.last_epoch - self.patience - 1,
                self.decay_times <= self.max_decays)
        return self.last_epoch, True

    def reduce_lr(self):
        """Apply learning rate decay on all parameters."""
        factor = self.factor  # ** self.decay_times
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = factor * float(param_group['base_lr'])
            param_group['base_lr'] = factor * float(param_group['base_lr'])

    def _is_better(self, curr, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return curr < best * rel_epsilon
        if self.mode == 'min' and self.threshold_mode == 'abs':
            return curr < best - self.threshold
        if self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return curr > best * rel_epsilon
        return curr > best + self.threshold

    def state_dict(self):
        """Return state dict except for optimizer."""
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.__dict__.update(state_dict)
