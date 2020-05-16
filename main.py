# -*- coding: utf-8 -*-
"""Model training/testing pipeline."""

import argparse
import json

from config import Config
from src.models import ggcnn, unet

MODELS = {
    'ggcnn': ggcnn,
    'unet': unet,
}


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    # Model to train/test and peculiar parameters
    parser.add_argument(
        '--model', dest='model', help='Model to train (see main.py)',
        type=str, default='ggcnn'
    )
    parser.add_argument(
        '--model_params', dest='model_params',
        help='Dictionary of params peculiar to a model',
        type=str, default="{}"
    )
    # Dataset/task parameters
    parser.add_argument(
        '--dataset', dest='dataset', help='Name of dataset to use',
        type=str, default='jacquard'
    )
    parser.add_argument(
        '--net_name', dest='net_name', help='Name of trained model',
        type=str, default=''
    )
    # Specific task parameters: data handling
    parser.add_argument(
        '--handle_as_ggcnn', dest='handle_as_ggcnn',
        help='Handle annotations as GGCNN does',
        action='store_true'
    )
    parser.add_argument(
        '--im_size', dest='im_size',
        help='Image size (always consider square images)',
        type=int, default=320
    )
    parser.add_argument(
        '--jaw_size', dest='jaw_size',
        help='Jaw size during evaluation, "half" or float',
        type=str, default='half'
    )
    parser.add_argument(
        '--jaw_size_policy', dest='jaw_size_policy',
        help='Jaw size during training, {"min". "max", "random"}',
        type=str, default='min'
    )
    parser.add_argument(
        '--num_of_bins', dest='num_of_bins',
        help='Number of angle bins to consider when creating target maps',
        type=int, default=3
    )
    parser.add_argument(
        '--use_binary_map', dest='use_binary_map',
        help='Binarize quality map',
        action='store_true'
    )
    parser.add_argument(
        '--use_rgbd_img', dest='use_rgbd_img',
        help='Use RGB-D image as input',
        action='store_true'
    )
    # Specific task parameters: loss function
    parser.add_argument(
        '--use_angle_loss', dest='use_angle_loss',
        help='Force trigonometric constraints',
        action='store_true'
    )
    parser.add_argument(
        '--use_bin_loss', dest='use_bin_loss',
        help='Use a bin classification loss',
        action='store_true'
    )
    parser.add_argument(
        '--use_bin_attention_loss', dest='use_bin_attention_loss',
        help='Supervise bin_cls * pos_map',
        action='store_true'
    )
    parser.add_argument(
        '--use_graspness_loss', dest='use_graspness_loss',
        help='Solve a binary graspness task',
        action='store_true'
    )
    # Training parameters
    parser.add_argument(
        '--batch_size', dest='batch_size',
        help='Batch size in terms of images',
        type=int, default=8
    )
    parser.add_argument(
        '--learning_rate', dest='learning_rate',
        help='Learning of classification layers (not backbone)',
        type=float, default=0.002
    )
    parser.add_argument(
        '--weight_decay', dest='weight_decay',
        help='Weight decay of optimizer',
        type=float, default=0.0
    )
    # Learning rate policy
    parser.add_argument(
        '--not_use_early_stopping', dest='not_use_early_stopping',
        help='Do not use early stopping learning rate policy',
        action='store_true'
    )
    parser.add_argument(
        '--not_restore_on_plateau', dest='not_restore_on_plateau',
        help='Do not restore best model on validation plateau',
        action='store_true'
    )
    parser.add_argument(
        '--patience', dest='patience',
        help='Number of epochs to consider a validation plateu',
        type=int, default=1
    )
    # Other data loader parameters
    parser.add_argument(
        '--num_workers', dest='num_workers',
        help='Number of workers employed by data loader',
        type=int, default=2
    )
    return parser.parse_args()


def main():
    """Train and test a network pipeline."""
    args = parse_args()
    model = MODELS[args.model]
    cfg = Config(
        dataset=args.dataset,
        net_name=args.net_name if args.net_name else args.model,
        handle_as_ggcnn=args.handle_as_ggcnn,
        im_size=args.im_size,
        jaw_size=args.jaw_size,
        jaw_size_policy=args.jaw_size_policy,
        num_of_bins=args.num_of_bins,
        use_binary_map=args.use_binary_map,
        use_rgbd_img=args.use_rgbd_img,
        use_angle_loss=args.use_angle_loss,
        use_bin_loss=args.use_bin_loss,
        use_bin_attention_loss=args.use_bin_attention_loss,
        use_graspness_loss=args.use_graspness_loss,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_early_stopping=not args.not_use_early_stopping,
        restore_on_plateau=not args.not_restore_on_plateau,
        patience=args.patience,
        num_workers=args.num_workers
    )
    model_params = eval(json.loads('"' + args.model_params + '"'))
    model.train_test(cfg, model_params)


if __name__ == "__main__":
    main()
