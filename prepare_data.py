# -*- coding: utf-8 -*-
"""Prepare annotations and create train/test dataset."""

import os
import sys

from config import Config, PATHS
from src.dataset_transformers import CornellTransformer, JacquardTransformer

TRANSFORMERS = {
    'cornell': CornellTransformer,
    'jacquard': JacquardTransformer
}


def main(datasets):
    """Run the data preprocessing and creation pipeline."""
    if not os.path.exists(PATHS['json_path']):
        os.mkdir(PATHS['json_path'])
    if not os.path.exists(PATHS['loss_path']):
        os.mkdir(PATHS['loss_path'])
    if not os.path.exists(PATHS['models_path']):
        os.mkdir(PATHS['models_path'])
    if not os.path.exists(PATHS['results_path']):
        os.mkdir(PATHS['results_path'])
    for dataset in datasets:
        print('Creating annotations for ' + dataset)
        TRANSFORMERS[dataset](Config(dataset=dataset), False).transform()
    print('Done.')


if __name__ == "__main__":
    if any(sys.argv[1:]):
        main(sys.argv[1:])
    else:
        main(['cornell', 'jacquard'])
