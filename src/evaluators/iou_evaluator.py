# -*- coding: utf-8 -*-
"""A class to evaluate IoU metrics."""

from matplotlib import pyplot as plt
import numpy as np
from skimage.draw import polygon
from skimage.feature import peak_local_max


class IoUEvaluator:
    """A class to evaluate accuracy using IoU."""

    def __init__(self, config):
        """Initialize counters."""
        self._handle_as_ggcnn = config.handle_as_ggcnn
        self._im_size = config.im_size
        self._jaw_size = config.jaw_size
        self._results_path = config.paths['results_path']
        self._detected = np.zeros(101)
        self._gt = 0

    def step(self, q_img, angle_img, width_img, targets):
        """Evaluate a single image."""
        max_iou = calculate_iou_match(
            q_img, angle_img, width_img,
            targets, self._handle_as_ggcnn, self._im_size, self._jaw_size)
        self._detected[:int(np.floor(max_iou)) + 1] += 1
        self._gt += 1
        return max_iou

    def print_stats(self, name=''):
        """Print accuracy."""
        acc_per_thres = 100 * self._detected / self._gt
        print("Accuracy@0.25: %f" % acc_per_thres[25])
        print("Accuracy@0.30: %f" % acc_per_thres[30])
        print("Accuracy@0.50: %f" % acc_per_thres[50])
        print("Accuracy@0.75: %f" % acc_per_thres[75])
        print("Avg. Accuracy: %f" % acc_per_thres.mean())
        plt.plot(acc_per_thres)
        plt.ylabel('Accuracy')
        plt.xlabel('Threshold')
        plt.savefig(self._results_path + name + '_accuracy.png')
        plt.close()


def calculate_iou_match(q_img, angle_img, width_img, ground_truth_bbs,
                        handle_as_ggcnn, im_size, jaw_size):
    """
    Calculate grasp success using the IoU (Jacquard) metric.

    Success: grasp rectangle has a 25% IoU with a ground truth
    and is within 30 degrees.
    """
    # Find local maximum
    if handle_as_ggcnn:
        local_max = peak_local_max(q_img, 20, threshold_abs=0.2, num_peaks=1)
    else:
        local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])
    if not local_max.tolist():
        local_max = np.array([np.unravel_index(q_img.argmax(), q_img.shape)])
    grasp_point = tuple(local_max[0])

    # Reconstruct detected box
    center, angle, length, width = [
        grasp_point[-2:], angle_img[grasp_point], width_img[grasp_point],
        _compute_jaw_size(width_img[grasp_point], jaw_size)]
    x_0, y_0 = (np.cos(angle), np.sin(angle))
    y_1 = center[0] + length / 2 * y_0
    x_1 = center[1] - length / 2 * x_0
    y_2 = center[0] - length / 2 * y_0
    x_2 = center[1] + length / 2 * x_0
    det = np.array([  # detected shape
        [y_1 - width / 2 * x_0, x_1 - width / 2 * y_0],
        [y_2 - width / 2 * x_0, x_2 - width / 2 * y_0],
        [y_2 + width / 2 * x_0, x_2 + width / 2 * y_0],
        [y_1 + width / 2 * x_0, x_1 + width / 2 * y_0],
    ]).astype(np.float)

    # Return max IoU
    return max(100 * iou(det, grasp) for grasp in ground_truth_bbs)


def iou(det, grasp):
    """Compute IoU between detected and ground-truth grasp."""
    angle = np.arctan2(-grasp[1, 0] + grasp[0, 0], grasp[1, 1] - grasp[0, 1])
    gt_angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    angle = np.arctan2(-det[1, 0] + det[0, 0], det[1, 1] - det[0, 1])
    det_angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    if abs((det_angle - gt_angle + np.pi / 2) % np.pi - np.pi / 2) > np.pi / 6:
        return 0
    rr1, cc1 = polygon(det[:, 0], det[:, 1])
    rr2, cc2 = polygon(grasp[:, 0], grasp[:, 1])
    if not all(itm.tolist() for itm in [rr1, cc1, rr2, cc2]):
        return 0
    r_max = max(rr1.max(), rr2.max()) + 1
    c_max = max(cc1.max(), cc2.max()) + 1
    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection / union


def _compute_jaw_size(width, jaw_size):
    if jaw_size == 'half':
        return width / 2
    if jaw_size == 'full':
        return width
    return float(jaw_size)
