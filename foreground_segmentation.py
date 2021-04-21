import matplotlib.pyplot as plt
import numpy as np

import cv2

from tqdm.notebook import tqdm

def get_jaccand_distance(labels0, labels1, target_label1):
    """
    Given labels for frames 0, 1, and a specified target label for frames
    0, 1, returns the jaccand difference of the label from frame 1 
    projected back to frame 0
    """
    pix1 = labels1 == target_label1

    pix0_overlap = labels0[pix1] 

    labels0_overlap = np.unique(pix0_overlap)

    jaccand_d = np.ones(labels0_overlap.shape[0], dtype=np.float32)

    # jaccand_d = []

    for (i,target_label0) in enumerate(labels0_overlap):
        pix0 = labels0 == target_label0

        intersection = np.logical_and(pix0, pix1)
        union = np.logical_or(pix0, pix1)

        intersection_sum = intersection.sum()
        union_sum = union.sum()

        if intersection_sum == 0:
            continue

        jaccand_d[i] = 1 - (intersection_sum / union_sum)

    return jaccand_d.min()

def get_motion_superpixels(labels0, labels1, jaccand_threshold):
    n_labels1 = np.unique(labels1).shape[0]

    jaccand_matrix = np.zeros(n_labels1, dtype=np.float32)

    for label in range(n_labels1):
        jaccand_matrix[label] = get_jaccand_distance(labels0, labels1, label)

    # print(jaccand_matrix)

    min_jaccand_above_threshold = (jaccand_matrix > jaccand_threshold).nonzero()[0]

    return min_jaccand_above_threshold
