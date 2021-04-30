import numpy as np

import cv2
import maxflow

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_ubyte
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import numpy as np

import cv2

from tqdm.notebook import tqdm


def slic_segment_image(img, n_segments=500, compactness=1, mask=None):
    H, W, C = img.shape

    img = img_as_float(img)

    labels = slic(img, n_segments=n_segments, start_label=0,
                  compactness=compactness, multichannel=True, mask=mask)

    return labels

def get_boundary_segments(img, labels):
    return img_as_ubyte(mark_boundaries(img, labels))

def get_jaccard_distance(labels0, labels1, target_label1):
    """
    Given labels for frames 0, 1, and a specified target label for frames
    0, 1, returns the jaccard difference of the label from frame 1 
    projected back to frame 0
    """
    pix1 = labels1 == target_label1

    pix0_overlap = labels0[pix1]

    labels0_overlap = np.unique(pix0_overlap)

    jaccard_d = np.zeros(labels0_overlap.shape[0], dtype=np.float32)

    for (i, target_label0) in enumerate(labels0_overlap):
        pix0 = labels0 == target_label0

        intersection = np.logical_and(pix0, pix1)
        union = np.logical_or(pix0, pix1)

        intersection_sum = intersection.sum()
        union_sum = union.sum()

        if intersection_sum == 0:
            continue

        jaccard_d[i] = 1 - (intersection_sum / union_sum)

    return jaccard_d.min()

def get_jaccard_motion_superpixels(labels0, labels1, jaccard_threshold):
    n_labels1 = np.unique(labels1).shape[0]

    jaccard_matrix = np.zeros(n_labels1, dtype=np.float32)

    for label in range(n_labels1):
        jaccard_matrix[label] = get_jaccard_distance(labels0, labels1, label)

    min_jaccard_above_threshold = (
        jaccard_matrix > jaccard_threshold).nonzero()[0]

    return min_jaccard_above_threshold

def calculate_background_priors(backgorund_models):
    priors = np.array([model.calculatePriorBackground() for model in backgorund_models])

    return priors / priors.sum()

def calculate_foreground_priors(t_k, N_f):
    if type(t_k) != type(np.array):
        t_k = np.ones(N_f) * t_k

    priors = 1 / (t_k + N_f)

    return priors/priors.sum()

def calculate_kl_divergence(P, Q, samples):
    P_x = P.score_samples(samples)
    Q_x = Q.score_samples(samples)

    KL = np.exp(P_x) * (P_x - Q_x)

    return KL.sum()

def calculate_local_similarity(model_tau, PHI, n_samples=100):
    tau_samples = model_tau.sample(n_samples)[0]

    P_x = model_tau.score_samples(tau_samples)
    P_x_e = np.exp(P_x)

    kl_divergences = np.array([
        (P_x_e * (P_x - phi.score_samples(tau_samples))).sum() for phi in PHI
    ])

    return kl_divergences.min()

def get_num_foreground_regions(superpixel_mask):
    """
    Given the superpixel mask
    """
    object_count = 0
    visited = np.zeros_like(superpixel_mask, dtype=np.bool)
    labels = np.zeros_like(superpixel_mask, dtype=np.int16) - 1

    H, W = superpixel_mask.shape 

    aa = [(0,1),(0,-1),(1,0),(-1,0)]

    def DFS(x, y):
        stack = []

        stack.append((x, y))
        
        while len(stack) > 0:
            i, j = stack.pop() 

            if i < 0 or j < 0 or i >= H or j >= W:
                continue
            if visited[i,j] or superpixel_mask[i,j] == 0:
                visited[i,j] = True
                continue

            visited[i,j] = True
            labels[i,j] = object_count

            for ii,jj in aa:
                stack.append((i + ii, j + jj))

    ones_x, ones_y = np.where(superpixel_mask == 1)

    for (x,y) in zip(ones_x, ones_y):        
        if visited[x,y]:
            continue

        DFS(x,y)

        object_count += 1

    return object_count, labels