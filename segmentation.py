import matplotlib.pyplot as plt
import numpy as np

import cv2

from tqdm import tqdm

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float,img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np

import cv2

from tqdm.notebook import tqdm

def slic_segment_image(img, n_segments=500, compactness=10, mask=None):
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

    jaccard_d = np.ones(labels0_overlap.shape[0], dtype=np.float32)

    # jaccard_d = []

    for (i,target_label0) in enumerate(labels0_overlap):
        pix0 = labels0 == target_label0

        intersection = np.logical_and(pix0, pix1)
        union = np.logical_or(pix0, pix1)

        intersection_sum = intersection.sum()
        union_sum = union.sum()

        if intersection_sum == 0:
            continue

        jaccard_d[i] = 1 - (intersection_sum / union_sum)

    return jaccard_d.min()

def get_motion_superpixels(labels0, labels1, jaccard_threshold):
    n_labels1 = np.unique(labels1).shape[0]

    jaccard_matrix = np.zeros(n_labels1, dtype=np.float32)

    for label in range(n_labels1):
        jaccard_matrix[label] = get_jaccard_distance(labels0, labels1, label)

    # print(jaccard_matrix)

    min_jaccard_above_threshold = (jaccard_matrix > jaccard_threshold).nonzero()[0]

    return min_jaccard_above_threshold

def get_superpixel_masks(frame, labels, motion_labels):
    H, W, C = frame.shape 

    superpixel_mask = np.zeros((H, W), dtype=np.uint8)

    for label in motion_labels:
        label_pix_x, label_pix_y = np.where(labels == label)

        superpixel_mask[label_pix_x,label_pix_y] = 1

    background_mask = (1 - superpixel_mask).astype(np.uint8)

    return superpixel_mask, background_mask
    

def get_background_images(frame, labels, motion_superpixel_labels):
    mask = np.isin(labels, motion_superpixel_labels, invert=True).nonzero()

    background_pixels = np.zeros((mask[0].shape[0], 5), dtype=np.uint32)
    background_pixels[:,0] = mask[0]
    background_pixels[:,1] = mask[1]
    background_pixels[:,2:] = frame[mask[0],mask[1],:]

    return background_pixels