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


def get_motion_superpixels(labels0, labels1, jaccard_threshold):
    n_labels1 = np.unique(labels1).shape[0]

    jaccard_matrix = np.zeros(n_labels1, dtype=np.float32)

    for label in range(n_labels1):
        jaccard_matrix[label] = get_jaccard_distance(labels0, labels1, label)

    min_jaccard_above_threshold = (
        jaccard_matrix > jaccard_threshold).nonzero()[0]

    return min_jaccard_above_threshold


def get_superpixel_masks(frame, labels, motion_labels):
    H, W, C = frame.shape

    superpixel_mask = np.zeros((H, W), dtype=np.uint8)

    for label in motion_labels:
        label_pix_x, label_pix_y = np.where(labels == label)

        superpixel_mask[label_pix_x, label_pix_y] = 1

    background_mask = (1 - superpixel_mask).astype(np.uint8)

    return superpixel_mask, background_mask

# def image_to_array(frame, labels, motion_superpixel_labels):
#     mask = np.isin(labels, motion_superpixel_labels, invert=True).nonzero()

#     background_pixels = np.zeros((mask[0].shape[0], 5), dtype=np.uint32)
#     background_pixels[:, 0] = mask[0]
#     background_pixels[:, 1] = mask[1]
#     background_pixels[:, 2:] = frame[mask[0], mask[1], :]

#     return background_pixels

def calculate_background_priors(background_labels):
    n_labels = background_labels.max() + 1

    priors = []
    sizes = []

    for label in range(n_labels):
        c_x, c_y = np.where(background_labels == label)

        priors.append(1 / c_x.shape[0])
        sizes.append(c_x.shape[0])

    priors = np.array(priors)

    return np.array(sizes), (priors/priors.sum())

def calculate_foreground_priors(foreground_labels, t_k, N_f):
    if type(t_k) != type(np.array):
        t_k = np.array(t_k)

    n_labels = foreground_labels.max() + 1

    sizes = []

    for label in range(n_labels):
        c_x, c_y = np.where(foreground_labels == label)

        sizes.append(c_x.shape[0])

    priors = 1 / (t_k + N_f)

    return np.array(sizes), (priors/priors.sum())

def calculate_unary_potential(frame, frame_superpixels, priors, models):
    # TODO: Background priors should be based on the previous frame
    n_superpixels = frame_superpixels.max() + 1

    # unary_potential_vals = np.zeros(frame.shape[:2])
    unary_potential_vals = np.zeros(n_superpixels)

    for s_idx in range(n_superpixels):
        c_x, c_y = np.where(frame_superpixels == s_idx)

        pix = frame[c_x,c_y,:]

        log_likelihood = np.log(priors) + np.array([
            np.log(np.exp(model.score_samples(pix)).mean() + 1e-6) for model in models
        ])

        # unary_potential_vals[c_x,c_y] = log_likelihood.min()
        unary_potential_vals[s_idx] = -1 * log_likelihood.max()

    return unary_potential_vals

def graph_segmentation_to_object_mask(H, W, labels, grid_segments):
    obj_mask = np.zeros((H, W))

    g_x = np.where(grid_segments == True)
    isin = np.isin(labels, g_x)

    c_x, c_y = np.where(isin == True)

    obj_mask[c_x, c_y] = 1

    return obj_mask

def calculate_adjacency_matrix(label_frame):
    H, W = label_frame.shape

    n_labels = label_frame.max() + 1

    blank_img = np.zeros_like(label_frame)

    white_lines = mark_boundaries(blank_img, label_frame, color=(1,1,1))[:,:,0]

    c_x, c_y = np.where(white_lines == 1)

    adjacency = np.zeros((n_labels, n_labels), dtype=np.bool)

    aa = [(0,1),(0,-1),(1,0),(-1,0)]

    for x,y in tqdm(zip(c_x,c_y)):
        for xx,yy in aa:
            i = x + xx
            j = y + yy 

            if i < 0 or j < 0 or i >= H or j >= W:
                continue
            
            if label_frame[x,y] == label_frame[i,j]:
                continue

            label0 = label_frame[x,y]
            label1 = label_frame[i,j]

            if adjacency[label0,label1]:
                continue

            adjacency[label0, label1] = True

    return adjacency

def calculate_kl_divergence(P, Q, samples):
    P_x = P.score_samples(samples)
    Q_x = Q.score_samples(samples)

    # print(np.exp(P_x), np.exp(Q_x))

    KL = np.exp(P_x) * (P_x - Q_x)

    return KL.sum()

def calculate_local_similarity(model_tau, PHI, n_samples=100):
    tau_samples = model_tau.sample(n_samples)[0]

    kl_divergences = np.array([
        calculate_kl_divergence(model_tau, phi, tau_samples) for phi in PHI
        ])

    # print(kl_divergences)

    # return 1 - kl_divergences.min()
    return kl_divergences.min()

def create_adjacency_edges(frame, label_frame, graph, f_mogs, b_mogs, n_samples=100):    
    PHI = f_mogs + b_mogs

    adjacency = calculate_adjacency_matrix(label_frame)

    c_x,c_y = np.where(adjacency == True)

    adjacency_models = []

    for label0,label1 in tqdm(zip(c_x,c_y)):
        union_x, union_y = np.where((label_frame == label0) | (label_frame == label1))

        union_pixels = frame[union_x, union_y, :]

        model_tau = GaussianMixture(n_components=3, init_params='random').fit(union_pixels)

        adjacency_models.append((label0,label1,model_tau))

    for l0,l1,model_tau in tqdm(adjacency_models):
        local_sim = calculate_local_similarity(model_tau, PHI, n_samples)

        graph.add_edge(l0, l1, local_sim, local_sim)


def generate_gaussians(frame, background_labels, foreground_labels, n_components=3):
    n_background = background_labels.max() + 1
    n_foreground = foreground_labels.max() + 1

    backgorund_mogs = []
    foreground_mogs = []

    for n in range(n_background):
        c_x, c_y = np.where(background_labels == n)

        color_pix = frame[c_x,c_y,:]

        mog = GaussianMixture(n_components=n_components).fit(color_pix)

        backgorund_mogs.append(mog)

    for n in range(n_foreground):
        c_x, c_y = np.where(foreground_labels == n)

        color_pix = frame[c_x,c_y,:]

        mog = GaussianMixture(n_components=n_components).fit(color_pix)

        foreground_mogs.append(mog)

    return backgorund_mogs, foreground_mogs        

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