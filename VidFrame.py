import numpy as np

import cv2
import maxflow

from skimage.segmentation import mark_boundaries,find_boundaries
from sklearn.mixture import GaussianMixture

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from VidFrame_utils import *

import numpy as np

import cv2

class VidFrame():
    def __init__(self, vid_frame, label_frame, memory_save_mode=False):
        self.memory_save_mode = memory_save_mode # If True, no values will be re-computed

        self.vid_frame = vid_frame
        self.label_frame = label_frame

        self.N_superpixels = label_frame.max() + 1

        self.adjacency = self.calculateAdjacencyMatrix()

    def calculateAdjacencyMatrix(self):
        """
        Calculates adjacency matrix A for labels of the video frame
        where A_ij = 1 iff i != j and i is adjacent to label j

        A is a symmetric matrix
        """
        if self.memory_save_mode and hasattr(self, 'adjacency'):
            return self.adjacency

        H, W = self.label_frame.shape

        boundaries = find_boundaries(self.label_frame)

        c_x, c_y = np.where(boundaries == True)

        adjacency = np.zeros((self.N_superpixels, self.N_superpixels), dtype=np.bool)

        aa = [(0,1),(0,-1),(1,0),(-1,0)]

        for x,y in zip(c_x,c_y):
            for xx,yy in aa:
                i = x + xx
                j = y + yy 

                if i < 0 or j < 0 or i >= H or j >= W:
                    continue
                
                if not boundaries[x,y] or not boundaries[i,j]:
                    continue

                if self.label_frame[x,y] == self.label_frame[i,j]:
                    continue

                label0 = self.label_frame[x,y]
                label1 = self.label_frame[i,j]

                if adjacency[label0,label1]:
                    continue

                adjacency[label0, label1] = True
                adjacency[label1, label0] = True

        self.adjacency = adjacency
        return self.adjacency

    def calculateLabeledAdjacency(self, labels):
        """
        Given a set of labels, computes an adjacency matrix B
        where B_ij = 1 iff i != j and i is adjacent to j and both i and j are labels
        """
        if self.memory_save_mode and hasattr(self, 'motion_superpixel_adjacency'):
            return self.motion_superpixel_adjacency
        
        if not hasattr(self, 'adjacency'):
            self.adjacency = self.calculateAdjacencyMatrix()
        
        motion_superpixel_adjacency = np.zeros_like(self.adjacency, dtype=np.bool)

        for i in range(len(labels)-1):
            for j in range(i+1, len(labels)):
                label0, label1 = labels[i], labels[j]
                
                motion_superpixel_adjacency[label0,label1] = self.adjacency[label0,label1]
                motion_superpixel_adjacency[label1,label0] = self.adjacency[label1,label0]

        self.motion_superpixel_adjacency = motion_superpixel_adjacency

        return self.motion_superpixel_adjacency

    def calculateMotionSuperpixels(self, prev_VidFrame, jaccard_threshold):
        """
        Calculates the motion superpixels based on the previous labeled image, and
        the jaccard threshold

        Performs these steps:
        1. Calculate initial motion superpixels based on jaccard threshold
        2. Computes adjacency matrix for initial motion superpixels
        3. Removes superpixel groups of size 2 or less
        4. Saves remaining superpixel labels and returns them
        """
        if self.memory_save_mode and hasattr(self, 'motion_superpixel_labels'):
            return self.motion_superpixel_labels

        initial_motion_superpixel_labels = get_jaccard_motion_superpixels(prev_VidFrame.label_frame, self.label_frame, jaccard_threshold)
        
        # print(f'initial_motion_superpixel_labels: {initial_motion_superpixel_labels}')

        motion_superpixel_adjacency = self.calculateLabeledAdjacency(initial_motion_superpixel_labels)

        graph = csr_matrix(motion_superpixel_adjacency)
        _, component_labels = connected_components(graph)

        c, component_counts = np.unique(component_labels, return_counts=True) 

        labels = c[component_counts > 2]
        labels = np.isin(component_labels, labels)
        labels = np.arange(self.N_superpixels)[labels]

        self.motion_superpixel_labels = labels
        return self.motion_superpixel_labels

    def calculateMotionSuperpixelMasks(self):
        """
        Returns {0,1} masks motion_superpixel_foreground_mask and motion_superpixel_background_mask
        """
        if self.memory_save_mode and hasattr(self, 'motion_superpixel_foreground_mask') and hasattr(self, 'motion_superpixel_background_mask'):
            return self.motion_superpixel_foreground_mask, self.motion_superpixel_background_mask

        if not hasattr(self, 'motion_superpixel_labels'):
            raise Exception('Cannot compute motion superpixel masks w/o having motion superpixel labels')

        H, W, C = self.vid_frame.shape

        foreground_mask = np.zeros((H, W), dtype=np.uint8)

        for label in self.motion_superpixel_labels:
            label_pix_x, label_pix_y = np.where(self.label_frame == label)

            foreground_mask[label_pix_x, label_pix_y] = 1

        background_mask = (1 - foreground_mask).astype(np.uint8)

        self.motion_superpixel_foreground_mask = foreground_mask
        self.motion_superpixel_background_mask = background_mask

        return self.motion_superpixel_foreground_mask, self.motion_superpixel_background_mask