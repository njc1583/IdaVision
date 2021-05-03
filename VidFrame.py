import numpy as np

import cv2
import maxflow

from skimage.segmentation import mark_boundaries,find_boundaries
from sklearn.mixture import GaussianMixture

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from frame_utils import *

import numpy as np

import cv2

from GaussianModel import *

class VidFrame():
    def __init__(self, vid_frame: np.ndarray, label_frame: np.ndarray, memory_save_mode:bool=False):
        """
        A VidFrame object

        @param vid_frame: the unprocessed video frame
        @param label_frame: the result of running slic on vid_frame
        @param memory_save_mode: if True, the instance of VidFrame will
            recalculate values only when absolutely necessary 
        """

        self.memory_save_mode = memory_save_mode

        self.vid_frame = vid_frame
        self.label_frame = label_frame

        self.N_superpixels = label_frame.max() + 1

        self.adjacency = self.calculateAdjacencyMatrix()

    def getSegmentedImage(self):
        """
        Returns a VidFrame's video frame by the frame labels
        """
        return img_as_ubyte(mark_boundaries(self.vid_frame, self.label_frame))

    def colorMotionSuperpixels(self):
        """
        Returns an image where motion superpixels are filled with the 
            color green
        """
        if not hasattr(self, 'motion_superpixel_labels'):
            return None

        boundary_img = self.getSegmentedImage()

        motion_superpix_img = np.copy(boundary_img)

        for label in self.motion_superpixel_labels:
            label_pix_x, label_pix_y = np.where(self.label_frame == label) 
                
            motion_superpix_img[label_pix_x,label_pix_y,:] = np.array([0,255,0])

        return motion_superpix_img


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

    def calculateMotionSuperpixels(self, prev_VidFrame, jaccard_threshold: float):
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
        
        motion_superpixel_adjacency = self.calculateLabeledAdjacency(initial_motion_superpixel_labels)

        graph = csr_matrix(motion_superpixel_adjacency)
        _, component_labels = connected_components(graph)

        c, component_counts = np.unique(component_labels, return_counts=True) 

        labels = c[component_counts > 2]

        self.N_foreground_regions = labels.shape[0]

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

    def calculateBackgroundLabels(self, N_background_segments: int, background_compactness: int):
        """
        Computes further superpixels for superpixels already labeled
        as background, making them ready for statistical modeling
        """
        if self.memory_save_mode and hasattr(self, 'background_superpixels'):
            return self.background_superpixels

        if not hasattr(self, 'motion_superpixel_background_mask'):
            self.calculateMotionSuperpixelMasks()

        self.background_superpixels = slic_segment_image(self.vid_frame, n_segments=N_background_segments, compactness=background_compactness, mask=self.motion_superpixel_background_mask)

        self.N_background_labels = self.background_superpixels.max() + 1

        return self.background_superpixels

    def calculateForegroundLabels(self):
        """
        Computes the labels for the foreground regions; namely
        calculates the number of foreground objects (before labeling) 
        and stores that information into a labeled image
        """
        if self.memory_save_mode and hasattr(self, 'N_foreground_regions') and hasattr(self, 'foreground_labels'):
            return self.N_foreground_regions, self.foreground_labels

        if not hasattr(self, 'motion_superpixel_foreground_mask'):
            self.calculateMotionSuperpixelMasks()

        N_foreground_labels, foreground_labels = get_num_foreground_regions(self.motion_superpixel_foreground_mask)

        self.N_foreground_regions = N_foreground_labels
        self.foreground_labels = foreground_labels

        return self.N_foreground_regions, self.foreground_labels

    def generateGaussians(self, n_components:int=3):
        if not hasattr(self, 'N_background_labels') or not hasattr(self, 'background_superpixels') or not hasattr(self, 'N_foreground_regions') or not hasattr(self, 'foreground_labels'):
            return None, None, None, None

        backgorund_mogs = []
        foreground_mogs = []

        for n in range(self.N_background_labels):
            c_x, c_y = np.where(self.background_superpixels == n)

            color_pix = self.vid_frame[c_x,c_y,:]

            mog = GaussianModel(color_pix, n_components)

            backgorund_mogs.append(mog)

        for n in range(self.N_foreground_regions):
            c_x, c_y = np.where(self.foreground_labels == n)

            color_pix = self.vid_frame[c_x,c_y,:]

            mog = GaussianModel(color_pix, n_components)

            foreground_mogs.append(mog)

        return backgorund_mogs, foreground_mogs, len(backgorund_mogs), len(foreground_mogs)

    def calculateUnaryPotential(self, models, priors):
        if not hasattr(self, 'N_superpixels'):
            return None

        unary_potential_vals = np.zeros(self.N_superpixels)

        for s_idx in range(self.N_superpixels):
            c_x, c_y = np.where(self.label_frame == s_idx)

            pix = self.vid_frame[c_x,c_y,:]

            log_likelihood = np.log(priors) + np.array([
                np.log(np.exp(model.score_samples(pix)).mean() + 1e-8) for model in models
            ])

            unary_potential_vals[s_idx] = -1 * log_likelihood.max()

        return unary_potential_vals

    def calculateAdjacencyModels(self):
        c_x,c_y = np.where(self.adjacency == True)

        adjacency_models = []

        for label0,label1 in zip(c_x,c_y):
            if label0 > label1:
                continue

            union_x, union_y = np.where((self.label_frame == label0) | (self.label_frame == label1))

            union_pixels = self.vid_frame[union_x, union_y, :]

            model_tau = GaussianMixture(n_components=3, init_params='random').fit(union_pixels)

            adjacency_models.append((label0,label1,model_tau))

        return adjacency_models

    def createMaxflowGraph(self, b_models, b_priors, f_models, f_priors, n_samples=500):
        if not hasattr(self, 'N_superpixels'):
            return

        graph = maxflow.Graph[float]()

        self.nodeids = graph.add_grid_nodes(self.N_superpixels)

        unary_background = self.calculateUnaryPotential(b_models, b_priors)
        unary_foreground = self.calculateUnaryPotential(f_models, f_priors)

        graph.add_grid_tedges(self.nodeids, unary_foreground, unary_background)

        adjacency_models = self.calculateAdjacencyModels()

        PHI = b_models + f_models

        for l0,l1,model_tau in adjacency_models:
            local_sim = calculate_local_similarity(model_tau, PHI, n_samples)
            
            graph.add_edge(l0, l1, local_sim, local_sim)

        self.maxflow_graph = graph 

    
    def createObjectSegmentationMask(self, grid_segments):
        H, W = self.vid_frame.shape[:2]

        obj_mask = np.zeros((H, W), dtype=np.uint8)

        g_x = np.where(grid_segments == True)
        isin = np.isin(self.label_frame, g_x)

        c_x, c_y = np.where(isin == True)

        obj_mask[c_x, c_y] = 1

        return obj_mask


    def createObjectSegmentation(self):
        if not hasattr(self, 'N_foreground_regions'):
            return -1,None

        if self.N_foreground_regions == 0:
            self.N_objects = 0
            
            H, W, C = self.vid_frame.shape

            self.object_mask = np.zeros((H, W), np.uint8)
            self.object_labels = self.object_mask - 1

            return 0,self.object_mask

        maxflow_val = self.maxflow_graph.maxflow()

        maxflow_segmentation = self.maxflow_graph.get_grid_segments(self.nodeids)

        self.object_mask = self.createObjectSegmentationMask(maxflow_segmentation)

        self.N_objects, self.object_labels = get_num_foreground_regions(self.object_mask)

        return maxflow_val,self.object_mask


    def getBackgroundPixels(self):
        c_x, c_y = np.where(self.object_mask == 0)

        return self.vid_frame[c_x, c_y]

    def getForegroundPixels(self):

        foreground_pixels = []

        for i in range(self.N_objects):
            c_x, c_y = np.where(self.object_labels == i)
            foreground_pixels.append(self.vid_frame[c_x, c_y])

        return foreground_pixels
