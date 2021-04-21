import matplotlib.pyplot as plt
import numpy as np

import cv2

from tqdm import tqdm

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

def segment_image(img, n_segments=500, compactness=10):
    H, W, C = img.shape 

    img = img_as_float(img)

    labels = slic(img, n_segments=n_segments, start_label=0,  compactness=compactness, multichannel=True)

    return labels

def get_segmented_image(img, labels):
    return mark_boundaries(img, labels)

def segment_frames(frame_imgs, frames, nrows, ncols, n_segments=500, compactness=10):
    assert nrows * ncols == len(frames)

    fig, ax = plt.subplots(nrows, ncols, figsize=(5*nrows, 5*ncols))

    for fidx,frame in enumerate(frames):
        labels = segment_image(frame_imgs[frame], n_segments, compactness)

        segmented_image = get_segmented_image(frame_imgs[frame], labels)

        row = fidx // ncols 
        col = fidx % ncols

        ax[row, col].imshow(segmented_image)

    for a in ax.ravel():
        a.set_axis_off()

    return fig, ax

def preprocess_video(vid_frames, tqdm, n_segments=500, compactness=10):
    F, H, W, C = vid_frames.shape 

    labels = np.zeros((F, H, W), dtype=np.uint32)

    for f in tqdm(range(F)):
        l = segment_image(vid_frames[f], n_segments, compactness)

        labels[f] = l 

    return labels
