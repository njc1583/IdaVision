import matplotlib.pyplot as plt
import numpy as np

from segmentation import *

def visualize_images(images, nrows, ncols):
    N, H, W, C = images.shape 

    assert N == nrows * ncols

    fig, ax = plt.subplots(nrows, ncols)

    for i,a in enumerate(ax.ravel()):
        a.imshow(images[i])
        a.set_axis_off()

    return fig

def segment_frames(frame_imgs, frames, nrows, ncols, n_segments=500, 
                   compactness=10, mask=None):
    
    N, H, W, C = frame_imgs.shape

    assert nrows * ncols == len(frames)

    segmented_frames = np.zeros((len(frames), H, W, C), dtype=frame_imgs.dtype)

    for fidx,frame in enumerate(frames):
        segment_labels = slic_segment_image(frame_imgs[frame], n_segments, compactness, mask)

        segmented_img = get_boundary_segments(frame_imgs[frame], segment_labels)

        segmented_frames[fidx] = segmented_img

    return visualize_images(segmented_frames, nrows, ncols)