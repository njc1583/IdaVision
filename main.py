from VidFrame import *
from frame_utils import *

from tqdm.notebook import tqdm

N_BACKGROUND_SEGMENTS = 25
BACKGROUND_COMPACTNESS = 10
JACCARD_THRESHOLD = 0.3


def process_frames(video_frames, label_frames, start_idx, end_idx):
    vid_frames = []

    for i in range(start_idx-1, end_idx):
        vid_frames.append(VidFrame(video_frames[i], label_frames[i]))

    vid_frames[1].calculateMotionSuperpixels(vid_frames[0], JACCARD_THRESHOLD)

    foreground_initialized = vid_frames[1].N_foreground_regions > 0

    if foreground_initialized:
        vid_frames[1].calculateMotionSuperpixelMasks()
        vid_frames[1].calculateBackgroundLabels(N_BACKGROUND_SEGMENTS, BACKGROUND_COMPACTNESS)
        vid_frames[1].calculateForegroundLabels()

        b_models, f_models, N_b, N_f = vid_frames[1].generateGaussians(3)

        b_priors = calculate_background_priors(b_models)
        f_priors = calculate_foreground_priors(f_models, N_f)

        vid_frames[1].createMaxflowGraph(b_models, b_priors, f_models, f_priors, 500)
    else:
        b_models, f_models = None, None
        N_b, N_f = 0, 0

    pbar = tqdm(total=end_idx - start_idx)

    vid_frames[1].createObjectSegmentation()

    pbar.update(1)

    """
    1. If foreground is not initialized, calculate the foreground motion superpixels
        1a. If there is at least one motion superpixel, generate foreground models
        1b. If none, set object segmentation to all-0 mask and iterate
    2. create the max flow graph based on models
    3. Create object segmentations
    4. Update foreground and background models
    """

    for i in range(2, len(vid_frames)):
        vid_frames[i].calculateMotionSuperpixels(vid_frames[i-1], JACCARD_THRESHOLD)

        foreground_initialized = vid_frames[i].N_foreground_regions > 0

        if foreground_initialized and b_models is None and f_models is None:
            vid_frames[i].calculateMotionSuperpixelMasks()
            vid_frames[i].calculateBackgroundLabels(N_BACKGROUND_SEGMENTS, BACKGROUND_COMPACTNESS)
            vid_frames[i].calculateForegroundLabels()

            b_models, f_models, N_b, N_f = vid_frames[i].generateGaussians(3)


        if b_models is not None and f_models is not None:
            b_priors = calculate_background_priors(b_models)
            f_priors = calculate_foreground_priors(f_models, N_f)

            vid_frames[i].createMaxflowGraph(b_models, b_priors, f_models, f_priors, 500)

            update_background_models(vid_frames[i-1], b_models)
            update_foreground_models(vid_frames[i-1], f_models, 0.8, 2) 

        vid_frames[i].createObjectSegmentation()

        pbar.update(1)
