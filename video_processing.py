from VidFrame import *
from frame_utils import *

from tqdm import tqdm

def process_frames(video_frames, label_frames, start_idx, end_idx, args):
    jaccard_threshold = args['jaccard_threshold']
    N_background_segments = args['N_background_segments']
    background_compactness = args['background_compactness']
    n_samples = args['N_samples'] 
    kl_threshold = args['kl_threshold']
    match_threshold = args['match_threshold']

    vid_frames_objs = []

    if start_idx == 0:
        start_idx = 1

    for i in range(start_idx-1, end_idx):
        vid_frames_objs.append(VidFrame(video_frames[i], label_frames[i]))

    vid_frames_objs[1].calculateMotionSuperpixels(vid_frames_objs[0], jaccard_threshold)

    foreground_initialized = vid_frames_objs[1].N_foreground_regions > 0

    if foreground_initialized:
        vid_frames_objs[1].calculateMotionSuperpixelMasks()
        vid_frames_objs[1].calculateBackgroundLabels(N_background_segments, background_compactness)
        vid_frames_objs[1].calculateForegroundLabels()

        b_models, f_models, N_b, N_f = vid_frames_objs[1].generateGaussians(3)

        b_priors = calculate_background_priors(b_models)
        f_priors = calculate_foreground_priors(f_models, N_f)

        vid_frames_objs[1].createMaxflowGraph(b_models, b_priors, f_models, f_priors, 500)
    else:
        b_models, f_models = None, None
        N_b, N_f = 0, 0

    pbar = tqdm(total=end_idx - start_idx)

    vid_frames_objs[1].createObjectSegmentation()

    pbar.update(1)

    """
    1. If foreground is not initialized, calculate the foreground motion superpixels
        1a. If there is at least one motion superpixel, generate foreground models
        1b. If none, set object segmentation to all-0 mask and iterate
    2. create the max flow graph based on models
    3. Create object segmentations
    4. Update foreground and background models
    """

    for i in range(2, len(vid_frames_objs)):
        vid_frames_objs[i].calculateMotionSuperpixels(vid_frames_objs[i-1], jaccard_threshold)

        foreground_initialized = vid_frames_objs[i].N_foreground_regions > 0

        if foreground_initialized and b_models is None and f_models is None:
            vid_frames_objs[i].calculateMotionSuperpixelMasks()
            vid_frames_objs[i].calculateBackgroundLabels(N_background_segments, background_compactness)
            vid_frames_objs[i].calculateForegroundLabels()

            b_models, f_models, N_b, N_f = vid_frames_objs[i].generateGaussians(3)


        if b_models is not None and f_models is not None:
            b_priors = calculate_background_priors(b_models)
            f_priors = calculate_foreground_priors(f_models, N_f)

            vid_frames_objs[i].createMaxflowGraph(b_models, b_priors, f_models, f_priors, n_samples)

            update_background_models(vid_frames_objs[i-1], b_models)
            update_foreground_models(vid_frames_objs[i-1], f_models, kl_threshold, match_threshold) 

        vid_frames_objs[i].createObjectSegmentation()

        pbar.update(1)

    F, H, W, C = video_frames.shape

    object_segmentation_masks = np.zeros((end_idx-start_idx,H,W), dtype=np.uint8)
    colored_motion_superpixel_imgs = np.zeros((end_idx-start_idx,H,W,C), dtype=np.uint8)

    for i in range(end_idx-start_idx):
        object_segmentation_masks[i] = vid_frames_objs[i+1].object_mask
        colored_motion_superpixel_imgs[i] = vid_frames_objs[i+1].colorMotionSuperpixels()

    return object_segmentation_masks,colored_motion_superpixel_imgs