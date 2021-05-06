import argparse
from frame_utils import get_boundary_segments
import numpy as np

from video_utils import load_video, mark_video_segments, vidwrite_from_numpy

from ida_constants import *
from io_utils import *

from tqdm import tqdm 

import os
from os import listdir
from os.path import isfile
from os.path import join as pjoin


def segment_objects(vid_frames, object_masks):
    F, H, W = object_masks.shape

    out = np.zeros((F, H, W, 3), dtype=np.uint8)

    for f in tqdm(range(F)):
        out[f] = get_boundary_segments(vid_frames[f], object_masks[f], color=(0, 1, 0))

    return out

def coalesce_videos(vid_frames, superpix_segment_frames, motion_superpixel_frames, object_segment_frames):
    assert len(vid_frames.shape) == 4
    assert len(superpix_segment_frames.shape) == 4
    assert len(motion_superpixel_frames.shape) == 4
    assert len(object_segment_frames.shape) == 4

    F, H, W, C = object_segment_frames.shape

    out = np.zeros((F, H*2, W*2, C), dtype=np.uint8)

    for f in tqdm(range(F)):
        out[f, :H, :W] = vid_frames[f]
        out[f, :H, W:] = superpix_segment_frames[f]
        out[f, H:, :W] = motion_superpixel_frames[f]
        out[f, H:, W:] = object_segment_frames[f]

    return out

def load_segmentations_info(directory, prefix, H, W, C):
    only_files = [f for f in listdir(directory) if isfile(pjoin(directory, f))]  

    prefix_files = list(filter(lambda f: f.startswith(prefix), only_files))
    prefix_files = sorted(prefix_files)

    idx_info = []

    for f in prefix_files:
        split = f.split('_')

        start_idx = int(split[-2])
        end_idx = int(split[-1].split('.')[0])

        idx_info.append((start_idx, end_idx))

    F = max([x[1] for x in idx_info])
    
    if C != 1:
        out = np.zeros((F, H, W, C), dtype=np.uint8)
    else:
        out = np.zeros((F, H, W), dtype=np.uint8)

    for (i,j),f in zip(idx_info,prefix_files):
        a = np.load(directory + '/' + f)

        if i == 0:
            out[i+1:j] = a
        else:
            out[i:j] = a

    return out

 
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_filename', '-vf', type=str, required=True, help='The original video to process')

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = parse_args()

    make_directory(OUTPUT_DIR)

    output_foldername = get_output_prefix(args['video_filename'])
    output_foldername = pjoin(OUTPUT_DIR, output_foldername)

    make_directory(output_foldername)

    print('Loading video...')

    try:
        segment_frames = np.load(pjoin(output_foldername, 'segment.npy'))
    except Exception as e:
        print(e)
        exit()

    F, H, W, C = segment_frames.shape

    try:
        vid_frames = load_video(filename=args['video_filename'], H=H, W=W)
    except Exception as e:
        print(e)
        exit()


    colored_motion_superpixels = load_segmentations_info(pjoin(output_foldername, 'segmentations'), 'motion_superpix', H, W, 3)
    object_segmentations = load_segmentations_info(pjoin(output_foldername, 'segmentations'), 'segmentation', H, W, 1)

    print('Segmentating objects...')

    segmented_video = segment_objects(vid_frames, object_segmentations)

    out_video = coalesce_videos(vid_frames, segment_frames, colored_motion_superpixels, segmented_video)

    vidwrite_from_numpy(pjoin(output_foldername, 'object_segmentation.mp4'), segmented_video)
    vidwrite_from_numpy(pjoin(output_foldername, 'coalesce.mp4'), out_video)