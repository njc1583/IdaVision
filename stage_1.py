import argparse
import numpy as np

from video_utils import load_video, mark_video_segments, segment_video, vidwrite_from_numpy

from ida_constants import *
from io_utils import *

from tqdm import tqdm 

from os.path import join as pjoin

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_filename', '-vf', type=str, required=True, help='The filename of the original video')
    
    parser.add_argument('--N_clusters', '-Nc', type=int, default=N_CLUSTERS_DEFAULT, help='The number of clusters for superpixel segmentation')
    parser.add_argument('--compactness', '-c', type=float, default=COMPACTNESS_DEFAULT, help='Compactness of superpixels')
    
    parser.add_argument('--H', '-H', type=int, default=H_DEFAULT, help='Height of video to process')
    parser.add_argument('--W', '-W', type=int, default=W_DEFAULT, help='Width of video to process')

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
        vid_frames = load_video(filename=args['video_filename'], H=args['H'], W=args['W'])
    except Exception as e:
        print(e)
        exit()

    print('Segmenting video...')

    try:
        label_frames = segment_video(vid_frames, tqdm, n_segments=args['N_clusters'], compactness=args['compactness'])
    except Exception as e:
        print(e)
        exit()

    print('Marking segment boundaries...')

    try:
        segmented_video = mark_video_segments(vid_frames, label_frames, tqdm)
    except Exception as e:
        print(e)
        exit()

    print('Saving information...')

    np.save(pjoin(output_foldername, 'label_frames.npy'), label_frames)
    np.save(pjoin(output_foldername, 'segment.npy'), segmented_video)
    vidwrite_from_numpy(pjoin(output_foldername, 'segment.mp4'), segmented_video)