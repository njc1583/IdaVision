import argparse
import numpy as np

from video_utils import load_video
from video_processing import process_frames

from ida_constants import *
from io_utils import *

from tqdm import tqdm 

from os.path import join as pjoin

def get_processing_jobs(start_frame, end_frame, T_init):
    F = end_frame - start_frame

    processing_jobs = []

    for f in range(F // T_init):
        job = [start_frame + (T_init * f), start_frame + (T_init * (f+1))]

        processing_jobs.append(job)

    if len(processing_jobs) == 0:
        processing_jobs = [[start_frame, end_frame]]
    else:
        processing_jobs[-1][-1] = end_frame

    return processing_jobs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_filename', '-vf', type=str, required=True, help='The original video to process')
    
    parser.add_argument('--T_init', type=int, default=T_INIT_DEFAULT, help='Number of frames to process in one chunk; after every T_init frames, the Gaussian models are re-initialized')
    parser.add_argument('--kl_threshold', type=float, default=KL_THRESHOLD_DEFAULT, help='When updating foreground models, superpixels which have a KL-divergence above this are considered to match to an object')
    parser.add_argument('--match_threshold', '-mt', type=int, default=MATCH_THRESHOLD_DEFAULT, help='Number of frames that a foreground model can lack a matched object before it is removed when updating')

    parser.add_argument('--N_background_segments', '-Nbs', type=int, default=N_BACKGROUND_SEGMENTS_DEFAULT, help='Number of background superpixels to model when generating Gaussian models')
    parser.add_argument('--background_compactness', '-bc', type=int, default=BACKGROUND_COMPACTNESS_DEFAULT, help='The background compactness when generating Gaussian models')
    parser.add_argument('--N_samples', '-ns', type=int, default=N_SAMPLES, help='The number of samples to use when performing KL-divergence. In general, a higher number will increase processing time significantly.')

    parser.add_argument('--start_frame', '-sf', type=int, default=0, help='The first frame to process in the video')
    parser.add_argument('--end_frame', '-ef', type=int, default=-1, help='The last frame of the video to process')

    parser.add_argument('--jaccard_threshold', '-jt', type=float, default=JACCARD_THRESHOLD_DEFAULT, help='threshold for a superpixel to be considered a motion superpixel')

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = parse_args()

    make_directory(OUTPUT_DIR)

    output_foldername = get_output_prefix(args['video_filename'])
    output_foldername = pjoin(OUTPUT_DIR, output_foldername)

    make_directory(output_foldername)

    print('Loading video info...')

    try:
        label_frames = np.load(pjoin(output_foldername, 'label_frames.npy'))
    except Exception as e:
        print(e)
        exit()

    F, H, W = label_frames.shape

    try:
        vid_frames = load_video(filename=args['video_filename'], H=H, W=W)
    except Exception as e:
        print(e)
        exit()

    output_foldername = pjoin(output_foldername, 'segmentations')

    make_directory(output_foldername)

    start_frame = args['start_frame']
    end_frame = args['end_frame']

    if end_frame == -1:
        end_frame = F

    T_init = args['T_init']

    processing_jobs = get_processing_jobs(start_frame, end_frame, T_init)

    print('Generating object segmentation masks...')

    obj_seg_fname_format = output_foldername + '/segmentation_%06d_%06d.npy'
    color_fname_format = output_foldername + '/motion_superpix_%06d_%06d.npy'

    for job_start,job_end in tqdm(processing_jobs):
        object_segmentation_masks,colored_motion_superpixel_imgs = process_frames(vid_frames, label_frames, job_start, job_end, args)

        obj_seg_fname = obj_seg_fname_format % (job_start, job_end)
        color_fname = color_fname_format % (job_start, job_end)

        np.save(obj_seg_fname, object_segmentation_masks)
        np.save(color_fname, colored_motion_superpixel_imgs)