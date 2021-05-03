import argparse
import numpy as np

from video_utils import load_video
from video_processing import process_frames

from ida_constants import *
from io_utils import *

from tqdm import tqdm 

def get_processing_jobs(start_frame, end_frame, T_init):
    F = end_frame - start_frame

    processing_jobs = []

    for f in range(F // T_init):
        job = [T_init * f, T_init * (f+1)]

        processing_jobs.append(job)

    if len(processing_jobs) == 0:
        processing_jobs = [[start_frame, end_frame]]
    else:
        processing_jobs[-1][-1] = F

    return processing_jobs

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_filename', '-vf', type=str, required=True)
    
    parser.add_argument('--T_init', type=int, default=T_INIT)
    parser.add_argument('--kl_threshold', type=float, default=KL_THRESHOLD)
    parser.add_argument('--match_threshold', '-mt', type=int, default=MATCH_THRESHOLD)

    parser.add_argument('--N_background_segments', '-Nbs', type=int, default=N_BACKGROUND_SEGMENTS)
    parser.add_argument('--background_compactness', '-bc', type=int, default=BACKGROUND_COMPACTNESS)
    parser.add_argument('--N_samples', '-ns', type=int, default=N_SAMPLES)

    parser.add_argument('--start_frame', '-sf', type=int, default=0)
    parser.add_argument('--end_frame', '-ef', type=int, default=-1)

    parser.add_argument('--jaccard_threshold', '-jt', type=float, default=JACCARD_THRESHOLD)

    args = parser.parse_args()

    return vars(args)

if __name__ == '__main__':
    args = parse_args()

    print(args)

    make_directory(OUTPUT_DIR)

    output_foldername = get_output_prefix(args['video_filename'])
    output_foldername = f'{OUTPUT_DIR}/{output_foldername}'

    make_directory(output_foldername)

    print('Loading video info...')

    try:
        label_frames = np.load(f'{output_foldername}/label_frames.npy')
    except Exception as e:
        print(e)
        exit()

    F, H, W = label_frames.shape

    try:
        vid_frames = load_video(filename=args['video_filename'], H=H, W=W)
    except Exception as e:
        print(e)
        exit()

    output_foldername += '/segmentations'

    start_frame = args['start_frame']
    end_frame = args['end_frame']

    if end_frame == -1:
        end_frame = F

    T_init = args['T_init']

    processing_jobs = get_processing_jobs(start_frame, end_frame, T_init)

    print(processing_jobs)

    print('Generating object segmentation masks...')

    obj_seg_fname_format = output_foldername + '/segmentation_%06d_%06d.npy'
    color_fname_format = output_foldername + '/motion_superpix_%06d_%06d.npy'

    for job_start,job_end in tqdm(processing_jobs):
        object_segmentation_masks,colored_motion_superpixel_imgs = process_frames(vid_frames, label_frames, job_start, job_end, args)

        obj_seg_fname = obj_seg_fname_format % (job_start, job_end)
        color_fname = color_fname_format % (job_start, job_end)

        np.save(obj_seg_fname, object_segmentation_masks)
        np.save(color_fname, colored_motion_superpixel_imgs)