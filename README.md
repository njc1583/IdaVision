# IdaVision

## Before you run:
1. Gather some data
2. Install the requirements in `requirements.txt`

## Stage 1:

Run `python stage_1.py --help` for more information about arguments. The only required argument is `video_filename`: which is a local path to the video to process. 

This stage will produce several outputs in the folder `output/video_filename`:
1. `segment.mp4`: the original video with boundaries marked between superpixels
2. `segment.npy`: superpixel boundaries stored in a convenient format
3. `label_frames.npy`: labeled superpixels stored in a convenient format

## Stage 2:

Run `python stage_2.py --help` for more information about arguments. The only required argument is `video_filename`: which is a local path to the video to process. 

This stage will split the video into several tasks which will be computed one at a time. Each chunk will be `T_init` frames large. After processing, several `.npy` files will be saved in the `output/video_filename/segmentations`.

As a specific example, say you process frames `0, 100` with `T_init = 10`. The frames will be processed in chunks of `[0, 10], [10, 20], ..., [90, 100]`. The files `motion_superpix_00000_000010.npy` and `segmentation_000000_000010.npy` and so on will be saved to the aforementioned folder.

**NOTE**: Running this will take a long time to complete. In general, 100 frames will take an hour to process. 

## Stage 3:

Run `python stage_3.py --help` for more information about arguments. The only required argument is `video_filename`: which is a local path to the video to process. 

This stage will take all the `.npy` files saved from the previous stage and coalesce them together. In `output/video_filename` you will find two new files: `object_segmentation.mp4`, a video of the object segmented by a green boundary marker, and `coalesce.mp4`, which contains the original video, the segmented superpixels, labeled motion superpixels, and the object segmentation. 