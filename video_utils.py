import cv2 
import numpy as np 
import ffmpeg

from segmentation import slic_segment_image

def load_video(filename, H=None, W=None):
    cap = cv2.VideoCapture(filename)

    frames = []

    while True:
        ret, bgr_frame = cap.read()

        if not ret:
            break 

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        if H is not None and W is not None:
            rgb_frame = cv2.resize(rgb_frame, (W, H))

        frames.append(rgb_frame)

    frames = np.stack(frames, axis=0)

    return frames

# Code borrowed from MP5
def vidwrite_from_numpy(fn, images, framerate=30, vcodec='libx264'):
    ''' 
      Writes a file from a numpy array of size nimages x height x width x RGB
      # source: https://github.com/kkroening/ffmpeg-python/issues/246
    '''
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n,height,width,channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame
                .astype(np.uint8)
                .tobytes()
        )
    process.stdin.close()
    process.wait()

def preprocess_video(vid_frames, tqdm, n_segments=500, compactness=10):
    F, H, W, C = vid_frames.shape 

    labels = np.zeros((F, H, W), dtype=np.uint32)

    for f in tqdm(range(F)):
        l = slic_segment_image(vid_frames[f], n_segments, compactness)

        labels[f] = l 

    return labels
