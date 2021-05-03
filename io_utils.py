import os

OUTPUT_DIR = './output'

def make_directory(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    
def get_output_prefix(vid_fname):
    folders = vid_fname.split('/')
    fname = folders[-1].split('.')

    return fname[0]

