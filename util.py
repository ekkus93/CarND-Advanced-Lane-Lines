import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import glob
from scipy.misc import imsave

def display_images(X, start_idx=0, end_idx=None,  step_val=1, 
                   columns = 5, use_gray=False, 
                   apply_fnc=None, figsize=(32,18)):
    """
    Display a set of images
    Parameters
    ----------
    X: numpy array or list of images
         Images to be displayed
    start_idx: int
         Start index for images
    end_idx: int
         End index for images
    step_val: int
        step value
    columns: int
         Number of columns of images
    use_gray: bool
         True for RGB images.  False for grayscale images.
    apply_fnc: function
         An function to apply to each image before displaying.
    figsize: tuple of int
         Display height and width of images.
    """
    if end_idx is None:
        if isinstance(X, (list,)):
            end_idx = len(X)
        else:
            end_idx = X.shape[0]

    if apply_fnc is None:
        apply_fnc = lambda image: image
        
    plt.figure(figsize=figsize)

    num_of_images = end_idx - start_idx
    rows = num_of_images / columns + 1
    
    cnt = 0
    for i in range(start_idx, end_idx, step_val):
        cnt += 1
        image = X[i]
        
        plt.subplot(rows, columns, cnt)
        
        if use_gray:
            plt.imshow(apply_fnc(image), cmap="gray")
        else:
            plt.imshow(apply_fnc(image)) 
            
    plt.tight_layout()
            
    plt.show()

def read_imgs(file_names):
    """
    Read list of images from disk.
    Parameters
    ----------
    file_names: list of str
         List of image file names.
    Returns
    -------
    numpy array of images:
         Images from disk.
    """
    img_arr = []
    
    for file_name in file_names:
        img = imread(file_name)
        img_arr.append(img)
        
    return np.stack(img_arr)

def pad_zeros(val, num_of_zeros=7):
    pad_str = '{:0>%d}' % num_of_zeros
    return pad_str.format(val)

def extract_video_imgs(video, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    vidcap = cv2.VideoCapture(video)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    success,image = vidcap.read()
    for i in tqdm(range(length-1)):
        if success:
            success, image = vidcap.read()
        
            if success:
                frame_file = "%s/frame_%s.jpg" % (output_dir, pad_zeros(i))
                cv2.imwrite(frame_file, image)     # save frame as JPEG file
        else:
            print("WARNING: frame #%d could not be read. Stopping.")
            break

