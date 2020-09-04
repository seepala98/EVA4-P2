import cv2
import numpy as np
from operator import itemgetter

import PIL
from PIL import Image
import matplotlib.pyplot as plt

from utils.skeleton import get_skeleton, JOINT_NAMES 

def vis_joints(img:PIL, output, save_filename=None):
    fig = plt.figure(figsize=(16, 16))
    for idx, joint_data in enumerate(output):
        ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
        plt.title(f'{idx}: {JOINT_NAMES[idx]}')
        H, W = joint_data.shape
        plt.imshow(img.resize((W, H)), cmap='gray', interpolation='bicubic')
        plt.imshow(joint_data, alpha=0.5, cmap='jet', )
        plt.axis('off')
    
    if save_filename:
      fig.savefig(save_filename)
    return

def vis_all_joints(img:PIL, output, figsize=(8, 8), save_filename=None):
    fig = plt.figure(figsize=figsize)
    H, W = output[0].shape
    plt.imshow(img.resize((W, H)), cmap='gray', interpolation='bicubic')
    joints = np.clip(output, 0.4, 1.0)
    joint_sum = np.sum(output, axis=0) # stacking all layers 
    plt.imshow(joint_sum, alpha=0.5, cmap='jet', interpolation='bicubic')
    plt.axis('off')
    
    if save_filename:
      fig.savefig(save_filename)
    return

def vis_skeleton(img:PIL, output, figsize=(16,8), save_filename=None):
    hpe_img = get_skeleton(img, output)
    fig = plt.figure(figsize=figsize); 

    # input image
    ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
    ax.set_title("Input Image")
    plt.imshow(img)

    # hpe image
    ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
    ax.set_title("HPE Image")
    plt.imshow(hpe_img)

    if save_filename:
      fig.savefig(save_filename)
    return
