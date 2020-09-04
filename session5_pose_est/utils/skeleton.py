import cv2
import numpy as np
from operator import itemgetter
import PIL
from PIL import Image

# 16 64X64 images output for joints are in these order for Resnet50 model
JOINT_NAMES = ['R-Ankle', 'R-Knee', 'R-Hip', 'L-Hip', 'L-Knee', 'L-Ankle', 'Pelvis', 'Thorax', 'Upperneck', 'Head', 'R-Wrist', 'R-Elbow', 'R-Shoulder', 'L-Shoulder', 'L-Elbow', 'L-Wrist']

# Based on Joint position as in JOINT_NAMES, these are the pair we need to join
JOINT_IDX_PAIRS = [[0,1], [1,2], [2,6],      # R-ankel -> R-Knee, R-Knee -> R-Hip, R-Hip -> Pelvis
                   [5,4], [4,3], [3,6],      # L-ankel -> L-Knee, L-Knee -> L-Hip, L-Hip -> Pelvis
                   [6,7], [7,8], [8,9],      # Pelvis -> Thorax, Thorax -> Upperneck, Upperneck -> Head
                   [7,12], [12,11], [11,10], # Thorax -> R-Shoulder, R-Shoulder -> R-Elbow, R-elbow -> R-Wrist
                   [7,13], [13,14], [14,15]] # Thorax -> L-Shoulder, L-Shoulder -> L-Elbow, L-elbow -> L-Wrist

'''
img: PIL
output: Numpy image with HPE Joints linked and plotted (numpy array)
'''
def get_skeleton(img:PIL, output):
    img = np.array(img)

    '''
    Get the x,y coordinate from joint image of size 64X64. Pick the position of max value
    Get pixel value and x,y corordinates: (val, (x,y))
    '''
    get_joint_location = lambda joints: map(itemgetter(1, 3), [cv2.minMaxLoc(joint) for joint in joints])
    joint_locations = list(get_joint_location(output))
        
    IMG_H, IMG_W, _ = img.shape         # Input Image Dimension
    OUT_W, OUT_H = output[0].shape      # Joints Image Dimension

    # Lambda fxn to get relative x,y, position based on image size
    get_X = lambda x: int(x * IMG_W / OUT_W)
    get_Y = lambda y: int(y * IMG_H / OUT_H)

    # Mark all joint positions on the original image
    color = (0, 0, 255)
    thickness = 2
    for _ , (x,y) in joint_locations:
        # Get relative x,y, position based on image size
        x, y = get_X(x), get_Y(y)
        cv2.ellipse(img, (x, y), (4, 4), 0, 0, 360, color, thickness) #cv2.FILLED)

    # iterate through Joint pair and draw line between two joint positions
    for start, end in JOINT_IDX_PAIRS:
        _ , (x1,y1) = joint_locations[start]
        _ , (x2,y2) = joint_locations[end]

        # Get relative x,y, position based on image size
        x1, y1 = get_X(x1), get_Y(y1)
        x2, y2 = get_X(x2), get_Y(y2)

        # Join the two points by drawing line
        color = (0, 255, 0)        
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
            
    return img
