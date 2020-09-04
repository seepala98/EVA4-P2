import numpy as np
import PIL

def img_normalize(img, means, stds):
    """
    Args:
        Numpy : Image of size (C, H, W) to be normalized.
    Returns:
        Numpy: Normalized image.
    """
    for i in range(3):
        img[i] = np.divide(np.subtract(img[i], means[i]), stds[i])
        img[i] = np.nan_to_num(img[i])
    return img

def HWC_2_CHW(img):
    H, W, C = img.shape 
    im = np.zeros((C,H,W),dtype=np.float32)
    for i in range(C):
      im[i] = img[:,:,i]      
    return im

# img: PIL Image data
def transform_image(img:PIL):
    img = img.resize((256,256),  PIL.Image.BICUBIC) # resizing image
    img = np.asarray(img)

    # converting from HWC to CHW format
    img_chw = HWC_2_CHW(img)

    # Convert image to floating point in the range 0 to 1
    img_chw = np.float32(img_chw)/255.0

    # Normalizing image data
    means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    img_chw = img_normalize(img_chw, means, stds)

    img_chw = np.expand_dims(img_chw, axis=0) # Making batch size of 1
    
    return img_chw
