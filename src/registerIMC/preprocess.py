import cv2 as cv
import numpy as np


def preprocess_IF(IF, scale, rotation_angle = None):
     """
     Preprocesses an IF reference image, including resizing, rotation (if necessary), and binarisation.
        
     Args:
         IF (numpy.ndarray): The input image.
         scale (float): The scaling factor for resizing the image.
         rotation_angle (str): Rotation angle (either 90, -90, or 180). Defaults to None.

     Returns:
         numpy.ndarray: The preprocessed image.

         """
    
     IF_resize = cv.resize(IF, dsize=None, fx=scale, fy=scale)
     IF_resize = cv.normalize(IF_resize, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
     

     if rotation_angle is not None:
          rotation = {
               90: cv.ROTATE_90_CLOCKWISE,
               -90: cv.ROTATE_90_COUNTERCLOCKWISE,
               180: cv.ROTATE_180
          }

          if rotation_angle not in rotation:
               raise ValueError("Invalid rotation angle. Must be 90, -90, or 180.")
          
          IF_resize_rot = cv.rotate(IF_resize, rotateCode=rotation[rotation_angle])
    

     IF_bin = cv.threshold(IF_resize_rot, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

     return IF_bin[1]


def contrast_stretch(img):
    """
    Constrast stretch input image.

    Args:
        img (numpy.ndarray): Input image.

    Returns:
        numpy.ndarray: The constrast-stretched image.

    """

    p1, p99 = np.percentile(img, (1, 99))
    
    stretched = (img - p1) / (p99 - p1) * 255
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    return stretched