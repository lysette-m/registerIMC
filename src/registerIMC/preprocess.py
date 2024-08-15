# import cv2 as cv
import numpy as np


# def preprocess_IF(IF, scale, rotation_angle = None):
#     """
#     Preprocesses an IF reference image, including resizing, rotation (if necessary), and binarisation.
        
#     Args:
#         IF (numpy.ndarray): The input image.
#         scale (float): The scaling factor for resizing the image.
#         rotate (bool, optional): Whether to rotate the image. Defaults to None.
#         angle (str): Rotation function for the image.

#     Returns:
#         numpy.ndarray: The preprocessed image.

#         """

#     IF_resize = cv.resize(IF, dsize=None, fx=scale, fy=scale)
#     IF_resize = cv.normalize(IF_resize, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
#     IF_bin = cv.threshold(IF_resize, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

#     if rotation_angle is not None:
#         IF_bin = cv.rotate(IF_bin, rotation_angle)

#     return IF_bin


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