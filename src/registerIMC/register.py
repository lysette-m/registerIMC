import numpy as np
import cv2 as cv
import math 
import os 
import tifffile
import glob

from preprocess import contrast_stretch

def get_markers(panel_path, marker_col):
    """
    Retrieves markers from a panel file based on the specified marker column.

    Args:
        panel_path (str): The path to the panel file (tab-delimited).
        marker_col (str): The name of the marker column.

    Returns:
        list: A list of markers extracted from the panel file.

    """

    with open(panel_path) as p: 
        headers = p.readline().split('\t')
        marker_index = headers.index(marker_col)
        markers = [line.split('\t')[marker_index] for line in p.readlines()[1:]]

    return markers


def rotate_and_resize(img, degrees):
    """
    Rotate an image whilst adding padding to avoid cropping.

    Args:
        img (numpy.ndarray): The input image.
        degrees (float): The rotation angle in degrees.

    Returns:
        tuple: A tuple containing:
            - rotationM (numpy.ndarray): The rotation matrix.
            - rotated_img (numpy.ndarray): The rotated image.
            - new_h (int): The height of the rotated image.
            - new_w (int): The width of the rotated image.
    """

    (h,w) = img.shape

    radians = math.radians(degrees)
    new_w = int(abs(w * math.cos(radians)) + abs(h * math.sin(radians)))
    new_h = int(abs(w * math.sin(radians)) + abs(h * math.cos(radians)))

    padded_img = np.zeros((new_h, new_w), dtype=np.uint8)
    
    vertical_offset = (new_h - h) // 2
    horitzontal_offset = (new_w - w) // 2

    padded_img[vertical_offset:vertical_offset + h, 
               horitzontal_offset:horitzontal_offset + w] = img

    center = (new_w / 2, new_h / 2)

    rotationM = cv.getRotationMatrix2D(center, degrees, 1)

    rotated_img = cv.warpAffine(padded_img, M=rotationM, dsize=(new_w, new_h))
    
    return rotationM, rotated_img, new_h, new_w



def get_transformation(ref, template, max_angle, angle_increment):
    """
    Calculates the best angle and location for template matching.

    Args:
        ref (numpy.ndarray): The reference image.
        template (numpy.ndarray): The template image.
        max_angle (float): The maximum angle for rotation in degrees.

    Returns:
        tuple: A tuple containing the best transformation angle and location.

    """

    c = cv.matchTemplate(ref, template, cv.TM_CCORR_NORMED)

    _, _, _, max_loc = cv.minMaxLoc(c)
    x,y = max_loc

    best_val = 0

    for degrees in np.arange(-(max_angle), max_angle, angle_increment):

        _, rotated_img, new_h, new_w = rotate_and_resize(img = template, degrees = degrees)

        # TODO: make this dependent on the size of the rotation
        y_search = y - 50
        x_search = x - 50
        h_search = new_h + 100
        w_search = new_w + 100

        ref_ROI = ref[y_search:y_search + h_search, x_search:x_search + w_search]
        
        c_rotated = cv.matchTemplate(ref_ROI, rotated_img, cv.TM_CCORR_NORMED)

        _, max_val, _, max_loc = cv.minMaxLoc(c_rotated)

        if max_val > best_val:
            best_val = max_val
            best_angle = degrees
            best_loc = (max_loc[0] + x_search, max_loc[1] + y_search)
    

    return best_angle, best_loc


def register_marker_image(IMC_path, ref_size, marker, ROI_transformations):
    """
    Register an IMC channel image onto a reference image.

    Args:
        IMC_path (str): The path to the IMC data.
        ref_size (tuple): The size of the reference image.
        marker (str): The marker image to register.
        ROI_transformations (dict): A dictionary containing the best angle and location for each region of interest (ROI) on the reference.

    Returns:
        registered_image (ndarray): The registered image.
    """

    
    registered_image = np.zeros(ref_size, dtype=np.uint8)

    for roi_name, (best_angle, best_loc) in ROI_transformations.items():

        marker_file = glob.glob(os.path.join(IMC_path, roi_name, f'*{marker}*.tiff'))

        if not marker_file:
            raise FileNotFoundError("No file found for marker {marker} in {roi_name}")
        
        marker_img = tifffile.imread(marker_file)

        _, final_img, final_h, final_w = rotate_and_resize(marker_img, best_angle)
        copy = final_img > 0
        
        registered_image[best_loc[1]:best_loc[1] + final_h, best_loc[0]:best_loc[0] + final_w][copy] = final_img[copy]

    return registered_image



# TODO: option to save as multi-channel ome.tiff?

def register_IF_IMC(IF_ref, IMC_path: str, ROI_prefix: str, output_path: str,
                    match_channel: str, panel_path: str = None, marker_col: str = None, 
                    custom_markers: list = None, max_angle: int = 1, 
                    angle_increment: float = 0.1, process_IMC = True):
    """
    Main function to register IMC images to IF image and save.
    
    Args:
        IF_ref (ndarray): Reference IF image.
        IMC_path (str): Path to IMC data.
        ROI_prefix (str): Prefix denoting ROI folders.
        match_channel (str): Channel used for registration. Should be common to both modalities.
        panel_path (str): Path to panel file (tab-delimited). Also requires marker_col.
        marker_col (str): Column name in panel file containing marker names. Also requires panel_path.
        output_path (str): Path to save registered images.
        custom_markers (list): List of custom markers. 
        max_angle (int, optional): Maximum rotation angle (-/+) to test. Defaults to 1.
        angle_increment (float, optional): Step size for angle increments for rotation search (i.e. the precision of the registration process). Defaults to 0.1.
        process_IMC (bool, optional): Flag to process IMC images (normalise and binarise). Defaults to True.
    
    Returns:
        None
    
    """
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ROIs = sorted([r for r in os.listdir(IMC_path) if r.startswith(ROI_prefix)])

    if not ROIs:
        raise ValueError("No ROIs found in specified directory.")
    

    if custom_markers is None:
        if panel_path is None or marker_col is None:
            raise ValueError("Please provide panel_path AND marker_col, OR custom_markers.")
        
        markers = get_markers(panel_path, marker_col)
    else:
        markers = custom_markers

    transformations = {}

    for ROI in ROIs:

        print(f"Calculating transformation for {ROI}")

        filename = glob.glob(os.path.join(IMC_path, ROI, f'*{match_channel}*.tiff'))

        IMC_temp = tifffile.imread(filename)
        
        # need both contrast stretch and cv.threshold?
        if process_IMC: 
            IMC_temp = contrast_stretch(IMC_temp)
            IMC_bin = cv.threshold(IMC_temp, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            IMC_bin = IMC_bin[1]

        best_angle, best_loc = get_transformation(IF_ref, IMC_bin, max_angle, angle_increment)

        transformations[ROI] = (best_angle, best_loc)


    for marker in markers:

        print(f"Stitching image for {marker}")

        registered_image = register_marker_image(IMC_path, IF_ref.shape, marker, transformations)
        
        tifffile.imwrite(os.path.join(output_path, f'{marker}_registered.tiff'), registered_image)

   


