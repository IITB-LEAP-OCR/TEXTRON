from src.utils import binarize_image, pure_binarize, get_boxes

import cv2
import numpy as np
import pandas as pd

from skimage import io, filters, morphology, img_as_ubyte
from skimage.util import invert
from skimage.morphology import convex_hull_image
from skimage.feature import canny
from scipy import ndimage as ndi

from PIL import ImageFilter
from doctr.io import DocumentFile
from pytesseract import Output, image_to_data


def get_convex_hull(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = binarize_image(image)

    convex_hull_1 = convex_hull_image(image)
    image = invert(image)
    convex_hull_2 = convex_hull_image(image)
    
    intersection_hull = np.bitwise_and(convex_hull_1, convex_hull_2)
    return intersection_hull


def get_image_edges(image, width_threshold, height_threshold, thickness):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = binarize_image(image)
    image = invert(image)
    edges = filters.sobel(image)
    edges = pure_binarize(edges)
    # Convert 2D numpy array to 3-channel image
    cv_image = cv2.cvtColor(img_as_ubyte(edges), cv2.COLOR_GRAY2BGR)
    return get_boxes(cv_image, width_threshold, height_threshold, thickness, "double")


def get_pillow_image_edges(image, width_threshold, height_threshold):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = image.convert("L")    # Converting Image to Gray
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges = np.array(edges)
    edges = pure_binarize(edges)
    # Convert 2D numpy array to 3-channel image
    cv_image = cv2.cvtColor(img_as_ubyte(edges), cv2.COLOR_GRAY2BGR)
    return get_boxes(cv_image, width_threshold, height_threshold, type="double")


def get_segmentation_labels(image, width_threshold, height_threshold, thickness):
    image = binarize_image(image)
    edges = canny(image)
    image = ndi.binary_fill_holes(edges)
    image = pure_binarize(image)
    # Convert 2D numpy array to 3-channel image
    cv_image = cv2.cvtColor(img_as_ubyte(image), cv2.COLOR_GRAY2BGR)
    return get_boxes(cv_image, width_threshold, height_threshold, thickness, "double")


def get_contour_labels(image, width_threshold, height_threshold, thickness):
    """
    Args:
        image (numpy.ndarray): An input image as a NumPy array.
        width_threshold (float): A float value to adjust the width of the extracted text boxes.
        height_threshold (float): A float value to adjust the height of the extracted text boxes.
        thickness (int): An integer value indicating the thickness of the bounding box around the text regions.

    Returns:
        numpy.ndarray: A binary image as a NumPy array with white pixels indicating the text regions.
    """
    return get_boxes(image, width_threshold, height_threshold, thickness, "double")


def get_title_contour_labels(image, width_threshold, height_threshold, thickness):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    return get_boxes(image, width_threshold, height_threshold,thickness, "double")


def get_doctr_labels(model, imgfile, image, width_threshold, height_threshold):
    """
    _summary_

    Args:
        model (_type_): _description_
        imgfile (_type_): _description_

    Returns:
        _type_: _description_
    """
    doc = DocumentFile.from_images(imgfile)
    result = model(doc)
    dim = tuple(reversed(result.pages[0].dimensions))
    values = []
    image = binarize_image(image)
    image = 0*image
    image = invert(image)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                geo = word.geometry
                a = list(a*b for a,b in zip(geo[0],dim))
                b = list(a*b for a,b in zip(geo[1],dim))
                w = (b[0] - a[0])*width_threshold
                h = (b[1] - a[1])*height_threshold
                values.append(a+b)
                cv2.rectangle(image, (int(a[0]), int(a[1])), (int(a[0]+w), int(a[1]+h)), (0, 0, 0),-1)
    image = pure_binarize(image)
    return image


def get_existing_doctr_labels(ann_dir,imgfile, image, width_threshold, height_threshold):
    """
    _summary_

    Args:
        model (_type_): _description_
        imgfile (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = binarize_image(image)
    image = 0*image
    image = invert(image)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    df = pd.read_csv(ann_dir + imgfile[:-4] + '.txt', sep=' ', names = ['class','confidence','X','Y','W','H','label'])
    df['W'] = df['W'].apply(lambda x : int(int(x)*width_threshold))
    df['H'] = df['H'].apply(lambda x : int(int(x)*height_threshold))
    for _,a in df.iterrows():
        cv2.rectangle(image, (int(a['X']), int(a['Y'])), (int(a['X']+a['W']), int(a['Y']+a['H'])), (0, 0, 0),-1)
    image = pure_binarize(image)
    print('Tess LF Output')
    print(image)
    return image


def get_tesseract_labels(image, width_threshold, height_threshold):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    d = image_to_data(image, output_type=Output.DICT)
    image = binarize_image(image)
    image = 0*image
    image = invert(image)
    width, height = image.shape
    image = np.ascontiguousarray(image, dtype=np.uint8)
    for i in range(len(d['level'])):
        if d['level'][i]==5:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            (x, y, w, h) = (int(x), int(y), int(w), int(h))
            w = int(w*width_threshold)
            h = int(h*height_threshold)
            if((h<(height/40)) and (w < width/15)):
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0),-1)
    return image


def get_mask_holes_labels(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask = binarize_image(image)
    mask = morphology.remove_small_holes(mask,100)
    mask = invert(mask)
    return mask

def get_mask_objects_labels(image, luminosity):
    """_summary_

    Args:
        image (_type_): _description_
        luminosity (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask = binarize_image(image)
    mask = morphology.remove_small_objects(mask < luminosity)
    return mask
