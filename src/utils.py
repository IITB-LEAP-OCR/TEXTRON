"""
_summary_

Returns:
    _type_: _description_
"""
import pickle
from skimage import color
from skimage.filters import threshold_otsu
import numpy as np


def get_pixels(image): 
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    X = []
    image = binarize_image(image)
    for i in range(len(image)):
        for j in range(len(image[0])):
            val = (i,j)
            X.append(val)
    X = np.array(X)
    return X


def pure_binarize(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    thresh = threshold_otsu(image)
    image = image > thresh
    return image    


def binarize_image(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = color.rgb2gray(image)
    thresh = threshold_otsu(image)
    image = image > thresh
    return image


def get_label(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = binarize_image(image)
    image = image*1
    image = image.ravel()
    return image


def resize_image(image, new_dim):
    """
    _summary_

    Args:
        image (_type_): _description_
        new_dim (_type_): _description_

    Returns:
        _type_: _description_
    """
    new_image = image.resize(new_dim)
    return new_image


def store_pickle(data, location):
    """
    _summary_

    Args:
        data (_type_): _description_
        location (_type_): _description_
    """
    with open(location, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
