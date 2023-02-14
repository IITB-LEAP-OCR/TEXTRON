"""
_summary_

Returns:
    _type_: _description_
"""
import cv2
import pickle
import numpy as np

from skimage import io, color
from skimage.util import invert
from skimage.filters import threshold_otsu


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



def get_boxes(image, width_threshold, height_threshold, thickness=4, type="double"):
    """_summary_

    Args:
        image (_type_): _description_
        width_threshold (_type_): _description_
        height_threshold (_type_): _description_
        type (str, optional): _description_. Defaults to "double".

    Returns:
        _type_: _description_
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white 
    # foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary
    width, height = inverted_binary.shape

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs.
    # hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

    if(type == "double"):
        #This is inmtermediate contour image having red contours plotted along the letters
        with_contours_int = cv2.drawContours(image, contours, -1,(0,0,255),thickness)

        #We again perform binarization of above image inorder to find contours again 
        gray_contour = cv2.cvtColor(with_contours_int, cv2.COLOR_BGR2GRAY)

        ret, binary_contour = cv2.threshold(gray_contour, 100, 255, 
        cv2.THRESH_OTSU)
        inverted_contour = ~binary_contour

        # We find contours again of this inverted binary map so that word boundaries are detected
        contours, hierarchy = cv2.findContours(inverted_contour,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Make sure contour area is large enough
        #For English DocBank use the below one
        # if (w < width/2):
        # For multilingual
        if (cv2.contourArea(c)) > (width*height)/100000 and h<(height/4) and (w < width/2) and cv2.contourArea(c)>35:
        #for funsd
        # if (cv2.contourArea(c)) > 30:
            bboxes.append([x, y, w, h])

    final_img = np.zeros((image.shape), dtype = np.uint8)
    for b in bboxes:
        x = b[0]
        y = b[1]
        w = int(b[2]*width_threshold)
        h = int(b[3]*height_threshold)
        cv2.rectangle(final_img,(x,y), (x+w,y+h), (255, 255, 255),-1)
    final_img = ~final_img
    final_img = binarize_image(final_img)
    final_img = final_img*1
    return final_img


def save_image(img, width_threshold, height_threshold, final_img, name):
    """_summary_

    Args:
        img (_type_): _description_
        width_threshold (_type_): _description_
        height_threshold (_type_): _description_
        final_img (_type_): _description_
        name (_type_): _description_
    """
    img = invert(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs. hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w = int(w*(0.95/width_threshold))
        h = int(h*(0.8/height_threshold))
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 25 and (cv2.contourArea(c) < 5000):
            bboxes.append(['text',1,x, y, w, h])

    for b in bboxes:
        x = b[2]
        y = b[3]
        w = int(b[4])
        h = int(b[5])
        if(x=='0' and y=='0'):
            bboxes.remove(b)
            continue
        else:
            cv2.rectangle(final_img,(x,y), (x+w,y+h), (0, 255, 128),1)

    io.imsave(name+'.jpg', final_img)