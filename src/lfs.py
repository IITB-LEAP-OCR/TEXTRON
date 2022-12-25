import cv2
import numpy as np

from src.utils import binarize_image, pure_binarize

from skimage import io, filters, morphology
from skimage.util import invert
from skimage.morphology import convex_hull_image
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
    # print("Convex Hull LF")
    image = binarize_image(image)

    convex_hull_1 = convex_hull_image(image)
    image = invert(image)
    convex_hull_2 = convex_hull_image(image)
    
    intersection_hull = np.bitwise_and(convex_hull_1, convex_hull_2)
    return intersection_hull


def get_image_edges(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print("EDGES")
    image = binarize_image(image)
    image = invert(image)
    edges = filters.sobel(image)
    edges = pure_binarize(edges)
    return edges


def get_pillow_image_edges(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print("PILLOW EDGES")
    image = image.convert("L")    # Converting Image to Gray
    edges = image.filter(ImageFilter.FIND_EDGES)
    edges = np.array(edges)
    edges = pure_binarize(edges)
    return edges


def get_contour_labels(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print("CONTOUR LABELS")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white 
    # foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs.
    # hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

    #This is inmtermediate contour image having red contours plotted along the letters
    with_contours_int = cv2.drawContours(image, contours, -1,(0,0,255),2)

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
        if (cv2.contourArea(c)) > 20:
            #cv2.rectangle(origimage,(x,y), (x+w,y+h), (0,0,0), cv2.FILLED)
            bboxes.append([x, y, w, h])

    final_img = np.zeros((1024, 1024, 3), dtype = np.uint8)
    for b in bboxes:
        x = b[0]
        y = b[1]
        w = b[2]
        h = b[3]
        cv2.rectangle(final_img,(x,y), (x+w,y+h), (255, 255, 255),3)

    final_img = ~final_img
    final_img = binarize_image(final_img)
    final_img = final_img*1
    io.imsave("contour.jpg",image)
    return final_img


def get_doctr_labels(model, imgfile, image):
    """
    _summary_

    Args:
        model (_type_): _description_
        imgfile (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print("DOCTR LABELS")
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
                values.append(a+b)
                cv2.rectangle(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 0, 0),3)
#     image = binarize_image(image)
#     image = image*1
    io.imsave("doctr.jpg",image)
    return image


def get_tesseract_labels(image):
    """
    _summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    # print("TESSERACT LABELS")
    d = image_to_data(image, output_type=Output.DICT)
    image = binarize_image(image)
    image = 0*image
    image = invert(image)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    for i in range(len(d['level'])):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        (x, y, w, h) = (int(x), int(y), int(w), int(h))
        # if(x==0 and y==0):
        #     continue
        # else:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0),3)
#     image = binarize_image(image)
#     image = image*1
    io.imsave("tesseract.jpg",image)    
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
    mask = binarize_image(image)
    mask = morphology.remove_small_objects(mask < luminosity)
    return mask
