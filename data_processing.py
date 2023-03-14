from src.config import *
import skimage.io as io
from src.lf_utils import *
from src.utils import get_pixels
from PIL import Image

import enum

class pixelLabels(enum.Enum):
    TEXT = 1
    NOT_TEXT = 0


class Labeling:
    def __init__(self,imgfile, model) -> None:
        self.imgfile = INPUT_IMG_DIR + imgfile
        image = io.imread(self.imgfile)
        image2 = Image.open(self.imgfile)
        image3 = cv2.imread(self.imgfile)
        self.CHULL        = get_convex_hull(image)
        self.EDGES        = get_image_edges(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        # self.PILLOW_EDGES = get_pillow_image_edges(image2, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.CONTOUR      = get_contour_labels(image3, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.TITLE_CONTOUR = get_title_contour_labels(image3, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, 7)
        self.DOCTR        = get_doctr_labels(model, self.imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        # self.DOCTR        = get_existing_doctr_labels(ANN_DOCTR_DIR, imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.TESSERACT    = get_tesseract_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.MASK_HOLES   = get_mask_holes_labels(image)
        self.MASK_OBJECTS = get_mask_objects_labels(image, LUMINOSITY)
        self.SEGMENTATION = get_segmentation_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.pixels = get_pixels(image)
        self.image = image