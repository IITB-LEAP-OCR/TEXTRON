from PIL import Image
import cv2
from doctr.models import ocr_predictor
import skimage.io as io
import numpy as np
from tqdm import tqdm

from src.lf_utils import * 
from src.config import *
from src.utils import get_pixels
import time

import warnings
warnings.filterwarnings("ignore")


MODEL = ocr_predictor(pretrained=True)



def get_bboxes(inverted_binary, file):
    width, height = inverted_binary.shape
    inverted_binary = inverted_binary.astype(np.uint8)
    inverted_binary = cv2.cvtColor(inverted_binary, cv2.COLOR_BGR2GRAY) if len(inverted_binary.shape) == 3 else inverted_binary
    contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w = int(w*(1/WIDTH_THRESHOLD))
        h = int(h*(1/HEIGHT_THRESHOLD))
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > (width*height)/100000 and h<(height/4) and (w < width/2) and cv2.contourArea(c)>30:
            bboxes.append([x, y, w, h])

    final_img = cv2.imread(INPUT_IMG_DIR + file)
    for b in bboxes:
        x = b[0]
        y = b[1]
        w = int(b[2])
        h = int(b[3])
        cv2.rectangle(final_img,(x,y), (x+w,y+h), (0, 255, 0),1)

    df = pd.DataFrame(bboxes, columns = ['X', 'Y', 'W', 'H'])
    name = file[:len(file) - 4]

    io.imsave(RESULTS_DIR + name + '_pred.jpg', final_img)
    df.to_csv(RESULTS_DIR + name + '.txt', sep=' ',index=False, header=False)


def binary_arrays_to_numbers(arr_list):
    # arr_list = np.where(np.array(arr_list) == None, 0, arr_list)  # Replace None with 0
    return np.array([int("".join(map(str, row)), 2) for row in arr_list])

class Labeling:
    def __init__(self,imgfile, model) -> None:
        # self.imgfile = INPUT_IMG_DIR + imgfile
        self.imgfile = imgfile
        image = io.imread(self.imgfile)
        image2 = cv2.imread(self.imgfile)
        image = image.astype(np.uint8)
        self.CONTOUR      = get_contour_labels(image2, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, CONTOUR_THICKNESS).ravel()
        self.DOCTR        = get_doctr_labels(model, self.imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD).ravel()
        self.TESSERACT    = get_tesseract_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD).ravel()
        self.SEGMENTATION = get_segmentation_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, SEGMENT_THICKNESS).ravel()
        self.pixels = get_pixels(image)
        self.image = image
        self.TESSERACT[self.TESSERACT == 255] = 1

### Main Code
if __name__ == "__main__":

    dir_list = os.listdir(INPUT_IMG_DIR)

    ### Predictions on Test
    test_data = sorted(dir_list)
    for img_file in tqdm(test_data):
        if not (os.path.exists(RESULTS_DIR + img_file)):
            start_time = time.time()
            lf = Labeling(imgfile=INPUT_IMG_DIR + img_file, model=MODEL)
            
            merged_arr = np.column_stack([lf.DOCTR, lf.TESSERACT, lf.CONTOUR, lf.SEGMENTATION])
            merged_arr = binary_arrays_to_numbers(merged_arr)
            merged_arr = np.where(merged_arr > 8, 1, np.where(merged_arr < 7, 0, np.where(merged_arr == 7, 1, 0)))

            labels = merged_arr.reshape(lf.image.shape[0], lf.image.shape[1])
            img = (labels * 255).astype(np.uint8)
            get_bboxes(img, img_file)
            end_time = time.time()
            
            # write the time taken to a file
            with open('log_cage_time_2.txt', 'a') as f:
                f.write(f"{end_time - start_time}\n")
    

