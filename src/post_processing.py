from src.config import *
import cv2
import skimage.io as io
import pandas as pd
import numpy as np
from numpy import invert
import json



### Postprocessing Step
def get_bboxes(file):

    img = cv2.imread(RESULTS_DIR + file)

    img = invert(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary
    width, height = inverted_binary.shape

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs. hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w = int(w*(1/WIDTH_THRESHOLD))
        h = int(h*(1/HEIGHT_THRESHOLD))
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > (width*height)/100000 and h<(height/4) and (w < width/2) and cv2.contourArea(c)>30:
            bboxes.append(['text',1,x, y, w, h])

    final_img = cv2.imread(INPUT_IMG_DIR + file)
    for b in bboxes:
        x = b[2]
        y = b[3]
        w = int(b[4])
        h = int(b[5])
        cv2.rectangle(final_img,(x,y), (x+w,y+h), (0, 255, 0),1)

    df = pd.DataFrame(bboxes, columns = ['label', 'confidence', 'X', 'Y', 'W', 'H'])
    name = file[:len(file) - 4]
    io.imsave(PREDICTIONS_DIR + name + '_pred.jpg', final_img)
    df.to_csv(OUT_TXT_DIR + name + '.txt', sep=' ',index=False, header=False)
    

def coco_conversion():
    coco_dict = {
    "info": {"description": "TEXTRON : Improving Multi-Lingual Text Detection through Data Programming"},
    "licenses": [],
    "categories": [],
    "images": [],
    "annotations": []
    }

    # Add category information
    category_dict = {
        "id": 1,
        "name": "word",
        "supercategory": "text"
    }

    img_id = 0
    for file in os.listdir(INPUT_IMG_DIR):
        # Add image and annotation information
        
        img = io.imread(INPUT_IMG_DIR + file)
        height, width, _ = img.shape
        img_id += 1
        image_dict = {
            "id": img_id,
            "width": width,
            "height": height,
            "file_name": file
        }
        coco_dict["images"].append(image_dict)
        
        # Read the annotations/ bbox info. from the corresponding CSV file
        name = file[:len(file) - 4]
        df = pd.read_csv(OUT_TXT_DIR + name + '.txt', sep=' ', names = ['label', 'confidence', 'X', 'Y', 'W', 'H'])

        # Create a list of annotations for the current image
        annotations = []
        for i, row in df.iterrows():
            annotation_dict = {
                "id": i + 1,
                "image_id": img_id,
                "category_id": 1,
                "bbox": [int(row['X']), int(row['Y']), int(row['W']), int(row['H'])],
                "area": int(row['W']) * int(row['H']),
                "iscrowd": 0
            }
            annotations.append(annotation_dict)
        coco_dict["annotations"].extend(annotations)    

    coco_dict["categories"].append(category_dict)


    # Write the dictionary to a JSON file
    with open(RESULTS_DATA_DIR + "coco_output.json", "w") as f:
        json.dump(coco_dict, f)
