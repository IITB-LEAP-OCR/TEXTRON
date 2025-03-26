import os
import torch
torch.cuda.empty_cache()


### Path to the Data and Results directories
INPUT_DATA_DIR = 'TEXTRON_datasets/Handwritten/PhDIndic11_Devanagari/'
RESULTS_DATA_DIR  = 'results/'


### Keep True if True labels are available, else False
GROUND_TRUTH_AVAILABLE = False

### Path to images and the ground truths
INPUT_IMG_DIR    = os.path.join(INPUT_DATA_DIR, 'images/')
GROUND_TRUTH_DIR = os.path.join(INPUT_DATA_DIR, 'txt/')


### Directories for resultant predictions
MODEL = 'SAP_8LF'
RESULT_VALUE = 'devanagari'

RESULTS_DIR     = os.path.join(RESULTS_DATA_DIR, 'cage/' + str(RESULT_VALUE) + '/')
OUT_TXT_DIR     = os.path.join(RESULTS_DATA_DIR, "txt/" + str(RESULT_VALUE) + '/')
PREDICTIONS_DIR = os.path.join(RESULTS_DATA_DIR, "predictions/" + str(RESULT_VALUE) + '/')

# Create a new directory if it does not exist already
if not os.path.exists(RESULTS_DIR):
   os.makedirs(RESULTS_DIR)
   os.makedirs(OUT_TXT_DIR)
   os.makedirs(PREDICTIONS_DIR)

## Train or Test Flag
PRED_ONLY = True
CAGE_EPOCHS = 50
PARAMS_LOADED = False

PARAMS_PATH     = os.path.join(RESULTS_DATA_DIR, "params/")
if not os.path.exists(PARAMS_PATH):   
   os.makedirs(PARAMS_PATH)

PARAMS_FILE = PARAMS_PATH + str(MODEL) + '_params.pkl'



### Hyperparameters for Shrinkage threshold on LF outputs
WIDTH_THRESHOLD = 0.9
HEIGHT_THRESHOLD = 0.7

### Hyperparameter for Contour thickness to generate bboxes
CONTOUR_THICKNESS = 5
SEGMENT_THICKNESS = 5

### This is used when already DL model results are present, to save time
ANN_DOCTR_DIR = './../testing_sample/doctr_txt/'

### Used in MASK model LF to change intensity as hyperparameter
LUMINOSITY = 1.0

### Choose the Labeling Functions which should be run
lab_funcs = [ 
   # "CONVEX_HULL_LABEL_PURE", 
   # "CONVEX_HULL_LABEL_NOISE", 
   #"EDGES_LABEL", 
   #"EDGES_LABEL_REVERSE",
   # "EDGES_LABEL_REVERSE", 
   # "PILLOW_EDGES_LABEL", 
   # "PILLOW_EDGES_LABEL_REVERSE",
   "DOCTR_LABEL",
   "DOCTR_LABEL_REVERSE",
   # "DOCTR_LABEL2",
   "TESSERACT_LABEL",
   "TESSERACT_LABEL_REVERSE",
   "CONTOUR_LABEL",
   "CONTOUR_LABEL_REVERSE",
   # "MASK_HOLES_LABEL",
   # "MASK_OBJECTS_LABEL",
   "SEGMENTATION_LABEL",
   "SEGMENTATION_LABEL_REVERSE"
]

QUALITY_GUIDE = [0.9, 0.95, 0.75, 0.95, 0.85, 0.95, 0.85, 0.95]

### Other Quality Guides that has been tried out for respective Chosen LFs
# [0.99, 0.99, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
#[0.9, 0.95, 0.75, 0.95, 0.85, 0.95, 0.85, 0.95]
# [0.99, 0.99, 0.01, 0.01, 0.99, 0.99, 0.01, 0.01]
# 8LF new [0.9, 0.95, 0.75, 0.95, 0.85, 0.95, 0.85, 0.95]
# 8LF [0.95, 0.99, 0.95, 0.99, 0.75, 0.99, 0.1, 0.99]
# [0.85, 0.9, 0.95, 0.99]
# QUALITY_GUIDE =  [0.8, 0.8, 0.85, 0.95, 0.8, 0.95, 0.8, 0.95, 0.85, 0.9]
# [0.9, 0.99, 0.9, 0.99]
# 11LF [0.9, 0.8, 0.8, 0.75, 0.95, 0.8, 0.95, 0.8, 0.99, 0.85, 0.9]

SPLIT_THRESHOLD = 0


