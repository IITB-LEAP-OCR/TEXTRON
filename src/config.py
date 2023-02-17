import os
import torch
torch.cuda.empty_cache()

### Path to the Data and Results directories
INPUT_DATA_DIR    = './../processed/docbank_100/'
RESULTS_DATA_DIR  = './../results/'

### Keep True if True labels are available, else False
GROUND_TRUTH_AVAILABLE = True

### Path to images and the ground truths
INPUT_IMG_DIR    = os.path.join(INPUT_DATA_DIR, 'images/')
GROUND_TRUTH_DIR = os.path.join(INPUT_DATA_DIR, 'txt/')


### Directories for resultant predictions
RESULT_VALUE    =  35
RESULTS_DIR     = os.path.join(RESULTS_DATA_DIR, 'cage/results' + str(RESULT_VALUE) + '/')
OUT_TXT_DIR     = os.path.join(RESULTS_DATA_DIR, "txt/txt" + str(RESULT_VALUE) + '/')
PREDICTIONS_DIR = os.path.join(RESULTS_DATA_DIR, "predictions/predictions" + str(RESULT_VALUE) + '/')


# Create a new directory if it does not exist already
if not os.path.exists(RESULTS_DIR):
   os.makedirs(RESULTS_DIR)
   os.makedirs(OUT_TXT_DIR)
   os.makedirs(PREDICTIONS_DIR)


### Hyperparameters for Shrinkage threshold on LF outputs
WIDTH_THRESHOLD = 0.90
HEIGHT_THRESHOLD = 0.75

### Hyperparameter for Contour thickness to generate bboxes
THICKNESS = 4

### This is used when already DL model results are present, to save time
ANN_DOCTR_DIR = './../testing_sample/doctr_txt/'

### Used in MASK model LF to change intensity as hyperparameter
LUMINOSITY = 1.0