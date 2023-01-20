import os

INPUT_DATA_DIR    = './../processed/'
RESULTS_DATA_DIR  = './../results/'
DATASET = 'docbank_100'

INPUT_IMG_DIR = os.path.join(INPUT_DATA_DIR, DATASET + '/images_resized/')
ANN_IMG_DIR   = os.path.join(INPUT_DATA_DIR, DATASET + '/ann/')
ORI_TXT_DIR   = os.path.join(INPUT_DATA_DIR, DATASET + '/txt/')


RESULT_VALUE =  3
RESULTS_DIR     = os.path.join(RESULTS_DATA_DIR, 'cage/results' + str(RESULT_VALUE) + '/')
OUT_TXT_DIR     = os.path.join(RESULTS_DATA_DIR, "txt/txt" + str(RESULT_VALUE) + '/')
PREDICTIONS_DIR = os.path.join(RESULTS_DATA_DIR, "predictions/predictions" + str(RESULT_VALUE) + '/')

if not os.path.exists(RESULTS_DIR):
   # Create a new directory because it does not exist
   os.makedirs(RESULTS_DIR)
   os.makedirs(OUT_TXT_DIR)
   os.makedirs(PREDICTIONS_DIR)


GROUND_TRUTH = True
GROUND_TRUTH_DIR = os.path.join(INPUT_DATA_DIR,  DATASET + '/gt_cage_sample100/gt_cage')

IS_DOCTR_AND = False

IS_EXPERIMENT    = False
EXPERIMENT_VALUE = len(os.listdir(RESULTS_DIR)) + 1


LUMINOSITY = 1.0

WIDTH_THRESHOLD = 0.9
HEIGHT_THRESHOLD = 0.9

