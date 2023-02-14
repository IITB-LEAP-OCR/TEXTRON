import os
import torch
torch.cuda.empty_cache()

INPUT_DATA_DIR    = ''
RESULTS_DATA_DIR  = './../results/'
DATASET = './../sample'

INPUT_IMG_DIR = os.path.join(INPUT_DATA_DIR, DATASET + '/images/')
ANN_IMG_DIR   = os.path.join(INPUT_DATA_DIR, DATASET + '/ann/')
ORI_TXT_DIR   = os.path.join(INPUT_DATA_DIR, DATASET + '/txt/')


RESULT_VALUE =  34
RESULTS_DIR     = os.path.join(RESULTS_DATA_DIR, 'cage/results' + str(RESULT_VALUE) + '/')
OUT_TXT_DIR     = os.path.join(RESULTS_DATA_DIR, "txt/txt" + str(RESULT_VALUE) + '/')
PREDICTIONS_DIR = os.path.join(RESULTS_DATA_DIR, "predictions/predictions" + str(RESULT_VALUE) + '/')

if not os.path.exists(RESULTS_DIR):
   # Create a new directory because it does not exist
   os.makedirs(RESULTS_DIR)
   os.makedirs(OUT_TXT_DIR)
   os.makedirs(PREDICTIONS_DIR)


GROUND_TRUTH = True
GROUND_TRUTH_DIR = os.path.join(INPUT_DATA_DIR,  DATASET + '/gt_pro')

IS_DOCTR_AND = False

IS_EXPERIMENT    = False
EXPERIMENT_VALUE = len(os.listdir(RESULTS_DIR)) + 1


ANN_DOCTR_DIR = './../testing_sample/doctr_txt/'

LUMINOSITY = 1.0

WIDTH_THRESHOLD = 0.90
HEIGHT_THRESHOLD = 0.75


THICKNESS = 4