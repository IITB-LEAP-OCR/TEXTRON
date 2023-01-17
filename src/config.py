import os

DATA_DIR    = '../docbank_processed/'

INPUT_DIR       = os.path.join(DATA_DIR, 'processed_data/spear_ori_black/')
LABELS_DIR      = os.path.join(DATA_DIR, 'processed_data/txt/')
ORI_TXT_DIR     = os.path.join(DATA_DIR, 'processed_data/txt/')


RESULT_VALUE = "18"
RESULTS_DIR     = os.path.join(DATA_DIR, 'cage_results/results' + RESULT_VALUE + '/')
OUT_TXT_DIR     = os.path.join(DATA_DIR, "txt_outputs/txt" + RESULT_VALUE + '/')
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions/predictions" + RESULT_VALUE + '/')

if not os.path.exists(RESULTS_DIR):
   # Create a new directory because it does not exist
   os.makedirs(RESULTS_DIR)
   os.makedirs(OUT_TXT_DIR)
   os.makedirs(PREDICTIONS_DIR)


GROUND_TRUTH = True
GROUND_TRUTH_DIR = os.path.join(DATA_DIR, 'gt_cage_sample100/gt_cage')

IS_DOCTR_AND = False

IS_EXPERIMENT    = False
EXPERIMENT_VALUE = len(os.listdir(RESULTS_DIR)) + 1


LUMINOSITY = 1.0

WIDTH_THRESHOLD = 0.75
HEIGHT_THRESHOLD = 0.75

