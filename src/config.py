import os

DATA_DIR    = './../docbank_processed/processed_data/'

INPUT_DIR       = os.path.join(DATA_DIR, 'spear_ori_black/')
LABELS_DIR      = os.path.join(DATA_DIR, 'ann/')
RESULTS_DIR     = os.path.join(DATA_DIR, 'cage_results9/')
ORI_TXT_DIR     = os.path.join(DATA_DIR, 'txt/')
OUT_TXT_DIR     = os.path.join(DATA_DIR, "out_txt9/")
PREDICTIONS_DIR = os.path.join(DATA_DIR, "predictions9/")

IS_EXPERIMENT    = False
EXPERIMENT_VALUE = len(os.listdir(RESULTS_DIR)) + 1


LUMINOSITY = 1.0

WIDTH_THRESHOLD = 0.65
HEIGHT_THRESHOLD = 0.65

