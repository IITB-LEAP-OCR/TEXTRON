import enum
import pandas as pd
import numpy as np
from src.lfs import get_contour_labels
from src.config import *

import os
import pickle
from skimage import io
import cv2

# from snorkel.labeling import labeling_function
# from snorkel.labeling import PandasLFApplier

from spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from spear.spear.labeling import LFAnalysis, LFSet, PreLabels
from spear.spear.utils import get_data, get_classes
from sklearn.model_selection import train_test_split

from spear.spear.cage import Cage


CHULL = None
EDGES = None
PILLOW_EDGES = None
DOCTR = None
TESSERACT = None
CONTOUR = None
LABELS = None

CHULL_RESULTS = None
EDGES_RESULTS = None
PILLOW_EDGES_RESULTS = None
DOCTR_RESULTS = None
TESSERACT_RESULTS = None
CONTOUR_RESULTS = None
LABELS_RESULTS = None
IMG = None




with open(RESULTS_DIR + 'img_processing/convex_hull_results.pkl', 'rb') as f:
    CHULL_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'img_processing/edges_results.pkl', 'rb') as f:
    EDGES_RESULTS = pickle.load(f)
    
with open(RESULTS_DIR + 'img_processing/pillow_edges_results.pkl', 'rb') as f:
    PILLOW_EDGES_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'img_processing/contour.pkl', 'rb') as f:
    CONTOUR_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'img_processing/mask_results_small_holes_100.pkl', 'rb') as f:
    MASK_HOLES_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'img_processing/mask_results_small_objects_lum_1.pkl', 'rb') as f:
    MASK_OBJECTS_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'doctr/pixels.pkl', 'rb') as f:
    DOCTR_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'tesseract/pixels.pkl', 'rb') as f:
    TESSERACT_RESULTS = pickle.load(f)

with open(RESULTS_DIR + 'labels/pixels.pkl', 'rb') as f:
    LABELS_RESULTS = pickle.load(f)


class pixelLabels(enum.Enum):
    TEXT = 1
    NOT_TEXT = 0


class Labeling:

    def __init__(self,imgfile) -> None:
        self.imgfile = imgfile
        imgfile2 = imgfile[:len(imgfile) - 7] + 'ann.jpg'
        self.CHULL        = CHULL_RESULTS[imgfile]
        self.EDGES        = EDGES_RESULTS[imgfile]
        self.PILLOW_EDGES = PILLOW_EDGES_RESULTS[imgfile]
        self.CONTOUR      = CONTOUR_RESULTS[imgfile]
        self.DOCTR        = DOCTR_RESULTS[imgfile]
        self.TESSERACT    = TESSERACT_RESULTS[imgfile] 
        self.MASK_OBJECTS = MASK_OBJECTS_RESULTS[imgfile]
        self.MASK_HOLES   = MASK_HOLES_RESULTS[imgfile]
        self.LABELS       = LABELS_RESULTS[imgfile2]



lf = None


@preprocessor()
def get_chull_info(x):
    return lf.CHULL[x[0]][x[1]]

@preprocessor()
def get_edges_info(x):
    return lf.EDGES[x[0]][x[1]]

@preprocessor()
def get_pillow_edges_info(x):
    return lf.PILLOW_EDGES[x[0]][x[1]]


@preprocessor()
def get_doctr_info(x):
    return lf.DOCTR[x[0]][x[1]]


@preprocessor()
def get_tesseract_info(x):
    return lf.TESSERACT[x[0]][x[1]]

@preprocessor()
def get_contour_info(x):
    return lf.CONTOUR[x[0]][x[1]]

@preprocessor()
def get_mask_holes_info(x):
    return lf.MASK_HOLES[x[0]][x[1]]

@preprocessor()
def get_mask_objects_info(x):
    return lf.MASK_OBJECTS[x[0]][x[1]]



@labeling_function(label = pixelLabels.NOT_TEXT, pre=[get_chull_info], name="CHULL_PURE")
def CONVEX_HULL_LABEL_PURE(pixel):
    if(pixel):
        return ABSTAIN
    else:
        return pixelLabels.NOT_TEXT
    
@labeling_function(label=pixelLabels.TEXT, pre=[get_chull_info], name="CHULL_NOISE")
def CONVEX_HULL_LABEL_NOISE(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN
    

@labeling_function(label=pixelLabels.TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES")
def EDGES_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN

@labeling_function(label = pixelLabels.NOT_TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES_REVERSE")
def EDGES_LABEL_REVERSE(pixel):
    if(pixel):
        return ABSTAIN
    else:
        return pixelLabels.NOT_TEXT
    
    
@labeling_function(label=pixelLabels.TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES")
def PILLOW_EDGES_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label = pixelLabels.NOT_TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES_REVERSE")
def PILLOW_EDGES_LABEL_REVERSE(pixel):
    if(pixel):
        return ABSTAIN
    else:
        return pixelLabels.NOT_TEXT


@labeling_function(label=pixelLabels.TEXT, pre=[get_doctr_info], name="DOCTR")
def DOCTR_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_tesseract_info], name="TESSERACT")
def TESSERACT_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.TEXT, pre=[get_contour_info], name="CONTOUR")
def CONTOUR_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_mask_holes_info], name="MASK_HOLES")
def MASK_HOLES_LABEL(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_mask_objects_info], name="MASK_OBJECTS")
def MASK_OBJECTS_LABEL(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN






def main():
    

    d = lf.CHULL

    X = []
    Y = []
    for i in range(len(d)):
        for j in range(len(d[0])):
            val = (i,j)
            X.append(val)
            Y.append(lf.LABELS[i][j])
    X = np.array(X)
    Y = np.array(Y)

    LFS = [DOCTR_LABEL, TESSERACT_LABEL, 
    CONTOUR_LABEL, MASK_HOLES_LABEL]

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    n_lfs = len(rules.get_lfs())

    R = np.zeros((X.shape[0],len(rules.get_lfs())))


    td_noisy_labels = PreLabels(name="TD",
                               data=X,
                               rules=rules,
                               gold_labels=Y,
                               labels_enum=pixelLabels,
                               num_classes=2)

    L,S = td_noisy_labels.get_labels()


    analyse = td_noisy_labels.analyse_lfs(plot=True)

    result = analyse.head(16)
    result["image"] = img

    print("===== All Done =====")
    print(result)
    return result





if __name__ == "__main__":
    dir_list = os.listdir(INPUT_DIR)
    df = pd.DataFrame()
    for img in dir_list:
        lf = Labeling(imgfile=img)
        result = main()
        df = df.append(result)

    df.to_csv("results2.csv",index=False)


# def cage(X, gold_label):


#     LFS = [ DOCTR_LABEL, TESSERACT_LABEL, CONTOUR_LABEL]

#     rules = LFSet("DETECTION_LF")
#     rules.add_lf_list(LFS)

#     n_lfs = len(rules.get_lfs())


#     path_json = 'sms_json.json'
#     T_path_pkl = 'pickle_T.pkl' #test data - have true labels
#     U_path_pkl = 'pickle_U.pkl' #unlabelled data - don't have true labels

#     log_path_cage_1 = 'sms_log_1.txt' #cage is an algorithm, can be found below
#     params_path = 'sms_params.pkl' #file path to store parameters of Cage, used below

#     X_train, X_test, y_train, y_test = train_test_split(X, gold_label ,random_state=104, test_size=0.2, shuffle=True)

#     sms_noisy_labels = PreLabels(name="sms",
#                                data=X_test,
#                                gold_labels=y_test,
#                                rules=rules,
#                                labels_enum=pixelLabels,
#                                num_classes=2)
#     sms_noisy_labels.generate_pickle(T_path_pkl)
#     sms_noisy_labels.generate_json(path_json) #generating json files once is enough

#     sms_noisy_labels = PreLabels(name="sms",
#                                 data=X_train,
#                                 rules=rules,
#                                 labels_enum=pixelLabels,
#                                 num_classes=2) #note that we don't pass gold_labels here, for the unlabelled data
#     sms_noisy_labels.generate_pickle(U_path_pkl)



#     data_U = get_data(path = U_path_pkl, check_shapes=True)
#     #check_shapes being True(above), asserts for relative shapes of arrays in pickle file
#     print("Number of elements in data list: ", len(data_U))
#     print("Shape of feature matrix: ", data_U[0].shape)
#     print("Shape of labels matrix: ", data_U[1].shape)
#     print("Shape of continuous scores matrix : ", data_U[6].shape)
#     print("Total number of classes: ", data_U[9])

#     classes = get_classes(path = path_json)
#     print("Classes dictionary in json file(modified to have integer keys): ", classes)

#     cage = Cage(path_json = path_json, n_lfs = n_lfs)


#     cage = Cage(path_json = path_json, n_lfs = n_lfs)

#     probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_test = T_path_pkl, path_log = log_path_cage_1, \
#                                     qt = 0.9, qc = 0.85, metric_avg = ['binary'], n_epochs = 200, lr = 0.01)
#     labels = np.argmax(probs, 1)
#     print("probs shape: ", probs.shape)
#     print("labels shape: ",labels.shape)
#     print(labels)
#     with open('results.pkl', 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(labels, outp, pickle.HIGHEST_PROTOCOL)