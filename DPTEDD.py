import enum
import pandas as pd
import numpy as np

import os
import pickle
from skimage import io

# from snorkel.labeling import labeling_function
# from snorkel.labeling import PandasLFApplier

from spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from spear.spear.labeling import LFAnalysis, LFSet, PreLabels
from spear.spear.utils import get_data, get_classes

# input_dir = './data/images/'
# results_dir = './data/results/'

# IMG = 'docbank_test_page-0001.jpg'




input_dir = './../temp_data/images/'
results_dir = './../temp_data/results/'

IMG = '10.tar_1701.04170.gz_TPNL_afterglow_evo_8_pro.jpg'
IMG2 = '10.tar_1701.04170.gz_TPNL_afterglow_evo_8_ann.jpg'



with open(results_dir + 'img_processing/convex_hull_results.pkl', 'rb') as f:
    val = pickle.load(f)

with open(results_dir + 'img_processing/edges_results.pkl', 'rb') as f:
    val2 = pickle.load(f)
    
with open(results_dir + 'img_processing/pillow_edges_results.pkl', 'rb') as f:
    val3 = pickle.load(f)

with open(results_dir + 'doctr/pixels.pkl', 'rb') as f:
    val4 = pickle.load(f)

with open(results_dir + 'tesseract/pixels.pkl', 'rb') as f:
    val5 = pickle.load(f)

with open(results_dir + 'labels/pixels.pkl', 'rb') as f:
    val6 = pickle.load(f)

LABELS = val6[IMG2]


CHULL = val[IMG]
EDGES = val2[IMG]
PILLOW_EDGES = val3[IMG]
DOCTR = val4[IMG]
TESSERACT = val5[IMG]



class pixelLabels(enum.Enum):
    TEXT = 1
    NOT_TEXT = 0



@preprocessor()
def get_chull_info(x):
    return CHULL[x[0]][x[1]]

@preprocessor()
def get_edges_info(x):
    return EDGES[x[0]][x[1]]

@preprocessor()
def get_pillow_edges_info(x):
    return PILLOW_EDGES[x[0]][x[1]]


@preprocessor()
def get_doctr_info(x):
    return DOCTR[x[0]][x[1]]


@preprocessor()
def get_tesseract_info(x):
    return TESSERACT[x[0]][x[1]]



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


@labeling_function(label=pixelLabels.TEXT, pre=[get_pillow_edges_info], name="PILLOW_EDGES")
def DOCTR_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN


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






if __name__ == "__main__":

    

    d = CHULL

    X = []
    Y = []
    for i in range(len(d)):
        for j in range(len(d[0])):
            val = (i,j)
            X.append(val)
            Y.append(LABELS[i][j])
    X = np.array(X)
    Y = np.array(Y)

    LFS = [CONVEX_HULL_LABEL_PURE, CONVEX_HULL_LABEL_NOISE, EDGES_LABEL, EDGES_LABEL_REVERSE, PILLOW_EDGES_LABEL, DOCTR_LABEL, TESSERACT_LABEL]

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

    print("===== All Done =====")
    print(result)
    # display(result)
    td_noisy_labels.generate_pickle()
    td_noisy_labels.generate_json()




    data_U = get_data(path = './TD_pickle.pkl', check_shapes=True)
    #check_shapes being True(above), asserts for relative shapes of arrays in pickle file
    print("Number of elements in data list: ", len(data_U))
    print("Shape of feature matrix: ", data_U[0].shape)
    print("Shape of labels matrix: ", data_U[1].shape)
    print("Shape of continuous scores matrix : ", data_U[6].shape)
    print("Total number of classes: ", data_U[9])

    classes = get_classes(path = './TD_json.json')
    print("Classes dictionary in json file(modified to have integer keys): ", classes)



    from spear.spear.cage import Cage

    cage = Cage(path_json = './TD_json.json', n_lfs = n_lfs)


    probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_test = T_path_pkl, path_log = log_path_cage_1, \
                                    qt = 0.9, qc = 0.85, metric_avg = ['binary'], n_epochs = 200, lr = 0.01)
    labels = np.argmax(probs, 1)
    print("probs shape: ", probs.shape)
    print("labels shape: ",labels.shape)


