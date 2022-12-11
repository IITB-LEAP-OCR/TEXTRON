import enum
import numpy as np
import pickle

from spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from spear.spear.labeling import LFAnalysis, LFSet, PreLabels

input_dir = './../temp_data/images/'
results_dir = './../temp_data/results/'
IMG = '2.tar_1801.00617.gz_idempotents_arxiv_4_pro.jpg'
IMG2 = '2.tar_1801.00617.gz_idempotents_arxiv_4_ann.jpg'

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

with open(results_dir + 'img_processing/contour-based.pkl', 'rb') as f:
    val7 = pickle.load(f)

CHULL = val[IMG]
EDGES = val2[IMG]
PILLOW_EDGES = val3[IMG]
DOCTR = val4[IMG]
TESSERACT = val5[IMG]
LABELS = val6[IMG2]
CONTOUR = val7[IMG]

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

@preprocessor()
def get_contour_info(x):
    return CONTOUR[x[0]][x[1]]



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

@labeling_function(label=pixelLabels.TEXT, pre=[get_contour_info], name="CONTOUR")
def CONTOUR_LABEL(pixel):
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

    LFS = [CONVEX_HULL_LABEL_PURE, CONVEX_HULL_LABEL_NOISE, EDGES_LABEL, EDGES_LABEL_REVERSE, PILLOW_EDGES_LABEL, DOCTR_LABEL, TESSERACT_LABEL, CONTOUR_LABEL]
    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)
    R = np.zeros((X.shape[0],len(rules.get_lfs())))
    td_noisy_labels = PreLabels(name="TD",
                               data=X,
                               rules=rules,
                               gold_labels=Y,
                               labels_enum=pixelLabels,
                               num_classes=2)

    L,S = td_noisy_labels.get_labels()

    analyse = td_noisy_labels.analyse_lfs()
    result = analyse.head(16)

    print(result)
    #display(result)
    td_noisy_labels.generate_pickle()
    td_noisy_labels.generate_json()
