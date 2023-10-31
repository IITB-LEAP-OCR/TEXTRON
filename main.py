from src.lf_utils import *
from src.config import *
from src.utils import get_label

from PIL import Image
from tqdm import tqdm

import subprocess
import random
import pandas as pd

from doctr.models import ocr_predictor

from spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from spear.spear.labeling import LFSet, PreLabels
from spear.spear.cage import Cage

from src.data_processing import Labeling, pixelLabels
from src.post_processing import get_bboxes, coco_conversion

import warnings
warnings.filterwarnings("ignore")


imgfile =  None
Y = None
lf = None
MODEL = ocr_predictor(pretrained=True)


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
def get_title_contour_info(x):
    return lf.TITLE_CONTOUR[x[0]][x[1]]

@preprocessor()
def get_mask_holes_info(x):
    return lf.MASK_HOLES[x[0]][x[1]]

@preprocessor()
def get_mask_objects_info(x):
    return lf.MASK_OBJECTS[x[0]][x[1]]

@preprocessor()
def get_segmentation_info(x):
    return lf.SEGMENTATION[x[0]][x[1]]



@labeling_function(label = pixelLabels.NOT_TEXT, pre=[get_chull_info], name="CHULL_PURE")
def CONVEX_HULL_LABEL_PURE(pixel):
    if(not pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN
    
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
    if(not pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN
    
    
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
    if(not pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_doctr_info], name="DOCTR_REVERSE")
def DOCTR_LABEL_REVERSE(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN


@labeling_function(label=pixelLabels.TEXT, pre=[get_tesseract_info], name="TESSERACT")
def TESSERACT_LABEL(pixel):
    if(not pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN
    
@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_tesseract_info], name="TESSERACT_REVERSE")
def TESSERACT_LABEL_REVERSE(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.TEXT, pre=[get_contour_info], name="CONTOUR")
def CONTOUR_LABEL(pixel):
    if(not pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_contour_info], name="CONTOUR_REVERSE")
def CONTOUR_LABEL_REVERSE(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.TEXT, pre=[get_title_contour_info], name="CONTOUR_TITLE")
def CONTOUR_TITLE_LABEL(pixel):
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

@labeling_function(label=pixelLabels.TEXT, pre=[get_segmentation_info], name="SEGMENTATION")
def SEGMENTATION_LABEL(pixel):
    if(not pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN
    
@labeling_function(label=pixelLabels.NOT_TEXT, pre=[get_segmentation_info], name="SEGMENTATION_REVERSE")
def SEGMENTATION_LABEL_REVERSE(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN


### Get LF Analysis of the input images
def analysis(img):

    ### Labeling Functions which should be run
    LFS = [globals()[LF] for LF in lab_funcs]

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    R = np.zeros((lf.pixels.shape[0],len(rules.get_lfs())))
    Y = io.imread(INPUT_IMG_DIR + img)
    
    name = img[:len(img) - 8]
    
    if GROUND_TRUTH_AVAILABLE:
        df = pd.read_csv(GROUND_TRUTH_DIR+name+'_pro.txt', delimiter=' ',
                     names=["token", "x0", "y0", "x1", "y1", "R", "G", "B", "font name", "label"])

        height, width, _ = Y.shape
        for i in range(df.shape[0]):
            x0, y0, x1, y1  = (df['x0'][i], df['y0'][i], df['x1'][i], df['y1'][i])
            x0, y0, x1, y1 = (int(x0*width/1000), int(y0*height/1000), int(x1*width/1000), int(y1*height/1000))
            w = int((x1-x0)*WIDTH_THRESHOLD)
            h = int((y1-y0)*HEIGHT_THRESHOLD)
            cv2.rectangle(Y, (x0, y0), (x0+w, y0+h), (0, 0, 0), cv2.FILLED)

    #gold_label = get_label(Y)

    td_noisy_labels = PreLabels(name="TD",
                               data=lf.pixels,
                               rules=rules,
                               #gold_labels=gold_label,
                               labels_enum=pixelLabels,
                               num_classes=2)

    L,S = td_noisy_labels.get_labels()


    analyse = td_noisy_labels.analyse_lfs(plot=True)

    result = analyse.head(16)
    result["image"] = img
    return result


### Get CAGE based output predictions
def cage(file, X, only_pred):

    ### Labeling Functions which should be run
    LFS = [globals()[LF] for LF in lab_funcs]

    prob_arr = np.array(QUALITY_GUIDE)

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    n_lfs = len(rules.get_lfs())

    Y = io.imread(INPUT_IMG_DIR + file)
    height, width, _ = Y.shape

    if(GROUND_TRUTH_AVAILABLE):
        if False:
            name = file[:len(file) - 4]
            df = pd.read_csv(GROUND_TRUTH_DIR+name+'.txt', delimiter=' ',
                            names=["token", "x0", "y0", "x1", "y1", "R", "G", "B", "font name", "label"])

            for i in range(df.shape[0]):
                x0, y0, x1, y1  = (df['x0'][i], df['y0'][i], df['x1'][i], df['y1'][i])
                x0, y0, x1, y1 = (int(x0*width/1000), int(y0*height/1000), int(x1*width/1000), int(y1*height/1000))
                w = int((x1-x0)*WIDTH_THRESHOLD)
                h = int((y1-y0)*HEIGHT_THRESHOLD)
                cv2.rectangle(Y, (x0, y0), (x0+w, y0+h), (0, 0, 0), cv2.FILLED)

        elif os.path.exists(GROUND_TRUTH_DIR+ file[:len(file) - 4] +'.txt'):
            #('Just' in INPUT_DATA_DIR) or 'testing_sample' in INPUT_DATA_DIR and 'cTDaR' not in file
            name = file[:len(file) - 4]
            df = pd.read_csv(GROUND_TRUTH_DIR+name+'.txt', delimiter=' ',
                            names=["label","x0","y0",'w','h'])   

            
            for i in range(df.shape[0]):
                x0, y0, w, h  = (df['x0'][i], df['y0'][i], df['w'][i], df['h'][i])
                w = int(w*WIDTH_THRESHOLD)
                h = int(h*HEIGHT_THRESHOLD)
                cv2.rectangle(Y, (x0, y0), (x0+w, y0+h), (0, 0, 0), cv2.FILLED)
        
        else:
            Y = io.imread(INPUT_IMG_DIR + file)
        
    
    gold_label = get_label(Y)


    path_json = 'sms_json.json'
    T_path_pkl = 'pickle_T.pkl' #test data - have true labels
    U_path_pkl = 'pickle_U.pkl' #unlabelled data - don't have true labels

    log_path_cage_1 = 'sms_log_1.txt' #cage is an algorithm, can be found below


    if not (only_pred):
        sms_noisy_labels = PreLabels(name="sms",
                                   data=X,
                                  gold_labels=gold_label,
                                  rules=rules,
                                   labels_enum=pixelLabels,
                                   num_classes=2)
        sms_noisy_labels.generate_pickle(T_path_pkl)
        sms_noisy_labels.generate_json(path_json) #generating json files once is enough

    sms_noisy_labels = PreLabels(name="sms",
                                data=X,
                                rules=rules,
                                labels_enum=pixelLabels,
                                num_classes=2) #note that we don't pass gold_labels here, for the unlabelled data
    sms_noisy_labels.generate_pickle(U_path_pkl)


    cage = Cage(path_json = path_json, n_lfs = n_lfs)

    print(PARAMS_FILE)
    
    if(os.path.exists(PARAMS_FILE)):
        cage.load_params(load_path = PARAMS_FILE)
        print('loaded params')

    if not (only_pred):
        probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_test = T_path_pkl, path_log = log_path_cage_1, \
                                    qt = prob_arr, qc = prob_arr, metric_avg = ['binary'], n_epochs = CAGE_EPOCHS, lr = 0.01)
    else:
        probs = cage.predict_proba(path_test = U_path_pkl, qc = prob_arr)

    labels = np.argmax(probs, 1)
    x,y,_ = Y.shape

    labels = labels.reshape(x,y)
    im = Image.fromarray((labels * 255).astype(np.uint8))
    im.save(RESULTS_DIR + file)

    if not only_pred:
        cage.save_params(save_path = PARAMS_FILE)
    # io.imsave(RESULTS_DIR + file, labels)


### Main Code
if __name__ == "__main__":
    dir_list = os.listdir(INPUT_IMG_DIR)

    random.shuffle(dir_list)

    data_size  = len(dir_list)
    test_split = int((data_size+1)*SPLIT_THRESHOLD)
    train_data = dir_list[:test_split] #Remaining 80% to training set
    test_data  = dir_list[test_split:] #Splits 20% data to test set

    ### CAGE Execution
    # for img_file in tqdm(train_data):
    #     # if not (os.path.exists(RESULTS_DIR + img_file)):
    #     lf = Labeling(imgfile=img_file, model=MODEL)
    #     cage(img_file, lf.pixels, only_pred=False)
    #     get_bboxes(img_file)

    ### Predictions on Test
    # print(test_data)
    for img_file in tqdm(test_data):
        if not (os.path.exists(RESULTS_DIR + img_file)):
            lf = Labeling(imgfile=img_file, model=MODEL)
            cage(img_file, lf.pixels, only_pred=PRED_ONLY)
            get_bboxes(img_file)
    
    # coco_conversion()

    #subprocess.run(["python3","./iou-results/pascalvoc.py","-gt", '../' + GROUND_TRUTH_DIR, "-det", '../' + OUT_TXT_DIR])

    ## SPEAR EXECUTION
    # df = pd.DataFrame()
    # for img in tqdm(dir_list):
    #    lf = Labeling(imgfile=img, model=MODEL)
    #    result = analysis(img)
    #    df = df.append(result)

    # df.to_csv("results_only_some.csv",index=False)

    
