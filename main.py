from src.lfs import *
from src.config import *
from src.utils import get_pixels, get_label
import enum
import pandas as pd
from doctr.models import ocr_predictor

from PIL import Image
import pickle
from tqdm import tqdm
import subprocess 

from spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from spear.spear.labeling import LFAnalysis, LFSet, PreLabels
from spear.spear.utils import get_data, get_classes

from sklearn.model_selection import train_test_split

from spear.spear.cage import Cage


class pixelLabels(enum.Enum):
    TEXT = 1
    NOT_TEXT = 0

class Labeling:

    def __init__(self,imgfile, model) -> None:
        self.imgfile = imgfile
        image = io.imread(imgfile)
        image2 = Image.open(imgfile)
        image3 = cv2.imread(imgfile)
        self.CHULL        = get_convex_hull(image)
        self.EDGES        = get_image_edges(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.PILLOW_EDGES = get_pillow_image_edges(image2, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.CONTOUR      = get_contour_labels(image3, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.DOCTR        = get_doctr_labels(model, self.imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.TESSERACT    = get_tesseract_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.MASK_HOLES   = get_mask_holes_labels(image)
        self.MASK_OBJECTS = get_mask_objects_labels(image, LUMINOSITY)
        self.SEGMENTATION = get_segmentation_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.pixels = get_pixels(image)
        self.image = image



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
    if(pixel):
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

@labeling_function(label=pixelLabels.TEXT, pre=[get_doctr_info], name="DOCTR2")
def DOCTR_LABEL2(pixel):
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

@labeling_function(label=pixelLabels.TEXT, pre=[get_segmentation_info], name="SEGMENTATION")
def SEGMENTATION_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN



def main(img):
    
    LFS = [ 
        CONVEX_HULL_LABEL_PURE, 
        CONVEX_HULL_LABEL_NOISE, 
        EDGES_LABEL, 
        EDGES_LABEL_REVERSE, 
        PILLOW_EDGES_LABEL, 
        PILLOW_EDGES_LABEL_REVERSE,
        DOCTR_LABEL, TESSERACT_LABEL, CONTOUR_LABEL,
        MASK_HOLES_LABEL, MASK_OBJECTS_LABEL,
        SEGMENTATION_LABEL
    ]

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    

    R = np.zeros((lf.pixels.shape[0],len(rules.get_lfs())))


    gold_label = get_label(Y)


    td_noisy_labels = PreLabels(name="TD",
                               data=lf.pixels,
                               rules=rules,
                               gold_labels=gold_label,
                               labels_enum=pixelLabels,
                               num_classes=2)

    L,S = td_noisy_labels.get_labels()


    analyse = td_noisy_labels.analyse_lfs(plot=True)

    result = analyse.head(16)
    result["image"] = img

    # data_U = get_data(path = './TD_pickle.pkl', check_shapes=True)
    # #check_shapes being True(above), asserts for relative shapes of arrays in pickle file
    # print("Number of elements in data list: ", len(data_U))
    # print("Shape of feature matrix: ", data_U[0].shape)
    # print("Shape of labels matrix: ", data_U[1].shape)
    # print("Shape of continuous scores matrix : ", data_U[6].shape)
    # print("Total number of classes: ", data_U[9])

    # classes = get_classes(path = './TD_json.json')
    # print("Classes dictionary in json file(modified to have integer keys): ", classes)
    print(result)
    return result


def cage(file, name, X, Y):

    # 1. CONVEX_HULL_LABEL_PURE, 
    # 2. CONVEX_HULL_LABEL_NOISE, 
    # 3. EDGES_LABEL, 
    # 4. EDGES_LABEL_REVERSE, 
    # 5. PILLOW_EDGES_LABEL, 
    # 6. PILLOW_EDGES_LABEL_REVERSE,
    # 7. DOCTR_LABEL,
    # 8. TESSERACT_LABEL,
    # 9. CONTOUR_LABEL,
    # 10. MASK_HOLES_LABEL,
    # 11. MASK_OBJECTS_LABEL
    # 12. SEGMENTATION_LABEL

    LFS = [ 
        CONVEX_HULL_LABEL_PURE, 
        CONVEX_HULL_LABEL_NOISE, 
        EDGES_LABEL, 
        # EDGES_LABEL_REVERSE, 
        # PILLOW_EDGES_LABEL, 
        # PILLOW_EDGES_LABEL_REVERSE,
        DOCTR_LABEL,
        DOCTR_LABEL2,
        # TESSERACT_LABEL,
        CONTOUR_LABEL,
        # MASK_HOLES_LABEL,
        # MASK_OBJECTS_LABEL,
        # SEGMENTATION_LABEL
    ]

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    n_lfs = len(rules.get_lfs())

    gold_label = get_label(Y)


    path_json = 'sms_json.json'
    T_path_pkl = 'pickle_T.pkl' #test data - have true labels
    U_path_pkl = 'pickle_U.pkl' #unlabelled data - don't have true labels

    log_path_cage_1 = 'sms_log_1.txt' #cage is an algorithm, can be found below
    params_path = 'sms_params.pkl' #file path to store parameters of Cage, used below


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



    data_U = get_data(path = U_path_pkl, check_shapes=True)
    classes = get_classes(path = path_json)

    cage = Cage(path_json = path_json, n_lfs = n_lfs)


    # cage = Cage(path_json = path_json, n_lfs = n_lfs)

    probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_test = T_path_pkl, path_log = log_path_cage_1, \
                                    qt = 0.9, qc = 0.85, metric_avg = ['binary'], n_epochs = 100, lr = 0.01)
    labels = np.argmax(probs, 1)
    x,y,_ = Y.shape

    labels = labels.reshape(x,y)
    if(GROUND_TRUTH==True):
        if(IS_EXPERIMENT):
            io.imsave(RESULTS_DIR + name + str(EXPERIMENT_VALUE) + '_ori_pro.jpg', labels)
        else:
            io.imsave(RESULTS_DIR + name + '_ori_pro.jpg', labels)
    else:
        io.imsave(RESULTS_DIR + file, labels)


def get_bboxes(file1,file, imgfile, model):


    if(GROUND_TRUTH == True):
        # df = pd.read_csv(ORI_TXT_DIR+file+'.txt', delimiter='\t')
        if(IS_EXPERIMENT):
            img = cv2.imread(RESULTS_DIR+file+ str(EXPERIMENT_VALUE) +'_ori_pro.jpg')
        else:
            img = cv2.imread(RESULTS_DIR+file +'_ori_pro.jpg')
    else:
        img = cv2.imread(RESULTS_DIR + file1)

    if(IS_DOCTR_AND == True):
        image = io.imread(imgfile)
        doctr = get_doctr_labels(model, imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        img2 = binarize_image(img)
        #img = np.logical_or(doctr, img2)
        img = doctr * img2
        #img = invert(img)
        io.imsave("temp.jpg",img)
        img = cv2.imread("temp.jpg")

    img = invert(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs. hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)

    #This is inmtermediate contour image having red contours plotted along the letters
    #with_contours_int = cv2.drawContours(img, contours, -1,(0,0,255),2)

    #We again perform binarization of above image inorder to find contours again 
    #gray_contour = cv2.cvtColor(with_contours_int, cv2.COLOR_BGR2GRAY)

    #ret, binary_contour = cv2.threshold(gray_contour, 100, 255, 
    #cv2.THRESH_OTSU)
    #inverted_contour = ~binary_contour

    # We find contours again of this inverted binary map so that word boundaries are detected
    #contours, hierarchy = cv2.findContours(inverted_contour,
    #cv2.RETR_TREE,
    #cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w = int(w*(0.95/WIDTH_THRESHOLD))
        h = int(h*(0.95/HEIGHT_THRESHOLD))
        # Make sure contour area is large enough
        if (cv2.contourArea(c)) > 25 and (cv2.contourArea(c) < 10000):
            bboxes.append(['text',1,x, y, w, h])

    final_img = cv2.imread(INPUT_DIR + file1)
    for b in bboxes:
        x = b[2]
        y = b[3]
        w = int(b[4])
        h = int(b[5])
        if(x=='0' and y=='0'):
            bboxes.remove(b)
            continue
        else:
            cv2.rectangle(final_img,(x,y), (x+w,y+h), (0, 255, 128),1)

    df = pd.DataFrame(bboxes, columns = ['label', 'confidence', 'x0', 'y0', 'w', 'h'])
    if(IS_EXPERIMENT):
        io.imsave(PREDICTIONS_DIR + file + str(EXPERIMENT_VALUE) + '_pred.jpg', final_img)
        df.to_csv(OUT_TXT_DIR + file + str(EXPERIMENT_VALUE) + '.txt', sep=' ',index=False)
    else:
        name = file1[:len(file1) - 4]
        io.imsave(PREDICTIONS_DIR + name + '_pred.jpg', final_img)
        df.to_csv(OUT_TXT_DIR + name + '.txt', sep=' ',index=False)


if __name__ == "__main__":
    dir_list = os.listdir(INPUT_DIR)

    ### CAGE Execution
    for img in tqdm(dir_list):
        # if not os.path.isfile(RESULTS_DIR + img):
        name = img[:len(img) - 12]
        # Y = io.imread(INPUT_DIR + name + '_ori_pro.jpg')
        Y = io.imread(INPUT_DIR + img)
        imgfile = INPUT_DIR + img
        lf = Labeling(imgfile=imgfile, model=MODEL)

        cage(img, name, lf.pixels, Y)
        get_bboxes(img, name, imgfile, model=MODEL)

    subprocess.run(["python","./iou-results/pascalvoc.py","-gt", '../' + GROUND_TRUTH_DIR, "-det", '../' + OUT_TXT_DIR])

    ### SPEAR EXECUTION
    df = pd.DataFrame()
    for img in dir_list:
        if(img == '100.tar_1705.04261.gz_main_11_ori_pro.jpg'):
            name = img[:len(img) - 11]
            Y = io.imread(LABELS_DIR + name + 'ann_pro.jpg')
            imgfile = INPUT_DIR + img
            lf = Labeling(imgfile=imgfile, model=MODEL)
            result = main(img)
            df = df.append(result)

    df.to_csv("results_only_some.csv",index=False)

    