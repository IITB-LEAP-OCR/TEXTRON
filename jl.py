import enum
import subprocess
from tqdm import tqdm

from doctr.models import ocr_predictor


from src.lf_utils import *
from src.config   import *
from src.utils    import get_pixels, get_label


from spear.spear.labeling import labeling_function, ABSTAIN, preprocessor
from spear.spear.labeling import LFSet, PreLabels
from spear.spear.utils import get_data, get_classes
from helper.utils import get_various_data
from spear.spear.jl import JL


import warnings
warnings.filterwarnings("ignore")



imgfile =  None
Y     = None
lf    = None
MODEL = ocr_predictor(pretrained=True)

class pixelLabels(enum.Enum):
    TEXT     = 1
    NOT_TEXT = 0
    
    
class Labeling:
    def __init__(self,imgfile, model) -> None:
        self.imgfile = INPUT_IMG_DIR + imgfile
        image = io.imread(self.imgfile)
        image3 = cv2.imread(self.imgfile)
        self.CHULL        = get_convex_hull(image)
        self.EDGES        = get_image_edges(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.CONTOUR      = get_contour_labels(image3, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.DOCTR        = get_doctr_labels(model, self.imgfile, image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.TESSERACT    = get_tesseract_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD)
        self.SEGMENTATION = get_segmentation_labels(image, WIDTH_THRESHOLD, HEIGHT_THRESHOLD, THICKNESS)
        self.pixels = get_pixels(image)
        self.image = image
        
        


@preprocessor()
def get_chull_info(x):
    return lf.CHULL[x[0]][x[1]]

@preprocessor()
def get_edges_info(x):
    return lf.EDGES[x[0]][x[1]]

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
def get_segmentation_info(x):
    return lf.SEGMENTATION[x[0]][x[1]]



@labeling_function(label = pixelLabels.NOT_TEXT, pre=[get_chull_info], name="CHULL_PURE")
def CONVEX_HULL_LABEL_PURE(pixel):
    if(pixel):
        return pixelLabels.NOT_TEXT
    else:
        return ABSTAIN

@labeling_function(label=pixelLabels.TEXT, pre=[get_edges_info], name="SKIMAGE_EDGES")
def EDGES_LABEL(pixel):
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

@labeling_function(label=pixelLabels.TEXT, pre=[get_segmentation_info], name="SEGMENTATION")
def SEGMENTATION_LABEL(pixel):
    if(pixel):
        return pixelLabels.TEXT
    else:
        return ABSTAIN
    
    
    
### Get LF Analysis of the input images
def analysis(img):

    ### Choose the Labeling Functions which should be run
    LFS = [ 
        CONVEX_HULL_LABEL_PURE, 
        EDGES_LABEL, 
        DOCTR_LABEL,
        TESSERACT_LABEL,
        CONTOUR_LABEL,
        SEGMENTATION_LABEL
    ]
    
    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    R = np.zeros((lf.pixels.shape[0],len(rules.get_lfs())))
    Y = io.imread(INPUT_IMG_DIR + img)
    name = img[:len(img) - 8]
    df = pd.read_csv(GROUND_TRUTH_DIR+name+'_pro.txt', delimiter=' ',
                     names=["token", "x0", "y0", "x1", "y1", "R", "G", "B", "font name", "label"])

    height, width, _ = Y.shape
    for i in range(df.shape[0]):
        x0, y0, x1, y1  = (df['x0'][i], df['y0'][i], df['x1'][i], df['y1'][i])
        x0, y0, x1, y1 = (int(x0*width/1000), int(y0*height/1000), int(x1*width/1000), int(y1*height/1000))
        w = int((x1-x0)*WIDTH_THRESHOLD)
        h = int((y1-y0)*HEIGHT_THRESHOLD)
        cv2.rectangle(Y, (x0, y0), (x0+w, y0+h), (0, 0, 0), cv2.FILLED)

    gold_label = get_label(Y)

    td_noisy_labels = PreLabels(name="TD", data=lf.pixels, rules=rules, gold_labels=gold_label, labels_enum=pixelLabels, num_classes=2)
    # L,S = td_noisy_labels.get_labels()
    analyse = td_noisy_labels.analyse_lfs(plot=True)

    result = analyse.head(16)
    result["image"] = img

    print(result)
    return result


### Postprocessing Step
def get_bboxes(file):

    img = cv2.imread(RESULTS_DIR + file)

    img = invert(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs. hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    bboxes = []
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        w = int(w*(1/WIDTH_THRESHOLD))
        h = int(h*(1/HEIGHT_THRESHOLD))
        # Make sure contour area is large enough
        if cv2.contourArea(c) > 30:
            bboxes.append(['text',1,x, y, w, h])

    final_img = cv2.imread(INPUT_IMG_DIR + file)
    for b in bboxes:
        x = b[2]
        y = b[3]
        w = int(b[4])
        h = int(b[5])
        cv2.rectangle(final_img,(x,y), (x+w,y+h), (0, 255, 0),1)

    df = pd.DataFrame(bboxes, columns = ['label', 'confidence', 'X', 'Y', 'W', 'H'])
    name = file[:len(file) - 4]
    io.imsave(PREDICTIONS_DIR + name + '_pred.jpg', final_img)
    df.to_csv(OUT_TXT_DIR + name + '.txt', sep=' ',index=False, header=False)


def jl(img_file, X):
    
    LFS = [ 
        CONVEX_HULL_LABEL_PURE, 
        EDGES_LABEL, 
        DOCTR_LABEL,
        TESSERACT_LABEL,
        CONTOUR_LABEL,
        SEGMENTATION_LABEL
    ]
    
    QUALITY_GUIDE = [0.85, 0.9, 0.95]
    prob_arr = np.array(QUALITY_GUIDE)

    rules = LFSet("DETECTION_LF")
    rules.add_lf_list(LFS)

    
    n_features = 3
    X_feats = np.random.uniform(low=-1, high=1, size=(X.shape[0],n_features))
    Y = get_label(io.imread(INPUT_IMG_DIR + img_file))
    

    validation_size = 1000
    test_size = 4000
    L_size = 100
    U_size = 4500
    n_lfs = len(rules.get_lfs())

    X_V, Y_V, X_feats_V,_, X_T, Y_T, X_feats_T,_, X_L, Y_L, X_feats_L,_, X_U, X_feats_U,_ = get_various_data(X, Y,\
        X_feats, n_lfs, validation_size, test_size, L_size, U_size)
    
    
    path_json = 'data_pipeline/JL/sms_json.json'
    V_path_pkl = 'data_pipeline/JL/sms_pickle_V.pkl' #validation data - have true labels
    T_path_pkl = 'data_pipeline/JL/sms_pickle_T.pkl' #test data - have true labels
    L_path_pkl = 'data_pipeline/JL/sms_pickle_L.pkl' #Labeled data - have true labels
    U_path_pkl = 'data_pipeline/JL/sms_pickle_U.pkl' #unlabelled data - don't have true labels

    log_path_jl_1 = 'log/JL/sms_log_1.txt' #jl is an algorithm, can be found below
    params_path = 'params/JL/sms_params.pkl' #file path to store parameters of JL, used below
    
    
    sms_noisy_labels = PreLabels(name="sms", data=X_V, gold_labels=Y_V, data_feats=X_feats_V, rules=rules, labels_enum=pixelLabels, num_classes=2)
    sms_noisy_labels.generate_pickle(V_path_pkl)
    sms_noisy_labels.generate_json(path_json) #generating json files once is enough

    sms_noisy_labels = PreLabels(name="sms", data=X_T, gold_labels=Y_T, data_feats=X_feats_T, rules=rules, labels_enum=pixelLabels, num_classes=2)
    sms_noisy_labels.generate_pickle(T_path_pkl)

    sms_noisy_labels = PreLabels(name="sms", data=X_L, gold_labels=Y_L, data_feats=X_feats_L, rules=rules, labels_enum=pixelLabels, num_classes=2)
    sms_noisy_labels.generate_pickle(L_path_pkl)

    sms_noisy_labels = PreLabels(name="sms", data=X_U, rules=rules, data_feats=X_feats_U, labels_enum=pixelLabels, num_classes=2) #note that we don't pass gold_labels here, for the unlabelled data
    sms_noisy_labels.generate_pickle(U_path_pkl)
    
    data_U = get_data(path = U_path_pkl, check_shapes=True)
    #check_shapes being True(above), asserts for relative shapes of arrays in pickle file
    print("Number of elements in data list: ", len(data_U))
    print("Shape of feature matrix: ", data_U[0].shape)
    print("Shape of labels matrix: ", data_U[1].shape)
    print("Shape of continuous scores matrix : ", data_U[6].shape)
    print("Total number of classes: ", data_U[9])

    classes = get_classes(path = path_json)
    print("Classes dictionary in json file(modified to have integer keys): ", classes)
    
    

    n_hidden = 512
    feature_model = 'nn'
    '''
    'nn' is neural network. other alternative is 'lr'(logistic regression) which doesn't need n_hidden to be passed
    during initialisation.
    ''' 

    loss_func_mask = [1,1,1,1,1,1,1] 
    '''
    One can keep 0s in places where he don't want the specific loss function to be part
    the final loss function used in training. Refer documentation(spear.JL.core.JL) to understand
    the which index of loss_func_mask refers to what loss function.

    Note: the loss_func_mask above may not be the optimal mask for sms dataset. We have to try
        some other masks too, to find the best one that gives good accuracies.
    '''
    batch_size = 150
    lr_fm = 0.0005
    lr_gm = 0.01
    use_accuracy_score = False

    jl = JL(path_json = path_json, n_lfs = n_lfs, n_features = n_features, feature_model = feature_model, \
            n_hidden = n_hidden)

    probs_fm, probs_gm = jl.fit_and_predict_proba(path_L = L_path_pkl, path_U = U_path_pkl, path_V = V_path_pkl, \
            path_T = T_path_pkl, loss_func_mask = loss_func_mask, batch_size = batch_size, lr_fm = lr_fm, lr_gm = \
        lr_gm, use_accuracy_score = use_accuracy_score, path_log = log_path_jl_1, return_gm = True, n_epochs = \
        100, start_len = 7,stop_len = 10, is_qt = True, is_qc = True, qt = 0.9, qc = 0.85, metric_avg = 'binary')

    labels = np.argmax(probs_fm, 1)
    print("probs_fm shape: ", probs_fm.shape)
    print("probs_gm shape: ", probs_gm.shape)


### Main Code
if __name__ == "__main__":
    dir_list = os.listdir(INPUT_IMG_DIR)

    ### JL Execution
    for img_file in tqdm(dir_list):
        lf = Labeling(imgfile=img_file, model=MODEL)
        jl(img_file, lf.pixels)
        get_bboxes(img_file)

    subprocess.run(["python","./iou-results/pascalvoc.py","-gt", '../' + GROUND_TRUTH_DIR, "-det", '../' + OUT_TXT_DIR])

    ### SPEAR EXECUTION
    df = pd.DataFrame()
    for img in tqdm(dir_list):
        lf = Labeling(imgfile=img, model=MODEL)
        result = analysis(img)
        df = df.append(result)

    df.to_csv("results_only_some.csv",index=False)