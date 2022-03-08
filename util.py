from IPython.display import Image as display_image
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from os import environ
import subprocess
import sys
import numpy as np
import json

score_threshold = 0.5

outputimage = "res.jpg"
label_json = "label2class.json"

if "LOCAL_CVFLOWBACKEND" in environ:
    cvb_libpath = environ["LOCAL_CVFLOWBACKEND"]
    cvb_ver = "local @ {}".format(cvb_libpath)
else:
    cvb_ver = None
    try:
        cvb_libpath = subprocess.check_output(['tv2', '-libpath', 'cvflowbackend'], 
            stderr=subprocess.STDOUT)
        cvb_libpath = cvb_libpath.decode().rstrip('\n')
    except Exception:
        raise OSError("No tv2 installation of cvflowbackend found.")
try:
    if cvb_libpath not in sys.path:
        sys.path.insert(0, cvb_libpath)
    import cvflowbackend 
except Exception:
    raise ImportError("Unable to import cvflowbackend from: %s" % cvb_libpath)



def draw_bbox(image_file, bboxes, scores, labels):

    def create_color_dict(classes, num_boxes):

        np.random.seed(1)

        # map distinct classes to unique colors
        uc     = list(set(classes[:num_boxes]))
        num_uc = len(uc)
        colors = np.random.randint(0, 0xFFFFFF, num_uc)

        colors_dict = {}
        for k,u in enumerate(uc):
            colors_dict[u] = "#%06x" % colors[k]

        return colors_dict

    # scores
    keep_idxs = scores > score_threshold
    scores = scores[keep_idxs]

    print(scores)
    
    num_boxes = 10

    # number of detected boxes
    num_boxes = num_boxes if num_boxes < len(scores) else len(scores)

    print(num_boxes)
    # classes -> color maps
    colors_dict = create_color_dict(labels, num_boxes)

    # bounding boxes -> x1,y1,x2,y2
    bboxes = bboxes.reshape([-1,4])
    bboxes = bboxes[keep_idxs,:]

    # labels
    if label_json is not None:
        with open(label_json) as f:
            lbl = json.load(f)
    else:
        lbl = None

    # read image
    img = Image.open(image_file)
    im = np.array(img, dtype=np.uint8)
    im_wd = img.width
    im_ht = img.height

    # Create figure and axes
    fig,ax = plt.subplots(1)
    ax.imshow(im)

    for i in range(num_boxes):

        b = bboxes[i,:]

        w_scale = im_wd/300.
        h_scale = im_ht/300.

        x_min = b[0] * w_scale
        y_min = b[1] * h_scale
        x_max = b[2] * w_scale
        y_max = b[3] * h_scale
        w = (x_max - x_min)
        h = (y_max - y_min)

        sc = scores[i]
        cl = labels[i]
        ch = colors_dict[cl]

        cl_info = str(cl)
        if lbl is not None:
            cl_info = lbl[cl_info]

        # Create a rectangle patch
        rect = patches.Rectangle((x_min,y_min), w, h, linewidth=2, edgecolor=ch, facecolor='none')

        ax.add_patch(rect)
        ax.text(x_min, y_min-3, "{0[0]:s}, {0[1]:.4f}".format([cl_info,sc]))

    if outputimage is None:
        basename, extn = os.path.splitext(image_file)
        out_img_fname = basename + '_bbox' + extn
    else:
        out_img_fname = outputimage

    plt.savefig(out_img_fname)

    return outputimage

def read_image(img_file, model_input_shape_tuple):
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img_array = cv2.resize(img, model_input_shape_tuple)
    img_array = np.transpose(img_array, (2,0,1))        
    img_array = np.expand_dims(img_array, 0)
    img_array = np.asarray(img_array, dtype=np.float32)
    img_array = img_array / 255.0
    
    return img_array