import os
import os
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

compound_coef = 3
weight_model = f'weights/efficientdet-d{compound_coef}_86_001.pth'  # efficientdet-d3_86_001.pth
print(weight_model)
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

# bdd
obj_list = ['bus', 'traffic light', 'traffic sign', 'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(weight_model)) # efficientdet-d3_001.pth
model.requires_grad_(False)
model.eval()
if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

def single_display(preds, img, img_path=''):
    if len(preds[0]['rois']) == 0:
        return

    for j in range(len(preds[0]['rois'])):
        x1, y1, x2, y2 = preds[0]['rois'][j].astype(np.int)
        obj = obj_list[preds[0]['class_ids'][j]]
        score = float(preds[0]['scores'][j])
        plot_one_box(img[0], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

    cv2.imwrite(img_path[:-4]+'_detected.jpg', img[0])

# images_path = '/data/bdd_100k/bdd2coco2020/test_hengtong'
images_path = '/data/bdd_100k/bdd2coco2020/test_zhangjiang'
images_path = '/data/bdd_100k/bdd2coco2020/test2020'
images = os.listdir(images_path)
print(len(images))
for img in images:
    img_full_path = os.path.join(images_path, img)
    print(img_full_path)
    ori_imgs, framed_imgs, framed_metas = preprocess(img_full_path, max_size=input_size)
    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
    with torch.no_grad():
        features, regression, classification, anchors = model(x)
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold)
    
    out = invert_affine(framed_metas, out)
    single_display(out, ori_imgs, img_full_path)




