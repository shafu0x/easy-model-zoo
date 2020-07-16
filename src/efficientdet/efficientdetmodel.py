import time
import torch
from torch.backends import cudnn

from .backbone import EfficientDetBackbone
import cv2
from PIL import Image
import numpy as np
from bounding_box import bounding_box as bb

from ..model import Model

from .efficientdet.utils import BBoxTransform, ClipBoxes
from .utils.utils import invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

def preprocess(ori_imgs, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


class EfficientDetModel(Model):
    def __init__(self, name, weights_id, device='GPU', coeff=0):
        self.compound_coef = coeff
        super().__init__(name, weights_id, device)

    def _init_model(self):
        self.input_size = input_sizes[self.compound_coef] if force_input_size is None else force_input_size

        model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(obj_list),
                                    ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load(self.weights_f))
        model.requires_grad_(False)
        model.eval()

        if self.use_cuda: model = model.cuda()
        if use_float16: model = model.half()
        return model

    def _parse_name(self, name):
        for i in range(8):
            if str(i) in name: return i

    def _preprocess(self, image):
        img = Model.img2arr(image)
        if len(img.shape) <= 3: img = np.expand_dims(img, axis=0)

        ori_imgs, framed_imgs, framed_metas = preprocess(img, max_size=self.input_size)

        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        return x, framed_metas

    def run(self, image):
        x, framed_metas = self._preprocess(image)
        
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)

            out = invert_affine(framed_metas, out)

            return out

    def visualize(self, image, pred):
        img = Model.img2arr(image)
        rois, class_ids = EfficientDetModel.parse(pred)
        for i, roi in enumerate(rois):
            bb.add(img, roi[0],roi[1],roi[2],roi[3], str(class_ids[i]))
        Image.fromarray(img).show()

    @staticmethod
    def parse(out):
        pred_objs = []
        boxes = out[0]['rois']
        classes =  out[0]['class_ids']
        for i, obj_cl in enumerate(classes):
            pass
        return out[0]['rois'], out[0]['class_ids']
