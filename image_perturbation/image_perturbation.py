import numpy as np
import cv2
import skimage
import re
from skimage import io
from pathlib import Path
from .utils import Config
from .modeling_frcnn import GeneralizedRCNN
from .preprocess_image import Preprocess


OBJ_ID_PATTERN = r'obj[\d+]_id'


class ImageBuffer:
    def __init__(self, scenes, dataset, num_detections, device='cuda', root_dir='./images/'):
        self.image = None
        self.root_dir = root_dir
        self.a_id = None
        self.feats = None
        self.boxes = None
        self.assignment = None
        self.device = device
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        # frcnn_cfg.model.device = self.device
        # self.num_detections = num_detections
        # frcnn_cfg.min_detections = self.num_detections
        # frcnn_cfg.max_detections = self.num_detections
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
        self.image_preprocess = Preprocess(frcnn_cfg)
        self.a_id_2_assignment = dict()
        for datum in dataset.data:
            a_id = '-'.join(datum['question_id'].split('-')[:3])
            if a_id not in self.a_id_2_assignment:
                self.a_id_2_assignment[a_id] = datum['assignment']
            else:
                assert datum['assignment'] == self.a_id_2_assignment[a_id]
        self.scenes = scenes

    def _get_img_id(self):
        return self.a_id.split('-')[0]

    def _get_a_id(self, question_id):
        return '-'.join(question_id.split('-')[:3])

    def _set_blurred_img(self):
        img_id = self._get_img_id()
        scene = self.scenes[img_id]
        orig_image = cv2.imread(get_img_file(img_id, self.root_dir))
        self.image = torch.Tensor(blur_img_objects(orig_image, scene, self.assignment, sigma=3))

    def __getitem__(self, key):
        a_id = self._get_a_id(key)
        if a_id != self.a_id:
            self.a_id = a_id
            self.assignment = self.a_id_2_assignment[a_id]
            self._set_blurred_img()
            image, size, scale_yx = self.image_preprocess([self.image, ])
            output_dict = self.frcnn(image, size, scales_yx=scale_yx, padding="max_detections",
                                     max_detections=self.num_detections, return_tensors="pt")
            self.feats = output_dict.get("roi_features")
            self.boxes = output_dict.get("normalized_boxes")
        return self.feats, self.boxes


def imshow(img):
    skimage.io.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
def get_img_file(img_id, root_dir):
    return Path(root_dir, img_id, '.jpg')


def get_bbox(obj, max_x, max_y, padding=6):
    x0, y0 = obj['x'], obj['y']
    x1, y1 = x0 + obj['w'], y0 + obj['h']
    return max(x0-padding, 0), min(x1+padding, max_x), max(y0-padding, 0), min(y1+padding, max_y)


def mask(img, bbox):
    x0, x1, y0, y1 = bbox
    m = np.zeros_like(img, dtype=bool)
    m[y0:y1, x0:x1:, :] = True
    return m


def blur_context(img, sigma, bboxes):
    ksize = 0
    img_blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    objs_mask = np.zeros_like(img, dtype=bool)
    for bbox in bboxes:
        objs_mask |= mask(img, bbox)
    objs_mask = objs_mask.astype('uint8')
    objs_mask_blurred = cv2.GaussianBlur(objs_mask, (ksize, ksize), sigma)
    return img_blurred * (1 - objs_mask_blurred) + objs_mask_blurred * img


def blur_img_objects(img, scene, assignment, sigma=3):
    max_x = scene['width']
    max_y = scene['height']
    obj_ids = list()
    for k in assignment:
        match = re.search(OBJ_ID_PATTERN, k)
        if match:
            if assignment[k]:
                obj_ids.append(assignment[k])
    bboxes = [get_bbox(scene['objects'][obj_id], max_x, max_y) for obj_id in obj_ids]
    a_id_img = blur_context(img, sigma, bboxes)
    return a_id_img


def black_context(img, bboxes):
    black = np.zeros_like(img)
    for bbox in bboxes:
        print(bbox)
        m = mask(img, bbox)
        black[m] = img[m]
    return black


def black_objects(img, bboxes):
    black = np.zeros_like(img)
    copy = img.copy()
    for bbox in bboxes:
        print(bbox)
        m = mask(img, bbox)
        copy[m] = black[m]
    return copy


def black_context_and_objects(img, bboxes):
    black = np.zeros_like(img)
    context = img.copy()
    objects = black.copy()
    for bbox in bboxes:
        print(bbox)
        m = mask(img, bbox)
        context[m] = black[m]
        objects[m] = img[m]
    return context, objects
