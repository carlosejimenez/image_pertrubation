import re
import torch
from pathlib import Path
import cv2
import skimage
import requests
from skimage import io
import numpy as np
from .utils import Config
from .modeling_frcnn import GeneralizedRCNN
from .preprocess_image import Preprocess


AVERAGES = np.array([106.12723969, 115.13348752, 119.6278144 ])
OBJ_ID_PATTERN = r'obj[\d+]_id'


class ImageProcessor:
    def __init__(self, questions, scenes, mode, root_dir='./images/', **mode_kwargs):
        self.root_dir = root_dir
        self.use_url = False
        if not Path(self.root_dir).is_dir():
            self.use_url = True
            self.root_dir = self.root_dir.rstrip('/')
        self.questions = questions if type(questions) is dict else {q['question_id']: q for q in questions}
        self.mode = mode
        self.scenes = scenes
        if mode == 'blur':
            self.mode_func = blur_context
        elif mode == 'avg':
            self.mode_func = avg_context
        elif mode =='crop':
            self.mode_func = crop_context
        self.mode_kwargs = mode_kwargs

    def set_mode(self, mode, **mode_kwargs):
        if mode == 'blur':
            self.mode = 'blur'
            self.mode_func = blur_context
            if mode_kwargs['sigma'] == 0:
                print('Warning: Parameter sigma set to 0. Output images will not be blurred.')
        elif mode == 'avg':
            self.mode = 'avg'
            self.mode_func = avg_context
        else:
            assert mode is None
            self.mode = None
            self.mode_func = None
        self.mode_kwargs = mode_kwargs

    def __getitem__(self, key):
        question = self.questions[key]
        img_id = question['img_id']
        assignment = question['assignment']
        scene = self.scenes[img_id]
        if self.use_url:
            arr = np.asarray(bytearray(requests.get(self.root_dir + '/%s.jpg' % img_id, stream=True).content), dtype=np.uint8)
            orig_image = cv2.imdecode(arr, -1)
        else:
            orig_image = cv2.imread(get_img_file(img_id, self.root_dir))
        if self.mode is None or (self.mode == 'blur' and self.mode_kwargs['sigma'] == 0):
            return np.array(orig_image)
        else:
            return np.array(apply_img_objects(self.mode_func, orig_image, scene, assignment, **self.mode_kwargs))


class ImageBuffer:
    def __init__(self, scenes, dataset, num_detections, mode, sigma=None, device='cuda', root_dir='./images/'):
        self.image = None
        self.root_dir = root_dir
        self.a_id = None
        self.feats = None
        self.boxes = None
        self.sigma = sigma
        self.mode = mode
        if mode == 'blur':
            self.mode_func = blur_context
            self.mode_kwargs = {'sigma': self.sigma}
        elif mode == 'avg':
            self.mode_func = avg_context
            self.mode_kwargs = dict()
        elif mode =='crop':
            self.mode_func = crop_context
            self.mode_kwargs = {'padding': 6}
        if self.sigma == 0:
            print('Warning: Parameter sigma set to 0. Output images will not be blurred.')
        self.assignment = None
        self.device = device
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn_cfg.model.device = self.device
        self.num_detections = num_detections
        # frcnn_cfg.min_detections = self.num_detections
        frcnn_cfg.max_detections = self.num_detections
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

    def _set_img(self):
        img_id = self._get_img_id()
        scene = self.scenes[img_id]
        orig_image = cv2.imread(get_img_file(img_id, self.root_dir))
        if self.mode == 'blur' and self.sigma == 0:
            self.image = torch.Tensor(orig_image)
        else:
            self.image = torch.Tensor(apply_img_objects(self.mode_func, orig_image, scene, self.assignment, **self.mode_kwargs))

#     def _set_blurred_img(self):
#         img_id = self._get_img_id()
#         scene = self.scenes[img_id]
#         orig_image = cv2.imread(get_img_file(img_id, self.root_dir))
#         if self.sigma > 0:
#             self.image = torch.Tensor(apply_img_objects(orig_image, scene, self.assignment, sigma=self.sigma))
#         else:
#             self.image = torch.Tensor(orig_image)

    def __getitem__(self, key):
        a_id = self._get_a_id(key)
        if a_id != self.a_id:
            self.a_id = a_id
            self.assignment = self.a_id_2_assignment[a_id]
            self._set_img()
            image, size, scale_yx = self.image_preprocess([self.image, ])
            output_dict = self.frcnn(image, size, scales_yx=scale_yx, padding="max_detections",
                                     max_detections=self.num_detections, return_tensors="pt")
            self.feats = output_dict.get("roi_features")
            self.boxes = output_dict.get("normalized_boxes")
        return self.feats, self.boxes


def imshow(img):
    skimage.io.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    
def get_img_file(img_id, root_dir):
    return Path(root_dir, img_id).with_suffix('.jpg').as_posix()


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
    objs_mask = objs_mask.astype('uint8') * 255
    objs_mask_blurred = cv2.GaussianBlur(objs_mask, (ksize, ksize), sigma) / 255
    return (img_blurred * (1 - objs_mask_blurred) + objs_mask_blurred * img).clip(0, 255).round().astype('uint8')


def avg_context(img, bboxes):
    img_avgd = np.ones_like(img) * AVERAGES
    objs_mask = np.zeros_like(img, dtype=bool)
    for bbox in bboxes:
        objs_mask |= mask(img, bbox)
    objs_mask = objs_mask.astype('uint8') * 255
    objs_mask_blurred = cv2.GaussianBlur(objs_mask, (0, 0), 6) / 255
    return (img_avgd * (1 - objs_mask_blurred) + objs_mask_blurred * img).clip(0, 255).round().astype('uint8')


def crop_context(img, bboxes):
    copy = img.copy()
    mx0, mx1 = np.inf, -np.inf
    my0, my1 = np.inf, -np.inf
    for bbox in bboxes:
        x0, x1, y0, y1 = bbox
        mx0 = min(x0, mx0)
        mx1 = max(x1, mx1)
        my0 = min(y0, my0)
        my1 = max(y1, my1)
    copy = copy[my0:my1,mx0:mx1]
    return copy

# def blur_img_objects(img, scene, assignment, sigma=3):
#     return apply_img_objects(blur_context, img, scene, assignment, sigma=3)
# 
# 
# def avg_img_objects(img, scene, assignment, sigma=3):
#    return apply_img_objects(avg_context, img, scene, assignment)


def apply_img_objects(apply_func, img, scene, assignment, **kwargs):
    max_x = scene['width']
    max_y = scene['height']
    obj_ids = list()
    padding = getattr(kwargs, 'padding', 6)
    for k in assignment:
        match = re.search(OBJ_ID_PATTERN, k)
        if match:
            if assignment[k]:
                obj_ids.append(assignment[k])
    bboxes = [get_bbox(scene['objects'][obj_id], max_x, max_y, padding=padding) for obj_id in obj_ids]
    a_id_img = apply_func(img=img, bboxes=bboxes, **kwargs)
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
