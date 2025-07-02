import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, cache_images=False, single_cls=False):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.mosaic = self.augment and not rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.rect = rect
        
        # 加载图像和标签路径
        self.img_files, self.label_files = self._load_files(path)
        self.n = len(self.img_files)  # number of images
        self.indices = range(self.n)
        
        # 设置矩形训练
        if rect:
            # 从标签中获取图像宽高比
            shapes = self._get_img_shapes()
            # 根据宽高比排序
            self.batch_indices = np.floor(np.arange(self.n) / batch_size).astype(np.int)
            self.sort_files_by_aspect_ratio(shapes)
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        
        # 马赛克数据增强
        if self.mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None
            
            # MixUp augmentation
            if random.random() < self.hyp['mixup']:
                img2, labels2 = self.load_mosaic(random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            
            # Letterbox
            shape = self.batch_shapes[self.batch_indices[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
            
            # Load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]
        
        if self.augment:
            # Augment imagespace
            if not self.mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=self.hyp['degrees'],
                                                 translate=self.hyp['translate'],
                                                 scale=self.hyp['scale'],
                                                 shear=self.hyp['shear'],
                                                 perspective=self.hyp['perspective'])
            
            # Augment colorspace
            augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])
            
            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)
        
        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
            
            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width
        
        if self.augment:
            # flip up-down
            if random.random() < self.hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]
            
            # flip left-right
            if random.random() < self.hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]
        
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes
    
    def load_mosaic(self, index):
        # loads images in a 4-mosaic
        
        labels4 = []
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        
        for i, index in enumerate(indices):
            # Load image
            img, _, (h, w) = self.load_image(index)
            
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Labels
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
            labels4.append(labels)
        
        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # clip labels
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
        
        # Augment
        img4, labels4 = random_perspective(img4, labels4,
                                         degrees=self.hyp['degrees'],
                                         translate=self.hyp['translate'],
                                         scale=self.hyp['scale'],
                                         shear=self.hyp['shear'],
                                         perspective=self.hyp['perspective'],
                                         border=self.mosaic_border)  # border to remove
        
        return img4, labels4    