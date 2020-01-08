"""COCO-Style类型数据集
"""

import os
import os.path as osp
import cv2
import numpy as np
from pycocotools.coco import COCO

import torch
from torch.utils.data import Dataset as tDataset


class COCODataset(tDataset):

    def __init__(self, root, ann_file, transforms=None):
        """
        Arguments:
            root {str} -- 图片目录
            ann_file {str} -- 标注json文件路径
        
        Keyword Arguments:
            transforms {transform} -- 图片增强方式 (default: {None})
        """
        self.root = root
        self.transforms = transforms
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        self.relabel_classes()

    def relabel_classes(self):
        """重新整理类别顺序
        """
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
        # 名字->id
        self.classes = {}
        # 重新排序的id->原id
        self.coco_labels = {}
        # 原id->重新排序的id
        self.coco_labels_inv = {}
        for cat in categories:
            self.coco_labels[len(self.classes)] = cat['id']
            self.coco_labels_inv[cat['id']] = len(self.classes)
            self.classes[cat['name']] = len(self.classes)

        # id->名字
        self.labels = {value: key for key, value in self.classes.items()}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        annotations = self.load_annotations(idx)
        sample = {
            "image": image,
            "bboxes": annotations[:, :4],
            "labels": annotations[:, 4]
        }
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, idx

    def load_image(self, image_id):
        """根据图片id读取对应的图片
        
        Arguments:
            image_id {int} -- 图片id
        
        Returns:
            numpy.ndarray -- BGR形式的numpy数组
        """
        image_info = self.coco.loadImgs(self.image_ids[image_id])[0]
        image = cv2.imread(osp.join(self.root, image_info["file_name"]))
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def load_annotations(self, image_id):
        """根据图片id读取其所有的标注框
        
        Arguments:
            image_id {int} -- 图片id
        
        Returns:
            numpy.ndarray -- Nx5数组，代表N个框，每个框前四个元素为位置，最后一个元素为类别
        """
        annotation_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_id], iscrowd=False)
        annotations = np.zeros((0, 5))

        # 如果没有标注
        if len(annotation_ids) == 0:
            return annotations

        coco_anns = self.coco.loadAnns(annotation_ids)
        for _, coco_ann in enumerate(coco_anns):
            # 过滤无用的标注
            if coco_ann["bbox"][2] < 1 or coco_ann["bbox"][3] < 1:
                continue
            annotation = np.zeros((1, 5))
            annotation[0, :4] = coco_ann["bbox"]
            annotation[0, 4] = self.relabel_coco_label(coco_ann["category_id"])
            annotations = np.append(annotations, annotation, axis=0)

        # 将[x,y,w,h]形式转化为[x1,y1,x2,y2]形式
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def relabel_coco_label(self, coco_label):
        return self.coco_labels_inv[coco_label]

    def return_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_id):
        image_info = self.coco.loadImgs(self.image_ids[image_id])[0]
        return float(image_info["width"]) / float(image_info["height"])

    @property
    def num_classes(self):
        return len(self.labels)
