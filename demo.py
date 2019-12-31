
import os
import cv2
import argparse
import torch
import copy
from models import EfficientDet
from data import COCODataset, build_transform
from config import get_default_cfg, model_kwargs
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description="EfficientDet Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--img')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = get_default_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    num_classes = COCODataset(cfg.data.train[0], cfg.data.train[1]).num_classes
    size_image = (512, 512)
    test_transforms = build_transform(False)
    model = EfficientDet(num_classes=num_classes, **model_kwargs(cfg))
    device = torch.device(cfg.device)
    model.to(device)
    ckpt = torch.load(cfg.model.resume)['model']
    ckpt_dict = OrderedDict()
    for key, value in ckpt.items():
        if 'module' in key:
            key = key[7:]
        ckpt_dict[key] = value
    
    model.load_state_dict(ckpt_dict)
    model.eval()

    img = cv2.imread(args.img)
    origin_img = copy.deepcopy(img)
    augmentation = test_transforms(image=img)
    img = augmentation['image']
    img = img.to(device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        scores, classification, transformed_anchors = model(img)
        bboxes = list()
        labels = list()
        bbox_scores = list()
        colors = list()
        for j in range(scores.shape[0]):
            bbox = transformed_anchors[[j], :][0].data.cpu().numpy()
            x1 = int(bbox[0]*origin_img.shape[1]/size_image[1])
            y1 = int(bbox[1]*origin_img.shape[0]/size_image[0])
            x2 = int(bbox[2]*origin_img.shape[1]/size_image[1])
            y2 = int(bbox[3]*origin_img.shape[0]/size_image[0])
            cv2.rectangle(origin_img, (x1, y1), (x2, y2), (179, 255, 179), 2, 1)

    cv2.imwrite("results.jpg", origin_img)

if __name__ == "__main__":
    main()
