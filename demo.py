

import os
import os.path as osp
import cv2
import numpy as np
import json
import torch

from data import COCODataset
from data.transforms import build_transforms
from config import get_default_cfg
from models import EfficientDet
from utils.checkpoint import Checkpointer


@torch.no_grad()
def inference(model, img, label_map, score_thr=0.5, transforms=None):

    img_copy = img.copy()
    if len(img_copy.shape) == 2:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    sample = {
        "image": img_copy,
        "bboxes": np.ones(1, 4),
        "labels": np.ones(1, dtype=np.int)
    }
    if transforms is not None:
        sample = transforms(sample)
    output = model(sample['image'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
    
    for item in output:
        # draw
        bbox = [int(e) for e in item['bbox']]
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cv2.rectangle(img, pt1, pt2, (0, 0, 255), 3)

        ptt = (bbox[0], bbox[1] - 20)
        cv2.putText(img, "{}: {:03f}".format(
            label_map[item['class']], item['score']), ptt, cv2.FONT_HERSHEY_COMPLEX, 6, (0, 255, 0), 25)

    return img


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientDet Demo")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--img", help="Directory or image path", default=None)
    parser.add_argument("--vid", help="Video file path", default=None)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save", type=str, default="./EfficientDet_result")
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    if args.img is None and args.vid is None:
        print("Please provide option --img or --vid...There is nothing to predict!")
        exit()

    if not osp.exists(args.save):
        os.makedirs(args.save)
    return args


def is_valid_file(filename):
    return filename.enswith(('.png', '.jpg', '.PNG', '.JPG'))


def main():
    args = parse_args()
    cfg = get_default_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    dataset = COCODataset(cfg.data.test[0], cfg.data.test[1])
    num_classes = dataset.num_classes
    label_map = dataset.coco_labels
    model = EfficientDet(num_classes=num_classes, model_name=cfg.model.name)
    device = torch.device(cfg.device)
    model.to(device)
    model.eval()

    inp_size = model.config['inp_size']
    transforms = build_transforms(False, inp_size=inp_size)

    output_dir = cfg.output_dir
    checkpointer = Checkpointer(model, None, None, output_dir, True)
    checkpointer.load(args.ckpt)

    images = []
    if args.img:
        if osp.isdir(args.img):
            for filename in os.listdir(args.img):
                if is_valid_file(filename):
                    images.append(osp.join(args.img, filename))
        else:
            images = [args.img]

    for img_path in images:
        img = cv2.imread(img_path)
        img = inference(model, img, label_map, score_thr=args.score_thr, transforms=transforms)
        save_path = osp.join(args.save, osp.basename(img_path))
        cv2.imwrite(save_path, img)

    if args.vid:
        vCap = cv2.VideoCapture(args.v)
        fps = int(vCap.get(cv2.CAP_PROP_FPS))
        height = int(vCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(vCap.get(cv2.CAP_PROP_FRAME_WIDTH))
        size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = osp.join(args.save, osp.basename(args.v))
        vWrt = cv2.VideoWriter(save_path, fourcc, fps, size)
        while True:
            flag, frame = vCap.read()
            if not flag:
                break
            frame = inference(model, frame, label_map, score_thr=args.score_thr, transforms=transforms)
            vWrt.write(frame)
        
        vCap.release()
        vWrt.release()

if __name__ == '__main__':
    main()
    
