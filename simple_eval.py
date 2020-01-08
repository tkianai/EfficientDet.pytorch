

from pycocotools.cocoeval import COCOeval
import json
import torch

from data import COCODataset
from data.transforms import build_transforms
from config import get_default_cfg
from models import EfficientDet
from utils.checkpoint import Checkpointer


def evaluate_coco(dataset, model, threshold=0.05):
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index][0]
            scale = data['scale']
            outputs = model(data['image'].cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            for item in outputs[0]:

                image_result = {
                    'image_id': dataset.image_ids[index],
                    'category_id': dataset.return_coco_label(item['class']),
                    'score': item['score'],
                    'bbox': [e / scale for e in item['bbox']],
                }

                results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('bbox_results.json', 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('bbox_results.json')

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == '__main__':
    cfg = get_default_cfg()
    num_classes = COCODataset(cfg.data.test[0], cfg.data.test[1]).num_classes
    model = EfficientDet(num_classes=num_classes, model_name=cfg.model.name)
    device = torch.device(cfg.device)
    model.to(device)

    inp_size = model.config['inp_size']
    transforms = build_transforms(False, inp_size=inp_size)
    dataset = COCODataset(cfg.data.test[0], cfg.data.test[1], transforms=transforms)
    
    output_dir = cfg.output_dir
    checkpointer = Checkpointer(model, None, None, output_dir, True)
    checkpointer.load()

    evaluate_coco(dataset, model)
