
import torch
from .transforms import build_transform, detection_collate
from .coco import COCODataset

def build_dataloader(cfg, is_train=True, distributed=False):
    transforms = build_transform(is_train, width=cfg.data.width, height=cfg.data.height)
    root, ann_file = cfg.data.train if is_train else cfg.data.test
    dataset = COCODataset(root, ann_file, transforms=transforms)
    sampler = torch.utils.data.DistributedSampler(
        dataset, shuffle=True) if distributed else torch.utils.data.RandomSampler(dataset)
    batch_size = cfg.train.batch_size if is_train else cfg.test.batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.dataloader.num_workers,
        batch_size=batch_size,
        collate_fn=detection_collate,
    )

    return dataloader
