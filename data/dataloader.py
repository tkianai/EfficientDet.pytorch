

import torch
from .coco import COCODataset
from .batch_collate import BatchCollater
from .transforms import build_transforms
from .samplers import build_sampler, build_batch_sampler


def build_dataloader(cfg, inp_size, is_train=True, distributed=False, start_iter=0):
    batch_size = cfg.solver.ims_per_batch
    if is_train:
        root, ann_file = cfg.data.train
        num_iters = cfg.solver.max_iter
    else:
        root, ann_file = cfg.data.train
        num_iters = None
    transforms = build_transforms(is_train, inp_size=inp_size)
    dataset = COCODataset(root, ann_file, transforms=transforms)
    sampler = build_sampler(dataset, is_train, distributed)
    batch_sampler = build_batch_sampler(
        sampler, batch_size, num_iters, start_iter)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.dataloader.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=BatchCollater,
    )

    return dataloader
