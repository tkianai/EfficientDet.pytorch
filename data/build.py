
import torch
from .transforms import build_transform, detection_collate
from .coco import COCODataset


class IterationBasedBatchSampler(torch.utils.data.sampler.BatchSampler):

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def build_sampler(dataset, shuffle, distributed):
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def build_batch_sampler(sampler, batch_size, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler


def build_dataloader(cfg, is_train=True, distributed=False, start_iter=0):
    batch_size = cfg.solver.batch_size
    if is_train:
        root, ann_file = cfg.data.train
        num_iters = cfg.solver.max_iter
    else:
        root, ann_file = cfg.data.train
        num_iters = None
    transforms = build_transform(is_train, width=cfg.data.width, height=cfg.data.height)
    dataset = COCODataset(root, ann_file, transforms=transforms)
    sampler = build_sampler(dataset, is_train, distributed)
    batch_sampler = build_batch_sampler(sampler, batch_size, num_iters, start_iter)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.dataloader.num_workers,
        batch_sampler=batch_sampler,
        collate_fn=detection_collate,
    )

    return dataloader
