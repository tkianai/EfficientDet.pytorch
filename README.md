# EfficientDet.pytorch
Accurate, fast EfficientDet implementation with pytorch


**Many pytorch implementations of EfficientDet only support training with multi-gpu using `DataParallel` wrapper, Which caused slow speed and gpu imbalanced!!!**

**We support Distributed training with multi-gpu using `DistributedDataParallel`!!!**

## Features

- *Efficient train: fast and functional*
- *Efficient dataset code: only supports coco-style*
- *Efficient structure: less codes, easy usage*


## Usage

- Train

`python torch.distributed.launch --nproc_per_node <N GPUS> train.py --config-file <config file>`

- Test

`python demo.py --img <image path> model.resume <checkpoint path>`

