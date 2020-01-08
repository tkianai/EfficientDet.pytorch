# EfficientDet.pytorch
Accurate, fast EfficientDet implementation with pytorch


**We support Distributed training with multi-gpu using `DistributedDataParallel`!!!**

> Compared to multi-gpu training using `DataParallel`, it's fast and gpu-balanced!

## Features

- *Efficient train: fast and functional*
- *Efficient dataset code: only supports coco-style*
- *Efficient structure: less codes, easy usage*


## Usage

- Train

`python torch.distributed.launch --nproc_per_node <N GPUS> train.py --config-file <config file>`

- Test

`python demo.py --img <image path> model.resume <checkpoint path>`


# TODO

- [x] distributed training
- [x] distributed evaluation
- [ ] speed up data loading
- [ ] inference demo for images and video
- [ ] model zoo
- [ ] tensorrt speed up
- [ ] time consuming analysis
- [ ] add chinese commnets
- [ ] clean code
- [ ] tensorboard visulize