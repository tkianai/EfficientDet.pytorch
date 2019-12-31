

from yacs.config import CfgNode as CN

__EFFICIENTDET = {
    'efficientdet-d0': {'input_size': 512, 'backbone': 'B0', 'W_bifpn': 64, 'D_bifpn': 2, 'D_class': 3},
    'efficientdet-d1': {'input_size': 640, 'backbone': 'B1', 'W_bifpn': 88, 'D_bifpn': 3, 'D_class': 3},
    'efficientdet-d2': {'input_size': 768, 'backbone': 'B2', 'W_bifpn': 112, 'D_bifpn': 4, 'D_class': 3},
    'efficientdet-d3': {'input_size': 896, 'backbone': 'B3', 'W_bifpn': 160, 'D_bifpn': 5, 'D_class': 4},
    'efficientdet-d4': {'input_size': 1024, 'backbone': 'B4', 'W_bifpn': 224, 'D_bifpn': 6, 'D_class': 4},
    'efficientdet-d5': {'input_size': 1280, 'backbone': 'B5', 'W_bifpn': 288, 'D_bifpn': 7, 'D_class': 4},
    'efficientdet-d6': {'input_size': 1408, 'backbone': 'B6', 'W_bifpn': 384, 'D_bifpn': 8, 'D_class': 5},
    'efficientdet-d7': {'input_size': 1636, 'backbone': 'B6', 'W_bifpn': 384, 'D_bifpn': 8, 'D_class': 5},
}


def get_default_cfg():
    cfg = CN()
    cfg.output_dir = "./work_dirs"
    cfg.device = "cuda"
    cfg.dtype = "float32"

    cfg.model = CN()
    cfg.model.name = "efficientdet-d0"
    cfg.model.resume = ""

    cfg.train = CN()
    cfg.train.max_epoch = 500
    cfg.train.batch_size = 12
    cfg.train.checkpoint_period = 1
    cfg.train.log_period = 20

    cfg.dataloader = CN()
    cfg.dataloader.num_workers = 8

    cfg.solver = CN()
    cfg.solver.lr = 1e-4
    cfg.solver.momentum = 0.9
    cfg.solver.weight_decay = 5e-4
    cfg.solver.gamma = 0.1




    cfg.solver.test_period = 100
    cfg.solver.checkpoint_period = 100
    cfg.solver.log_period = 50
    cfg.solver.ims_per_gpu = 4
    cfg.solver.max_iter = 25000
    
    cfg.solver.bias_lr = 0.01
    cfg.solver.bias_weight_decay = 1e-5
    
    cfg.solver.steps = [1000, 3000]
    
    cfg.solver.warmup_factor = 0.1
    cfg.solver.warmup_iters = 500
    cfg.solver.warmup_method = "linear"

    cfg.test = CN()
    cfg.test.test_period = 0
    cfg.test.ims_per_gpu = 1

    cfg.test.bbox_aug = CN()
    cfg.test.bbox_aug.enabled = False
    cfg.test.bbox_aug.h_flip = False
    cfg.test.bbox_aug.scales = ()
    cfg.test.bbox_aug.max_size = 4000
    cfg.test.bbox_aug.scale_h_flip = False

    cfg.test.expected_results = []
    cfg.test.expected_results_sigma_tol = 4

    

    cfg.data = CN()
    cfg.data.train = ("./datasets/coco2017/train2017", "./datasets/coco2017/annotations/instances_train2017.json")
    cfg.data.test = ("./datasets/coco2017/val2017", "./datasets/coco2017/annotations/instances_val2017.json")
    cfg.data.width = __EFFICIENTDET[cfg.model.name]["input_size"]
    cfg.data.height = cfg.data.width

    cfg.data.min_size_train = (800, )
    cfg.data.max_size_train = 1333
    cfg.data.min_size_test = 800
    cfg.data.max_size_test = 1333
    cfg.data.pixel_mean = [102.9801, 115.9465, 122.7717]
    cfg.data.pixel_std = [1., 1., 1.]
    cfg.data.to_bgr255 = True
    cfg.data.brightness = 0.0
    cfg.data.contrast = 0.0
    cfg.data.saturation = 0.0
    cfg.data.hue = 0.0
    cfg.data.horizontal_flip_prob = 0.5
    cfg.data.vertical_flip_prob = 0.0

    return cfg


def lr_scheduler_kwargs(cfg):
    return {
        'warmup_factor': cfg.solver.warmup_factor,
        'warmup_iters': cfg.solver.warmup_iters,
        'warmup_method': cfg.solver.warmup_method,
    }


def optimizer_kwargs(cfg):
    return {
        'lr': cfg.data.lr,
        'weight_decay': cfg.data.weight_decay,
        'bias_lr': cfg.data.bias_lr,
        'bias_weight_decay': cfg.data.bias_weight_decay,
        'momentum': cfg.data.momentum,
    }


def model_kwargs(cfg):
    return {
        "network": cfg.model.name,
        "W_bifpn": __EFFICIENTDET[cfg.model.name]['W_bifpn'],
        "D_bifpn": __EFFICIENTDET[cfg.model.name]['D_bifpn'],
        "D_class": __EFFICIENTDET[cfg.model.name]['D_class'],
    }
