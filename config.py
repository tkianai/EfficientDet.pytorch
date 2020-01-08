

from yacs.config import CfgNode as CN


def get_default_cfg():
    cfg = CN()
    cfg.output_dir = "./work_dirs"
    cfg.device = "cuda"
    cfg.dtype = "float32"

    cfg.model = CN()
    cfg.model.name = "efficientdet-d0"
    cfg.model.resume = ""

    cfg.dataloader = CN()
    cfg.dataloader.num_workers = 8

    cfg.solver = CN()
    cfg.solver.lr = 0.001
    cfg.solver.bias_lr = 0.002
    cfg.solver.weight_decay = 5e-4
    cfg.solver.bias_weight_decay = 5e-4
    cfg.solver.momentum = 0.9
    cfg.solver.gamma = 0.1
    cfg.solver.ims_per_batch = 8

    cfg.solver.checkpoint_period = 10000
    cfg.solver.log_period = 50
    
    cfg.solver.max_iter = 70000
    

    cfg.solver.steps = [30000, 60000]

    cfg.solver.warmup_factor = 0.1
    cfg.solver.warmup_iters = 3000
    cfg.solver.warmup_method = "linear"

    cfg.test = CN()
    cfg.test.test_period = 0
    cfg.test.ims_per_batch = 8

    cfg.data = CN()
    cfg.data.train = ("./datasets/coco2017/train2017",
                      "./datasets/coco2017/annotations/instances_train2017.json")
    cfg.data.test = ("./datasets/coco2017/val2017",
                     "./datasets/coco2017/annotations/instances_val2017.json")

    return cfg


def lr_scheduler_kwargs(cfg):
    return {
        'steps': cfg.solver.steps,
        'gamma': cfg.solver.gamma,
        'warmup_factor': cfg.solver.warmup_factor,
        'warmup_iters': cfg.solver.warmup_iters,
        'warmup_method': cfg.solver.warmup_method,
    }


def optimizer_kwargs(cfg):
    return {
        'lr': cfg.solver.lr,
        'weight_decay': cfg.solver.weight_decay,
        'bias_lr': cfg.solver.bias_lr,
        'bias_weight_decay': cfg.solver.bias_weight_decay,
        'momentum': cfg.solver.momentum,
    }
