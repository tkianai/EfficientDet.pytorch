
import os
import argparse
from apex import amp
import torch
import utils.comm as comm
import utils.misc as misc
from data import COCODataset
from data import build_dataloader
from models import EfficientDet
from config import get_default_cfg, optimizer_kwargs, lr_scheduler_kwargs
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from engine import do_train
from solver import build_optimizer, build_lr_scheduler



def train(cfg, local_rank, distributed):

    num_classes = COCODataset(cfg.data.train[0], cfg.data.train[1]).num_classes
    model = EfficientDet(num_classes=num_classes, model_name=cfg.model.name)
    inp_size = model.config['inp_size']
    device = torch.device(cfg.device)
    model.to(device)

    optimizer = build_optimizer(model, **optimizer_kwargs(cfg))
    lr_scheduler = build_lr_scheduler(optimizer, **lr_scheduler_kwargs(cfg))

    use_mixed_precision = cfg.dtype == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.output_dir
    save_to_disk = comm.get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, lr_scheduler, output_dir, save_to_disk)
    extra_checkpoint_data = checkpointer.load(cfg.model.resume)
    arguments.update(extra_checkpoint_data)

    train_dataloader = build_dataloader(
        cfg, inp_size, is_train=True, distributed=distributed, start_iter=arguments["iteration"])

    test_period = cfg.test.test_period
    if test_period > 0:
        val_dataloader = build_dataloader(
            cfg, inp_size, is_train=False, distributed=distributed)
    else:
        val_dataloader = None

    checkpoint_period = cfg.solver.checkpoint_period
    log_period = cfg.solver.log_period

    do_train(
        cfg,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        lr_scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        log_period,
        arguments
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientDet Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
        )
        comm.synchronize()

    cfg = get_default_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.output_dir
    if output_dir:
        misc.mkdir(output_dir)

    logger = setup_logger("EfficientDet", output_dir, comm.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(output_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    misc.save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
