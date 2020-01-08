
import torch
from bisect import bisect_right


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def build_optimizer(model, lr=0.01, weight_decay=5e-4, bias_lr=0.01, bias_weight_decay=5e-4, momentum=0.9):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        _lr = lr
        _weight_decay = weight_decay
        if "bias" in key:
            _lr = bias_lr
            _weight_decay = bias_weight_decay
        params += [{"params": [value], "lr": _lr, "weight_decay": _weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


def build_lr_scheduler(optimizer, steps=None, gamma=0.1, warmup_factor=0.1, warmup_iters=500, warmup_method="linear"):
    lr_scheduler = WarmupMultiStepLR(
        optimizer,
        steps,
        gamma,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method
    )
    return lr_scheduler
