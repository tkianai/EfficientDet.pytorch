import torch
import torch.nn as nn


def calc_iou(a, b):
    """计算两组矩形框的IoU值
    
    Arguments:
        a {tensor}} -- (N, 4) mode: xyxy
        b {tensor} -- (M, 4) mode: xyxy
    
    Returns:
        tensor -- (N, M)
    """    
    # 计算各自的矩形框面积
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    # 将一组的N个和另一组的M个两两对比，找出相交部分的长宽
    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])
    # 没有相交的部分置0
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    # 计算相交部分的面积
    intersection = iw * ih
    # 计算IOU值
    ua = torch.unsqueeze(area_a, dim=1) + area_b - intersection
    ua = torch.clamp(ua, min=1e-8)
    IoU = intersection / ua
    return IoU


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.5, cls_eps=1e-4, neg_thr=0.4, pos_thr=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_eps = cls_eps
        self.neg_thr = neg_thr
        self.pos_thr = pos_thr

    def forward(self, classifications, regressions, anchors, annotations):
        """Focal Loss计算
        
        Arguments:
            classifications {tensor} -- (B, N, C) B表示batch size， N表示anchor数目， C表示类别数
            regressions {tensor} -- (B, N, 4) xyxy
            anchors {[tensor} -- (B, N, 4) xyxy
            annotations {tensor} -- (B, M, 5) xyxyc
        
        Returns:
            tuple -- (类别损失，回归损失)
        """        
        # 确定计算的设备类型
        device = anchors.device
        # 确定 batch size
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        # 获取anchor的属性
        anchor = anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            # 过滤掉无用的标注
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            # 如果没有可用标注，则跳过
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().to(device))
                classification_losses.append(torch.tensor(0).float().to(device))
                continue
            # 限制类别分数范围
            classification = torch.clamp(classification, self.cls_eps, 1.0 - self.cls_eps)
            # 计算anchor和标注之间的IoU
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
            # 对每个anchor，选择IoU最大的标注
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # NOTE 计算类别损失
            # targets: (anchor的数目, 类别数目)
            targets = torch.ones(classification.shape) * -1
            targets = targets.to(device)
            # IoU小于阈值作为负样本
            targets[torch.lt(IoU_max, self.neg_thr), :] = 0
            # IoU大于阈值作为正样本
            positive_indices = torch.ge(IoU_max, self.pos_thr)
            num_positive_anchors = positive_indices.sum()
            # 已经匹配到anchor的标注框
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape) * self.alpha
            alpha_factor = alpha_factor.to(device)
            # 详见focal loss中的alpha定义
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            # 确定focal loss中的gamma项
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            # 总体的权重
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            # 计算交叉熵损失
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            # focal loss
            cls_loss = focal_weight * bce
            # 去除未定义处的损失(置0)
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).to(device))
            # 记录第j个样本的类别损失
            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # NOTE 计算回归损失
            # 确定存在正样本
            if positive_indices.sum() > 0:
                # 取出正样本的标注
                assigned_annotations = assigned_annotations[positive_indices, :]
                # 取出对应anchor的属性
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                # 取出真值
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                # 长宽最小为1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                # 计算anchor和真值的偏差
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                # 整合四个偏差值用于计算loss, (4, 正样本数目)
                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                # 转置后, (正样本数目, 4)
                targets = targets.t()
                # variance
                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
                # 计算回归损失
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                # smooth
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                # 记录第j个样本的矩形框回归损失
                regression_losses.append(regression_loss.mean())
            else:
                # 没有标注
                regression_losses.append(torch.tensor(0).float().to(device))
        
        return (
            torch.stack(classification_losses).mean(dim=0, keepdim=True), 
            torch.stack(regression_losses).mean(dim=0, keepdim=True)
        )