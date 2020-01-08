
import math
import torch
import torch.nn as nn
from torchvision.ops.boxes import nms as nms_torch

from .efficientnet import EfficientNet
from .layers import BiFPN, Regressor, Classifier
from .loss import FocalLoss
from .utils import Anchors, BBoxTransform, ClipBoxes


_EFFICIENTDET = {
    'efficientdet-d0': {'backbone': 'efficientnet-b0', 'W_bifpn': 64, 'D_bifpn': 2, 'D_class': 3, 'inp_size': 512},
    'efficientdet-d1': {'backbone': 'efficientnet-b1', 'W_bifpn': 88, 'D_bifpn': 3, 'D_class': 3, 'inp_size': 640},
    'efficientdet-d2': {'backbone': 'efficientnet-b2', 'W_bifpn': 112, 'D_bifpn': 4, 'D_class': 3, 'inp_size': 768},
    'efficientdet-d3': {'backbone': 'efficientnet-b3', 'W_bifpn': 160, 'D_bifpn': 5, 'D_class': 4, 'inp_size': 896},
    'efficientdet-d4': {'backbone': 'efficientnet-b4', 'W_bifpn': 224, 'D_bifpn': 6, 'D_class': 4, 'inp_size': 1024},
    'efficientdet-d5': {'backbone': 'efficientnet-b5', 'W_bifpn': 288, 'D_bifpn': 7, 'D_class': 4, 'inp_size': 1280},
    'efficientdet-d6': {'backbone': 'efficientnet-b6', 'W_bifpn': 384, 'D_bifpn': 8, 'D_class': 5, 'inp_size': 1408},
    'efficientdet-d7': {'backbone': 'efficientnet-b7', 'W_bifpn': 384, 'D_bifpn': 8, 'D_class': 5, 'inp_size': 1636},
}


class EfficientDet(nn.Module):

    def __init__(self, num_classes, model_name='efficientdet-d0', score_thr=0.1, iou_thr=0.5):
        super().__init__()
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        print(model_name)
        self.config = _EFFICIENTDET[model_name]
        self.backbone = EfficientNet.from_pretrained(self.config['backbone'])
        self.trans_conv = []
        for in_channel in self.backbone.get_list_features()[-5:]:
            self.trans_conv.append(nn.Conv2d(in_channel, self.config['W_bifpn'], kernel_size=1, stride=1, padding=0))
        self.trans_conv = nn.ModuleList(self.trans_conv)
        self.bifpn = nn.Sequential(*[
            BiFPN(self.config['W_bifpn']) for _ in range(self.config['D_bifpn'])
        ])

        self.regressor = Regressor(
            in_channels=self.config['W_bifpn'],
            num_anchors=9,
            num_layers=self.config['D_class'],
        )
        self.classifier = Classifier(
            in_channels=self.config['W_bifpn'],
            num_anchors=9,
            num_classes=num_classes,
            num_layers=self.config['D_class'],
        )

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)
        
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def forward(self, inputs, targets=None):
        bone_feats = self.backbone(inputs)
        trans_feats = []
        for feat, conv in zip(bone_feats, self.trans_conv):
            trans_feats.append(conv(feat))

        features = self.bifpn(trans_feats)
        regression = torch.cat([self.regressor(feature) for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature) for feature in features], dim=1)
        anchors = self.anchors(inputs)

        if self.training:
            cls_loss, reg_loss = self.focalLoss(classification, regression, anchors, targets)
            return {
                'classification_loss': cls_loss,
                'regression_loss': reg_loss
            }
        else:
            # 增加batch预测功能
            transformed_anchors = self.regressBoxes(anchors, regression)
            # anchor预测结果 (B, N, 4)
            transformed_anchors = self.clipBoxes(transformed_anchors, inputs)
            # 每个anchor对应的类别的最大得分
            scores = torch.max(classification, dim=2, keepdim=True)[0]
            # 通过得分阈值筛选有效anchor位置
            scores_over_thresh = (scores > self.score_thr)[:, :, 0]
            # 对每一张图像分别进行后处理
            batch_size = scores.shape[0]
            outputs = []
            for i in range(batch_size):
                result = []
                # 第i个样本的有效anchor位置
                scores_over_thresh_i = scores_over_thresh[i, :]
                # 有矩形框进行后处理
                if scores_over_thresh_i.sum() > 0:
                    # 第i个样本的每个类别得分
                    classification_i = classification[i, scores_over_thresh_i, :]
                    # 第i个样本的举行框
                    transformed_anchors_i = transformed_anchors[i, scores_over_thresh_i, :]
                    # 第i个样本的所有有效anchor分数
                    scores_i = scores[i, scores_over_thresh_i, 0]
                    # 通过分数进行nms（所有类别，此处不是multi-label nms）
                    anchors_nms_idx = nms_torch(transformed_anchors_i, scores_i, self.iou_thr)
                    # 获取通过nms后保留下来的anchor分数以及类别
                    nms_scores_i, nms_class_i = classification_i[anchors_nms_idx, :].max(dim=1)
                    # nms后保留下来的矩形框
                    nms_bboxes_i = transformed_anchors_i[anchors_nms_idx, :]
                    # xyxy -> xywh
                    nms_bboxes_i[:, 2] -= nms_bboxes_i[:, 0]
                    nms_bboxes_i[:, 3] -= nms_bboxes_i[:, 1]

                    for j in range(nms_bboxes_i.shape[0]):
                        _score = float(nms_scores_i[j])
                        _label = int(nms_class_i[j])
                        _box = nms_bboxes_i[j, :]

                        item = {
                            'class': _label,
                            'score': _score,
                            'bbox': _box.tolist(),
                        }
                        result.append(item)

                outputs.append(result)

            return outputs
