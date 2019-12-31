import torch
import numpy as np
import albumentations as albu
from albumentations.pytorch.transforms import ToTensor



def build_transform(is_train, width=512, height=512, min_area=0., min_visibility=0.):
    """数据增强方式
    """    
    list_transforms = []
    if is_train:
        list_transforms.extend([
            albu.augmentations.transforms.LongestMaxSize(max_size=width, always_apply=True),
            albu.PadIfNeeded(
                min_height=height, 
                min_width=width,
                always_apply=True, 
                border_mode=0, 
                value=[0, 0, 0]
            ),
            albu.augmentations.transforms.RandomResizedCrop(height=height, width=width, p=0.3),
            albu.augmentations.transforms.Flip(),
            albu.augmentations.transforms.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.4),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=20, b_shift_limit=15, g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5),
                albu.NoOp()
            ]),
            albu.CLAHE(p=0.8),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
        ])
    else:
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    
    list_transforms.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        ToTensor()
    ])
    if is_train:
        bbox_params = albu.BboxParams(
            format='pascal_voc', 
            min_area=min_area,
            min_visibility=min_visibility, 
            label_fields=['labels']
        )
        return albu.Compose(list_transforms, bbox_params=bbox_params)
    return albu.Compose(list_transforms)


def detection_collate(batch):
    """将数据组成一个可以送入网络的batch形式
    
    Arguments:
        batch {list} -- 原始的batch形式
    
    Returns:
        tuple -- 两个元素的元组，第一个为图片的batch(B,3,H,W)，第二个为标注信息batch(B,N,5)
    """    
    images = [e[0]['image'] for e in batch]
    anns = [e[0]['bboxes'] for e in batch]
    labels = [e[0]['labels'] for e in batch]
    idxs = [e[1] for e in batch]

    max_num_anns = max(len(ann) for ann in anns)
    anns_padded = np.ones((len(anns), max_num_anns, 5)) * -1

    if max_num_anns > 0:
        for idx, (ann, lab) in enumerate(zip(anns, labels)):
            if len(ann) > 0:
                anns_padded[idx, :len(ann), :4] = ann
                anns_padded[idx, :len(ann), 4] = lab
    return (torch.stack(images, 0), torch.FloatTensor(anns_padded), idxs)
