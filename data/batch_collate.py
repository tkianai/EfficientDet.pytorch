
import torch


def BatchCollater(batch):
    """将数据组成一个可以送入网络的batch形式
    
    Arguments:
        batch {list} -- 原始的batch形式
    
    Returns:
        tuple -- 两个元素的元组，第一个为图片的batch(B,3,H,W)，第二个为标注信息batch(B,N,5)
    """
    images = [e[0]['image'] for e in batch]
    anns = [e[0]['bboxes'] for e in batch]
    labels = [e[0]['labels'] for e in batch]
    scales = [e[0]['scale'] for e in batch]
    idxs = [e[1] for e in batch]

    max_num_anns = max(len(ann) for ann in anns)
    anns_padded = torch.ones((len(anns), max_num_anns, 5)) * -1

    if max_num_anns > 0:
        for idx, (ann, lab) in enumerate(zip(anns, labels)):
            if len(ann) > 0:
                anns_padded[idx, :len(ann), :4] = ann
                anns_padded[idx, :len(ann), 4] = lab

    return (
        torch.stack(images, 0).permute(0, 3, 1, 2), 
        torch.FloatTensor(anns_padded),
        {
            'scale': scales,
            'index': idxs
        }
    )
