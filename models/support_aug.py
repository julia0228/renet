import torch
import torch.nn.functional as F


def build_support_augmentation(data_shot, data_query, probs, renet, args):
    """
    Select top-1 confident query per class, append to support set,
    and estimate per-support reliability via SupportWeightNet.
    """
    if renet.support_weight_net is None:
        raise RuntimeError('SupportWeightNet is not initialized. Enable --use_support_aug during training/testing.')

    way, shot = args.way, args.shot
    device = data_shot.device
    C, H, W = data_shot.shape[1], data_shot.shape[2], data_shot.shape[3]

    support = data_shot.view(way, shot, C, H, W)
    base_labels = torch.arange(way, device=device).unsqueeze(1).repeat(1, shot).view(-1)

    aug_feats, aug_labels = [], []
    for c in range(way):
        class_scores = probs[:, c]
        _, idx = class_scores.max(dim=0)
        feat_map = data_query[idx]
        aug_feats.append(feat_map)
        aug_labels.append(torch.tensor(c, device=device))
    aug_feats = torch.stack(aug_feats, dim=0)
    aug_labels = torch.stack(aug_labels, dim=0)

    support_aug = torch.cat([support, aug_feats.unsqueeze(1)], dim=1).view(-1, C, H, W)
    labels_all = torch.cat([base_labels, aug_labels], dim=0)

    logits = renet.support_weight_net(support_aug)
    reliability = F.softmax(logits, dim=-1).gather(1, labels_all.unsqueeze(1)).squeeze(1)

    total_shot = shot + 1  # one augmented sample per class
    support_weights = reliability.view(way, total_shot)
    support_weights = support_weights / (support_weights.sum(dim=-1, keepdim=True) + 1e-8)

    return support_aug, support_weights

