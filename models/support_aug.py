import torch
import torch.nn.functional as F


def build_support_augmentation(data_shot, data_query, probs, args):
    """
    Select top-1 confident query per class, append to support set,
    and assign weights based on class probabilities.
    """
    way, shot = args.way, args.shot
    topk = max(1, getattr(args, 'pseudo_topk', 1))
    device = data_shot.device
    C, H, W = data_shot.shape[1], data_shot.shape[2], data_shot.shape[3]

    support = data_shot.view(way, shot, C, H, W)
    base_weights = torch.ones(way, shot, device=device, dtype=data_shot.dtype)

    aug_feats = []
    pseudo_weights = []
    pseudo_indices = []
    for c in range(way):
        class_scores = probs[:, c]
        k = min(topk, class_scores.size(0))
        conf, idx = torch.topk(class_scores, k=k, largest=True)
        feat_maps = data_query[idx]
        aug_feats.append(feat_maps)
        pseudo_weights.append(conf)      # [k]
        pseudo_indices.append(idx)       # [k]
    aug_feats = torch.stack(aug_feats, dim=0)        # way, topk, C, H, W
    pseudo_weights = torch.stack(pseudo_weights, dim=0)  # way, topk
    pseudo_indices = torch.stack(pseudo_indices, dim=0)  # way, topk

    total_shot = shot + aug_feats.size(1)
    support_aug = torch.cat([support, aug_feats], dim=1).view(-1, C, H, W)

    # weights: original shots weight=1, pseudo shots use their probabilities (<=1)
    support_weights = torch.cat([base_weights, pseudo_weights], dim=1)
    support_weights = support_weights / (support_weights.sum(dim=-1, keepdim=True) + 1e-8)

    return support_aug, support_weights, pseudo_indices