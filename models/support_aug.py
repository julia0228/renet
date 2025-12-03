import torch
import torch.nn.functional as F


def build_support_augmentation(data_shot, data_un, probs, args):
    way, shot = args.way, args.shot
    topk = max(1, getattr(args, 'pseudo_topk', 1))
    device = data_shot.device
    C, H, W = data_shot.shape[1], data_shot.shape[2], data_shot.shape[3]
    
    # 将data_shot从样本优先排列转换为类别优先排列
    # 原始: [c1s1, c2s1, ..., c_ways1, c1s2, c2s2, ..., c_ways2, ...]
    # 目标: [c1s1, c1s2, ..., c1s_shot, c2s1, c2s2, ..., c2s_shot, ...]
    data_shot_class_wise = data_shot.view(shot, way, C, H, W).permute(1, 0, 2, 3, 4).contiguous()
    data_shot_class_wise = data_shot_class_wise.view(way * shot, C, H, W)
    
    # 为每个类别选择topk个伪标签样本
    aug_feats = []
    pseudo_weights = []
    pseudo_indices = []
    
    for c in range(way):
        class_scores = probs[:, c]
        k = min(topk, class_scores.size(0))
        conf, idx = torch.topk(class_scores, k=k, largest=True)
        feat_maps = data_un[idx]
        aug_feats.append(feat_maps)
        # pseudo_weights.append(conf)
        thresh_weights = torch.where(conf > 0.7, torch.ones_like(conf), torch.full_like(conf, 0.9))
        pseudo_weights.append(thresh_weights)
        pseudo_indices.append(idx)
    
    # 将伪标签样本转换为类别优先排列
    aug_feats = torch.stack(aug_feats, dim=0)  # [way, topk, C, H, W]
    aug_feats = aug_feats.view(way * topk, C, H, W)  # 展平为类别优先
    
    # 将原始支持样本和伪标签样本合并
    support_aug = []
    base_weights = []
    
    for c in range(way):
        # 获取当前类别的原始支持样本
        start_idx = c * shot
        end_idx = (c + 1) * shot
        class_support = data_shot_class_wise[start_idx:end_idx]
        
        # 获取当前类别的伪标签样本
        start_aug_idx = c * topk
        end_aug_idx = (c + 1) * topk
        class_aug = aug_feats[start_aug_idx:end_aug_idx]
        
        # 合并原始支持样本和伪标签样本
        class_combined = torch.cat([class_support, class_aug], dim=0)
        support_aug.append(class_combined)
        
        # 创建当前类别的权重
        class_base_weights = torch.ones(shot, device=device, dtype=data_shot.dtype)
        class_pseudo_weights = pseudo_weights[c]
        class_weights = torch.cat([class_base_weights, class_pseudo_weights], dim=0)
        base_weights.append(class_weights)
    
    # 合并所有类别的样本
    support_aug = torch.cat(support_aug, dim=0)
    
    # 合并所有权重
    support_weights = torch.stack(base_weights, dim=0)  # [way, shot + topk]
    
    # 类内归一化
    support_weights = support_weights / (support_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    # 将支持集从类别优先转换回样本优先
    total_shot = shot + topk
    support_aug = support_aug.view(way, total_shot, C, H, W)
    support_aug = support_aug.permute(1, 0, 2, 3, 4).contiguous()
    support_aug = support_aug.view(way * total_shot, C, H, W)
    
    # 打印形状以验证
    # print("Final support_aug shape:", support_aug.shape)
    
    return support_aug, support_weights, pseudo_indices