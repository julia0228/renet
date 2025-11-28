import torch
import torch.nn as nn
import torch.nn.functional as F


class SupportWeightNet(nn.Module):
    """
    Learn per-support reliability scores conditioned on support feature maps.
    Input: tensor of shape (num_samples, C, H, W)
    Output: (num_samples, num_classes) reliability logits
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        inter_channels = max(in_channels // 4, 32)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(inter_channels, inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(inter_channels, num_classes)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        feats = self.feature_extractor(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).view(b, -1)
        logits = self.classifier(pooled)
        return logits

