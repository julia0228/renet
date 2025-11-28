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
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, 1, kernel_size=3, padding=1, bias=True)
        self.fc1 = nn.Linear(1, max(inter_channels // 8, 4))
        self.fc2 = nn.Linear(max(inter_channels // 8, 4), num_classes)
        self.reset_parameters()

    def forward(self, x):
        b, c, h, w = x.shape
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, 1).view(b, -1)
        x = F.relu(self.fc1(x), inplace=True)
        logits = self.fc2(x)
        return logits

    def reset_parameters(self):
        for m in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

