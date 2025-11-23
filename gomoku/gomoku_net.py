import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class SpatialAttention(nn.Module):
    """空间注意力模块 - 让网络更关注棋盘上的关键区域"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(2, in_channels // reduction, 1)  # 固定输入通道为2
        self.conv2 = nn.Conv2d(in_channels // reduction, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 使用全局平均池化和最大池化来生成注意力权重
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接两种池化结果
        x_combined = torch.cat([avg_out, max_out], dim=1)
        
        # 生成注意力权重图
        attention = self.conv2(F.relu(self.conv1(x_combined)))
        attention = self.sigmoid(attention)
        
        return x * attention


class EnhancedResBlock(nn.Module):
    """增强的残差块，包含空间注意力机制"""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.attention = SpatialAttention(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.attention(y)  # 应用空间注意力
        return self.act(identity + y)


class GomokuNet(nn.Module):
    def __init__(self, board_size=15, mode: Literal["policy", "value"] = "policy"):
        super().__init__()
        self.board_size = board_size
        self.mode = mode
        
        # 增强的主干网络 - 增加通道数以提升表达能力
        self.stem = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1, bias=False),  # 增加通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 使用增强的残差块
        self.body = nn.Sequential(*[EnhancedResBlock(128) for _ in range(6)])
        
        if mode == "policy":
            self.head = nn.Sequential(
                nn.Conv2d(128, 4, 1),  # 增加输出通道
                nn.Flatten(),
                nn.Linear(4 * board_size**2, board_size**2),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256),  # 增加隐藏层大小
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh(),
            )

    def forward(self, x):
        # x: [N, H, W] -> [N, 1, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = x.to(next(self.parameters()).device)
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)
