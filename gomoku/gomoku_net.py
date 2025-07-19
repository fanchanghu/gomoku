import torch.nn as nn
from typing import Literal


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class GomokuNet(nn.Module):
    def __init__(self, board_size=15, mode: Literal["policy", "value"] = "policy"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(64) for _ in range(8)])
        if mode == "policy":
            self.head = nn.Sequential(
                nn.Conv2d(64, 2, 1),
                nn.Flatten(),
                nn.Linear(2 * board_size**2, board_size**2),
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Tanh(),
            )

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)
