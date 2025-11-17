# coding: utf-8

# Standard imports
from functools import reduce
import operator

# External imports
import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=2, stride=2, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]




class FancyCNN(nn.Module):
    """
    A fancy CNN model with :
        - stacked 3x3 convolutions
        - convolutive down sampling
        - a global average pooling at the end
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        layers = []
        cin = input_size[0]
        base_ch = cfg.get("base_channels", 16)
        # Build N blocks. Each block: conv_relu_bn(cin -> out_ch),
        # conv_relu_bn(out_ch -> out_ch), conv_down(out_ch -> out_ch*2)
        for i in range(cfg["num_blocks"]):
            out_ch = base_ch * (2 ** i)
            layers += conv_relu_bn(cin, out_ch)
            layers += conv_relu_bn(out_ch, out_ch)
            layers += conv_down(out_ch, out_ch * 2)
            cin = out_ch * 2

        # Global Average Pooling -> (B, C, 1, 1)
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        self.model = nn.Sequential(*layers)

        # Final classifier
        self.classifier = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.model(x)            # (B, C, 1, 1)
        x = torch.flatten(x, 1)     # (B, C)
        x = self.classifier(x)      # (B, num_classes)
        return x
