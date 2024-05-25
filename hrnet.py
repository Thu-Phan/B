import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class HRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HRNet, self).__init__()
        self.layer1 = self._make_layer(BasicBlock, in_channels, 64, 4)
        self.transition1 = self._make_transition_layer(64, [32, 64])
        self.stage2 = self._make_stage([32, 64], 4)

        self.final_layer = nn.Conv2d(32, out_channels, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        layers.append(block(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_channels, out_channels):
        return nn.ModuleList([nn.Conv2d(in_channels, out_channels[i], kernel_size=3, stride=1, padding=1) for i in range(len(out_channels))])

    def _make_stage(self, out_channels, blocks):
        return nn.ModuleList([self._make_layer(BasicBlock, out_channels[i], out_channels[i], blocks) for i in range(len(out_channels))])

    def forward(self, x):
        x = self.layer1(x)
        x_list = [self.transition1[i](x) for i in range(len(self.transition1))]
        x = self.stage2[0](x_list[0])
        x = self.final_layer(x)
        return x

