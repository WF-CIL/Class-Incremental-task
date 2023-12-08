import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out  
        return self.sigmoid(out)


class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention1D, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SEFeatureAt1D(nn.Module):
    def __init__(self, inplanes, type, at_res):
        super(SEFeatureAt1D, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(inplanes, inplanes // 16, 1),
            nn.ReLU(),
            nn.Conv1d(inplanes // 16, inplanes, 1),
            nn.Sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.type = type
        self.at_res = at_res
        self.ca = ChannelAttention1D(inplanes)
        self.sa = SpatialAttention1D()

    def forward(self, x):
        residual = x
        if self.type == "se":
            attention = self.se(x)
            x = x * attention
        elif self.type == "ffm":
            x = self.ca(x) * x
            x = self.sa(x) * x
        if self.at_res:
            x += residual
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=(1, 1), remove_last_relu=False):
        super(BasicBlock, self).__init__()
        self.remove_last_relu = remove_last_relu
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dilation=dilation[0])
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, dilation=dilation[1])
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        if not self.remove_last_relu:
            out = self.relu(out)
        return out


class resnet18(nn.Module):
    def __init__(self, remove_last_relu=False, zero_init_residual=True):
        super(resnet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        blocks = [2, 2, 2, 2]
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks[0], remove_last_relu)
        self.layer2 = self._make_layer(128, blocks[1])
        self.layer3 = self._make_layer(256, blocks[2])
        self.layer4 = self._make_layer(512, blocks[3], remove_last_relu)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 512
        self.fc = nn.Linear(512,100)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, out_channels, blocks, remove_last_relu=False):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride=2))
        self.in_channels = out_channels
        for i in range(1, blocks):
            if i == blocks - 1 and remove_last_relu:
                layers.append(BasicBlock(self.in_channels, out_channels, remove_last_relu=True))
            else:
                layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

    def forward_fc(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    """
    model = resnet18(**kwargs)
    # model = torch.nn.DataParallel(model)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model
