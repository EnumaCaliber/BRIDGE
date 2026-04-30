'''AlexNet for Tiny ImageNet in Pytorch.'''
import torch
import torch.nn as nn


class AlexNetTinyImageNet(nn.Module):
    def __init__(self, num_classes=200):
        super(AlexNetTinyImageNet, self).__init__()
        # Adapted for Tiny ImageNet (64x64 images, 200 classes)
        self.features = nn.Sequential(
            # Conv1: adapted for 64x64 input
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # After the conv layers: 64 -> 32 -> 15 -> 7 -> 3
        # So final feature map is 256 * 3 * 3 = 2304
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def test():
    net = AlexNetTinyImageNet()
    x = torch.randn(2, 3, 64, 64)
    y = net(x)
    print(y.size())
    print(f"Expected output shape: torch.Size([2, 200])")

# test()
