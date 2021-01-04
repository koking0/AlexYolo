import torch

from torch import nn


class YoloV2Model(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloV2Model, self).__init__()
        self.S, self.B, self.C = S, B, C

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convLayer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convLayer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )
        self.convLayer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )
        self.flatten = nn.Sequential(
            nn.Flatten(),
        )
        self.connLayer1 = nn.Sequential(
            nn.Linear(in_features=self.S * self.S * 1024, out_features=4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
        )
        self.connLayer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=self.S * self.S * (self.C + self.B * 5)),
            nn.Sigmoid(),
        )

    def forward(self, data):
        convLayer1 = self.convLayer1(data)
        convLayer2 = self.convLayer2(convLayer1)
        convLayer3 = self.convLayer3(convLayer2)
        convLayer4 = self.convLayer4(convLayer3)
        convLayer5 = self.convLayer5(convLayer4)
        convLayer6 = self.convLayer6(convLayer5)
        flatten = self.flatten(convLayer6)
        connLayer1 = self.connLayer1(flatten)
        connLayer2 = self.connLayer2(connLayer1)
        return connLayer2


if __name__ == '__main__':
    images = torch.rand(2, 3, 448, 448)
    YoloModel = YoloV2Model()
    predict = YoloModel(images)
    print(predict.size())
