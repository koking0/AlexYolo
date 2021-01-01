from torch import nn


class YoloV1Model(nn.Module):
    def __init__(self):
        super(YoloV1Model, self).__init__()
        self.classes = 20
        self.convLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convLayer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convLayer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convLayer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convLayer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convLayer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.convLayer7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.flatten = nn.Sequential(
            nn.Flatten()
        )
        self.connLayer1 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.connLayer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=7 * 7 * 24),
            nn.Sigmoid(),
        )

    def forward(self, data):
        convLayer1 = self.convLayer1(data)
        convLayer2 = self.convLayer2(convLayer1)
        convLayer3 = self.convLayer3(convLayer2)
        convLayer4 = self.convLayer4(convLayer3)
        convLayer5 = self.convLayer5(convLayer4)
        convLayer6 = self.convLayer6(convLayer5)
        convLayer7 = self.convLayer7(convLayer6)
        flatten = self.flatten(convLayer7)
        connLayer1 = self.connLayer1(flatten)
        connLayer2 = self.connLayer2(connLayer1)
        return connLayer2
