import torch
from torch import nn
from torch.nn import functional


class YoloV1Loss(nn.Module):
    def __init__(self, Lambda=4, S=7, B=1, C=20):
        super(YoloV1Loss, self).__init__()
        self.Lambda = Lambda
        self.S, self.B, self.C = S, B, C

    def forward(self, predict, target):
        batches = target.size(0)
        length = self.C + self.B * 4

        predict = predict.view(batches, -1, length)
        target = target.view(batches, -1, length)

        boxPredict = predict[:, self.C:].contiguous().view(-1, 4)
        boxTarget = target[:, self.C:].contiguous().view(-1, 4)

        # 定位误差
        xyLoss = functional.mse_loss(boxPredict[:, :2], boxTarget[:, :2], reduction='sum')
        predictBoxWh = torch.sqrt(torch.abs(boxPredict[:, 2:] - boxPredict[:, :2]) + 1e-7)
        targetBoxWh = torch.sqrt(torch.abs(boxTarget[:, 2:] - boxTarget[:, :2]) + 1e-7)
        whLoss = functional.mse_loss(predictBoxWh, targetBoxWh, reduction='sum')
        positionLoss = xyLoss + whLoss

        # 分类误差
        predictClass = predict[:, :self.C]
        targetClass = target[:, :self.C]
        classLoss = functional.mse_loss(predictClass, targetClass, reduction='sum')

        totalLoss = self.Lambda * positionLoss + classLoss
        return totalLoss


if __name__ == '__main__':
    pre = torch.rand(64, 7, 7, 24)
    tar = torch.rand(64, 7, 7, 24)
    YoloLoss = YoloV1Loss()
    loss = YoloLoss(pre, tar)
    print(loss.item())
