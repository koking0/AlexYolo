import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import box_iou, box_convert


class YoloV2Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, coord=5, noobj=0.5):
        super(YoloV2Loss, self).__init__()
        self.S, self.B, self.C, self.coord, self.noobj = S, B, C, coord, noobj

    def forward(self, predict, target):
        """
        :param predict: [batch,SxSx(C+Bx5))]
        :param target: [batch,S,S,C+Bx5]
        :return: total loss
        """
        length = self.C + self.B * 5
        batch = target.size(0)
        predict = predict.view(batch, -1, length)
        target = target.view(batch, -1, length)

        coordMask = target[:, :, -1] > 0    # ground truth 中含有物体的 mask
        noobjMask = target[:, :, -1] == 0   # ground truth 中不含有物体的 mask
        coordMask = coordMask.unsqueeze(-1).expand_as(target)
        noobjMask = noobjMask.unsqueeze(-1).expand_as(target)

        coordPredict = predict[coordMask].view(-1, length)
        coordTarget = target[coordMask].view(-1, length)

        boxPredict = coordPredict[:, self.C:].contiguous().view(-1, 5)
        boxTarget = coordTarget[:, self.C:].contiguous().view(-1, 5)

        # compute loss which contain objects
        coordResponseMask = torch.BoolTensor(boxTarget.size())
        coordNotResponseMask = torch.BoolTensor(boxTarget.size())
        coordResponseMask.zero_()
        coordNotResponseMask = ~coordNotResponseMask.zero_()
        for i in range(0, boxTarget.size()[0], self.B):
            box1 = boxPredict[i:i+self.B].detach()
            box2 = boxTarget[i:i+self.B].detach()

            box1[:, 2:4] = torch.pow(box1[:, 2:4], 2)
            box2[:, 2:4] = torch.pow(box2[:, 2:4], 2)
            box1[:, :4] = box_convert(boxes=box1[:, :4], in_fmt="cxcywh", out_fmt="xyxy")
            box2[:, :4] = box_convert(boxes=box2[:, :4], in_fmt="cxcywh", out_fmt="xyxy")

            iou = box_iou(box1[:, :4], box2[:, :4])
            maxIou, maxIndex = iou.max(0)
            maxIndex = maxIndex.data
            coordResponseMask[i + maxIndex] = 1
            coordNotResponseMask[i + maxIndex] = 0

        # response loss
        boxPredictResponse = boxPredict[coordResponseMask].view(-1, 5)
        boxTargetResponse = boxTarget[coordResponseMask].view(-1, 5)
        containLoss = functional.mse_loss(boxPredictResponse[:, 4], boxTargetResponse[:, 4], reduction='sum')
        xyLoss = functional.mse_loss(boxPredictResponse[:, :2], boxTargetResponse[:, :2], reduction='sum')
        whLoss = functional.mse_loss(boxPredictResponse[:, 2:4], boxTargetResponse[:, 2:4], reduction='sum')
        localizationLoss = xyLoss + whLoss

        noobjPredict = predict[noobjMask].view(-1, length)
        noobjTarget = target[noobjMask].view(-1, length)

        # compute loss which do not contain objects
        noobjTargetMask = torch.ByteTensor(noobjTarget.size()).bool()
        noobjTargetMask.zero_()
        for i in range(self.B):
            noobjTargetMask[:, i * 5 + 4] = 1
        noobj_target_c = noobjTarget[noobjTargetMask]
        noobj_pred_c = noobjPredict[noobjTargetMask]
        noobjLoss = functional.mse_loss(noobj_pred_c, noobj_target_c, reduction='sum')

        # compute class prediction loss
        classPredict = coordPredict[:, :self.C]
        classTarget = coordTarget[:, :self.C]
        classLoss = functional.mse_loss(classPredict, classTarget, reduction='sum')

        # compute total loss
        totalLoss = self.coord * localizationLoss + containLoss + self.noobj * noobjLoss + classLoss
        return totalLoss


if __name__ == '__main__':
    pre = torch.rand(64, 7, 7, 30)
    tar = torch.rand(64, 7, 7, 30)
    YoloLoss = YoloV2Loss()
    loss = YoloLoss(pre, tar)
    print(loss.item())
