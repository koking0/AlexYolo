import torch

from PIL import ImageDraw

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.ops import box_convert
from torchvision.datasets import VOCDetection

from utils.Color import getRandomColor


class YoloVOCDetection(VOCDetection):
    def __init__(self,
                 root,
                 S=7,
                 B=2,
                 C=20,
                 imageSize=448,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None):
        self.S, self.B, self.C = S, B, C
        self.VocLabelMap = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                            "bus", "car", "cat", "chair", "cow",
                            "diningtable", "dog", "horse", "motorbike", "person",
                            "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        self.transform = transform or transforms.Compose([
            transforms.Resize([imageSize, imageSize]),
            transforms.ColorJitter(brightness=(1, 3), contrast=(1, 3), saturation=(1, 3), hue=(0.1, 0.3)),
            transforms.ToTensor(),
        ])
        super().__init__(root, year, image_set, download, self.transform, self.getTargetTransform)

    def getTargetTransform(self, data):
        length = self.B * 5 + self.C
        target = torch.zeros(self.S, self.S, length)

        boxes, labels = torch.zeros(len(data["annotation"]["object"]), 4), []
        width, height = int(data["annotation"]["size"]["width"]), int(data["annotation"]["size"]["height"])
        for index, obj in enumerate(data["annotation"]["object"]):
            x0, y0, x1, y1 = obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"]
            box = torch.tensor([int(x0) / width, int(y0) / height, int(x1) / width, int(y1) / height])
            box = box_convert(boxes=box, in_fmt="xyxy", out_fmt="cxcywh")
            box[2:] = torch.sqrt(box[2:])
            boxes[index] = box
            labels.append(self.VocLabelMap.index(obj["name"]))
        labels = torch.tensor(labels)

        # 每个网格的宽度
        cellSize = 1.0 / float(self.S)
        # 每个 bounding box 的中心点坐标
        boxesXy = boxes[:, :2]
        for box in range(labels.size()[0]):
            xy = boxesXy[box]
            # 表示其在网格上的位置的 y&x 索引
            ij = (xy / cellSize).ceil() - 1
            i, j = int(ij[1]), int(ij[0])
            # 对应类别 confidence 设置为1
            target[i, j, labels[box].item()] = 1
            # 将最后 5 * self.B 位设置为 bounding box 的坐标
            target[i, j, self.C:] = torch.cat((boxes[box], torch.tensor([1])), 0).repeat(1, self.B)
        return target


if __name__ == '__main__':
    VocTrainSet = YoloVOCDetection(root="../../datasets/PASCAL-VOC", year="2012", image_set="train")
    VocTrainLoader = DataLoader(VocTrainSet, batch_size=8)

    trainIter = iter(VocTrainLoader)
    feature, label = trainIter.__next__()
    print(feature[0].size())
    print(label[0].size())

    image = transforms.functional.to_pil_image(feature[0])
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for i in range(7):
        for j in range(7):
            if 1 in label[0, i, j, :20]:
                index = label[0, i, j, :20].tolist().index(1)
                b = label[0, i, j, 20:24]
                b[2:] = torch.pow(b[2:], 2)
                x0, y0, x1, y1 = box_convert(boxes=b, in_fmt="cxcywh", out_fmt="xyxy")
                x0, y0, x1, y1 = x0 * w, y0 * h, x1 * w, y1 * h
                print(x0, y0, x1, y1)
                color = getRandomColor()
                draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=color, width=3)
                draw.text([x0 + 5, y0 + 5], VocTrainSet.VocLabelMap[index], fill=color)
    image.show()
