import torch
import numpy as np

from torch import nn
from torchvision import transforms

from PIL import Image, ImageDraw
from torchvision.ops import box_convert, batched_nms


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class YoloLayer(nn.Module):
    def __init__(self, anchors, numClasses, inputDim):
        super(YoloLayer, self).__init__()
        self.anchors = anchors
        self.numAnchors = len(anchors)
        self.numClasses = numClasses
        self.inputDim = inputDim
        self.gridSize = 0

    def forward(self, predicts):
        FloatTensor = torch.FloatTensor
        batchSize, gridSize = predicts.size(0), predicts.size(2)
        predicts = predicts.view(batchSize, self.numAnchors, 5 + self.numClasses, gridSize, gridSize)
        predicts = predicts.permute(0, 1, 3, 4, 2).contiguous()

        x = torch.sigmoid(predicts[..., 0])
        y = torch.sigmoid(predicts[..., 1])
        w = predicts[..., 2]
        h = predicts[..., 3]
        predictConfidence = torch.sigmoid(predicts[..., 4])
        predictClass = torch.sigmoid(predicts[..., 5:])

        if gridSize != self.gridSize:
            self.computeGridOffsets(gridSize)

        predictBoxes = FloatTensor(predicts[..., :4].shape)
        predictBoxes[..., 0] = x.data + self.gridX
        predictBoxes[..., 1] = y.data + self.gridY
        predictBoxes[..., 2] = torch.exp(w.data) * self.anchorW
        predictBoxes[..., 3] = torch.exp(h.data) * self.anchorH

        output = torch.cat((predictBoxes.view(batchSize, -1, 4) * self.stride,
                            predictConfidence.view(batchSize, -1, 1),
                            predictClass.view(batchSize, -1, self.numClasses)), -1)
        return output

    def computeGridOffsets(self, gridSize):
        FloatTensor = torch.FloatTensor
        self.gridSize = gridSize
        self.stride = self.inputDim // self.gridSize
        self.gridX = torch.arange(self.gridSize).repeat(self.gridSize, 1).view(1, 1, self.gridSize, self.gridSize)
        self.gridX = self.gridX.type(FloatTensor)
        self.gridY = torch.arange(self.gridSize).repeat(self.gridSize, 1).t().view(1, 1, self.gridSize, self.gridSize)
        self.gridY = self.gridY.type(FloatTensor)
        self.scaleAnchors = FloatTensor([(w / self.stride, h / self.stride) for w, h in self.anchors])
        self.anchorW = self.scaleAnchors[:, 0:1].view(1, self.numAnchors, 1, 1)
        self.anchorH = self.scaleAnchors[:, 1:2].view(1, self.numAnchors, 1, 1)


class Darknet(nn.Module):
    def __init__(self, cfgFilePath):
        super(Darknet, self).__init__()
        self.cfgFile = cfgFilePath
        self.blocks = self.parseCfg()
        self.netInfo, self.moduleList = self.createModules()

    def forward(self, x):
        layerOutputs, yoloOutputs = [], []
        for index, (block, module) in enumerate(zip(self.blocks, self.moduleList)):
            blockType = block["type"]
            if blockType in ["convolutional", "upsample"]:
                x = module(x)
            elif blockType == "route":
                x = torch.cat([layerOutputs[i] for i in block["layers"]], 1)
            elif blockType == "shortcut":
                x = layerOutputs[index - 1] + layerOutputs[index + int(block["from"])]
            elif blockType == "yolo":
                x = module(x)
                yoloOutputs.append(x)
            layerOutputs.append(x)
        yoloOutputs = torch.cat(yoloOutputs, 1)
        return yoloOutputs

    def parseCfg(self):
        """
        Takes a configuration file, return a list of blocks, each blocks describe a block in neural network to be built.
        Block is represented as a dictionary in the list.
        :return: list of block in neural network
        """
        with open(self.cfgFile, mode="r") as fp:
            lines = fp.read().split("\n")  # store lines in a list
            lines = [x for x in lines if len(x) > 0]  # filter empty lines
            lines = [x for x in lines if x[0] != '#']  # filter comments
            lines = [x.rstrip().lstrip() for x in lines]  # rid of fringe whitespaces

        block, blocks = {}, []
        for line in lines:
            if line[0] == '[':  # this marks the start of a new block
                if len(block) != 0:  # if block is not empty, implies it is storing values of previous block
                    blocks.append(block)  # add it to blocks list
                    block = {}  # re-init the block
                block["type"] = line[1:-1].rstrip().lstrip()
            else:
                key, value = line.split('=')  # according to '=' segmentation
                block[key.rstrip().lstrip()] = value.rstrip().lstrip()
        blocks.append(block)
        return blocks

    def createModules(self):
        netInfo = self.blocks.pop(0)
        moduleList = nn.ModuleList()
        previousFilters = 3
        outputFilters = []

        for index, block in enumerate(self.blocks):
            module = nn.Sequential()

            if block["type"] == "convolutional":
                activation = block["activation"]
                try:
                    batchNormalize = int(block["batch_normalize"])
                    bias = False
                except:
                    batchNormalize = 0
                    bias = True
                filters = int(block["filters"])
                padding = int(block["pad"])
                kernelSize = int(block["size"])
                stride = int(block["stride"])
                pad = (kernelSize - 1) // 2 if padding else 0

                # add the convolutional layer
                conv = nn.Conv2d(in_channels=previousFilters,
                                 out_channels=filters,
                                 kernel_size=kernelSize,
                                 stride=stride,
                                 padding=pad,
                                 bias=bias)
                module.add_module(f"conv_{index}", conv)

                # add batch norm layer
                if batchNormalize:
                    bn = nn.BatchNorm2d(num_features=filters)
                    module.add_module(f"batch_norm_{index}", bn)

                # add activation
                if activation == "leaky":
                    atv = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    module.add_module(f"leaky_{index}", atv)
            elif block["type"] == "upsample":
                upSample = nn.Upsample(scale_factor=int(block["stride"]), mode="bilinear", align_corners=True)
                module.add_module(f"upsample_{index}", upSample)
            elif block["type"] == "route":
                block["layers"] = list(map(int, block["layers"].split(',')))
                start = int(block["layers"][0])
                end = int(block["layers"][1]) if len(block["layers"]) == 2 else 0
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index

                filters = outputFilters[index + start]
                filters += outputFilters[index + end] if end < 0 else 0

                route = EmptyLayer()
                module.add_module(f"route_{index}", route)
            elif block["type"] == "shortcut":
                shortcut = EmptyLayer()
                module.add_module(f"shortcut_{index}", shortcut)
            elif block["type"] == "yolo":
                mask = list(map(int, block["mask"].split(',')))
                anchors = list(map(int, block["anchors"].split(',')))
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                numClasses = int(block["classes"])
                imageSize = int(netInfo["width"])

                yoloLayer = YoloLayer(anchors=anchors, numClasses=numClasses, inputDim=imageSize)
                module.add_module(f"yolo_{index}", yoloLayer)

            moduleList.append(module)
            previousFilters = filters
            outputFilters.append(filters)
        return netInfo, moduleList

    def encodePredict(self, prediction, confThreshold=0.5, nmsThreshold=0.4):
        boxCorner = prediction[..., :4].detach()
        # (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[..., :4] = box_convert(boxes=boxCorner, in_fmt="cxcywh", out_fmt="xyxy")

        outputs = [None for _ in range(len(prediction))]
        for idx, img in enumerate(prediction):
            img = img[img[:, 4] > confThreshold]    # filter out confidence score greater than threshold
            if img.size(0):     # surplus after filter
                score = img[:, 4] * img[:, 5:].max(1)[0]
                img = img[(-score).argsort()]
                classesConfidences, classPredicts = img[:, 5:].max(1, keepdim=True)
                # detections [x1, y1, x2, y2, confidence, class confidence, class predict]
                detections = torch.cat([img[:, :5], classesConfidences.float(), classPredicts.float()], 1)
                # print(f"detections = {detections} \n detections.size() = {detections.size()}")

                boxes, boxesScores, classesIndex = detections[:, :4], detections[:, 4], detections[:, 6]
                keep = batched_nms(boxes=boxes, scores=boxesScores, idxs=classesIndex, iou_threshold=nmsThreshold)
                # print(f"keep = {keep}, keep.size() = {keep.size()}")
                if keep.size()[0]:
                    outputs[idx] = detections[keep]
        return outputs

    def loadWeights(self, weightFilePath):
        with open(weightFilePath, mode="rb") as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]

            weights = np.fromfile(fp, dtype=np.float32)
            ptr = 0
            for index, (block, module) in enumerate(zip(self.blocks, self.moduleList)):
                if block["type"] == "convolutional":
                    convLayer = module[0]
                    if block.get("batch_normalize", None):
                        bnLayer = module[1]
                        bnBiases = bnLayer.bias.numel()
                        # Bias
                        bnBias = torch.from_numpy(weights[ptr: ptr + bnBiases]).view_as(bnLayer.bias)
                        bnLayer.bias.data.copy_(bnBias)
                        ptr += bnBiases
                        # Weight
                        bnWeight = torch.from_numpy(weights[ptr: ptr + bnBiases]).view_as(bnLayer.weight)
                        bnLayer.weight.data.copy_(bnWeight)
                        ptr += bnBiases
                        # Running Mean
                        bnRunningMean = torch.from_numpy(weights[ptr: ptr + bnBiases]).view_as(bnLayer.running_mean)
                        bnLayer.running_mean.data.copy_(bnRunningMean)
                        ptr += bnBiases
                        # Running Var
                        bnRunningVar = torch.from_numpy(weights[ptr: ptr + bnBiases]).view_as(bnLayer.running_var)
                        bnLayer.running_var.data.copy_(bnRunningVar)
                        ptr += bnBiases
                    else:
                        convBiases = convLayer.bias.numel()
                        convBias = torch.from_numpy(weights[ptr: ptr + convBiases]).view_as(convLayer.bias)
                        convLayer.bias.data.copy_(convBias)
                        ptr += convBiases
                    convWeights = convLayer.weight.numel()
                    convWeight = torch.from_numpy(weights[ptr: ptr + convWeights]).view_as(convLayer.weight)
                    convLayer.weight.data.copy_(convWeight)
                    ptr += convWeights


if __name__ == '__main__':
    image = Image.open("../../images/dog-cycle-car.png")
    transform = transforms.Compose([
        transforms.Resize([608, 608]),
        transforms.ToTensor(),
    ])
    imageTensor = transform(image).unsqueeze(0)
    print(f"image.size() = {imageTensor.size()}")
    model = Darknet("../../cfg/yolov3.cfg")
    model.loadWeights("../../weights/yolov3.weights")
    predict = model(imageTensor)
    print(f"predict = {predict}, predict.size() = {predict.size()}")
    output = model.encodePredict(prediction=predict)
    print(f"output = {output}, len(output) = {len(output)}")

    image = image.resize(size=(608, 608))
    print(image.size)
    draw = ImageDraw.Draw(image)
    for b in output:
        for i in b:
            draw.rectangle([int(i[0]), int(i[1]), int(i[2]), int(i[3])], outline=(0, 0, 255), width=3)
    image.show()
