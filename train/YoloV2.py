import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models.Input.LoadVocDataSet import YoloVOCDetection
from models.Loss.YoloV2 import YoloV2Loss
from models.Network.YoloV2 import YoloV2Model


def train(epochs, model, criterion, optimizer, scheduler, trainLoader):
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        print(f"This is the {epoch + 1} / {epochs} iteration(lr = {optimizer.state_dict()['param_groups'][0]['lr']}):")
        for i, (inputs, labels) in enumerate(trainLoader):
            inputs, labels = inputs.to(device), labels.to(device)
            predict = model(inputs)
            loss = criterion(predict, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"\tThe {i} / {len(trainLoader)} batches loss = {loss.item():.7f}")
                writer.add_scalar("Train Loss", loss.item(), epoch * len(trainLoader) + i)
        scheduler.step()
        torch.save(model.state_dict(), f"./pth/YoloV2Model-epoch-{epoch}.pth")


if __name__ == '__main__':
    batchSize = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    writer = SummaryWriter()

    Voc2012TrainSet = YoloVOCDetection(root="../datasets/PASCAL-VOC", year="2012", image_set="trainval")
    Voc2007TrainSet = YoloVOCDetection(root="../datasets/PASCAL-VOC", year="2007", image_set="trainval")
    Voc2012TrainLoader = DataLoader(Voc2012TrainSet, batch_size=batchSize)
    Voc2007TrainLoader = DataLoader(Voc2007TrainSet, batch_size=batchSize)

    YoloModel = YoloV2Model().to(device)
    YoloLoss = YoloV2Loss().to(device)
    optimizer = torch.optim.SGD(YoloModel.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    train(epochs=70, model=YoloModel, criterion=YoloLoss, optimizer=optimizer, scheduler=scheduler,
          trainLoader=Voc2012TrainLoader)
    train(epochs=70, model=YoloModel, criterion=YoloLoss, optimizer=optimizer, scheduler=scheduler,
          trainLoader=Voc2007TrainLoader)
