import torch
import torch.nn as nn
from tqdm import tqdm
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from resnets import ResNet50, ResNet101

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in tqdm(train_loader, desc = "Training Epoch {}".format(epoch)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, device, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    return total_loss / len(test_loader), correct / len(test_loader.dataset)
    