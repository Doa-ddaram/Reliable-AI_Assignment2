from utils.train import train, evaluate
import torch
from torchvision import datasets, transforms
from utils.resnets import ResNet50, ResNet101
from torch.utils.data import DataLoader
from torch import optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = ResNet50(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(50):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = evaluate(model, device, test_loader)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    main()