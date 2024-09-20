import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from ae import ConvAutoencoder


def train(model, optimizer, train_loader, device, num_epochs=20):

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

if __name__ == '__main__':


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, optimizer, train_loader, device)

