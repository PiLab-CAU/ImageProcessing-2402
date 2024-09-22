import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from ae import ConvAutoencoder

# Function to train the autoencoder model
def train(model, optimizer, train_loader, device, num_epochs=20):
    # Loop over the number of epochs
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Loop over the training data in batches
        for images, _ in train_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)  # Reconstruct images through the model
            loss = criterion(outputs, images)  # Calculate loss between original and reconstructed images

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average loss per epoch
        train_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

if __name__ == '__main__':

    # Convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load CIFAR-10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Load CIFAR-10 test dataset
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Set device to 'cuda' if GPU is available, otherwise 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    model = ConvAutoencoder().to(device)

    # Initialize the loss function
    criterion = nn.MSELoss()

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, optimizer, train_loader, device)