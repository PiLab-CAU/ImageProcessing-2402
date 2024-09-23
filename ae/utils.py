import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Function to visualize t-SNE of original and reconstructed images
def visualize_tsne(model, images, labels, device, dataset):
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)

    # Flatten the images and outputs for t-SNE (batch_size, channels, height, width) -> (batch_size, height*width*channels)
    images_flat = images.view(images.size(0), -1).cpu().numpy()
    outputs_flat = outputs.view(outputs.size(0), -1).cpu().numpy()

    # Concatenate original and reconstructed images
    combined_data = np.concatenate([images_flat, outputs_flat], axis=0)

    # Perform t-SNE on both original and reconstructed data
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_data)

    # Split t-SNE results back into original and reconstructed
    tsne_original = tsne_results[:len(images)]
    tsne_reconstructed = tsne_results[len(images):]

    # Plot the t-SNE results
    plt.figure(figsize=(10, 5))

    # Original images in t-SNE space
    scatter = plt.scatter(tsne_original[:, 0], tsne_original[:, 1], c=labels, cmap='tab10', marker='o', alpha=0.6, label='Original')

    # Reconstructed images in t-SNE space
    plt.scatter(tsne_reconstructed[:, 0], tsne_reconstructed[:, 1], c=labels, cmap='tab10', marker='x', alpha=0.6, label='Reconstructed')

    # Add colorbar
    cbar = plt.colorbar(scatter, ticks=range(len(dataset.classes)))

    # Replace colorbar ticks with class names
    cbar.set_ticklabels(dataset.classes)
    cbar.set_label('Classes')

    plt.legend()
    plt.title("t-SNE of Original and Reconstructed Images")
    plt.show()

# Function to visualize original and reconstructed images with class names
def visualize_reconstruction(model, images, labels, device, dataset):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)  # Reconstruct images

    # Convert the tensors to numpy for visualization
    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        # Plot original images
        axes[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
        axes[0, i].set_title(f"Original: {dataset.classes[labels[i]]}")
        axes[0, i].axis('off')

        # Plot reconstructed images
        axes[1, i].imshow(np.transpose(outputs[i], (1, 2, 0)))
        axes[1, i].set_title(f"Reconstructed: {dataset.classes[labels[i]]}")
        axes[1, i].axis('off')

    # Adjust layout for better visualization
    plt.tight_layout()
    plt.show()
