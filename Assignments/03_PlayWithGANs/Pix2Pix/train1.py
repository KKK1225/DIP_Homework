import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR

# 超参数
input_dim = 10   # x 的维度
condition_dim = 10  # 条件的维度
noise_dim = 5   # 随机噪声 z 的维度
output_dim = 10  # 生成目标 y 的维度
batch_size = 32
epochs = 1000
lr = 0.0002

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(modelG, modelD, dataloader, optimizerG, optimizerD, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    modelG.train()
    modelD.train()
    # total_d_loss = 0.0
    # total_g_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # ---- 训练判别器 ----
        optimizerD.zero_grad()

        
        z = torch.randn(image_rgb.size(0), 100, 1, 1).to(device)        # 随机噪声
        real_score = modelD(image_semantic, image_rgb)
        real_loss = torch.log(real_score + 1e-8).mean()
        fake_y = modelG(image_semantic, z)
        fake_score = modelD(image_semantic, fake_y.detach())
        fake_loss = torch.log(1 - fake_score + 1e-8).mean()

        # 判别器总损失
        d_loss = -(real_loss + fake_loss)
        d_loss.backward()
        optimizerD.step()

        # ---- 训练生成器 ----
        optimizerG.zero_grad()

        fake_score = modelD(image_semantic, fake_y)
        g_loss = -torch.log(fake_score + 1e-8).mean()
        g_loss.backward()
        optimizerG.step()

        total_d_loss = d_loss.item()
        total_g_loss = g_loss.item()
        ave_d_loss = total_d_loss / len(dataloader)
        ave_g_loss = total_g_loss / len(dataloader)


        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, fake_y, 'train_results', epoch)

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], G - Loss: {ave_g_loss:.4f}, D - Loss: {ave_d_loss:.4f}')

def validate(modelG, modelD, dataloader, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    modelG.eval()
    modelD.eval()

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            z = torch.randn(image_semantic.size(0), 100, 1, 1).to(device)

            real_score = modelD(image_semantic, image_rgb)
            real_loss = torch.log(real_score + 1e-8).mean()

            # 判别器对伪造数据的损失
            fake_y = modelG(image_semantic, z)
            fake_score = modelD(image_semantic, fake_y.detach())
            fake_loss = torch.log(1 - fake_score + 1e-8).mean()

            # 判别器总损失
            d_loss = -(real_loss + fake_loss)
            g_loss = -torch.log(fake_score + 1e-8).mean()

            total_d_loss = d_loss.item()
            total_g_loss = g_loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, fake_y, 'val_results', epoch)

    # Calculate average validation loss
    ave_d_loss = total_d_loss / len(dataloader)
    ave_g_loss = total_g_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], G - Loss: {ave_g_loss:.4f}, D - Loss: {ave_d_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')
    print(device)

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    # model = Generator().to(device)
    # Initialize model, loss function, and optimizer
    Generate = Generator().to(device)
    Discriminate = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(Generate.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(Discriminate.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer_G, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs+1):
        train_one_epoch(Generate, Discriminate, train_loader, optimizer_G, optimizer_D, device, epoch, num_epochs)
        validate(Generate, Discriminate, val_loader, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(Generate.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
