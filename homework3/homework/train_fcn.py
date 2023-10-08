import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
from torchvision import transforms
from .models import FCN, save_model
from .utils import load_dense_data, ConfusionMatrix, dense_transforms
from os import path
import numpy as np
import torch.nn as nn
from .dense_transforms import RandomRotation

import torch.nn.functional as F  # Import F for activation functions

def train(args):
    # Initialize the FCN model
    #model = FCN()
    model = FCN(num_classes = 5)
    

    # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set up data augmentation transforms for training data
    transform = dense_transforms.Compose([
        dense_transforms.RandomRotation(15),
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        dense_transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        dense_transforms.ToTensor(),
    ])

    # Data loading and preprocessing with data augmentation
    train_loader, valid_loader = load_dense_data(args.train_data, args.valid_data, batch_size=args.batch_size, num_workers=args.num_workers, transform=transform)

    # Set up TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        confusion_matrix = ConfusionMatrix()

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # Ensure batch_labels has the correct shape (batch_size, height, width)
            assert batch_labels.dim() == 3
            
            # Calculate loss
            #loss = criterion(outputs, batch_labels)
            loss = criterion(outputs, batch_labels.long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update confusion matrix and calculate IoU
            confusion_matrix.add(outputs.argmax(1), batch_labels)
            #iou = confusion_matrix.iou()
            iou = confusion_matrix.iou


        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Log training metrics
        if train_logger:
            train_logger.add_scalar('train/loss', avg_loss, epoch)
            train_logger.add_scalar('train/iou', iou, epoch)

        print(f'Epoch [{epoch + 1}/{args.epochs}] - Avg. Loss: {avg_loss:.4f} - IoU: {iou:.4f}')

    # Save the trained model
    save_model(model)
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Custom arguments
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_data', default='dense_data/train')  # Updated train_data path
    parser.add_argument('--valid_data', default='dense_data/valid')  # Updated valid_data path
    parser.add_argument('--num_workers', type=int, default=4)  # Ensure it's of type int

    args = parser.parse_args()
  

    train(args)