import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
import argparse
import numpy as np

from torchvision import transforms
from PIL import Image

from .models import FCN, save_model
from .utils import load_dense_data, ConfusionMatrix
from glob import glob
#import dense_transforms

# Define a function to calculate class weights based on class distribution
def calculate_class_weights(class_distribution):
    total_samples = sum(class_distribution)
    class_weights = [total_samples / (class_distribution[i] + 1e-6) for i in range(len(class_distribution))]
    
    # Normalize the class weights
    sum_weights = sum(class_weights)
    class_weights = [weight / sum_weights for weight in class_weights]
    
    return class_weights

# Define a function to calculate Intersection over Union (IoU)
def compute_iou(confusion_matrix):
    class_iou = confusion_matrix.class_iou
    return class_iou.mean()

def train(args):
    # Initialize your FCN model
    model = FCN()

    # Create data loaders for training and validation sets
    train_loader, valid_loader = load_dense_data('dense_data/train', 'dense_data/valid', batch_size=32, num_workers=0, transform=None)

    # Initialize TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Calculate class distribution for training dataset
    train_class_distribution = DENSE_CLASS_DISTRIBUTION  # Use the provided class distribution

    # Define loss function (CrossEntropyLoss) and optimizer (e.g., Adam)
    criterion = torch.nn.CrossEntropyLoss()

    # Calculate class weights based on the class distribution
    class_weights = calculate_class_weights(train_class_distribution)

    # Use class weights in the loss function
    criterion = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weights))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(args.num_epochs):
        model.train()  # Set the model to training mode
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            # Forward pass
            logits = model(imgs)

            # Calculate the loss
            loss = criterion(logits, lbls)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute IoU using ConfusionMatrix
            confusion_matrix = ConfusionMatrix()
            confusion_matrix.add(logits.argmax(1), lbls)
            iou = compute_iou(confusion_matrix)

            # Log performance metrics and visualize results using log function
            global_step = epoch * len(train_loader) + batch_idx
            log(train_logger, imgs, lbls, logits, global_step)

            # Print progress
            print(f"Epoch [{epoch+1}/{args.num_epochs}] | "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"IoU: {iou:.4f}")

        # Validation loop (similar to training loop) to evaluate the model on the validation set
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for valid_batch_idx, (valid_imgs, valid_lbls) in enumerate(valid_loader):
                valid_logits = model(valid_imgs)

                # Compute IoU for validation
                valid_confusion_matrix = ConfusionMatrix()
                valid_confusion_matrix.add(valid_logits.argmax(1), valid_lbls)
                valid_iou = compute_iou(valid_confusion_matrix)

                # Log validation performance metrics and visualize results using log function
                valid_global_step = epoch * len(valid_loader) + valid_batch_idx
                log(valid_logger, valid_imgs, valid_lbls, valid_logits, valid_global_step)

                # Print validation progress
                print(f"Validation | "
                      f"Epoch [{epoch+1}/{args.num_epochs}] | "
                      f"Batch [{valid_batch_idx+1}/{len(valid_loader)}] | "
                      f"IoU: {valid_iou:.4f}")

        # Save the trained model at appropriate intervals
        if (epoch + 1) % args.save_interval == 0:
            save_model(model, epoch)  # Save the model with a unique identifier (e.g., epoch number)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('--num_epochs', type=int, default=100)  # Set the number of training epochs
    parser.add_argument('--save_interval', type=int, default=10)  # Save model every 'save_interval' epochs
    args = parser.parse_args()
    train(args)
