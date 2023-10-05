"""import torch
import numpy as np

from .models import FCN, save_model
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    

    # Data loading and preprocessing (replace with your dataset loading code)
    train_loader, valid_loader = load_dense_data(args.train_data, args.valid_data, batch_size=args.batch_size)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        confusion_matrix = ConfusionMatrix()

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update confusion matrix and calculate IoU
            confusion_matrix.add(outputs.argmax(1), batch_labels)
            iou = confusion_matrix.iou()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Log training metrics
        if train_logger:
            train_logger.add_scalar('train/loss', avg_loss, epoch)
            train_logger.add_scalar('train/iou', iou, epoch)

        print(f'Epoch [{epoch + 1}/{args.epochs}] - Avg. Loss: {avg_loss:.4f} - IoU: {iou:.4f}')


    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, default=0.001)  # Adjust learning rate
    parser.add_argument('--batch_size', type=int, default=32)  # Adjust batch size
    parser.add_argument('--epochs', type=int, default=10)  # Adjust the number of training epochs
    parser.add_argument('--train_data', default='data/train')  # Path to training data
    parser.add_argument('--valid_data', default='data/valid')  # Path to validation data


    args = parser.parse_args()
    train(args) """

import torch
import torch.optim as optim
import torch.utils.tensorboard as tb
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from .models import FCN, save_model
from .utils import load_dense_data, ConfusionMatrix, dense_transforms
from os import path
import numpy as np

def train(args):
    # Initialize the FCN model
    model = FCN()

    # Define the loss function (CrossEntropyLoss) and optimizer (Adam)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Set up TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Data loading and preprocessing
    train_loader, valid_loader = load_dense_data(args.train_data, args.valid_data, batch_size=args.batch_size)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        confusion_matrix = ConfusionMatrix()

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update confusion matrix and calculate IoU
            confusion_matrix.add(outputs.argmax(1), batch_labels)
            iou = confusion_matrix.iou()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Log training metrics
        if train_logger:
            train_logger.add_scalar('train/loss', avg_loss, epoch)
            train_logger.add_scalar('train/iou', iou, epoch)

        print(f'Epoch [{epoch + 1}/{args.epochs}] - Avg. Loss: {avg_loss:.4f} - IoU: {iou:.4f}')

    # Save the trained model
    save_model(model)

def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                         convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                              label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                              convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Custom arguments
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_data', default='data/train')
    parser.add_argument('--valid_data', default='data/valid')
    parser.add_argument('--num_workers', type=int, default=4)  # Ensure it's of type int

    args = parser.parse_args()
    train(args)


