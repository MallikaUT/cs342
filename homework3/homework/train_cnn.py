import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from .models import CNNClassifier, save_model
from .utils import accuracy, load_data

import torch.utils.tensorboard as tb

# Define the ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  
        out = self.relu(out)
        return out

def train(args):
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNNClassifier().to(device)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    """
    Your code here, modify your HW1 / HW2 code
    """

    # Define data transformations and data loaders with input normalization
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = load_data(args.train_data, batch_size=args.batch_size, transform=train_transforms)
    valid_loader = load_data(args.valid_data, batch_size=args.batch_size)

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma)

    best_valid_accuracy = 0.0
    no_improvement_count = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if train_logger:
            train_logger.add_scalar('train/loss', avg_loss, epoch)

        print(f'Epoch [{epoch + 1}/{args.epochs}] - Loss: {avg_loss:.4f}')
        scheduler.step()

        model.eval()
        valid_acc_vals = []
        for batch_data, batch_labels in valid_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            valid_outputs = model(batch_data)
            valid_accuracy = accuracy(valid_outputs, batch_labels).detach().cpu().numpy()
            valid_acc_vals.append(valid_accuracy)

        avg_valid_accuracy = sum(valid_acc_vals) / len(valid_acc_vals)

        if valid_logger:
            valid_logger.add_scalar('valid/accuracy', avg_valid_accuracy, epoch)

        print(f'Validation Accuracy: {avg_valid_accuracy:.4f}')

        if avg_valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = avg_valid_accuracy
            no_improvement_count = 0
            save_model(model)
        else:
            no_improvement_count += 1

        if no_improvement_count >= args.early_stopping_patience:
            print("No improvement")
            break

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--train_data', default='data/train')
    parser.add_argument('--valid_data', default='data/valid')
    parser.add_argument('--weight_decay', type=float, default=0.001)  # L2 weight regularization
    parser.add_argument('--lr_scheduler_step', type=int, default=10)  # Learning rate scheduler step size
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5)  # Learning rate scheduler gamma
    parser.add_argument('--early_stopping_patience', type=int, default=10)

    args = parser.parse_args()
    train(args)
