import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, ColorJitter, RandomHorizontalFlip, ToTensor

from .models import TCN, save_model
from .utils import SpeechDataset, one_hot, load_data

def calculate_accuracy(predictions, targets):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0) * targets.size(1)
    accuracy = correct / total
    return accuracy

def train(args):
    from os import path
    
    model = TCN()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    Crossloss = nn.CrossEntropyLoss()
    
    train_data = SpeechDataset('data/train.txt', transform=one_hot)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    if args.log_dir is not None:
        train_logger = SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    for epoch in range(args.num_epoch):
        model.train()
        total_loss = 0.0

        for batch_data in train_loader:
            batch_data = batch_data[:, :, :-1]   # remove last column
            batch_labels = batch_data.argmax(dim=1)

            prediction = model(batch_data)

            loss_val = Crossloss(prediction, batch_labels)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            total_loss += loss_val.item()

        average_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch}, average loss is {average_loss}')

        if args.log_dir is not None:
            train_logger.add_scalar('Loss', average_loss, epoch)

    # Save the model after each epoch
    save_model(model)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)

    args = parser.parse_args()
    train(args)
