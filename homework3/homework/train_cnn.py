"""from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    
    #Your code here, modify your HW1 / HW2 code
    

    # Define your dataset and dataloaders
    train_loader = load_data(args.train_data, batch_size=args.batch_size)
    valid_loader = load_data(args.valid_data, batch_size=args.batch_size)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Log the training loss
        if train_logger:
            train_logger.add_scalar('train/loss', avg_loss, epoch)

        print(f'Epoch [{epoch + 1}/{args.epochs}] - Avg. Loss: {avg_loss:.4f}')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_data', default='data/train')
    parser.add_argument('--valid_data', default='data/valid')

    args = parser.parse_args()
    train(args)"""


from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('data/train')
    valid_data = load_data('data/valid')

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        acc_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = sum(acc_vals) / len(acc_vals)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(acc_vals) / len(acc_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)
    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')

    args = parser.parse_args()
    train(args)



