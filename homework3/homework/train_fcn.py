import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms as T
import torch.utils.tensorboard as tb
from torchvision import transforms

def train(args):
    from os import path

    # Initialize your FCN model
    model = FCN()

    # Initialize TensorBoard loggers
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    loss_func = ClassificationLoss()
    loss_func.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.016, momentum=0.92, weight_decay=1e-4)

    # Number of training epochs
    epochs = 30

    # Define data augmentation transformations
    train_trans = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96),
        transforms.ToTensor()
    ])

    # Load training and validation data
    train_data = load_dense_data('dense_data/train', transform=train_trans)
    val_data = load_dense_data('dense_data/valid')

    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        count = 0
        total_loss = 0

        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x)

            # Calculate the loss
            loss = loss_func(y_pred, y.long())
            total_loss += loss.item()
            count += 1

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training loss for this epoch
        print(f"Epoch: {epoch}, Loss: {total_loss / count:.4f}")

        # Set the model to evaluation mode
        model.eval()
        count = 0
        accuracy = 0

        for image, label in val_data:
            image = image.to(device)
            label = label.to(device)

            # Forward pass for validation
            pred = model(image)

            # Calculate accuracy
            accuracy += (pred.argmax(1) == label).float().mean().item()
            count += 1

        # Print validation accuracy for this epoch
        print(f"Epoch: {epoch}, Accuracy: {accuracy / count:.4f}")

        # Check if the desired accuracy is reached and break if so
        if accuracy / count > 0.87:
            print("Accuracy threshold reached. Training completed.")
            break

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
    # Put custom arguments here

    args = parser.parse_args()
    train(args)