import torch
import numpy as np
import torch.optim as optim
from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix
from . import dense_transforms as T
import torch.utils.tensorboard as tb
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FCN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.016, momentum=0.92, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_trans = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(96),
        transforms.ToTensor()
    ])
    
    train_data = load_dense_data('dense_data/train', transform=train_trans)
    val_data = load_dense_data('dense_data/valid')

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(args.log_dir)

    epochs = 30
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Epoch {epoch}: Train Loss={total_loss/len(train_loader):.4f}, Validation Accuracy={val_accuracy:.2f}%")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(model)

    print("Training completed.")


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