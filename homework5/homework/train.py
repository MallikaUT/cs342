from .planner import Planner, save_model
from torch import device, cuda, load, optim, nn
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms as DT


def train(args):
    from os import path
    print("Initializing Model =================================")
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    
    """
    print("installing torch ...")
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)
    if args.use_existing:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    loss = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("installing inspect ...")
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    print("loading data ...")
    train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)

    global_step = 0
    print("Begin Training =================================")
    for epoch in range(args.num_epoch):
        model.train()
        losses = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            pred = model(img)
            loss_val = loss(pred, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        if train_logger is None:
            print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
        save_model(model)

    save_model(model)


def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)]) / 2
    ax.add_artist(plt.Circle(WH2 * (label[0].cpu().detach().numpy() + 1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2 * (pred[0].cpu().detach().numpy() + 1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)