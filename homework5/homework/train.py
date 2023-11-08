from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms


def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = Planner().to(device)
    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-6)

    train_data = load_data('drive_data', transform=transform, num_workers=args.num_workers)

    for epoch in range(50):
        model.train()
        for image, heatmap, delta in train_data:
            image = image.to(device)
            heatmap = heatmap.to(device)

            pred_heatmap = model(image)

            l = loss(pred_heatmap, heatmap).mean()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
    #raise NotImplementedError('train')
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
