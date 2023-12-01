from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms



def collate_tensor_fn(batch):
    # Find the maximum height and width in the batch
    max_height = max(img.shape[1] for img in batch)
    max_width = max(img.shape[2] for img in batch)

    # Pad each image to the maximum height and width
    padded_batch = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in batch]

    # Stack the padded tensors
    stacked_batch = torch.stack(padded_batch, dim=0)

    return stacked_batch


def train(args):
    from os import path
    model = Planner()

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    #w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    loss = torch.nn.MSELoss()  #(weight=w / w.mean()).to(device)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_data(num_workers=2, transform=transform)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        losst = 0
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            losst += loss_val
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        print('\nEpoch: ' + str(epoch + 1) + " Loss: " + str(losst))

    save_model(model)

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
                        default='Compose([ColorJitter(0.4, 0.4, 0.4, 0.1), RandomHorizontalFlip(), ToTensor()])')
    args = parser.parse_args()
    train(args)