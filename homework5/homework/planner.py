import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import save, load
from os import path

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class Planner(nn.Module):
    def __init__(self):
        super(Planner, self).__init__()

        # Define your planner model using an encoder-decoder structure or any architecture of your choice.
        # Example architecture:
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers as needed
        )

        self.decoder = nn.Sequential(
            nn.Linear(64 * 24 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output 2D aim point
        )

    def forward(self, img):
        # Pass the input image through the encoder
        x = self.encoder(img)
        # Flatten the feature map
        x = x.view(x.size(0), -1)
        # Pass through the decoder to get the predicted aim point
        aim_point = self.decoder(x)
        return aim_point

def save_model(model):
    if isinstance(model, Planner):
        save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    else:
        raise ValueError("Model type '%s' not supported!" % str(type(model)))

def load_model():
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r

if __name__ == '__main__':
    #from controller import control
    from utils import PyTux
    from argparse import ArgumentParser

    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
