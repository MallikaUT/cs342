import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 48 * 64, 64)
        self.fc2 = nn.Linear(64, 2)  # Output size is 2 for the aim point

    def forward(self, img):
        x = F.relu(self.pool(self.conv1(img)))
        x = x.view(-1, 32 * 48 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output size is 2
        return x


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    else:
        raise ValueError("Model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    model = Planner()
    model.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return model

if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
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
