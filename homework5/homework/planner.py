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
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()

        def conv_block(c, h):
            return [nn.BatchNorm2d(h), nn.Conv2d(h, c, 5, 2, 2), nn.ReLU(True)]

        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c
        _conv.append(nn.Conv2d(h, 1, 1))
        self._conv = nn.Sequential(*_conv)

    def forward(self, img):
        x = self._conv(img)
        return spatial_argmax(x[:, 0])


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
