import torch
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

class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        layers = []
        ic = 3
        structure = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        for l in structure:
            if l == 'M':
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(torch.nn.Conv2d(in_channels=ic, out_channels=l, padding=1, kernel_size=3, stride=1))
                layers.append(torch.nn.BatchNorm2d(l))
                layers.append(torch.nn.ReLU(inplace=True))
                ic = l

        layers.append(torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(128, 1, 1))

        self._conv = torch.nn.Sequential(*layers)

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
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r

if __name__ == '__main__':
    from controller import control
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
