import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNetV2

class Planner(nn.Module):
    def __init__(self):
        super(Planner, self).__init()
        # Use a lightweight base model, such as MobileNetV2
        self.base_model = MobileNetV2(features_only=True)
        
        # Additional layers to adapt the model for your task
        self.conv1 = nn.Conv2d(1280, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1)
        
    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def save_model(model):
    # Save the model using torch.save as before
    torch.save(model.state_dict(), 'planner.th')

def load_model():
    # Load the model using torch.load as before
    model = Planner()
    model.load_state_dict(torch.load('planner.th'))
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
