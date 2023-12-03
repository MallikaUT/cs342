import numpy as np
import torch
from torch.serialization import load
import torchvision
import time
from PIL import Image
from os import path
from torchvision.transforms import functional as F

from image_agent.models import load_model, Detector

GOALS = np.float32([[0, 75], [0, -75]])

LOST_STATUS_STEPS = 10
LOST_COOLDOWN_STEPS = 10
START_STEPS = 25
LAST_PUCK_DURATION = 4
MIN_SCORE = 0.2
MAX_DET = 15
MAX_DEV = 0.7
MIN_ANGLE = 20
MAX_ANGLE = 120
TARGET_SPEED = 15
STEER_YIELD = 15
DRIFT_THRESH = 0.7
TURN_CONE = 100

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')


def norm(vector):
    return torch.norm(torch.tensor(vector))


class Team:
    agent_type = 'image'

    def __init__(self):
        self.kart = 'wilber'
        self.initialize_vars()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = torch.jit.load(path.join(path.dirname(path.abspath(__file__)), 'detector.pt'), map_location=device)
        self.model = load_model(Detector, self.model_path, device='cpu')
        self.model.to(device)
        self.model.eval()
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((128, 128)),
                                                         torchvision.transforms.ToTensor()])
    def __init__(self):
        self.kart = 'wilber'
        self.initialize_vars()

        # Make sure to import 'path' and 'torchvision'
        self.model_path = path.join(path.dirname(path.abspath(__file__)), 'detector.pt')
        try:
            self.model = load_model(Detector, self.model_path, device='cpu')
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading the model: {e}")
            # Handle the error (e.g., provide a default model)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])



    def new_match(self, team: int, num_players: int) -> list:
        self.team, self.num_players = team, num_players
        self.initialize_vars()
        print(f"Using {device} Match Started: {time.strftime('%H-%M-%S')}")
        return [self.kart] * num_players

    def initialize_vars(self):
        self.step = 0
        self.timer1 = 0

        self.puck_prev1 = 0
        self.last_seen1 = 0
        self.recover_steps1 = 0
        self.use_puck1 = True
        self.cooldown1 = 0

        self.timer2 = 0

        self.puck_prev2 = 0
        self.last_seen2 = 0
        self.recover_steps2 = 0
        self.use_puck2 = True
        self.cooldown2 = 0

    # Assuming self.model is an instance of Detector
    def calculate_goal_parameters(self, team, loc, dir):
        own_goal = torch.tensor(GOALS[team])
        dist_own_goal = torch.norm(loc - own_goal)
        goal_dir = own_goal - loc
        goal_dist = torch.norm(goal_dir)
        goal_dir = goal_dir / torch.norm(goal_dir)
        
        goal_angle_rad = torch.acos(torch.clamp(torch.dot(dir, goal_dir), -1, 1))
        signed_goal_angle = torch.deg2rad(torch.sign(torch.cross(torch.tensor(dir.numpy()), torch.tensor(goal_dir.numpy()))) * goal_angle_rad)
        
        return dist_own_goal, signed_goal_angle


    def act(self, player_state, player_image):
        player_info = player_state[0]
        image = player_image[0]

        try:
            with torch.no_grad():
                img = F.to_tensor(Image.fromarray(image)).to(device)
                img = img.unsqueeze(0)  # Add batch dimension
                img = img[:, :3, :, :]  # Keep only the first 3 channels if there are more

                # Ensure img is a 4D tensor
                img = img.squeeze(0) if img.size(0) == 1 else img

                pred_boxes = self.model.detect(img, max_pool_ks=7, min_score=MIN_SCORE, max_det=MAX_DET)

            #print(f"Prediction boxes: {pred_boxes}")

        except Exception as e:
            print(f"Error during detection: {e}")
            pred_boxes = None


        if pred_boxes is not None and len(pred_boxes) > 0:
            front_raw = np.array(player_info['kart']['front'])
            loc_raw = np.array(player_info['kart']['location'])

            # Convert NumPy array to PyTorch tensor for front and location
            front = torch.tensor(np.float32(front_raw)[[0, 2]])
            loc = torch.tensor(np.float32(loc_raw)[[0, 2]])

        # execute when we find puck on screen
        if len(pred_boxes) > 0:
            print("Puck seen")
            print(f"front shape: {front.shape}")
            print(f"loc shape: {loc.shape}")
            puck_loc = torch.mean(torch.tensor([cx[1] for cx in pred_boxes], dtype=torch.float32)) / 64 - 1

            if self.use_puck1 and torch.abs(puck_loc - self.puck_prev1) > MAX_DEV:
                puck_loc = self.puck_prev1
                self.use_puck1 = False
            else:
                self.use_puck1 = True

            self.puck_prev1 = puck_loc
            self.last_seen1 = self.step
        # if puck not seen then use prev location or start lost actions
        elif self.step - self.last_seen1 < LAST_PUCK_DURATION:
            self.use_puck1 = False
            puck_loc = self.puck_prev1
        else:
            puck_loc = None
            self.recover_steps1 = LOST_STATUS_STEPS
            print("Puck not seen")
        # calculate direction vector
        dir = front - loc
        dir = dir / torch.norm(dir)

        # calculate angle and distance to own goal
        dist_own_goal, signed_goal_angle = self.calculate_goal_parameters(self.team, loc, dir)

        # calculate angle and distance to opp goal
        goal_dir = torch.tensor(GOALS[self.team]) - loc
        dist_opp_goal, signed_opp_goal_angle = self.calculate_goal_parameters(self.team + 1, loc, dir)

        # restrict dist between [1,2] so we can use a weight function
        goal_dist = ((torch.clamp(dist_opp_goal, 10, 100) - 10) / 90) + 1

        # set aim point if not cooldown or in recovery
        if (self.cooldown1 == 0 or len(pred_boxes) > 0) and self.recover_steps1 == 0:
            # if angle isn't extreme then weight our attack angle by dist
            if MIN_ANGLE < torch.abs(signed_goal_angle) < MAX_ANGLE:
                distW = 1 / goal_dist ** 3
                aim_point = puck_loc + torch.sign(puck_loc - signed_goal_angle / TURN_CONE) * 0.3 * distW
            # if two tight then just chase puck
            else:
                aim_point = puck_loc
            # sets the speed as const if found
            if self.last_seen1 == self.step:
                brake = False
                acceleration = 0.75 if torch.norm(player_info['kart']['velocity']) < TARGET_SPEED else 0
            else:
                brake = False
                acceleration = 0
        # cooldown actions
        elif self.cooldown1 > 0:
            self.cooldown1 -= 1
            brake = False
            acceleration = 0.5
            aim_point = signed_goal_angle / TURN_CONE
        # recovery actions
        else:
            # if not a goal keep backing up
            if dist_own_goal > 10:
                aim_point = signed_goal_angle / TURN_CONE
                acceleration = 0
                brake = True
                self.recover_steps1 -= 1
            # if at goal then cooldown on reversing
            else:
                self.cooldown1 = LOST_COOLDOWN_STEPS
                aim_point = signed_goal_angle / TURN_CONE
                acceleration = 0.5
                brake = False
                self.recover_steps1 = 0

        # set steering/drift
        steer = torch.clamp(aim_point * STEER_YIELD, -1, 1)
        drift = torch.abs(aim_point) > DRIFT_THRESH

        p1 = {
            'steer': signed_goal_angle if self.step < START_STEPS else steer.item(),
            'acceleration': 1 if self.step < START_STEPS else acceleration.item(),
            'brake': brake,
            'drift': drift,
            'nitro': False,
            'rescue': False
        }

        
        # Player 2 (same agent for now)
        player_info = player_state[1]
        image = player_image[1]

        print("Now enter in Player2")

        try:
            with torch.no_grad():
                img = F.to_tensor(Image.fromarray(image)).to(device)
                img = img[:, :3, :, :]  # Keep only the first 3 channels if there are more

                print(f"Shape of input tensor before squeeze (Player 2): {img.shape}")

                # Check if squeezing is necessary
                if img.size(0) == 1:
                    img = img.squeeze(0)

                print(f"Shape of input tensor after squeeze (Player 2): {img.shape}")

                # Add this line to check the size of the input tensor before detection (Player 2)
                print(f"Shape of input tensor before detection (Player 2): {img.shape}")

                pred_boxes = self.model.detect(img, max_pool_ks=7, min_score=MIN_SCORE, max_det=MAX_DET)

                print(f"Shape of pred_boxes tensor (Player 2): {pred_boxes.shape}")

        except Exception as e:
            print(f"Error during detection (Player 2): {e}")
            pred_boxes = None



        front_raw = np.array(player_info['kart']['front'])
        loc_raw = np.array(player_info['kart']['location'])

        front = torch.tensor(np.float32(front_raw)[[0, 2]])
        loc = torch.tensor(np.float32(loc_raw)[[0, 2]])

        puck_found = len(pred_boxes) > 0
        if puck_found:
            
            puck_loc = torch.mean(torch.tensor([cx[1] for cx in pred_boxes], dtype=torch.float32)) / 64 - 1


            if self.use_puck2 and torch.abs(puck_loc - self.puck_prev2) > MAX_DEV:
                puck_loc = self.puck_prev2
                self.use_puck2 = False
            else:
                self.use_puck2 = True

            self.puck_prev2 = puck_loc
            self.last_seen2 = self.step
        elif self.step - self.last_seen2 < LAST_PUCK_DURATION:
            self.use_puck2 = False
            puck_loc = self.puck_prev2
        else:
            puck_loc = None
            self.recover_steps2 = LOST_STATUS_STEPS

        dir = front - loc
        dir = dir / torch.norm(dir)

        # calculate angle and distance to own goal for player 2
        dist_own_goal, signed_own_goal_deg = self.calculate_goal_parameters(self.team + 1, loc, dir)

        # calculate angle and distance to opp goal for player 2
        goal_dir = torch.tensor(GOALS[self.team]) - loc
        goal_dist = torch.norm(goal_dir)
        goal_dir = goal_dir / torch.norm(goal_dir)

        goal_angle = torch.acos(torch.clamp(torch.dot(dir, goal_dir), -1, 1))
        signed_goal_angle = torch.degrees(-torch.sign(torch.cross(dir.numpy(), goal_dir.numpy())) * goal_angle)

        goal_dist = ((torch.clamp(goal_dist, 10, 100) - 10) / 90) + 1

        if self.recover_steps2 == 0 and (self.cooldown2 == 0 or puck_found):
            if MIN_ANGLE < torch.abs(signed_goal_angle) < MAX_ANGLE:
                distW = 1 / goal_dist ** 3
                aim_point = puck_loc + torch.sign(puck_loc - signed_goal_angle / TURN_CONE) * 0.3 * distW
            else:
                aim_point = puck_loc
            if self.last_seen2 == self.step:
                brake = False
                acceleration = 0.75 if torch.norm(player_info['kart']['velocity']) < TARGET_SPEED else 0
            else:
                acceleration = 0
                brake = False
        elif self.cooldown2 > 0:
            self.cooldown2 -= 1
            brake = False
            acceleration = 0.5
            aim_point = signed_goal_angle / TURN_CONE
        else:
            if dist_own_goal > 10:
                acceleration = 0
                brake = True
                aim_point = signed_own_goal_deg / TURN_CONE
                self.recover_steps2 -= 1
            else:
                self.cooldown2 = LOST_COOLDOWN_STEPS
                self.step_back = 0
                aim_point = signed_goal_angle / TURN_CONE
                acceleration = 0.5
                brake = False

        steer = torch.clamp(aim_point * STEER_YIELD, -1, 1)
        drift = torch.abs(aim_point) > DRIFT_THRESH

        p2 = {
            'steer': signed_goal_angle if self.step < START_STEPS else steer.item(),
            'acceleration': 1 if self.step < START_STEPS else acceleration.item(),
            'brake': brake,
            'drift': drift,
            'nitro': False,
            'rescue': False
        }

        # Return both player actions
        return p1, p2