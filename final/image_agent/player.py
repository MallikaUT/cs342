import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


class FCNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNModel, self).__init__()

        # Define the encoder (downsampling) part of the U-Net
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the decoder (upsampling) part of the U-Net
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # Forward pass through the encoder
        x = self.encoder(x)

        # Forward pass through the decoder
        x = self.decoder(x)

        return x

class Team:
    agent_type = 'image'

    def __init__(self):
        """
          TODO: Load your agent here. Load network parameters, and other parts of our model
          We will call this function with default arguments only
        """
        self.team = None
        self.num_players = None

        self.fcn_model = FCNModel(in_channels=3, out_channels=2)  # Assuming 3 input channels for RGB images and 2 output channels for x, y coordinates
        self.fcn_model.eval()

        self.game_width = 100.0  
        self.game_height = 50.0 
        
    def preprocess_image(self, player_image):
        # Define image preprocessing transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            
        ])

        img = transform(player_image)
        img = img.unsqueeze(0)

        return img
    def locate_puck(self, img):
        # Forward pass through the FCN model
        with torch.no_grad():
            output = self.fcn_model(img)

        # Use spatial_argmax to get the puck location
        puck_location = spatial_argmax(output[:, 0])

        return puck_location

    def screen_to_world_coordinates(self, screen_point):
        """
        Convert screen coordinates to game world coordinates
        :param screen_point: Point in screen coordinate frame [-1..1]
        :return: Point in game world coordinates
        """
        world_x = screen_point[0] * 0.5 * self.game_width
        world_y = screen_point[1] * 0.5 * self.game_height
        return world_x, world_y


    def new_match(self, team: int, num_players: int) -> list:
        """
        Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
        type (name) for each player.
        :param team: What team are you playing on RED=0 or BLUE=1
        :param num_players: How many players are there on your team
        :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
                 'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
                 'wilber', 'xue'. Default: 'tux'
        """
        """
           TODO: feel free to edit or delete any of the code below
        """
        self.team, self.num_players = team, num_players
        return ['tux'] * num_players

    def act(self, player_state, player_image):
        """
        This function is called once per timestep. You're given a list of player_states and images.

        DO NOT CALL any pystk functions here. It will crash your program on your grader.

        :param player_state: list[dict] describing the state of the players of this team. The state closely follows
                             the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
                             See HW5 for some inspiration on how to use the camera information.
                             camera:  Camera info for each player
                               - aspect:     Aspect ratio
                               - fov:        Field of view of the camera
                               - mode:       Most likely NORMAL (0)
                               - projection: float 4x4 projection matrix
                               - view:       float 4x4 view matrix
                             kart:  Information about the kart itself
                               - front:     float3 vector pointing to the front of the kart
                               - location:  float3 location of the kart
                               - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
                               - size:      float3 dimensions of the kart
                               - velocity:  float3 velocity of the kart in 3D

        :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
                             player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
                             from where the image was taken.

        :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
                 acceleration: float 0..1
                 brake:        bool Brake will reverse if you do not accelerate (good for backing up)
                 drift:        bool (optional. unless you want to turn faster)
                 fire:         bool (optional. you can hit the puck with a projectile)
                 nitro:        bool (optional)
                 rescue:       bool (optional. no clue where you will end up though.)
                 steer:        float -1..1 steering angle
        """
        # TODO: Change me. I'm just cruising straight
        aim_point_screen_coordinates = player_state[0]['aim_point']
        aim_point_world_coordinates = self.screen_to_world_coordinates(aim_point_screen_coordinates)

        # Preprocess the image
        img = self.preprocess_image(player_image[0])  # Assuming there's only one player

        # Use the FCN model to locate the puck
        puck_location = self.locate_puck(img)

        # Convert puck location from normalized coordinates to game world coordinates
        puck_location_world_coordinates = self.screen_to_world_coordinates(puck_location[0])

        # TODO: Implement your logic here based on puck_location_world_coordinates and aim_point_world_coordinates
        # Example: Setting actions based on relative positions of aim point and puck location
        actions = []
        for i in range(self.num_players):
            # Placeholder logic: Accelerate if the puck is in front, steer towards the puck
            acceleration = 1.0 if puck_location_world_coordinates[0] > 0 else 0.0
            steer = (puck_location_world_coordinates[0] - aim_point_world_coordinates[0]) * 0.5

            # Ensure steer is within valid range
            steer = max(-1.0, min(1.0, steer))

            actions.append(dict(acceleration=acceleration, steer=steer))

        return actions