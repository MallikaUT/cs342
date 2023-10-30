import pystk


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """


    # Set target velocity (e.g., 20)
    target_velocity = 20

    # Calculate steering weight based on the aim point (turn towards the aim point)
    steer_weight = aim_point[0]

    # Set acceleration based on the difference between current velocity and target velocity
    velocity_error = target_velocity - current_vel
    action.acceleration = max(0, min(1, velocity_error / 10))  # Adjust 10 as needed

    # Adjust steering based on the aim point
    action.steer = steer_weight

    # Enable drift for wide turns (based on aim point)
    drift_weight = 0.5  # You can adjust this value
    if abs(aim_point[0]) > drift_weight:
        action.drift = True

    return action
    """action.steer = aim_point[0] * steer_weight
    action.acceleration = 1 - abs(aim_point[0])
    if (abs(aim_point[0]) > drift_weight):
        action.drift = True
    return action """



if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    
    """def test_controller(args):
        import numpy as np
        pytux = PyTux()
        steps = np.zeros((4, 6))
        how_far = np.zeros((4, 6))
        for t in args.track:
            for i in range(1, 5):
                for j in range(1, 7):
                    steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
        print(steps, how_far)
        pytux.close()
    
    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('track')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)"""

    def test_controller(args):
        pytux = PyTux()
        steps = pytux.rollout(args.track, control, max_frames=1000, verbose=args.verbose)
        print(steps)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
