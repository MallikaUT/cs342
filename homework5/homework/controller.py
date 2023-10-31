import pystk

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    # Target a constant velocity (adjust as needed)
    target_velocity = 20  # You can tune this value
    acceleration = 0.0
    brake = False

    # Calculate the steering angle based on the aim point
    steering_angle = aim_point[0]  # Use the x-coordinate of the aim point
    steering_angle = max(-1, min(1, steering_angle))  # Clip to -1..1

    # Use drift for wide turns
    drift = abs(steering_angle) > 0.5

    # Set the values in the Action object
    action.acceleration = acceleration
    action.brake = brake
    action.steer = steering_angle
    action.drift = drift

    return action

if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
