import pystk

def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()

    # Fine-tune the target velocity for each track
    # You can adjust this value based on the specific track
    target_velocity = 20

    # Calculate the steering angle based on the aim point
    steering_angle = aim_point[0]  # Use the x-coordinate of the aim point

    # Adjust the steering angle based on current velocity and target velocity
    # You may need to experiment with this to maintain a constant velocity
    max_steering_angle = 1.0  # Maximum allowed steering angle
    min_steering_angle = -1.0  # Minimum allowed steering angle
    steering_gain = 0.5  # Adjust this gain to control steering sensitivity
    steering_angle = max(min_steering_angle, min(max_steering_angle, steering_gain * (steering_angle - current_vel / target_velocity)))

    # Use brake if the kart is going too fast or needs to slow down
    brake = current_vel > target_velocity

    # Use drift for wide turns
    drift = abs(steering_angle) > 0.5

    # Set the values in the Action object
    action.acceleration = 0.0  # Adjust acceleration as needed
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
            print(f"Track: {t}, Steps: {steps}, Distance: {how_far}")
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+', help='List of tracks to test the controller')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose mode')
    args = parser.parse_args()
    test_controller(args)
