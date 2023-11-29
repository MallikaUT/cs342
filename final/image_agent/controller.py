import numpy as np

def control(aim_point, player_state):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    current_vel=np.linalg.norm(player_state['kart']['velocity'])
    target_vel=20
    steer_factor=2.21
    drift_thresh=0.65
    brake_factor=0.75
    
    acceleration = target_vel - current_vel
    acceleration = np.clip(acceleration, 0, 1)

    steer = np.clip(aim_point[0][0] * steer_factor, -1, 1)

    brake = steer > brake_factor
    
    drift = abs(steer) > drift_thresh
    
    return acceleration, steer, brake