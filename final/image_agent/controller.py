import pystk
import numpy as np


def control(aim_point, current_vel):

    action = pystk.Action()

    #Steering and relative aim point use different units. Use the aim point 
    #and a tuned scaling factor to select the amount of normalized steering.
    
    steer_dir = aim_point[0]

    action.steer = steer_dir  
    action.brake = False
    action.nitro = True
    action.acceleration = 1
    
    if aim_point[1] < 0:
      t=1
      action.nitro = True
      #print ("Facing Front")

    if aim_point[1] > 0:
      t=1
      action.acceleration = 0
      #print ("LOOK BEHIND YOU")

    if abs(steer_dir)<.2:
      action.nitro = True         #NITROOOOOOOO
      #print("NITRO")

    #if abs(aim_point[0])<.2:
     # action.acceleration = 100

    #print (current_vel)
    if abs(steer_dir)<=.5:
      action.drift = False
    
    #comment 
    
    direction_steer = np.sign(aim_point[0])

    

    #print (aim_point[0])

    if steer_dir > 1:
      steer_dir = 1

    if steer_dir < -1:
      steer_dir = -1
    

    if abs(steer_dir)>.9:
      action.steer = (abs(steer_dir)*direction_steer)
      action.acceleration = 0
      action.acceleration = False
      #action.nitro = False

    if (abs(steer_dir)>.45) and (abs(aim_point[0])<=.7):
      action.drift = True
      action.acceleration = 0.1  #.1

    if (abs(steer_dir)>.7):
      #print (f'SLIGHT tight curve ahead, speed is {current_vel}, steering at {aim_point[0]}, acceleration {action.acceleration}, brake {action.brake}')
      action.drift = True
      action.acceleration = 0.00
      #action.nitro = False 
      
   
    if (abs(steer_dir)>.7) and current_vel > 15:
      action.brake = True 
      action.nitro = False
      #print (f'tight curve ahead, speed is {current_vel}, steering at {aim_point[0]}, acceleration {action.acceleration}, brake {action.brake}')

    if current_vel > 23:    #24
      action.brake = True
      #action.nitro = False

    
    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
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