#from __future__ import annotations

import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import torch
import torchvision
import cv2
import time
import os
import PIL.Image
from cnn.center_dataset import TEST_TRANSFORMS
from jetcam.csi_camera import CSICamera

A = 1
H = 1
Q = 1
R = 4
x = 0
P = 6

def kalman_filter(z_meas, x_esti, P):
    """Kalman Filter Algorithm for One Variable."""
    # (1) Prediction.
    x_pred = A * x_esti
    P_pred = A * P * A + Q

    # (2) Kalman Gain.
    K = P_pred * H / (H * P_pred * H + R)

    # (3) Estimation.
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # (4) Error Covariance.
    P = P_pred - K * H * P_pred

    return x_esti, P

def get_lane_model():
        lane_model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
        return lane_model

def preprocess(image: PIL.Image):
        device = torch.device('cuda')    
        image = TEST_TRANSFORMS(image).to(device)
        return image[None, ...]

car = NvidiaRacecar()
car.steering_gain = -1.0
car.steering_offset = 0.2                                                  #do not change
car.throttle_gain = 0.5
#throttle_range = (-0.5, 0.6)
steering_range = (-1.0+car.steering_offset, 1.0+car.steering_offset)
throttle_range = (-0.65,0.65)
car.throttle = 0.0
car.steering = 0.0

device = torch.device('cuda')
lane_model = get_lane_model()
lane_model.load_state_dict(torch.load('road_following_model_alexnet.pth'))
lane_model = lane_model.to(device)

camera = CSICamera(capture_width=1280, capture_height=720, downsample = 2, capture_fps=30)

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()
#set the  values for throttle, steering and control 
Kp = 2.5  
Kd = 0.0
Ki = 0.2
turn_threshold = 0.7
integral_threshold = 0.2
integral_range = (-0.4/Ki, 0.4/Ki)                                          #integral threshold for saturation prevention
cruise_speed = 0.40
slow_speed = 0.35

execution = True

#initializing values
now = time.time()
previous_err = 0.0
integral = 0.0

print("Ready...")
#execute
try:
    while execution:
        if joystick.get_button(11): #for shutoff: press start button
            print("terminated'")
            execution = False
        image = camera.read()
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_coverted)
        pil_width = pil_image.width
        pil_height = pil_image.height
        with torch.no_grad():
            image = preprocess(image=pil_image)
            output = lane_model(image).detach().cpu().numpy()
        err, y = output[0]
        #apply Kalman Filter
        #x, P = kalman_filter(err,x,P)
        #err = x
        time_interval = time.time()-now
        now = time.time()
        
        #Anti-windup
        if abs(err)> integral_threshold: integral = 0                            #prevent output saturation
        elif previous_err * err< 0: integral = 0                                 #zero-crossing reset
        else:
            integral += err * time_interval
            integral = max(integral_range[0], min(integral_range[1], integral)) #prevent integral saturation
        steering = float(Kp*err+Kd*(err-previous_err)/time_interval + Ki*integral)
        #steering = float(Kp*err)
        steering = max(steering_range[0], min(steering_range[1], steering))
        previous_err = err
        #throttle = -joystick.get_axis(1)
        #throttle = max(throttle_range[0], min(throttle_range[1], throttle))
        if abs(steering)<turn_threshold: throttle = cruise_speed
        else: throttle = slow_speed
        #only for troubleshooting - disable for actual test
        print(round(now),": ",steering)
        car.steering = steering
        car.throttle = throttle
        pygame.event.pump()

finally:
    camera.release()
    print("terminated")
    car.throttle = 0.0
    car.steering = 0.0

camera.release()
print("terminated")
car.throttle = 0.0
car.steering = 0.0