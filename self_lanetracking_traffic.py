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
from ultralytics import YOLO

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
#set the  values for throttle, steering and control 
car.steering_gain = -1.0
car.steering_offset = 0.2                                                  #do not change
car.throttle_gain = 0.5
#throttle_range = (-0.5, 0.6)
steering_range = (-1.0+car.steering_offset, 1.0+car.steering_offset)

car.throttle = 0.0
car.steering = 0.0

device = torch.device('cuda')
lane_model = get_lane_model()
lane_model.load_state_dict(torch.load('road_following_model_alexnet.pth'))
lane_model = lane_model.to(device)

traffic_model = YOLO('./yolo_best.pt')
intersection_model = YOLO('./best_intersection.pt')

'''
traffic_model
0: bus
1: crosswalk
2: left
3: right
4: straight

intersection
0: intersection
1: not intersection
'''

camera = CSICamera(capture_width=1280, capture_height=720, downsample = 2, capture_fps=30)

# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()

car.steering_gain = -1.0
car.throttle_gain = 0.3
#PID controller Gain
Kp = 2.7  
Kd = 0.1
Ki = 0.3
turn_threshold = 0.7
integral_threshold = 0.2
integral_range = (-0.4/Ki, 0.4/Ki)                                           #integral threshold for saturation prevention

execution = True
running = False
crosswalk_stop = False
intersection = False

#initializing values
now = time.time()
stopTime = now
intersectionTime = now
previous_err = 0.0
integral = 0.0

print("Ready...")
#execute
try:
    while execution:
        pygame.event.pump()
        
        if joystick.get_button(11): #for shutoff: press start button
            print("terminated'")
            execution = False
        #lane detection
        frame = camera.read()
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_coverted)
        pil_width = pil_image.width
        pil_height = pil_image.height
        with torch.no_grad():
            image = preprocess(image=pil_image)
            output = lane_model(image).detach().cpu().numpy()
        err, y = output[0]
        traffic_result = traffic_model(frame)
        intersection_result = intersection_model(frame)

        #apply Kalman Filter
        #x, P = kalman_filter(err,x,P)
        #err = x
        time_interval = time.time()-now
        now = time.time()
        #reset bool flag if sufficient time has passed since last crosswalk detection
        if now-stopTime > .0: crosswalk_stop = False
        
        #Anti-windup
        if abs(err)> 0.8: integral = 0                                          #prevent output saturation
        elif previous_err * err< 0: integral = 0                                #zero-crossing reset
        else:
            integral += err * time_interval
            integral = max(integral_range[0], min(integral_range[1], integral)) #prevent integral saturation
        #steering = float(Kp*err)
        steering = float(Kp*err+Kd*(err-previous_err)/time_interval + Ki*integral)
        steering = max(steering_range[0], min(steering_range[1], steering))
        previous_err = err

        if len(traffic_result[0].boxes.data) != 0:
            traffic_sign = traffic_result[0].boxes.data[0][5]
            # print(traffic_result[0].boxes.data[0])
        else: traffic_sign = 5

        if len(intersection_result[0].boxes.data) != 0:
            is_intersection = intersection_result[0].boxes.data[0][5]

        if traffic_sign == 1 and not crosswalk_stop:                            #crosswalk detected
            print("crosswalk")
            crosswalk_stop = True
            stopTime = now
            while time.time()-stopTime>1.0:                                     #stop for 1 sec
                car.throttle = 0.0
            stopTime = now
        
        elif traffic_sign == 0:                                                 #bus lane detected
            print("bus")
            throttle = 0.632                                                    #slow down

        #within intersection
        elif is_intersection == 0 and not intersection:
            print("intersection")
            intersectionTime = now
            intersection = True
            if traffic_sign ==2:                                                #left
                print("left")
                while(time.time()-intersectionTime> 2.0):
                    steering = -1.0+car.steering_offset
                    throttle = 0.632
            elif traffic_sign ==3:                                              #right
                print("right")
                while(time.time()-intersectionTime> 2.0):
                    steering = 1.0+car.steering_offset
                    throttle = 0.632
            elif traffic_sign ==4:                                              #forward
                print("forward")
                while(time.time()-intersectionTime>2.0):
                    steering = 0.0
                    throttle = 0.635
            else:
                print("HELP")
                while(time.time()-intersectionTime>2.0):
                    steering = 0.0
                    throttle = 0.635

        # Throttle control
        if len(traffic_result[0].boxes.data) != 0:
            traffic_sign = traffic_result[0].boxes.data[0][5]
            print(traffic_result[0].boxes.data[0])
            
            # if traffic_sign == 1:
            #     car.throttle = 0.0
            #     print("STOP")
            #     pygame.event.pump()
                
            #     for i in range(500000):
            #         print(i)
                
            #     car.throttle = 0.635
            #     pygame.event.pump()
            #     for i in range(500000):
            #         print(i)
            traffic_sign = traffic_result[0].boxes.data[0][5]
            #print(traffic_result[0].boxes.data[0][5])
            # if isinstance(traffic_sign, int):                                   # Check if it's an integer
            #     print(traffic_model.names[traffic_sign])
        
            if traffic_sign == 1 and not crosswalk_stop:
                print("crosswalk")
                crosswalk_stop = True
                stopTime = now
                while time.time()-stopTime>1.0:
                    car.throttle = 0.0

            elif traffic_sign == 0: #bus lane detected
                print("bus")
                throttle = 0.632  #slow down
                
            elif traffic_sign == 2:
                if len(intersection_result[0].boxes.data) != 0:
                    is_intersection = intersection_result[0].boxes.data[0][5]
                if is_intersection == 0:
                    print("intersection")
                
            else:
                throttle = 0.635
        
        else:
            throttle = 0.635
        
        #only for troubleshooting - disable for actual test
        # print(round(now),": ",steering)
        
        #temporary
        car.steering = steering
        #car.throttle = throttle
        #print(car.throttle)
    
finally:
    camera.release()
    print("terminated")
    car.throttle = 0.0
    car.steering = 0.0