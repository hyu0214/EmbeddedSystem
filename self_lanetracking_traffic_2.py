import os
import pygame
from jetracer.nvidia_racecar import NvidiaRacecar
import torch
import torchvision
import cv2
import time
import os
import PIL.Image
import logging
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

logging.getLogger('ultralytics').setLevel(logging.WARNING)

traffic_model = YOLO('./yolo_best.pt',verbose=False)
intersection_model = YOLO('./best_intersection.pt',verbose=False)

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
cruiseSpeed = 0.47
slowSpeed = 0.46
#PID controller Gain
Kp = 2.0
Kd = 0.1
Ki = 0.3
turn_threshold = 0.7
integral_threshold = 0.2
integral_range = (-0.2/Ki, 0.2/Ki)                                           #integral threshold for saturation prevention

#initializing values
execution = True
running = False
crosswalk_flag = False
intersection_flag = False
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
        traffic_result = traffic_model(source=frame, conf=0.6)
        intersection_result = intersection_model(source=frame, conf=0.6)

        #apply Kalman Filter
        #x, P = kalman_filter(err,x,P)
        #err = x
        time_interval = time.time()-now
        now = time.time()

        #reset bool flag if sufficient time has passed since last detection
        if now-stopTime > 5.0: crosswalk_flag = False
        intersection_flag = False
        
        #Anti-windup
        if abs(err)> 0.8: integral = 0                                              #prevent output saturation
        elif previous_err * err< 0: integral = 0                                    #zero-crossing reset
        else:
            integral += err * time_interval
            integral = max(integral_range[0], min(integral_range[1], integral))     #prevent integral saturation
        #steering = float(Kp*err)
        steering = float(Kp*err+Kd*(err-previous_err)/time_interval + Ki*integral)
        steering = max(steering_range[0], min(steering_range[1], steering))
        previous_err = err

        if len(traffic_result[0].boxes.data) != 0:                                  #if traffic sign detected
            traffic_sign = traffic_result[0].boxes.data[0][5]
        else: traffic_sign = 5

        if len(intersection_result[0].boxes.data) != 0:                             #if intersection detected
            is_intersection = intersection_result[0].boxes.data[0][5]
        else: is_intersection = 1
        
        #within intersection
        if is_intersection == 0 and not intersection_flag:
            intersectionTime = now
            intersection_flag = True
            if traffic_sign ==2:                                                    #left
                print("intersection-left")
                while(time.time()-intersectionTime< 2.5):
                    steering = -1.0+car.steering_offset
                    throttle = slowSpeed
                print("exited intersection")
            elif traffic_sign ==3:                                                  #right
                print("intersection-right")
                while(time.time()-intersectionTime< 2.5):
                    steering = 1.0+car.steering_offset
                    throttle = slowSpeed
                print("exited intersection")
            elif traffic_sign ==4:                                                  #forward
                print("intersection-forward")
                while(time.time()-intersectionTime< 1.5):
                    steering = 0.0
                    throttle = cruiseSpeed
                print("exited intersection")
            else:
                print("intersection-HELP")                                          #intersect detected, but no traffic sign
                while(time.time()-intersectionTime< 2.0):
                    steering = 0.0
                    throttle = cruiseSpeed
                print("exited intersection")
        
        elif traffic_sign == 1 and not crosswalk_flag:                              #crosswalk detected
            print(now, ": crosswalk")
            crosswalk_flag = True
            stopTime = now
            while time.time()-stopTime<2.5:                                         #stop for 2.5 sec
                car.throttle = 0.0          
            stopTime = time.time()
            print(stopTime, ": crosswalk passed")
            throttle = cruiseSpeed
        
        elif traffic_sign == 0:                                                     #bus lane detected
            print("bus")
            throttle = slowSpeed                                                    #slow down

        else: throttle = cruiseSpeed
        
        #calculate steering if not within intersection
        if not crosswalk_flag and not intersection_flag:
            if abs(err)> 0.8: integral = 0                                          #prevent output saturation
            elif previous_err * err< 0: integral = 0                                #zero-crossing reset
            else:
                integral += err * time_interval
                integral = max(integral_range[0], min(integral_range[1], integral)) #prevent integral saturation
            #steering = float(Kp*err)
            steering = float(Kp*err+Kd*(err-previous_err)/time_interval + Ki*integral)
            steering = max(steering_range[0], min(steering_range[1], steering))
            previous_err = err
        else:                                                                       #if just escaped crosswalk or intersection
            integral = 0.0
            previous_err = err
            now = time.time()
            steering = 0.0

        #only for troubleshooting - disable for actual test
        #print(round(now),": ",steering,"   ",throttle)
        print(throttle)
        car.steering = steering
        car.throttle = throttle
    
finally:
    camera.release()
    print("terminated")
    car.throttle = 0.0
    car.steering = 0.0