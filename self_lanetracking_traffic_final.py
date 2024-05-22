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

# Set a Kalman Filter parameters
A = 1
H = 1
Q = 0.95
R = 2.38
x = 0
P = 2

# Kalman filter
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

# Get the lane model of alexnet
def get_lane_model():
        lane_model = torchvision.models.alexnet(num_classes=2, dropout=0.3)
        return lane_model

# Normalize and to tensor 
def preprocess(image):
        device = torch.device('cuda')    
        image = TEST_TRANSFORMS(image).to(device)
        return image[None, ...]

# set the  values for throttle, steering and control 
car = NvidiaRacecar()
car.steering_gain = -1.0
car.steering_offset = 0.2                                                  #do not change
car.throttle_gain = 0.5
#throttle_range = (-0.5, 0.6)
steering_range = (-1.0+car.steering_offset, 1.0+car.steering_offset)

car.throttle = 0.0
car.steering = 0.0

# Get the lane model
device = torch.device('cuda')
lane_model = get_lane_model()
lane_model.load_state_dict(torch.load('road_following_model_alexnet_best_0.3_lbatch.pth'))
lane_model = lane_model.to(device)

logging.getLogger('ultralytics').setLevel(logging.WARNING)

traffic_model = YOLO('./best (2).pt',verbose=False)
intersection_model = YOLO('./best_intersection.pt',verbose=False)

'''
traffic_model
0: bus
1: crosswalk
2: left
3: right
4: straight

intersection_model
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
car.throttle_gain = 0.50
k = 0

throttle = 0.256
#0.26
#0.248
cruiseSpeed = 0.2481
slowSpeed = cruiseSpeed - 0.005
#PID controller Gain
Kp = 2.50
Kd = 0.1
Ki = 0.3
turn_threshold = 0.7
integral_threshold = 0.4
integral_range = (-0.4/Ki, 0.4/Ki)                                           #integral threshold for saturation prevention

#initializing values
execution = True
running = False
crosswalk_flag = False
intersection_flag = False
bus_flag = False
now = time.time()
stopTime = now
intersectionTime = now - 100.0
busTime = now - 100.0
previous_err = 0.0
integral = 0.0

print("Ready...")
#ii = 400
img_filename_fmt = 'dataset3/frame_{:09d}.jpg'
dirname = os.path.dirname(img_filename_fmt)
os.makedirs(dirname, exist_ok=True)

# execute the code of self-driving
try:
    while execution:
        #os.system('clear')
        pygame.event.pump()
        
        if joystick.get_button(11): #for shutoff: press start button
            print("terminated'")
            execution = False        
        
        # lane detection
        frmae_1 = 0
        frame = camera.read()
        
        #cv2.imwrite(img_filename_fmt.format(ii), frame)
        #ii += 1
        color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PIL.Image.fromarray(color_coverted)
        pil_width = pil_image.width
        pil_height = pil_image.height
        with torch.no_grad():
            image = preprocess(image=pil_image)
            output = lane_model(image).detach().cpu().numpy()
        err, y = output[0]
        traffic_result = traffic_model(source=frame, conf=0.7)
        intersection_result = intersection_model(source=frame, conf=0.7)

        #apply Kalman Filter
        x, P = kalman_filter(err,x,P)
        #err = x
        time_interval = time.time()-now
        now = time.time()

        #reset bool flag if sufficient time has passed since last detection
        if crosswalk_flag or bus_flag or intersection_flag:
            if now-stopTime > 10.0: crosswalk_flag = False
            if now-busTime > 10.0: bus_flag = False
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

        xywh = [0]*4
        
        if len(traffic_result[0].boxes.data) != 0:                                  #if traffic sign detected
            traffic_sign = traffic_result[0].boxes.data[0][5]
            tbox = traffic_result[0].boxes[0]
            xywh = tbox.xywh.cpu().detach().numpy().squeeze().tolist()
            if len(xywh) == 0:
                xywh = [0]*4
            crosswalk_area = xywh[2]*xywh[3]
        else: traffic_sign = 5

        if len(intersection_result[0].boxes.data) != 0:                             #if intersection detected
            is_intersection = intersection_result[0].boxes.data[0][5]
            box = intersection_result[0].boxes[0]
            xywh = box.xywh.cpu().detach().numpy().squeeze().tolist()
            if len(xywh) == 0:
                xywh = [0]*4
            intersection_area = xywh[2]*xywh[3]
            y_bar = xywh[1]
            print(intersection_area,y_bar)
        else: is_intersection = 1
        
        area_min = 1100
        y_bar_min = 130
        # within intersection
        if is_intersection == 0 and not intersection_flag :
            if intersection_area > area_min and y_bar > y_bar_min:
                # intersectionTime = now
                # intersection_flag = True
                if not intersection_flag and (traffic_sign==2 or traffic_sign==3):
                    intersection_flag = True
                    intersectionTime = now
                if traffic_sign ==2:                                                    #left
                    print("intersection-left")
                    while(time.time()-intersectionTime< 1.8):
                        car.steering = -1.0+car.steering_offset
                        car.throttle = cruiseSpeed + 0.013
                    print("exited intersection")
                elif traffic_sign ==3:                                                  #right
                    print("intersection-right")
                    while(time.time()-intersectionTime< 1.8):
                        car.steering = 1.0+car.steering_offset
                        car.throttle = cruiseSpeed + 0.013
                    print("exited intersection")
                elif traffic_sign ==4:                                                  #forward
                    print("intersection-forward")
                    car.steering = float(err)
                    car.throttle = cruiseSpeed
                    print("exited intersection")
                else:
                    print("intersection-HELP")                                          #intersect detected, but no traffic sign
                    car.steering = float(err)
                    car.throttle = cruiseSpeed
                integral = 0.0
                previous_err = err
                now = time.time()
                steering = 0.0
        else:
            #calculate steering if not within intersection
            if abs(err)> integral_threshold: integral = 0                                               #prevent output saturation
            elif previous_err * err< 0: integral = 0                                                    #zero-crossing reset
            else:
                integral += err * time_interval
                integral = max(integral_range[0], min(integral_range[1], integral))                     #prevent integral saturation
            #steering = float(Kp*err)
            steering = float(Kp*err+Kd*(err-previous_err)/time_interval + Ki*integral)
            steering = max(steering_range[0], min(steering_range[1], steering))
            previous_err = err

            if traffic_sign == 1 and not crosswalk_flag:                                                #crosswalk detected
                if crosswalk_area > 1800:
                    print(now, ": crosswalk")
                    crosswalk_flag = True
                    stopTime = now
                    while time.time()-stopTime<2.5:                                                     #stop for 2.5 sec
                        car.throttle = 0.0          
                    stopTime = time.time()
                    print(stopTime, ": crosswalk passed")
                    throttle = cruiseSpeed
                    car.throttle = throttle
                    for _ in range(100):
                        i = 1
            
            elif traffic_sign == 0:                                                                     #bus lane detected
                if not bus_flag:
                    print("bus detected")
                    bus_flag = True
                    busTime = now
                    #cruiseSpeed -= 0.002
                    #slowSpeed -= 0.001

            else:
                if bus_flag and now-busTime < 5.0:
                    print("slowing down")
                    throttle = slowSpeed
                elif abs(steering-0.2)>0.75: throttle = cruiseSpeed + 0.01
                else: throttle = cruiseSpeed

        #only for troubleshooting - disable for actual test
        #print(round(now),": ",steering,"   ",throttle)
        #print("Center of lane is ", err)
        #print(throttle)
        car.steering = steering
        car.throttle = throttle
        
        frame_1 = frame
        #os.system('clear')
    
finally:
    camera.release()
    print("terminated")
    car.throttle = 0.0
    car.steering = 0.0