import cv2
import os

from collections import OrderedDict
from ipywidgets import IntSlider, Label, Button, HBox, Checkbox
from ipycanvas import MultiCanvas, hold_canvas

thickness = 3
y_ratio = 0.5     # percentile of y-position from the top
    
# Input images
img_filename_fmt = 'dataset/images/frame_{:09d}.jpg'
ann_filename = 'dataset/annotation.txt'
ann_dict = OrderedDict()

num_frames = len(os.listdir(os.path.dirname(img_filename_fmt)))

cur_index = 0
height, width = cv2.imread(img_filename_fmt.format(cur_index)).shape[:2]
y_value = int(height * y_ratio)

def set_image():        
    image = cv2.imread(img_filename_fmt.format(cur_index))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
    image[y_value-thickness//2:y_value+thickness//2] = (255, 0, 0)

    canvas[0].clear()
    canvas[0].put_image_data(image, 0, 0)
    canvas[0].flush()

    pos = ann_dict.get(img_filename_fmt.format(cur_index))
    if pos is not None:
        if pos == 'False':
            no_lane_checkbox.value = True
            canvas[1].clear()
            canvas[2].clear()
        else:
            handle_mouse_down(pos[0], pos[1])
            no_lane_checkbox.value = False
    else:
        no_lane_checkbox.value = False

    cur_fname.value = 'Current image: {:s} | '.format(img_filename_fmt.format(cur_index))

def handle_mouse_move(xpos, ypos):
    with hold_canvas():
        canvas[1].clear()  # Clear the old animation step
        canvas[1].fill_style = "yellow"
        canvas[1].fill_circle(xpos, y_value, 5)  # Draw the new frame    

def handle_mouse_down(xpos, ypos):    
    with hold_canvas():
        canvas[2].clear()
        canvas[2].fill_style = "green"
        canvas[2].fill_circle(xpos, y_value, 5)  # Draw the new frame    

    cur_pos.value = "({:d}, {:d}) ".format(xpos, y_value)    
    ann_dict[img_filename_fmt.format(cur_index)] = (xpos, y_value)
    
def handle_slider_change(change):
    global y_value
    y_value = change.new
    set_image()
    canvas[1].clear()
    canvas[2].clear()

def handle_save_button(b):
    with open(ann_filename, 'w') as f:
        for k, v in ann_dict.items():            
            if v == 'False':
                f.write("{:s}\t{:s}\n".format(k, v))
            else:
                f.write("{:s}\t{:d}\t{:d}\n".format(k, v[0], v[1]))

def handle_prev_button(b):
    global cur_index
    cur_index = max(0, cur_index - 1)
    canvas.clear()
    set_image()

def handle_next_button(b):
    global cur_index
    cur_index = min(num_frames - 1, cur_index + 1)
    canvas.clear()
    set_image()

def handle_checkbox_change(change):
    if change.new:
        ann_dict[img_filename_fmt.format(cur_index)] = 'False'
        canvas[1].clear()
        canvas[2].clear()
    else:
        if img_filename_fmt.format(cur_index) in ann_dict:
            del ann_dict[img_filename_fmt.format(cur_index)]
    set_image()

canvas = MultiCanvas(3, width=width, height=height)
cur_fname = Label(value='', disabled=False)
cur_pos = Label(value='', disabled=True)
yslider = IntSlider(description="Y-bar: ", style={'description_width': 'initial'}, value=y_value, min=1, max=height-2, step=1)
prev_btn = Button(description='Prev', icon='arrow-left')
next_btn = Button(description='Next', icon='arrow-right')
save_btn = Button(description='Save labels', icon='check')
no_lane_checkbox = Checkbox(description='No Lane', value=False)

set_image()
canvas.on_mouse_move(handle_mouse_move)
canvas.on_mouse_down(handle_mouse_down)
yslider.observe(handle_slider_change, names='value')
no_lane_checkbox.observe(handle_checkbox_change, names='value')

prev_btn.on_click(handle_prev_button)
next_btn.on_click(handle_next_button)
save_btn.on_click(handle_save_button)

display(canvas, HBox([cur_fname, cur_pos, yslider]), HBox([prev_btn, next_btn, save_btn, no_lane_checkbox]))