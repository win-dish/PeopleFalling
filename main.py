import matplotlib.pyplot as plt
import torch
import cv2
import math
from torchvision import transforms
import numpy as np
import os
import logging

from tqdm import tqdm
import json
from ultralytics import YOLO

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def fall_detection(poses):
    for pose in poses:
        if pose.boxes and pose.keypoints:
            xmin = pose.boxes.xyxy[0][0]
            xmax = pose.boxes.xyxy[0][2]
            ymin = pose.boxes.xyxy[0][1]
            ymax = pose.boxes.xyxy[0][3]
            left_shoulder_y = pose.keypoints.xy[0][5][1]
            left_shoulder_x = pose.keypoints.xy[0][5][0]
            right_shoulder_y = pose.keypoints.xy[0][6][1]
            left_body_y = pose.keypoints.xy.data[0][11][1]
            left_body_x = pose.keypoints.xy.data[0][11][0]
            right_body_y = pose.keypoints.xy.data[0][12][1]
            len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
            left_foot_y = pose.keypoints.xy.data[0][15][1]
            right_foot_y = pose.keypoints.xy.data[0][16][1]
            dx = pose.boxes.xywh[0][2]
            dy = pose.boxes.xywh[0][3]
            difference = dy - dx
            if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                    len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                    right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                    len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                    or difference < 0:
                return True, (xmin, ymin, xmax, ymax)
    return False, None


def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)


def get_pose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print("device: ", device)
    model = YOLO('yolo11l-pose.pt')
    #if torch.cuda.is_available():
    #    model = model.half().to(device)
    return model, device


def get_pose(image, model, device):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output = model(image)
    #output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['kpt_shape'][0],
    #                                 kpt_label=True)
    #with torch.no_grad():
        output = output_to_keypoint(output)
    return image, output


def prepare_image(image):
    _image = image[0].permute(1, 2, 0) * 255
    _image = _image.cpu().numpy().astype(np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    return _image


def prepare_vid_out(video_path, vid_cap):
    vid_write_image = letterbox(vid_cap.read()[1], 960, stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{video_path.split('/')[-1].split('.')[0]}_keypoint.mp4"
    out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))
    return out

def write_data_to_json(fall_start, fall_end, video, file):
    event_info = f"Fall event detected from {fall_start:.2f} sec to {fall_end:.2f} sec"

    # Load existing data or initialize empty dict if file doesn't exist
    try:
        with open(file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    # Add or update entry with filename as key
    data[video] = event_info

    # Write updated data back to the JSON file
    with open(file, "w") as f:
        json.dump(data, f, indent=4)


def process_video(video_path):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    vid_cap = cv2.VideoCapture(video_path)

    if not vid_cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return

    # Get FPS from the video to compute timestamps
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Fallback to 30 if FPS is not available

    model, device = get_pose_model()
    vid_out = prepare_vid_out(video_path, vid_cap)

    # Read all frames into a list
    success, frame = vid_cap.read()
    _frames = []
    while success:
        _frames.append(frame)
        success, frame = vid_cap.read()

    # Variables to track fall events
    fall_event_active = False
    fall_event_start = 0
    fall_event_end = 0
    fall_events = []  # List to store (start, end) times for fall events

    logger.info("processing video %s", video_path)
    # Process each frame with its index (to compute timestamp)
    for frame_idx, image in enumerate(tqdm(_frames)):
        current_time = frame_idx / fps  # Timestamp in seconds
        image_tensor, output = get_pose(image, model, device)
        _image = prepare_image(image_tensor)
        is_fall, bbox = fall_detection(output)
        
        if is_fall:
            # Start a new fall event if one isn't active already.
            if not fall_event_active:
                fall_event_active = True
                fall_event_start = current_time
            fall_event_end = current_time  # Update the end time continuously
            falling_alarm(_image, bbox)
        else:
            # If we were in a fall event, check its duration now that it has ended.
            if fall_event_active:
                duration = fall_event_end - fall_event_start
                if duration >= 0.2:
                    fall_events.append((fall_event_start, fall_event_end))
                    write_data_to_json(fall_start=fall_event_start, fall_end=fall_event_end, video = video_path, file="detected_falls.json")
                # Reset the fall event flag
                fall_event_active = False

        vid_out.write(_image)

    # In case the video ends while a fall event is active:
    if fall_event_active:
        duration = fall_event_end - fall_event_start
        if duration >= 0.2:
            fall_events.append((fall_event_start, fall_event_end))
            write_data_to_json(fall_start=fall_event_start, fall_end=fall_event_end, video = video_path, file="detected_falls.json")

    vid_out.release()
    vid_cap.release()



if __name__ == '__main__':
    videos_path = 'fall_dataset/custom'
    for video in os.listdir(videos_path):
        video_path = os.path.join(videos_path, video)
        process_video(video_path)
