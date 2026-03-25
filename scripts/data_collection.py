"""Dataset collection script using Intel RealSense RGB & Depth camera. 
Captures color and depth frames at set time intervals (sec) and stores them in a 
directory: name of directory can be changed by editting global variable 'SAVE_AS_FOLDER'. 
Each run of the script saves the depth scale and saves depth data as raw `.npy` 
arrays for future processing."""
# # IMPORTS
import cv2
import numpy as np
import os
import pyrealsense2 as rs
import time
time.sleep(2)
# # GLOBAL VARIABLES
# Name of folder data will be saved to (str)
SAVE_AS_FOLDER = ""
# How many data points do you want to collect (int)
DATASET_SIZE = 00

# Interval (in seconds) between each data point capture (number)
INTERVAL_SEC = 0.5


def get_folder_nr():
    """Returns and keeps record of the number of folders"""
    try:
        with open("dataset_number.txt", "r") as file:
            current_value = int(file.read().strip())
    except FileNotFoundError:
        current_value = 0
        with open("dataset_number.txt", "w") as file:
            file.write(str(current_value))
    else:
        current_value += 1
        with open("dataset_number.txt", "w") as file:
            file.write(str(current_value))
    return current_value


def main():
    global DATASET_SIZE
    global INTERVAL_SEC
    global SAVE_AS_FOLDER

    foldername = f"{get_folder_nr()}_{SAVE_AS_FOLDER}"
    os.makedirs(foldername, exist_ok=True)

    # Start pipline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    
    # Get and save depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_scale_dir = f"{foldername}/depth_scale.txt"
    
    with open(depth_scale_dir, "w") as file:
        file.write(str(depth_scale))
    

    for data_number in range(0,DATASET_SIZE):
        rgb_dir = f"{foldername}/rgb_{data_number}.jpg"
        depth_dir = f"{foldername}/depth_{data_number}.npy"
        

        time.sleep(INTERVAL_SEC)
        # Get frames
        frames = pipeline.wait_for_frames()
        rgb_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Save color image
        rgb_image = np.asanyarray(rgb_frame.get_data())
        cv2.imwrite(rgb_dir, rgb_image)

        # Save depth data
        depth_image = np.asanyarray(depth_frame.get_data())
        np.save(depth_dir, depth_image)

    # Stop pipeline
    pipeline.stop()


if __name__ == "__main__":
    main()
