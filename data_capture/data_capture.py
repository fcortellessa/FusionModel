import time
import json
import os
import numpy as np
import cv2
import pyrealsense2 as rs
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from digit_interface import Digit

# --- CONFIGURATION ---
OUTPUT_DIR = "usb_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardware Connection IDs
ROBOT_IP = "192.168.1.100" 
DIGIT_SERIAL = "D12345" # Replace with DIGIT's specific serial number on the back of the camera

# Base fixture position:
# The exact calibrated reference pose [x,y,z,rx,ry,rz] = (0,0,0,0,0,0) of the USB connector in Robot Base Frame
REFERENCE_POSE = [0.4, 0.1, 0.2, 3.14, 0.0, 0.0] #??? ToDo

# Initialize RealSense D405
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

# Initialize DIGIT Tactile Sensor
digit = Digit(DIGIT_SERIAL)
digit.connect()     # Upon connection, the DIGIT defaults to VGA: 640x480 @ 30fps

# Initialize UR5 Manipulator
rtde_c = RTDEControl(ROBOT_IP)
rtde_r = RTDEReceive(ROBOT_IP)

def capture_sensors(grasp_id, frame_id, target_labels, actual_tcp_pose):
    """Synchronously captures data from D405 + DIGIT and outputs individual JSON metadata."""
    
    # 1. Capture RealSense D405 frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    
    if not depth_frame or not color_frame:
        print(f"Warning: Dropped RealSense frame at grasp {grasp_id}, frame {frame_id}")
        return
    
    # 2. Capture DIGIT tactile frame
    digit_frame = digit.get_frame() # Returns a standard OpenCV-compatible image array
    
    # Process RealSense arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    # Define file names
    base_name = f"grasp_{grasp_id:05d}_frame_{frame_id:02d}"
    
    # Save Image/Array Payloads
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_rgb.png"), color_image)
    np.save(os.path.join(OUTPUT_DIR, f"{base_name}_depth.npy"), depth_image)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_digit.png"), digit_frame)
    
    # Build cleanly structured individual JSON dictionary
    # Fabian's metadata structure:
    #{
    #     "timestamp": 81.264519847,
    #     "position": {
    #         "x": 0.45825660566342485,
    #         "y": -0.19091270940739116,
    #         "z": 0.16064320240764116
    #     },
    #     "orientation": {
    #         "x": -0.010598285995331571,
    #         "y": 0.9956966320083175,
    #         "z": 0.09206072090125639,
    #         "w": 0.0008467634826020699
    #     }
    # }
    metadata = {
        "timestamp": time.time(),
        "grasp_id": grasp_id,
        "frame_id": frame_id,
        "ground_truth_labels": {
            "translation_x_m": target_labels['x'],
            "translation_y_m": target_labels['y'],
            "translation_z_m": target_labels['z'],
            "rotation_x_rad": target_labels['rx']
        },
        "robot_actual_tcp": {
            "position": {
                "x": actual_tcp_pose[0],
                "y": actual_tcp_pose[1],
                "z": actual_tcp_pose[2]
            },
            "rotation_vector": {
                "rx": actual_tcp_pose[3],
                "ry": actual_tcp_pose[4],
                "rz": actual_tcp_pose[5]
            }
        }
    }
    
    # Save JSON file atomically
    json_path = os.path.join(OUTPUT_DIR, f"{base_name}_metadata.json")
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

# --- MAIN AUTOMATION LOOP ---
try:
    for grasp_id in range(1000): # Goal: 1000 randomized grasp trials
        print(f"\n--- Starting Grasp Sequence {grasp_id:04d} ---")
        
        # Randomize target planar variations relative to the nominal fixture pose
        gt_y = np.random.uniform(-0.01, 0.01)   # +/- 1 cm translation in Y
        gt_z = np.random.uniform(-0.005, 0.005) # +/- 5 mm translation in Z
        gt_rx = np.random.uniform(-0.2, 0.2)    # +/- ~11.5 deg rotation about X
        
        target_labels = {'y': gt_y, 'z': gt_z, 'rx': gt_rx} #??? ToDo
        
        # Calculate target approach pose
        target_pose = REFERENCE_POSE.copy()
        target_pose[1] += gt_y
        target_pose[2] += gt_z
        target_pose[3] += gt_rx
        
        # Safely approach from above
        approach_pose = target_pose.copy()
        approach_pose[2] += 0.05 # 5cm stand-off height
        rtde_c.moveL(approach_pose, 0.2, 0.2)
        rtde_c.moveL(target_pose, 0.05, 0.05)
        
        print("Gripping USB connector...") 
        time.sleep(1.0) # -> Trigger your specific gripper close command here <-
        
        # Lift slightly vertically to break contact with the fixture base before measuring
        lift_pose = target_pose.copy()
        lift_pose[2] += 0.03 # 3cm lift
        rtde_c.moveL(lift_pose, 0.1, 0.1)
        time.sleep(0.3) # Wait briefly for arm vibrations to dampen
        
        # Frame 00: Initial grasp state 
        current_tcp = rtde_r.getActualTCPPose()
        capture_sensors(grasp_id, frame_id=0, target_labels=target_labels, actual_tcp_pose=current_tcp)
        
        # Frames 01 - 03: Move around to vary backgrounds/lighting environments
        for bg_idx in range(1, 4):
            current_pose = rtde_r.getActualTCPPose()
            
            # Subtle translation/rotation adjustments to sweep varied backdrops
            rand_bg_pose = current_pose.copy()
            rand_bg_pose[0] += np.random.uniform(-0.05, 0.05)
            rand_bg_pose[1] += np.random.uniform(-0.05, 0.05)
            rand_bg_pose[2] += np.random.uniform(-0.02, 0.05)
            
            rtde_c.moveL(rand_bg_pose, 0.2, 0.2)
            time.sleep(0.3) # Dampen physical vibrations
            
            current_tcp = rtde_r.getActualTCPPose()
            capture_sensors(grasp_id, frame_id=bg_idx, target_labels=target_labels, actual_tcp_pose=current_tcp)
            
        # Disengage, return, and release cleanly
        rtde_c.moveL(lift_pose, 0.2, 0.2)
        rtde_c.moveL(target_pose, 0.05, 0.05)
        print("Releasing USB connector...")
        time.sleep(1.0) # -> Trigger your specific gripper open command here <-
        rtde_c.moveL(approach_pose, 0.2, 0.2)

except KeyboardInterrupt:
    print("\nData collection interrupted by user.")

finally:
    # Safe shutdown routine to release hardware handles cleanly
    pipeline.stop()
    digit.disconnect()
    rtde_c.disconnect()
    rtde_r.disconnect()
    print("All hardware interfaces disconnected cleanly.")