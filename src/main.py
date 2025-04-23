#!/usr/bin/env python3
"""
Driver Drowsiness Detection System - Main Application
"""

import argparse
import cv2
import time
import yaml
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector
from src.eye_detector import EyeDetector
from src.drowsiness_detector import DrowsinessDetector
from src.audio_alerts import AudioAlerts
from src.utils import FPS, draw_status, display_eye_tracking_data

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--config", type=str, default="../config/config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device ID (overrides config file)")
    parser.add_argument("--gemini-api-key", type=str, default=None,
                        help="Gemini API key (overrides config file)")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def main():
    """Main function to run the drowsiness detection system"""
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Override camera device if specified in command line
    if args.camera is not None:
        config['camera']['device_id'] = args.camera
    
    # Get Gemini API key
    gemini_api_key = args.gemini_api_key or config.get('gemini', {}).get('api_key')
    if not gemini_api_key:
        print("Warning: No Gemini API key provided. Voice analysis will be limited.")
    
    # Initialize components
    face_detector = FaceDetector(
        confidence_threshold=config['detection']['face_confidence']
    )
    
    eye_detector = EyeDetector(
        landmarks_model=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "models/shape_predictor_68_face_landmarks.dat")
    )
    
    # Use more sensitive threshold values for drowsiness detection
    drowsiness_detector = DrowsinessDetector(
        eye_aspect_ratio_threshold=config['detection'].get('eye_aspect_ratio_threshold', 0.3),
        consecutive_frames_threshold=config['detection'].get('consecutive_frames_threshold', 10),
        normal_duration_threshold=config['drowsiness']['normal'].get('duration_threshold', 1.5),
        extreme_duration_threshold=config['drowsiness']['extreme'].get('duration_threshold', 0.8),
        normal_ear_threshold=config['drowsiness']['normal'].get('ear_threshold', 0.3),
        extreme_ear_threshold=config['drowsiness']['extreme'].get('ear_threshold', 0.25)
    )
    
    audio_alerts = AudioAlerts(
        normal_message=config['drowsiness']['normal']['message'],
        extreme_message=config['drowsiness']['extreme']['message'],
        volume=config['alerts']['volume'],
        gemini_api_key=gemini_api_key,
        gemini_api_url=config.get('gemini', {}).get('api_url', 
                                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent")
    )
    
    # Initialize camera
    camera = cv2.VideoCapture(config['camera']['device_id'])
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera']['resolution'][0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera']['resolution'][1])
    camera.set(cv2.CAP_PROP_FPS, config['camera']['fps'])
    
    if not camera.isOpened():
        print(f"Error: Could not open camera {config['camera']['device_id']}")
        sys.exit(1)
    
    # Initialize FPS counter
    fps_counter = FPS()
    
    print("Driver Drowsiness Detection System Started")
    print("Press 'q' to quit")

    # Variables for face detection alerts
    face_detected = False
    face_detection_start_time = time.time()
    last_no_face_alert_time = time.time()
    no_face_alert_interval = config['face_detection']['alert_interval']  # seconds between no-face alerts
    
    # Main loop
    while True:
        # Read frame from camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Start FPS calculation
        fps_counter.start()
        
        # Get current time
        current_time = time.time()
        
        # Detect face in the frame
        faces = face_detector.detect(frame)
        
        # Current drowsiness level (defaults to AWAKE if no face detected)
        current_drowsiness_level = "AWAKE"
        
        # Process each detected face
        if faces:
            face_detected = True
            last_no_face_alert_time = time.time()
            
            for face in faces:
                # Detect eyes landmarks
                landmarks = eye_detector.detect(frame, face)
                
                # Calculate eye aspect ratio
                left_ear, right_ear = eye_detector.calculate_eye_aspect_ratio(landmarks)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Draw eye landmarks
                eye_detector.draw_eyes(frame, landmarks)
                
                # Check for drowsiness
                current_drowsiness_level = drowsiness_detector.detect(avg_ear)
                
                # Draw drowsiness status on frame
                draw_status(frame, current_drowsiness_level, avg_ear)
                
                # Display eye tracking data
                display_eye_tracking_data(frame, left_ear, right_ear, avg_ear, 
                                        drowsiness_detector.eye_aspect_ratio_threshold)
        else:
            face_detected = False
            
            # Check if we should play the no-face alert
            if current_time - last_no_face_alert_time >= no_face_alert_interval:
                # Only play alert if the system has been running for at least x seconds
                # to give time for camera initialization
                if current_time - face_detection_start_time >= no_face_alert_interval:
                    print(f"No face detected for {current_time - last_no_face_alert_time:.1f} seconds")
                    audio_alerts.play_no_face_alert(config['face_detection']['message'])
                    last_no_face_alert_time = current_time
        
        # Update audio alerts based on current drowsiness level
        audio_alerts.update(current_drowsiness_level)
        
        # End FPS calculation
        fps = fps_counter.update()
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display face detection status
        status_text = "Face detected" if face_detected else f"No face detected for {current_time - last_no_face_alert_time:.1f}s"
        
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if face_detected else (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Driver Drowsiness Detection", frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    camera.release()
    cv2.destroyAllWindows()
    audio_alerts.cleanup()
    print("Driver Drowsiness Detection System Stopped")

if __name__ == "__main__":
    main()