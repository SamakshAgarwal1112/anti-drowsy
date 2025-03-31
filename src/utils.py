"""
Utility functions for driver drowsiness detection system
"""

import time
import cv2

class FPS:
    """
    Class to calculate frames per second
    """
    
    def __init__(self):
        """Initialize the FPS counter"""
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
    
    def start(self):
        """Start counting a new frame"""
        if self.start_time is None:
            self.start_time = time.time()
        self.frame_count += 1
    
    def update(self):
        """Update FPS calculation"""
        if self.frame_count >= 10:
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps


def draw_status(frame, drowsiness_level, ear_value):
    """
    Draw drowsiness status on frame
    
    Args:
        frame (numpy.ndarray): Input image
        drowsiness_level (str): Current drowsiness level - "AWAKE", "NORMAL", or "EXTREME"
        ear_value (float): Current eye aspect ratio value
    
    Returns:
        None (modifies frame in-place)
    """
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Set status bar position and size
    status_bar_height = 60
    status_bar_y = h - status_bar_height
    
    # Set status bar color based on drowsiness level
    if drowsiness_level == "AWAKE":
        color = (0, 255, 0)  # Green
        status_text = "AWAKE"
    elif drowsiness_level == "NORMAL":
        color = (0, 165, 255)  # Orange
        status_text = "DROWSY - Warning"
    else:  # EXTREME
        color = (0, 0, 255)  # Red
        status_text = "EXTREMELY DROWSY - DANGER"
    
    # Draw status bar background
    cv2.rectangle(frame, (0, status_bar_y), (w, h), (0, 0, 0), -1)
    
    # Draw colored status indicator
    indicator_width = 60
    cv2.rectangle(frame, (10, status_bar_y + 10), 
                 (10 + indicator_width, status_bar_y + status_bar_height - 10), 
                 color, -1)
    
    # Draw status text
    cv2.putText(frame, status_text, (10 + indicator_width + 10, status_bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw EAR value
    ear_text = f"EAR: {ear_value:.2f}"
    ear_text_size = cv2.getTextSize(ear_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    ear_text_x = w - ear_text_size[0] - 10
    cv2.putText(frame, ear_text, (ear_text_x, status_bar_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def create_roi(frame, rect):
    """
    Create a region of interest (ROI) from a frame
    
    Args:
        frame (numpy.ndarray): Input image
        rect (list): Rectangle coordinates [x, y, width, height]
    
    Returns:
        numpy.ndarray: ROI image
    """
    x, y, w, h = rect
    return frame[y:y+h, x:x+w]


def display_eye_tracking_data(frame, left_ear, right_ear, avg_ear, eye_thresh):
    """
    Display eye tracking data on frame
    
    Args:
        frame (numpy.ndarray): Input image
        left_ear (float): Left eye aspect ratio
        right_ear (float): Right eye aspect ratio
        avg_ear (float): Average eye aspect ratio
        eye_thresh (float): Threshold for eye aspect ratio
    
    Returns:
        None (modifies frame in-place)
    """
    # Set metrics position
    y_pos = 100
    left_x = 10
    right_x = frame.shape[1] - 170
    
    # Draw background rectangle for metrics
    cv2.rectangle(frame, (left_x - 5, y_pos - 25), 
                 (left_x + 155, y_pos + 65), (0, 0, 0, 0.7), -1)
    cv2.rectangle(frame, (right_x - 5, y_pos - 25), 
                 (right_x + 155, y_pos + 65), (0, 0, 0, 0.7), -1)
    
    # Draw metrics
    cv2.putText(frame, "Left Eye EAR:", (left_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{left_ear:.2f}", (left_x + 115, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, "Right Eye EAR:", (left_x, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{right_ear:.2f}", (left_x + 115, y_pos + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, "Avg. EAR:", (left_x, y_pos + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Color code the average EAR value based on threshold
    if avg_ear < eye_thresh:
        ear_color = (0, 0, 255)  # Red if below threshold
    else:
        ear_color = (0, 255, 0)  # Green if above threshold
    
    cv2.putText(frame, f"{avg_ear:.2f}", (left_x + 115, y_pos + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
    
    # Draw threshold value
    cv2.putText(frame, "Threshold:", (right_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{eye_thresh:.2f}", (right_x + 90, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def annotate_frame(frame, text, position=(10, 30), font_scale=0.7, color=(0, 255, 0), thickness=2):
    """
    Add text annotation to frame
    
    Args:
        frame (numpy.ndarray): Input image
        text (str): Text to display
        position (tuple): Position (x, y) of text
        font_scale (float): Font scale
        color (tuple): RGB color of text
        thickness (int): Thickness of text
    
    Returns:
        None (modifies frame in-place)
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)