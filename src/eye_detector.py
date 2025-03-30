"""
Eye detection and analysis module using dlib facial landmarks
"""

import dlib
import numpy as np
import cv2
from scipy.spatial import distance as dist

class EyeDetector:
    """
    Class to detect eyes and calculate eye aspect ratio (EAR) using facial landmarks
    """
    
    def __init__(self, landmarks_model):
        """
        Initialize the eye detector
        
        Args:
            landmarks_model (str): Path to the facial landmarks predictor model
        """
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmarks_model)
        
        # Define indices of facial landmarks for the left and right eyes
        # Based on the 68-point facial landmark detector
        self.LEFT_EYE_INDICES = [36, 37, 38, 39, 40, 41]
        self.RIGHT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
    
    def detect(self, frame, face_rect):
        """
        Detect facial landmarks
        
        Args:
            frame (numpy.ndarray): Input image
            face_rect (list): Face bounding box [x, y, width, height]
            
        Returns:
            numpy.ndarray: Detected facial landmarks
        """
        # Convert OpenCV rectangle to dlib rectangle
        x, y, w, h = face_rect
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Get facial landmarks
        shape = self.predictor(frame, dlib_rect)
        
        # Convert landmarks to numpy array
        landmarks = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        
        return landmarks
    
    def calculate_eye_aspect_ratio(self, landmarks):
        """
        Calculate the eye aspect ratio (EAR) for both eyes
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        where p1,...,p6 are the coordinates of the eye landmarks
        
        Args:
            landmarks (numpy.ndarray): Facial landmarks
            
        Returns:
            tuple: Left and right eye aspect ratios
        """
        # Get coordinates of left eye landmarks
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        
        # Get coordinates of right eye landmarks
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        # Calculate eye aspect ratio for left eye
        left_ear = self._calculate_ear(left_eye)
        
        # Calculate eye aspect ratio for right eye
        right_ear = self._calculate_ear(right_eye)
        
        return left_ear, right_ear
    
    def _calculate_ear(self, eye):
        """
        Calculate the eye aspect ratio for a single eye
        
        Args:
            eye (numpy.ndarray): Eye landmarks
            
        Returns:
            float: Eye aspect ratio
        """
        # Calculate euclidean distance between horizontal eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Calculate euclidean distance between vertical eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        
        # Calculate eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def draw_eyes(self, frame, landmarks):
        """
        Draw eye contours and pupil detection on the frame
        
        Args:
            frame (numpy.ndarray): Input image
            landmarks (numpy.ndarray): Facial landmarks
            
        Returns:
            None (modifies frame in-place)
        """
        # Get coordinates of left eye landmarks
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        
        # Get coordinates of right eye landmarks
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        # Draw left eye contour
        left_eye_hull = cv2.convexHull(left_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 255), 1)
        
        # Draw right eye contour
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 255), 1)