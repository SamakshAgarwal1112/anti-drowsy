"""
Face detection module using OpenCV DNN face detector
"""

import cv2
import numpy as np
import os

class FaceDetector:
    """
    Class to detect faces in images using OpenCV's DNN module with a 
    pre-trained Caffe model for face detection.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the face detector
        
        Args:
            confidence_threshold (float): Minimum confidence to consider a detection valid
        """
        self.confidence_threshold = confidence_threshold
        
        # Load the model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        proto_path = os.path.join(current_dir, "..", "models", "deploy.prototxt")
        model_path = os.path.join(current_dir, "..", "models", "res10_300x300_ssd_iter_140000.caffemodel")
        
        # If model files don't exist, download them
        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            self._download_model(proto_path, model_path)
        
        # Load face detection model
        self.face_net = cv2.dnn.readNet(proto_path, model_path)
        
    def _download_model(self, proto_path, model_path):
        """Download face detection model files if they don't exist"""
        os.makedirs(os.path.dirname(proto_path), exist_ok=True)
        
        # Download prototxt
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        os.system(f"wget {proto_url} -O {proto_path}")
        
        # Download model
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        os.system(f"wget {model_url} -O {model_path}")
    
    def detect(self, frame):
        """
        Detect faces in a frame
        
        Args:
            frame (numpy.ndarray): Input image
            
        Returns:
            list: List of detected face bounding boxes [x, y, width, height]
        """
        # Get frame dimensions
        (h, w) = frame.shape[:2]
        
        # Create a blob from the frame
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        
        # Pass the blob through the network
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Process detections
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence < self.confidence_threshold:
                continue
            
            # Compute bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Ensure the bounding box falls within the frame dimensions
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Extract face region
            face = [startX, startY, endX - startX, endY - startY]
            faces.append(face)
            
            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Display confidence
            text = f"{confidence * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        return faces