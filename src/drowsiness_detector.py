"""
Drowsiness detection module using eye aspect ratio analysis
"""

import time

class DrowsinessDetector:
    """
    Class to detect driver drowsiness based on eye aspect ratio (EAR) analysis
    """
    
    def __init__(self, eye_aspect_ratio_threshold=0.3, consecutive_frames_threshold=10,
                 normal_duration_threshold=1.5, extreme_duration_threshold=0.8,
                 normal_ear_threshold=0.3, extreme_ear_threshold=0.25):
        """
        Initialize drowsiness detector
        
        Args:
            eye_aspect_ratio_threshold (float): Threshold for eye aspect ratio to consider eyes closed
            consecutive_frames_threshold (int): Number of consecutive frames with closed eyes to trigger alert
            normal_duration_threshold (float): Duration in seconds for normal drowsiness level
            extreme_duration_threshold (float): Duration in seconds for extreme drowsiness level
            normal_ear_threshold (float): EAR threshold for normal drowsiness level
            extreme_ear_threshold (float): EAR threshold for extreme drowsiness level
        """
        self.eye_aspect_ratio_threshold = eye_aspect_ratio_threshold
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.normal_duration_threshold = normal_duration_threshold
        self.extreme_duration_threshold = extreme_duration_threshold
        self.normal_ear_threshold = normal_ear_threshold
        self.extreme_ear_threshold = extreme_ear_threshold
        
        # Initialize counters and timers
        self.closed_eyes_frames = 0
        self.drowsy_start_time = None
        self.last_alert_time = 0
        self.current_drowsiness_level = "AWAKE"
        
        # Add eye closure percentage tracking (for gradual detection)
        self.eye_closure_history = []
        self.history_size = 30  # Track last 30 frames
        
    def _calculate_eye_closure_percentage(self):
        """Calculate percentage of recent frames where eyes were considered closed"""
        if not self.eye_closure_history:
            return 0
        
        closed_count = sum(1 for is_closed in self.eye_closure_history if is_closed)
        return (closed_count / len(self.eye_closure_history)) * 100
    
    def detect(self, eye_aspect_ratio):
        """
        Detect drowsiness based on eye aspect ratio
        
        Args:
            eye_aspect_ratio (float): Current eye aspect ratio
            
        Returns:
            str: Drowsiness level - "AWAKE", "NORMAL", or "EXTREME"
        """
        # Update eye closure history
        is_closed = eye_aspect_ratio < self.eye_aspect_ratio_threshold
        self.eye_closure_history.append(is_closed)
        
        # Keep history at fixed size
        if len(self.eye_closure_history) > self.history_size:
            self.eye_closure_history.pop(0)
        
        # Calculate closure percentage over recent frames
        closure_percentage = self._calculate_eye_closure_percentage()
        
        # Check if eyes are closed based on EAR
        if is_closed:
            # Increment counter for consecutive frames with closed eyes
            self.closed_eyes_frames += 1
            
            # Start timer if not already started
            if self.drowsy_start_time is None:
                self.drowsy_start_time = time.time()
                
            # Calculate drowsiness duration
            drowsiness_duration = time.time() - self.drowsy_start_time
            
            # Determine drowsiness level based on duration, EAR, and closure pattern
            if (eye_aspect_ratio <= self.extreme_ear_threshold and 
                drowsiness_duration >= self.extreme_duration_threshold) or closure_percentage > 70:
                self.current_drowsiness_level = "EXTREME"
            elif (eye_aspect_ratio <= self.normal_ear_threshold and 
                  drowsiness_duration >= self.normal_duration_threshold) or closure_percentage > 50:
                self.current_drowsiness_level = "NORMAL"
            else:
                # Keep current level to avoid flickering between states
                pass
        else:
            # If eyes are open but we have a pattern of frequent closures, maintain alert state
            if closure_percentage > 40:
                # Don't reset immediately to avoid stopping alerts too early
                pass
            elif closure_percentage > 20 and self.current_drowsiness_level != "AWAKE":
                # Downgrade from EXTREME to NORMAL if eyes are opening but still concerning
                if self.current_drowsiness_level == "EXTREME":
                    self.current_drowsiness_level = "NORMAL"
            else:
                # Reset counter and timer if eyes are consistently open
                self.closed_eyes_frames = 0
                self.drowsy_start_time = None
                self.current_drowsiness_level = "AWAKE"
        
        return self.current_drowsiness_level