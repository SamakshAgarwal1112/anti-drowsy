"""
Audio alerts module for driver drowsiness detection system with continuous playback
"""

import time
import threading
import pygame
import os
from gtts import gTTS
import speech_recognition as sr

class AudioAlerts:
    """
    Class to handle audio alerts for drowsiness detection with continuous playback
    and voice response detection
    """
    
    def __init__(self, normal_message="Hey, are you awake?", 
                 extreme_message="Alert! Wake up now!", 
                 volume=0.8):
        """
        Initialize audio alerts
        
        Args:
            normal_message (str): Message for normal drowsiness level
            extreme_message (str): Message for extreme drowsiness level
            volume (float): Volume level (0.0 to 1.0)
        """
        self.normal_message = normal_message
        self.extreme_message = extreme_message
        self.success_message = "Great! You're awake now. Stay alert."
        self.volume = volume
        
        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.set_num_channels(3)
        
        # Set up channels
        self.normal_channel = pygame.mixer.Channel(0)
        self.extreme_channel = pygame.mixer.Channel(1)
        self.success_channel = pygame.mixer.Channel(2)
        
        # Set up alert states
        self.normal_alert_active = False
        self.extreme_alert_active = False
        
        # Initialize voice recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate recognizer for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        
        # Thread for voice detection
        self.voice_detection_thread = None
        self.stop_voice_detection = False
        
        # Thread for periodic normal alerts
        self.normal_alert_thread = None
        self.stop_normal_alert_thread = False
        
        # Timer for periodic normal alerts
        self.last_normal_alert_time = 0
        self.normal_alert_interval = 5.0  # seconds between normal alerts
        
        # Current drowsiness state
        self.current_drowsiness = "AWAKE"
        
        # Generate audio files if they don't exist
        self._generate_audio_files()
    
    def _generate_audio_files(self):
        """Generate audio files for alerts using gTTS"""
        # Create audio directory if it doesn't exist
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate normal alert audio
        normal_audio_path = os.path.join(audio_dir, "alert_normal.mp3")
        if not os.path.exists(normal_audio_path):
            tts = gTTS(text=self.normal_message, lang='en')
            tts.save(normal_audio_path)
        
        # Generate extreme alert audio
        extreme_audio_path = os.path.join(audio_dir, "alert_extreme.mp3")
        if not os.path.exists(extreme_audio_path):
            tts = gTTS(text=self.extreme_message, lang='en')
            tts.save(extreme_audio_path)
        
        # Generate success message audio
        success_audio_path = os.path.join(audio_dir, "alert_success.mp3")
        if not os.path.exists(success_audio_path):
            tts = gTTS(text=self.success_message, lang='en')
            tts.save(success_audio_path)
        
        # Load audio files
        self.normal_alert_sound = pygame.mixer.Sound(normal_audio_path)
        self.extreme_alert_sound = pygame.mixer.Sound(extreme_audio_path)
        self.success_alert_sound = pygame.mixer.Sound(success_audio_path)
        
        # Set volume
        self.normal_alert_sound.set_volume(self.volume)
        self.extreme_alert_sound.set_volume(self.volume)
        self.success_alert_sound.set_volume(self.volume)
    
    def _run_periodic_normal_alerts(self):
        """Run normal alerts periodically"""
        self.stop_normal_alert_thread = False
        while not self.stop_normal_alert_thread:
            current_time = time.time()
            if (self.current_drowsiness == "NORMAL" and 
                not self.normal_channel.get_busy() and 
                current_time - self.last_normal_alert_time >= self.normal_alert_interval):
                self.normal_channel.play(self.normal_alert_sound)
                self.last_normal_alert_time = current_time
            time.sleep(0.5)
    
    def _listen_for_response(self):
        """Listen for voice response while alerts are active"""
        print("Voice detection running - speak to acknowledge alerts")
        self.stop_voice_detection = False
        
        # Set sensitivity for voice detection
        self.recognizer.energy_threshold = 1000
        self.recognizer.dynamic_energy_threshold = True
        
        while not self.stop_voice_detection:
            # Only listen if alerts are active
            if not (self.normal_alert_active or self.extreme_alert_active):
                time.sleep(0.5)
                continue
                
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=3.0)
                    
                try:
                    # Try to recognize speech
                    text = self.recognizer.recognize_google(audio)
                    print(f"Voice detected: {text}")
                    self._handle_user_response()
                    # Don't break the loop - keep listening for future alerts
                except sr.UnknownValueError:
                    # Check energy level in audio for non-speech sounds
                    audio_data = audio.get_raw_data()
                    energy = sum([abs(int.from_bytes(audio_data[i:i+2], byteorder='little', signed=True)) 
                                  for i in range(0, len(audio_data), 2)]) / (len(audio_data)/2)
                    
                    if energy > 300:  # Threshold for detecting non-speech sounds
                        print(f"Sound detected (energy: {energy})")
                        self._handle_user_response()
                except sr.RequestError:
                    # API error, fallback to just detecting sound
                    if audio and any(abs(x) > 500 for x in audio.get_raw_data()):
                        print("Sound detected through fallback method")
                        self._handle_user_response()
            except (sr.WaitTimeoutError, Exception) as e:
                # Timeout or other error, continue
                pass
            
            time.sleep(0.1)
    
    def _handle_user_response(self):
        """Handle user's voice response by stopping alerts and playing success message"""
        # Play success message
        self.success_channel.play(self.success_alert_sound)
        print("User responded - Playing success message: Great! You're awake now. Stay alert.")
        
        # Stop all active alerts
        self.stop_all_alerts()
        
        # Update state to AWAKE temporarily to break alert cycle
        # The main system will update with the actual state on next frame
        self.current_drowsiness = "AWAKE"
    
    def start_voice_detection(self):
        """Start voice detection in a separate thread if not already running"""
        if self.voice_detection_thread is None or not self.voice_detection_thread.is_alive():
            self.voice_detection_thread = threading.Thread(target=self._listen_for_response, daemon=True)
            self.voice_detection_thread.start()
    
    def play_normal_alert(self):
        """Start playing normal alert periodically"""
        if not self.normal_alert_active and not self.extreme_alert_active:
            self.normal_alert_active = True
            
            # Play initial alert immediately
            self.normal_channel.play(self.normal_alert_sound)
            self.last_normal_alert_time = time.time()
            
            # Start periodic alert thread if not already running
            if self.normal_alert_thread is None or not self.normal_alert_thread.is_alive():
                self.normal_alert_thread = threading.Thread(target=self._run_periodic_normal_alerts, daemon=True)
                self.normal_alert_thread.start()
    
    def play_extreme_alert(self):
        """Start playing extreme alert in a loop"""
        # Stop normal alert if playing
        if self.normal_alert_active:
            self.stop_normal_alert()
        
        if not self.extreme_alert_active:
            self.extreme_alert_active = True
            self.extreme_channel.play(self.extreme_alert_sound, loops=-1)  # -1 means loop indefinitely
    
    def stop_normal_alert(self):
        """Stop normal alert if playing"""
        if self.normal_alert_active:
            self.normal_channel.stop()
            self.normal_alert_active = False
    
    def stop_extreme_alert(self):
        """Stop extreme alert if playing"""
        if self.extreme_alert_active:
            self.extreme_channel.stop()
            self.extreme_alert_active = False
    
    def stop_all_alerts(self):
        """Stop all alerts"""
        self.stop_normal_alert()
        self.stop_extreme_alert()
    
    def update(self, drowsiness_level):
        """
        Update alerts based on current drowsiness level
        
        Args:
            drowsiness_level (str): Current drowsiness level ("AWAKE", "NORMAL", or "EXTREME")
        """
        # Update current drowsiness state
        self.current_drowsiness = drowsiness_level
        
        if drowsiness_level == "EXTREME":
            self.play_extreme_alert()
            # Make sure voice detection is running
            self.start_voice_detection()
        elif drowsiness_level == "NORMAL":
            self.play_normal_alert()
            # Make sure voice detection is running
            self.start_voice_detection()
        else:  # AWAKE
            self.stop_all_alerts()
    
    def cleanup(self):
        """Clean up pygame mixer and stop threads"""
        self.stop_voice_detection = True
        self.stop_normal_alert_thread = True
        self.stop_all_alerts()
        pygame.mixer.quit()

    def play_no_face_alert(self, message="No face detected! Please position yourself in front of the camera."):
        """
        Play a one-time alert when no face is detected
        
        Args:
            message (str): Message to play when no face is detected
        """
        # Create audio directory if it doesn't exist
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate no face alert audio
        no_face_audio_path = os.path.join(audio_dir, "alert_no_face.mp3")
        tts = gTTS(text=message, lang='en')
        tts.save(no_face_audio_path)
        
        # Play the alert once (not looping)
        no_face_sound = pygame.mixer.Sound(no_face_audio_path)
        no_face_sound.set_volume(self.volume)
        
        # Use a different channel for the no-face alert
        # to avoid conflicts with drowsiness alerts
        pygame.mixer.Channel(2).play(no_face_sound)