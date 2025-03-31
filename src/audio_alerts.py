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
        self.volume = volume
        
        # Initialize pygame mixer
        pygame.mixer.init()
        pygame.mixer.set_num_channels(3)
        
        # Set up channels
        self.normal_channel = pygame.mixer.Channel(0)
        self.extreme_channel = pygame.mixer.Channel(1)
        
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
        
        # Generate audio files if they don't exist
        self._generate_audio_files()
    
    def _generate_audio_files(self):
        """Generate audio files for alerts using gTTS"""
        # Create audio directory if it doesn't exist
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
        os.makedirs(audio_dir, exist_ok=True)
        
        # Generate normal alert audio
        normal_audio_path = os.path.join(audio_dir, "alert_normal.mp3")
        if not os.path.exists(normal_audio_path) or True:  # Always regenerate for testing
            tts = gTTS(text=self.normal_message, lang='en')
            tts.save(normal_audio_path)
        
        # Generate extreme alert audio
        extreme_audio_path = os.path.join(audio_dir, "alert_extreme.mp3")
        if not os.path.exists(extreme_audio_path) or True:  # Always regenerate for testing
            tts = gTTS(text=self.extreme_message, lang='en')
            tts.save(extreme_audio_path)
        
        # Load audio files
        self.normal_alert_sound = pygame.mixer.Sound(normal_audio_path)
        self.extreme_alert_sound = pygame.mixer.Sound(extreme_audio_path)
        
        # Set volume
        self.normal_alert_sound.set_volume(self.volume)
        self.extreme_alert_sound.set_volume(self.volume)
    
    def _listen_for_response(self):
        """Listen for voice response to stop alerts"""
        print("Voice detection started - say something to stop alerts")
        self.stop_voice_detection = False
        
        while not self.stop_voice_detection:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=3.0)
                    
                try:
                    # Just detecting any speech is enough to stop alerts
                    self.recognizer.recognize_google(audio)
                    print("Voice detected - stopping alerts")
                    self.stop_all_alerts()
                    break
                except sr.UnknownValueError:
                    # No speech detected, continue listening
                    pass
                except sr.RequestError:
                    # API error, fallback to just detecting sound
                    if audio and any(abs(x) > 1000 for x in audio.get_raw_data()):
                        print("Sound detected - stopping alerts")
                        self.stop_all_alerts()
                        break
            except (sr.WaitTimeoutError, Exception) as e:
                # Timeout or other error, continue
                pass
            
            time.sleep(0.1)
    
    def start_voice_detection(self):
        """Start voice detection in a separate thread"""
        if self.voice_detection_thread is None or not self.voice_detection_thread.is_alive():
            self.voice_detection_thread = threading.Thread(target=self._listen_for_response, daemon=True)
            self.voice_detection_thread.start()
    
    def stop_voice_detection(self):
        """Stop voice detection thread"""
        self.stop_voice_detection = True
        if self.voice_detection_thread and self.voice_detection_thread.is_alive():
            self.voice_detection_thread.join(timeout=1.0)
    
    def play_normal_alert(self):
        """Start playing normal alert in a loop"""
        if not self.normal_alert_active and not self.extreme_alert_active:
            self.normal_alert_active = True
            self.normal_channel.play(self.normal_alert_sound, loops=-1)  # -1 means loop indefinitely
            self.start_voice_detection()
    
    def play_extreme_alert(self):
        """Start playing extreme alert in a loop"""
        # Stop normal alert if playing
        if self.normal_alert_active:
            self.normal_channel.stop()
            self.normal_alert_active = False
        
        if not self.extreme_alert_active:
            self.extreme_alert_active = True
            self.extreme_channel.play(self.extreme_alert_sound, loops=-1)  # -1 means loop indefinitely
            self.start_voice_detection()
    
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
        if drowsiness_level == "EXTREME":
            self.play_extreme_alert()
        elif drowsiness_level == "NORMAL":
            self.play_normal_alert()
        else:  # AWAKE
            self.stop_all_alerts()
    
    def cleanup(self):
        """Clean up pygame mixer and stop threads"""
        self.stop_voice_detection = True
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