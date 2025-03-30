# Driver Drowsiness Detection System

A real-time driver drowsiness detection system for Raspberry Pi 5 that monitors driver alertness using computer vision and provides audio warnings when drowsiness is detected.

## Features

- Real-time face and eye detection
- Calculates Eye Aspect Ratio (EAR) to determine drowsiness
- Two-level alertness monitoring (normal and extreme drowsiness)
- Customizable audio alerts for different drowsiness levels
- Performance optimized for Raspberry Pi 5
- Easy to use and configure

## How It Works

The system uses computer vision techniques to:
1. Detect the driver's face using a pre-trained model
2. Locate facial landmarks to identify eye regions
3. Calculate the Eye Aspect Ratio (EAR) to determine if the eyes are closing
4. Track duration of eye closure to detect drowsiness
5. Provide audio alerts when drowsiness is detected

## Requirements

### Hardware
- Raspberry Pi 5 (or compatible Linux computer)
- USB camera or Raspberry Pi Camera Module
- Speaker or headphones for audio alerts
- (Optional) Small display for monitoring

### Software
- Python 3.7+
- OpenCV
- dlib
- pygame (for audio playback)
- Additional dependencies listed in `requirements.txt`

## Installation

1. Clone this repository to your Raspberry Pi 5:
```bash
git clone https://github.com/yourusername/driver-drowsiness-system.git
cd driver-drowsiness-system
```

2. Run the setup script to install dependencies:
```bash
chmod +x setup.sh
./setup.sh
```

This will install all required libraries and download the necessary model files.

## Usage

1. Connect a camera to your Raspberry Pi 5
2. Connect speakers or headphones for audio alerts
3. Run the system:
```bash
python src/main.py
```

4. Optional command-line arguments:
```bash
python src/main.py --config config/custom_config.yaml --camera 0
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Example configuration adjustments
detection:
  eye_aspect_ratio_threshold: 0.25  # Adjust based on testing

drowsiness:
  normal:
    duration_threshold: 3.0  # Seconds of eye closure for normal alert
    message: "Hey, are you awake?"  # Custom message
  extreme:
    duration_threshold: 1.0  # Seconds of eye closure for extreme alert
    message: "Alert! Wake up now!"  # Custom message
```

## Customizing Alert Messages

You can customize the audio alert messages by modifying the `message` fields in the configuration file. The system will automatically generate new audio files when you change these messages.

## Performance Optimization

For optimal performance on Raspberry Pi 5:
- Use a lower camera resolution (e.g., 640x480)
- Consider using the Raspberry Pi Camera Module for better performance
- Close unnecessary applications while running the system

## Troubleshooting

### Camera Issues
- Ensure your camera is properly connected and recognized by the system
- Try changing the `device_id` in the configuration file

### Audio Issues
- Check that your audio output is properly connected and set as default
- Adjust the `volume` parameter in the configuration file

### Performance Issues
- Lower the camera resolution in the configuration file
- Consider enabling the GPU for inference if available

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.