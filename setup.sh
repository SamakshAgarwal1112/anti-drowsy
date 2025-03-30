#!/bin/bash

echo "Setting up Driver Drowsiness Detection System..."

# Update package lists
sudo apt-get update

# Install Python and pip if not already installed
sudo apt-get install -y python3 python3-pip

# Install required system dependencies
sudo apt-get install -y \
    libatlas-base-dev \
    libjasper-dev \
    libcap-dev \
    libqtgui4 \
    libqt4-test \
    libhdf5-dev \
    libhdf5-serial-dev \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    v4l-utils \
    python3-opencv \
    python3-picamera \
    alsa-utils

# Install Python packages
pip3 install -r requirements.txt

# Download face landmark predictor model
mkdir -p models
if [ ! -f models/shape_predictor_68_face_landmarks.dat ]; then
    echo "Downloading facial landmark predictor model..."
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O models/shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2
fi

# Test audio output
echo "Testing audio output..."
aplay /usr/share/sounds/alsa/Front_Center.wav

echo "Setup complete! Run 'python3 src/main.py' to start the system."