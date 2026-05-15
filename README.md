# Goslip

AI that detects when you're falling asleep and plays Wake Up Mf sound.

## How it works

- Tracks your eye openness in real-time using MediaPipe
- Calculates a rolling median of eye aspect ratio over the last 60 seconds
- When the median drops below a threshold you are falling asleep and music plays

## Stack

- Python
- MediaPipe
- OpenCV (cv2)

## Setup

pip install mediapipe opencv-python numpy

## Run

python main.py

## Requirements

- Webcam or phone camera via DroidCam
- Eyes
- A reason to stay awake
