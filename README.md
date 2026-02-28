
# Sign Language Detector (PC)

This project implements a real-time Sign Language Detector compliant with the requested efficient and low-latency skeleton tracking.

## Requirements

- Python 3.8+
- Webcam

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

## Features

- **Holistic Tracking**: Utilizes MediaPipe Holistic to track face, pose, and hands simultaneously.
- **Real-time Performance**: Optimized for low latency.
- **Visualization**: Draws landmarks on the video feed.

## File Structure

- `main.py`: Core logic for video capture and holistic processing.
- `requirements.txt`: Python dependencies.
