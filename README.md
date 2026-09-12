# Small Object Detection - VisionAI

A deep learning-based Small Object Detection project that integrates multiple object detection models into a unified Streamlit application.

The project combines YOLO-based detection with CFINet for small-object detection and provides a hybrid detection mode for comparing and combining predictions.

## Features

- YOLOv5m object detection
- YOLOv8n object detection
- YOLOv8s object detection
- YOLO11n object detection
- CFINet small-object detection
- Hybrid detection using two selected models
- IoU-based duplicate filtering
- Confidence threshold control
- Streamlit web interface
- GPU/CUDA support
- Modular model integration

## Models

| Model | Purpose |
|---|---|
| YOLOv5m | General object detection |
| YOLOv8n | Lightweight object detection |
| YOLOv8s | Improved YOLO detection |
| YOLO11n | Modern lightweight YOLO detection |
| CFINet | Small-object detection |
| Hybrid | Combines predictions from two models |

## Project Structure

```text
Small-Object-Detection/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|
|-- cfinet/
|-- esod/
|-- models/
|-- Images/
|-- work_dirs/
|
|-- yolov5m.pt
|-- yolov8n.pt
|-- yolov8s.pt
|-- yolo11n.pt
`-- yolov5mu.pt

Large model weights and datasets are excluded from Git using `.gitignore`.

## Environment

The project has been tested locally with:

- Python 3.8.20
- PyTorch 2.0.1 + CUDA 11.7
- TorchVision 0.15.2
- Ultralytics 8.4.149
- Streamlit 1.40.1
- MMCV 1.7.2
- NumPy 1.24.4

A CUDA-capable NVIDIA GPU is recommended for CFINet inference.

## Installation

Clone the repository:

```bash
git clone https://github.com/ssatyabrata228-sketch/Small-Object-Detection.git
cd Small-Object-Detection

Create a virtual environment:

python -m venv venv

Windows:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

For GPU-enabled PyTorch, use a PyTorch build compatible with your NVIDIA driver and CUDA environment.

Run the Application
streamlit run app.py

The application provides:

Login
Single Model Detection
Hybrid Detection
Model selection
Confidence threshold adjustment
Detection result visualization
Login Configuration

Credentials are read from environment variables rather than being stored directly in the source code.

Example:

$env:VISIONAI_USERNAME="admin"
$env:VISIONAI_PASSWORD="YourStrongPassword"

Then run:

streamlit run app.py

Never commit real passwords, API keys, or other secrets to GitHub.

CFINet

CFINet is integrated as the dedicated small-object detection model.

Implementation:

cfinet/

Configuration:

cfinet/configs/cfinet/faster_rcnn_r50_fpn_cfinet_1x.py

Checkpoint:

models/cfinet/latest.pth

Large checkpoints are intentionally excluded from version control.

CFINet Paper

Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning

Published at ICCV 2023.

The original CFINet attribution and licensing information are retained in the cfinet directory.

Hybrid Detection

The Hybrid Detection module allows two supported models to be selected simultaneously.

Input Image
     |
     +-- Model 1 --> Predictions --+
     |                             |
     +-- Model 2 --> Predictions --+
                                   |
                                   v
                             IoU Filtering
                                   |
                                   v
                           Final Detections

This allows predictions from different detection approaches to be compared and combined.

Dataset

The project is designed for small-object detection research and experimentation.

Related datasets include:

SODA-D
SODA-A
COCO
VisDrone

Dataset files are not included in the repository when they are large or subject to separate distribution terms.

Testing Status

The following functionality has been verified locally:

YOLOv8n inference
YOLOv8s inference
YOLOv5m inference
YOLO11n inference
CFINet inference
Hybrid detection
Streamlit application startup
CFINet configuration and checkpoint loading
GPU inference

These tests verify the application inference pipeline. They are not official benchmark results or trained-model evaluation metrics.

Important Notes
Model weights (.pt, .pth) are ignored by Git.
Dataset directories are ignored by Git.
CFINet requires compatible deep-learning dependencies.
GPU inference is recommended for CFINet.
Run the application from the project root.
Do not commit passwords, API keys, or other secrets.
Citation

If you use the CFINet implementation, please cite the original work:

Small Object Detection via Coarse-to-fine Proposal Generation
and Imitation Learning.
ICCV 2023.

See the CFINet directory and citation files for the original attribution and licensing information.

License

This repository contains integrated third-party research code.

Please refer to:

LICENSE
cfinet/LICENSE

for applicable license and attribution information.

Project

Small Object Detection - VisionAI

A Computer Science Engineering project focused on deep learning, computer vision, and small-object detection.


