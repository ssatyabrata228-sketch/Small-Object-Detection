# Small Object Detection — VisionAI

A deep learning-based **Small Object Detection** project that integrates multiple object detection models into a unified **Streamlit web application**.

VisionAI combines YOLO-based detectors with **CFINet** for small-object detection and provides a **Hybrid Detection** mode for comparing and combining predictions from two selected models.

---

## 🎯 Project Objective

The objective of this project is to build an accessible computer-vision application for detecting objects, particularly objects that occupy relatively small regions of an image.

The system provides:

* Multiple pretrained detection models
* Small-object detection using CFINet
* Model-to-model prediction comparison
* Hybrid detection
* IoU-based duplicate filtering
* Adjustable confidence thresholds
* GPU-accelerated inference
* A simple Streamlit interface

---

## ✨ Features

* 🔹 YOLOv5m object detection
* 🔹 YOLOv8n object detection
* 🔹 YOLOv8s object detection
* 🔹 YOLO11n object detection
* 🔹 CFINet small-object detection
* 🔹 Hybrid detection using two selected models
* 🔹 IoU-based duplicate filtering
* 🔹 Adjustable confidence threshold
* 🔹 Detection visualization
* 🔹 Streamlit web interface
* 🔹 CUDA/GPU support
* 🔹 Modular model integration
* 🔹 Environment-based login configuration

---

## 🤖 Supported Models

| Model           | Role                                          |
| --------------- | --------------------------------------------- |
| **YOLOv5m**     | General-purpose object detection              |
| **YOLOv8n**     | Lightweight object detection                  |
| **YOLOv8s**     | Higher-capacity YOLO detection                |
| **YOLO11n**     | Modern lightweight YOLO detection             |
| **CFINet**      | Small-object detection                        |
| **Hybrid Mode** | Combines predictions from two selected models |

> Model weight files are intentionally excluded from Git because of their size.

---

## 🧠 Hybrid Detection

Hybrid Detection allows two supported models to process the same input image.

```text
                    Input Image
                         │
              ┌──────────┴──────────┐
              │                     │
           Model 1               Model 2
              │                     │
        Predictions           Predictions
              │                     │
              └──────────┬──────────┘
                         │
                  IoU Comparison
                         │
                 Duplicate Filtering
                         │
                         ▼
                  Final Detections
```

The hybrid pipeline can be used to compare predictions from different detection approaches and reduce overlapping duplicate detections according to the configured IoU threshold.

---

## 🛠️ Technologies Used

### Programming & Application

* Python
* Streamlit

### Deep Learning

* PyTorch
* TorchVision
* Ultralytics YOLO
* MMDetection
* MMCV

### Data & Computer Vision

* NumPy
* Pandas
* OpenCV
* Pillow
* SciPy
* Matplotlib
* COCO-format annotations

### Hardware Acceleration

* NVIDIA CUDA
* GPU-accelerated inference

---

## 📁 Project Structure

```text
Small-Object-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── cfinet/
│   ├── configs/
│   ├── mmdet/
│   ├── tools/
│   ├── demo/
│   └── tests/
│
├── esod/
│
├── models/
│   └── cfinet/
│       └── latest.pth          # Local only
│
├── Images/                     # Local input images
│
├── work_dirs/                  # Training/output directory
│
├── yolov5m.pt                 # Local only
├── yolov8n.pt                 # Local only
├── yolov8s.pt                 # Local only
├── yolo11n.pt                 # Local only
└── yolov5mu.pt                # Local only
```

Large model weights, datasets, generated outputs, and environment files are excluded from version control using `.gitignore`.

---

## 💻 Environment

The application has been tested locally with:

| Component   | Version           |
| ----------- | ----------------- |
| Python      | 3.8.20            |
| PyTorch     | 2.0.1 + CUDA 11.7 |
| TorchVision | 0.15.2            |
| Ultralytics | 8.4.149           |
| Streamlit   | 1.40.1            |
| MMCV        | 1.7.2             |
| NumPy       | 1.24.4            |

A CUDA-capable NVIDIA GPU is recommended for CFINet inference.

> Exact dependency compatibility may vary depending on the GPU, CUDA runtime, operating system, and installed Python packages.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ssatyabrata228-sketch/Small-Object-Detection.git
cd Small-Object-Detection
```

### 2. Create a virtual environment

Using Python:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU-enabled PyTorch, use a PyTorch build compatible with your NVIDIA driver and CUDA environment.

---

## ▶️ Run VisionAI

From the project root:

```bash
streamlit run app.py
```

The application provides:

1. Login
2. Single Model Detection
3. Hybrid Detection
4. Model selection
5. Confidence threshold adjustment
6. Detection visualization

---

## 🔐 Login Configuration

VisionAI reads login credentials from environment variables instead of storing real credentials in the source code.

### Windows PowerShell

```powershell
$env:VISIONAI_USERNAME="admin"
$env:VISIONAI_PASSWORD="YourStrongPassword"
```

Then start the application:

```powershell
streamlit run app.py
```

Never commit real passwords, API keys, tokens, or other secrets to GitHub.

---

## 🔬 CFINet Integration

CFINet is integrated as the dedicated small-object detection model.

## Datasets

This project supports small-object detection datasets including **SODA-D**, **SODA-A**, **COCO**, and **VisDrone**.

### SODA Benchmark

The project uses the **SODA (Small Object Detection Dataset)** benchmark, which contains two subsets:

* **SODA-D** — Small object detection dataset
* **SODA-A** — Small object detection dataset for aerial imagery

#### SODA-D Download

* [OneDrive](https://nwpueducn-my.sharepoint.com/:f:/g/personal/gcheng_nwpu_edu_cn/EhXUvvPZLRRLnmo0QRmd4YUBvDLGMixS11_Sr6trwJtTrQ?e=PellK6)
* [Baidu NetDisk](https://pan.baidu.com/s/1aqmqkG_GzDKBTM_NK5ecqA?pwd=SODA)

#### SODA-A Download

* [OneDrive](https://nwpueducn-my.sharepoint.com/:f:/g/personal/gcheng_nwpu_edu_cn/EqJBjheHJXVOrMQWcr8dOt0BZJAfn1bkUSEQwIKHkVE0Vg?e=Hhcnoi)
* [Baidu NetDisk](https://pan.baidu.com/s/1G6x-hslv5C02WikZCzsNlA?pwd=SODA)

### Dataset Preparation

SODA requires an image-splitting step before conventional object-detection processing.

For **SODA-D**, image-splitting scripts are available in:

```text
cfinet/tools/img_split/
```

For **SODA-A**, refer to the SODA-mmrotate project for the corresponding preparation workflow.

For additional information about the SODA benchmark, visit the [SODA Dataset Homepage](https://shaunyuan22.github.io/SODA/).

> Dataset files are not included in this repository because of their size and distribution requirements.


### Implementation

```text
cfinet/
```

### Configuration

```text
cfinet/configs/cfinet/faster_rcnn_r50_fpn_cfinet_1x.py
```

### Local checkpoint

```text
models/cfinet/latest.pth
```

The CFINet checkpoint is intentionally excluded from Git because of its large file size.

---

## 📊 Dataset Support

The project is designed for small-object detection research and experimentation.

Related datasets include:

* **SODA-D**
* **SODA-A**
* **COCO**
* **VisDrone**

Dataset files are not included in the repository when they are large or subject to separate distribution terms.

---
## Dataset Preparation

For COCO-format annotation files, the repository includes `fix_dataset.py` to remove invalid bounding boxes.

Example:

```bash
python fix_dataset.py --input path/to/train.json --output path/to/train_fixed.json

## 🧪 Testing Status

The following application functionality has been verified locally:

* ✅ YOLOv8n inference
* ✅ YOLOv8s inference
* ✅ YOLOv5m inference
* ✅ YOLO11n inference
* ✅ CFINet inference
* ✅ Hybrid Detection
* ✅ Streamlit application startup
* ✅ CFINet configuration loading
* ✅ CFINet checkpoint loading
* ✅ GPU inference
* ✅ Python syntax compilation

These checks verify the **application inference pipeline**.

They should **not** be interpreted as official benchmark results, training results, or scientific evaluation metrics such as mAP, precision, or recall.

---

## 📌 Important Notes

* Model weight files such as `.pt` and `.pth` are ignored by Git.
* Dataset directories are ignored by Git.
* Generated detection outputs are ignored by Git.
* CFINet requires compatible deep-learning dependencies.
* GPU inference is recommended for CFINet.
* Run the application from the project root.
* Do not commit passwords, API keys, tokens, or other secrets.
* Model checkpoints must be downloaded or placed locally before using models that require them.

---

## 🔮 Future Improvements

Potential future improvements include:

* [ ] Add quantitative benchmark results
* [ ] Add precision, recall, mAP, and IoU evaluation
* [ ] Add automated test coverage
* [ ] Add model performance comparison charts
* [ ] Add additional small-object datasets
* [ ] Improve authentication and user management
* [ ] Add Docker support
* [ ] Add CI/CD workflows
* [ ] Improve application deployment support
* [ ] Add more visualization and analytics features

---

## 📚 CFINet Citation

This project integrates the CFINet implementation for small-object detection.

**Paper:**

> *Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning.*

Published at **ICCV 2023**.

Please refer to the `cfinet/` directory and its accompanying citation and license files for the original research attribution.

---

## ⚖️ License

This repository contains integrated third-party research code.

Please refer to:

```text
LICENSE
cfinet/LICENSE
```

for the applicable license terms, attribution requirements, and third-party notices.

---

## 👨‍💻 Project

**Small Object Detection — VisionAI**

A Computer Science Engineering project focused on:

* Deep Learning
* Computer Vision
* Object Detection
* Small Object Detection
* Model Comparison
* Hybrid Detection
* Streamlit Application Development

---

## ⭐ Acknowledgements

This project builds upon existing open-source object-detection research and frameworks, including CFINet, MMDetection, and Ultralytics YOLO.

Please refer to the respective project licenses and citation information when using their code or models.
