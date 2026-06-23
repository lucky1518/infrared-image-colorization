# Infrared Image Colorization and Enhancement for Improved Object Interpretation using Landsat-8 Satellite Imagery and Deep Learning

## Overview

This project focuses on converting infrared satellite imagery into realistic RGB images using a U-Net-based deep learning architecture. The system utilizes Landsat-8 multispectral satellite data collected from multiple Indian cities and learns the mapping between infrared bands and visible RGB bands.

The generated colorized images improve visual interpretation and can assist in remote sensing applications such as:

* Urban Planning
* Environmental Monitoring
* Land Cover Analysis
* Agricultural Assessment
* Disaster Management

---

## Problem Statement

Satellite remote sensing frequently relies on infrared imagery for monitoring the Earth's surface. However, infrared images are difficult for humans to interpret because they lack natural color information and semantic textures.

This project addresses the challenge of converting infrared imagery into visually meaningful RGB representations using deep learning techniques, thereby improving object interpretation and scene understanding.

---

## Objectives

* Collect Landsat-8 infrared and RGB satellite imagery using Google Earth Engine.
* Preprocess and enhance infrared images for model training.
* Generate training patches from multispectral satellite data.
* Develop a U-Net-based deep learning model for infrared-to-RGB colorization.
* Evaluate model performance using PSNR and SSIM metrics.
* Improve object interpretation through enhanced RGB visualization.

---

## Dataset

### Source

* Google Earth Engine (GEE)

### Satellite Dataset

* Landsat-8 Collection 2 Level-2

### Cities Used

* Mumbai
* Nagpur
* Delhi
* Hyderabad
* Bangalore
* Ahmedabad
* Pune
* Jaipur
* Lucknow
* Chennai
* Kolkata
* Bhopal
* Chandigarh
* Kochi
* Guwahati

### Input Bands

| Band | Description                    |
| ---- | ------------------------------ |
| B5   | Near Infrared (NIR)            |
| B6   | Short Wave Infrared 1 (SWIR-1) |
| B7   | Short Wave Infrared 2 (SWIR-2) |

### Target RGB Bands

| Band | Description |
| ---- | ----------- |
| B4   | Red         |
| B3   | Green       |
| B2   | Blue        |

### Dataset Statistics

| Parameter       | Value     |
| --------------- | --------- |
| Total Cities    | 15        |
| Total Patches   | 380       |
| Patch Size      | 256 × 256 |
| Input Channels  | 3         |
| Output Channels | 3         |

---

## Project Workflow

```text
Google Earth Engine
        ↓
Landsat-8 Download
        ↓
Image Enhancement
(CLAHE + Denoising + Sharpening)
        ↓
Patch Generation
        ↓
Dataset Loader
        ↓
U-Net Training
        ↓
Inference
        ↓
PSNR & SSIM Evaluation
```

---

## Project Structure

```text
Infrared-Image-Colorization/

├── data/
│   ├── raw/
│   │   ├── input/
│   │   └── rgb/
│   │
│   └── processed/
│       └── train/
│           ├── input/
│           └── rgb/
│
├── models/
│   └── unet.pth
│
├── outputs/
│   └── prediction.png
│
├── src/
│   ├── colorization/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   ├── test_dataset.py
│   │   └── test_model.py
│   │
│   ├── enhancement/
│   │   ├── clahe.py
│   │   ├── denoise.py
│   │   └── sharpen.py
│   │
│   ├── metrics/
│   │   ├── psnr.py
│   │   └── ssim.py
│   │
│   └── utils/
│       └── read_landsat.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Torchvision

### Image Processing

* OpenCV
* Rasterio
* NumPy
* Pillow

### Visualization

* Matplotlib

### Satellite Data

* Google Earth Engine
* Landsat-8

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Infrared-Image-Colorization
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Verify Dataset

```bash
python src/colorization/test_dataset.py
```

### Verify U-Net Architecture

```bash
python src/colorization/test_model.py
```

### Train Model

```bash
python src/colorization/train.py
```

### Run Inference

```bash
python src/colorization/inference.py
```

### Calculate PSNR

```bash
python src/metrics/psnr.py
```

### Calculate SSIM

```bash
python src/metrics/ssim.py
```

---

## Results

### Model

* U-Net

### Training Configuration

| Parameter     | Value       |
| ------------- | ----------- |
| Epochs        | 20          |
| Dataset Size  | 380 Patches |
| Input Size    | 256 × 256   |
| Optimizer     | Adam        |
| Loss Function | MSE Loss    |

### Evaluation Metrics

| Metric | Score    |
| ------ | -------- |
| PSNR   | 19.39 dB |
| SSIM   | 0.699    |

### Observations

* Successfully learned infrared-to-RGB mapping.
* Preserved major scene structures.
* Water bodies were accurately identified.
* Land-cover regions were reconstructed reasonably well.
* Provides a strong baseline for future improvements.

---

## Conclusion

The proposed system successfully demonstrates the feasibility of converting infrared satellite imagery into RGB images using deep learning.

A U-Net architecture was trained on Landsat-8 multispectral imagery collected from multiple Indian cities. Experimental results achieved:

* PSNR = 19.39 dB
* SSIM = 0.699

The model successfully learned scene structures and generated visually meaningful colorized outputs. This project establishes a strong baseline for future improvements using higher-resolution satellite imagery and advanced neural network architectures.

---

## Future Scope

### Sentinel-2 Integration

* Use Sentinel-2 imagery with 10-meter spatial resolution.

### Advanced Architectures

* Attention U-Net
* Residual U-Net

### Additional Features

* NDVI Integration
* Multi-Spectral Feature Fusion
* Data Augmentation

### Deployment

* Streamlit Web Application
* Real-Time Satellite Image Colorization

---

## Author

Lokesh Gakhre

Master of Computer Applications (MCA)
Specialization: Artificial Intelligence & Machine Learning

---

## Version History

### Version 1.0

* Landsat-8 Dataset
* 15 Indian Cities
* U-Net Architecture
* PSNR: 19.39 dB
* SSIM: 0.699

### Version 2.0 (Planned)

* Sentinel-2 Dataset
* Attention U-Net
* Higher Resolution Imagery
* Improved Color Reconstruction
