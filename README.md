# 🚦 **Bogotá Traffic Vision**

[![Ultralytics CI](https://img.shields.io/badge/Ultralytics%20CI-passing-brightgreen)](https://github.com/ultralytics/ultralytics/actions)  [![Open in Kaggle](https://img.shields.io/badge/Open_in-Kaggle-blue)]() [![Paper (PDF)](https://img.shields.io/badge/Paper-PDF-green?logo=google-drive)](https://drive.google.com/file/d/1voYwoui9uE1eeHH7lskjdRdDJow13RoI/view?usp=sharing)

An intelligent, real-time vehicle flow detection system for optimizing urban mobility in Bogotá using state-of-the-art computer vision and deep learning techniques. Leveraging Python, OpenCV, Ultralytics YOLO, and BYTETRACK, this framework ingests live or recorded video streams from existing CCTV infrastructure, processes each frame for object detection, tracks vehicle trajectories, and performs line-cross counting to yield accurate vehicle flow statistics in under 10 ms per frame. Its modular design allows seamless benchmarking across multiple YOLO variants, facilitating data-driven decisions for adaptive traffic signaling and congestion management. 🏙️🚗

<p align="center">
  <img src="https://raw.githubusercontent.com/BCCRO/vision-urbana-bogota/main/media/4K%20Road%20traffic%20video_count.gif" alt="Model Demo" />
</p>

---

## 📖 Overview

This repository provides a modular pipeline to detect, track, and count vehicles in live or recorded video streams leveraging YOLO and BYTETRACK. Key objectives include:

* Leveraging existing urban camera infrastructure
* Comparing YOLOv8/9/10/11 for accuracy and throughput
* Implementing robust line-cross counting logic
* Enabling data-driven adaptive traffic control and alerts

---

## 📈 Results

Evaluation across YOLOv8–YOLOv11 models revealed that YOLOv10-m achieved the most favorable trade-off between accuracy and speed, reporting a mAP₅₀–₉₅ of 0.848, precision of 0.906, recall of 0.881, and an average throughput of 250 FPS on 4 K video input. Inference trials on a 500-frame test sequence recorded 19 vehicles with a mean latency of 4 ms per frame, demonstrating consistent performance in high-resolution urban scenarios. These metrics underscore the suitability of the proposed system for real-world deployment in Bogotá’s traffic monitoring network.

---

## 🗂️ Repository Structure

```text
vision-urbana-bogota/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── data.yaml            # Dataset configuration
├── data/
│   ├── raw/
│   │   ├── train/           # Raw training images & labels
│   │   └── val/             # Raw validation images & labels
│   ├── interim/             # Intermediate processed data
│   └── processed/           # Final data for training
├── media/                   # GIFs, figures, and demo assets
├── notebooks/
│   ├── 01_exploracion.ipynb
│   ├── 02_data_prep.ipynb
│   ├── 03_train_compare.ipynb
│   └── 04_video_inference.ipynb
├── src/
│   ├── train_yolo.py        # Training script
│   └── val_yolo.py          # Validation & metrics extraction
├── models/                  # Trained model weights
├── results/
│   ├── runs_compare_all/    # Training logs & plots
│   └── video_inference/     # Inference videos
└──   
```

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/BCCRO/vision-urbana-bogota.git
   cd vision-urbana-bogota
   ```
2. **Set up environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start

**Train all models:**

```bash
python src/train_yolo.py
```

**Validate and collect metrics:**

```bash
python src/val_yolo.py
```

**Run inference demo:**

```bash
jupyter nbconvert --to notebook --execute notebooks/04_video_inference.ipynb
```

---

## 📜 Resources

* **Ultralytics YOLO** for object detection framework
* **Kaggle Dataset**: Car Detection and Tracking Dataset (Kaggle)

---

## 🖋️ Authors

* **Briyid Catalina Cruz Ostos** ([bccruzo@udistrital.edu.co](mailto:bccruzo@udistrital.edu.co))
