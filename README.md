# DiseaseDetection
# Plant Disease Detector App

A desktop plant disease detection app built with **PyQt5**, **PyTorch inference**, **OpenCV**, and **PIL**.  
It supports both:

- **Image upload diagnosis**
- **Live camera diagnosis**

The app can also:

- highlight the detected leaf area
- show top prediction confidence
- display all class probabilities
- optionally show **Grad-CAM explanations**
- flag **unknown / low-confidence** predictions

---

## Features

- Upload an image and diagnose plant disease
- Use a live webcam feed for real-time capture
- Optional leaf detection before classification
- Bounding box around the detected plant/leaf
- Green box for healthy predictions
- Red box for disease predictions
- Cleaner disease label formatting
  - example: `Potato___Late_blight` → `Potato - Late Blight`
- Styled UI with plant-themed colours
- Adjustable:
  - unknown confidence threshold
  - entropy threshold
  - temperature scaling

---

## UI Theme

This version uses:

- **Top/header colour:** `#034732`
- **Main body colour:** `#9DC4B5`

It also improves:

- header readability
- probability bar visibility
- disease name formatting

---

## Requirements

Install the required packages before running the app.

```bash
pip install pyqt5 opencv-python pillow numpy pyyaml
```
Working application<img width="1286" height="722" alt="Screenshot 2026-03-31 193350" src="https://github.com/user-attachments/assets/088f1a09-4862-4915-b9f6-3eabfea4af00" />

