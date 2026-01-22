# Exercise Form Correctness Detection using Pose Estimation

## 📌 Overview

This project implements a **rule-based exercise form correctness detection pipeline** using **MediaPipe Pose** and **OpenCV**. The system analyzes short workout videos (3–5 seconds) and provides **frame-wise feedback** on posture correctness for exercises such as **bicep curls** and **lateral raises**.

The solution relies on **pose keypoint extraction**, **joint angle computation**, and **time-series smoothing**, followed by **human-interpretable posture rules**.

---

## 🎯 Objective

* Detect human body keypoints from exercise videos
* Analyze posture correctness using rule-based logic
* Provide real-time visual feedback on exercise form
* Ensure reproducibility and ease of setup

---

## 🧠 Key Features

* MediaPipe-based pose estimation
* Joint angle computation (elbow, spine, symmetry)
* Temporal smoothing using moving averages
* Frame-wise visual feedback overlay
* Works on pre-recorded short video clips

---

## 🏋️ Posture Rules Implemented

### 1. Elbow Angle (Bicep Curl)

* Expected range: **30° – 160°**
* Flags incorrect elbow flexion or overextension

### 2. Back Posture (Spine Alignment)

* Shoulder–Hip–Knee angle should be **> 160°**
* Detects excessive forward or backward bending

### 3. Arm Symmetry

* Difference between left and right elbow angles should be **< 15°**
* Ensures balanced movement during bilateral exercises

---

## 🛠️ Tech Stack

* **Python 3.9**
* **MediaPipe Pose**
* **OpenCV**
* **NumPy**

---

## 📂 Project Structure

```
project-folder/
│
├── exercise_form_detection.py   # Main script
├── input_video.mp4              # Sample exercise video
├── README.md                    # Project documentation
```

---

## ⚙️ Installation (Windows)

### 1. Create Virtual Environment

```powershell
py -3.9 -m venv pose_env
pose_env\Scripts\activate
```

### 2. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```powershell
python -m pip install mediapipe==0.10.32 opencv-python numpy
```

---

## ▶️ How to Run

1. Place a **3–5 second exercise video** in the project folder
2. Rename it to:

```text
input_video.mp4
```

3. Run the script:

```powershell
python exercise_form_detection.py
```

## 🖥️ Output

* Live video window with pose skeleton overlay
* On-screen feedback messages such as:

  * ❌ Left elbow angle out of range
  * ❌ Back bending detected
  * ✅ Good Form

---

## 📈 Design Choices

* **Rule-based analysis** ensures interpretability
* **Moving average smoothing** reduces keypoint noise
* **Single-person pose detection** for focused analysis

---

## 🚀 Possible Extensions

* Rep counting and tempo analysis
* Window-based scoring instead of frame-wise feedback
* Support for multiple exercise types
* MLflow integration for experiment tracking
* CSV export of joint angles

---

## 🧪 Compatibility Notes

* Tested on **Windows 10 / 11**
* Requires **Python 3.9**
* MediaPipe does not fully support Python 3.12

