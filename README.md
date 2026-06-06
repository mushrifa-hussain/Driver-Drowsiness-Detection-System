# 🚗 Driver Drowsiness Detection System

A real-time Computer Vision application that detects driver drowsiness by monitoring eye movements using facial landmarks and Eye Aspect Ratio (EAR). The system triggers an alert when prolonged eye closure is detected, helping reduce the risk of accidents caused by fatigue.

## 🚀 Features

* Real-time face detection using Dlib
* Eye landmark detection and tracking
* Eye Aspect Ratio (EAR) calculation
* Driver drowsiness detection
* Visual alert on screen
* Audio alarm when drowsiness is detected
* Live webcam monitoring

## 📸 Screenshots

### Driver Awake

<img width="700" alt="WhatsApp Image 2026-06-06 at 9 24 46 PM" src="https://github.com/user-attachments/assets/c6fc69a3-47e7-4554-af45-b8334d6abe55" />

### Driver Drowsiness Detected

<img width="700" alt="WhatsApp Image 2026-06-06 at 9 25 29 PM" src="https://github.com/user-attachments/assets/6072a731-660a-46dd-ba2b-5b4d4e252dcf" />

## 🛠 Technologies Used

* Python
* OpenCV
* Dlib
* NumPy
* SciPy
* Imutils

## 📂 Project Structure

```text
Driver-Drowsiness-Detection-System/
│
├── drowsiness_detector.py
├── shape_predictor_68_face_landmarks.dat
├── requirements.txt
├── README.md
└── images/
    ├── awake.png
    └── drowsy.png
```

## ▶️ How to Run

### Clone Repository

```bash
git clone <repository-url>
cd Driver-Drowsiness-Detection-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Facial Landmark Model

Download:

shape_predictor_68_face_landmarks.dat

and place it in the project root folder.

### Run Application

```bash
python drowsiness_detector.py
```

## 📊 Working Principle

1. Capture live video from webcam.
2. Detect face using Dlib's frontal face detector.
3. Extract facial landmarks.
4. Track both eyes.
5. Calculate Eye Aspect Ratio (EAR).
6. Detect prolonged eye closure.
7. Trigger visual and audio alerts when drowsiness is detected.

## 🎯 Future Improvements

* Blink counter
* Head pose estimation
* Mobile notifications
* Driver monitoring dashboard
* Streamlit web interface

## 👩‍💻 Author

**Mushrifa Hussain**

GitHub: https://github.com/mushrifa-hussain

LinkedIn: https://linkedin.com/in/mushrifa-hussain
