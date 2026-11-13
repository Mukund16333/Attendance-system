# Attendance System — Face Recognition

A real-time face recognition-based attendance system built with Python, OpenCV, and Flask. Supports multiple recognition backends including `face_recognition` (dlib) and `DeepFace` (ArcFace).

## Features

- **Real-time face detection & recognition** via webcam
- **Automatic attendance marking** with duplicate prevention (one entry per person per day)
- **Flask web dashboard** for live attendance viewing
- **Multiple recognition pipelines:**
  - `attendance.py` — dlib/face_recognition with confidence scoring
  - `attendance_deepface.py` — DeepFace with ArcFace model
  - `app.py` — Flask app with background recognition thread
- **Video-to-encoding pipeline** — generate face encodings from video samples
- **Face extraction utility** — extract face crops from videos using RetinaFace
- **Campus surveillance mode** — ESP32 stream with fight detection via MediaPipe Pose (`studentdetect.py`)

## Project Structure

```
attendance-system/
├── app.py                    # Flask web app with live attendance dashboard
├── attendance.py             # Standalone attendance via webcam (face_recognition)
├── attendance_deepface.py    # Standalone attendance via webcam (DeepFace/ArcFace)
├── encode_faces.py           # Generate face encodings from video files
├── extractfacesfromvideos.py  # Extract face images from videos (RetinaFace)
├── convert_images.py         # Convert images to RGB format
├── studentdetect.py          # ESP32 campus surveillance + fight detection
├── test_cam.py               # Camera test utility
├── templates/
│   └── index.html            # Flask dashboard template
├── requirements.txt          # Python dependencies
└── .gitignore
```

### Data directories (not tracked — generate locally):

| Directory | Purpose |
|---|---|
| `videos/` | Source videos of people for encoding generation |
| `dataset/` | Extracted face images per person |
| `*.pkl` | Face encoding files (sensitive) |
| `*.db` / `*.csv` | Attendance records |

## Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/<your-username>/attendance-system.git
cd attendance-system
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `dlib` may require CMake and a C++ compiler. On Windows, you can use a pre-built wheel.

### 3. Prepare face data

1. Place video files (`.mp4`) in a `videos/` folder, one per person (filename = person's name).
2. Generate encodings:
   ```bash
   python encode_faces.py
   ```
   This creates `<name>_encoding.pkl` files.

3. *(Optional)* Extract face images for DeepFace:
   ```bash
   python extractfacesfromvideos.py
   ```

### 4. Run

**Flask dashboard:**
```bash
python app.py
```
Open `http://localhost:5000` to see the live attendance table.

**Standalone recognition:**
```bash
python attendance.py           # face_recognition backend
python attendance_deepface.py  # DeepFace backend
```

## Tech Stack

- **Python 3.10+**
- **OpenCV** — camera capture & display
- **face_recognition / dlib** — face encoding & matching
- **DeepFace / ArcFace** — alternative recognition pipeline
- **MediaPipe** — pose estimation for fight detection
- **Flask** — web dashboard
- **SQLite** — attendance storage

## License

This project is for educational purposes.
