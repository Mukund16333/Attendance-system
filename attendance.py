import cv2
import face_recognition
import pickle
import numpy as np
import sqlite3
import os
from datetime import datetime

# ==============================
# CONFIGURATION (Improved)
# ==============================
TOLERANCE = 0.48              # Slightly stricter
FRAME_SCALE = 0.75
PROCESS_EVERY_N_FRAMES = 2
MIN_CONFIDENCE = 55           # Prevent weak matches

# ==============================
# LOAD ENCODINGS
# ==============================
print("[INFO] Loading encoding files...")

known_encodings = []
known_names = []

for file in os.listdir("."):
    if file.endswith("_encoding.pkl"):

        name = file.replace("_encoding.pkl", "")
        print(f"[INFO] Reading {file}")

        with open(file, "rb") as f:
            data = pickle.load(f)

            if isinstance(data, np.ndarray) and len(data.shape) == 2:
                for enc in data:
                    if enc.shape == (128,):
                        known_encodings.append(enc)
                        known_names.append(name)

            elif isinstance(data, np.ndarray) and len(data.shape) == 1:
                if data.shape == (128,):
                    known_encodings.append(data)
                    known_names.append(name)

            elif isinstance(data, list):
                for enc in data:
                    enc = np.array(enc)
                    if enc.shape == (128,):
                        known_encodings.append(enc)
                        known_names.append(name)

print(f"[INFO] Total encodings loaded: {len(known_encodings)}")

if len(known_encodings) == 0:
    print("[ERROR] No valid encodings found.")
    exit()

# ==============================
# DATABASE (REAL DUPLICATE PREVENTION)
# ==============================
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL,
    UNIQUE(name, date)
)
""")
conn.commit()

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    try:
        cursor.execute(
            "INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)",
            (name, today, current_time)
        )
        conn.commit()
        print(f"[ATTENDANCE MARKED] {name}")

    except sqlite3.IntegrityError:
        # Duplicate for same day
        pass

# ==============================
# CAMERA START
# ==============================
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

print("[INFO] Recognition Started. Press Q to quit.")

frame_count = 0
face_locations = []
face_names = []
face_confidences = []

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    display_frame = frame.copy()

    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Better far-distance detection
    face_locations = face_recognition.face_locations(
        rgb_small,
        model="hog",
        number_of_times_to_upsample=2
    )

    if frame_count % PROCESS_EVERY_N_FRAMES == 0:

        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        face_confidences = []

        for face_encoding in face_encodings:

            distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(distances) == 0:
                face_names.append("Unknown")
                face_confidences.append(0)
                continue

            best_index = np.argmin(distances)
            best_distance = distances[best_index]

            confidence = round((1 - best_distance) * 100, 2)

            # DOUBLE CHECK using compare_faces (important)
            matches = face_recognition.compare_faces(
                known_encodings, face_encoding, tolerance=TOLERANCE
            )

            if matches[best_index] and confidence >= MIN_CONFIDENCE:
                name = known_names[best_index]
                mark_attendance(name)
            else:
                name = "Unknown"

            face_names.append(name)
            face_confidences.append(confidence)

    # Multi-person drawing
    for (top, right, bottom, left), name, conf in zip(
        face_locations, face_names, face_confidences
    ):

        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

        label = f"{name} ({conf}%)"
        cv2.putText(display_frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Attendance System", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
conn.close()