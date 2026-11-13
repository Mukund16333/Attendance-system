from flask import Flask, render_template
import cv2
import pickle
import numpy as np
import face_recognition
import datetime
import sqlite3
import os
import threading
import time

app = Flask(__name__)

# -----------------------------------
# Load All Encoding Files
# -----------------------------------
known_encodings = []
known_names = []

print("[INFO] Loading encoding files...")

for file in os.listdir():
    if file.endswith("_encoding.pkl"):
        name = file.replace("_encoding.pkl", "")
        
        with open(file, "rb") as f:
            encodings = pickle.load(f)

        encodings = np.array(encodings)

        if encodings.ndim == 1:
            encodings = encodings.reshape(1, 128)

        if encodings.shape[1] != 128:
            raise ValueError(f"{file} has invalid encoding shape {encodings.shape}")

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(name)

        print(f"[INFO] Loaded {len(encodings)} encodings for {name}")

known_encodings = np.array(known_encodings)

print(f"[INFO] Total encodings loaded: {len(known_encodings)}")

# -----------------------------------
# Database Setup
# -----------------------------------
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    time TEXT
)
""")
conn.commit()

marked_today = set()

# -----------------------------------
# Face Recognition Background Loop
# -----------------------------------
def recognition_loop():
    print("[INFO] Starting camera...")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("[ERROR] Could not open webcam")
        return

    print("[INFO] Camera started successfully")

    while True:
        success, frame = camera.read()

        if not success:
            print("[WARNING] Failed to grab frame. Retrying...")
            time.sleep(1)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        for encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            if face_distances[best_match_index] < 0.6:
                name = known_names[best_match_index]

                if name not in marked_today:
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute(
                        "INSERT INTO attendance (name, time) VALUES (?, ?)",
                        (name, now)
                    )
                    conn.commit()
                    marked_today.add(name)
                    print(f"[INFO] Attendance marked for {name}")

        time.sleep(0.2)  # Reduce CPU usage


# Start recognition thread
threading.Thread(target=recognition_loop, daemon=True).start()

# -----------------------------------
# Routes
# -----------------------------------
@app.route('/')
def index():
    cursor.execute("SELECT name, time FROM attendance ORDER BY id DESC")
    records = cursor.fetchall()
    return render_template("index.html", records=records)

# -----------------------------------
# Run App
# -----------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)