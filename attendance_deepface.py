import cv2
import os
import sqlite3
from datetime import datetime
from deepface import DeepFace

# ==============================
# CONFIGURATION
# ==============================
DATASET_PATH = "dataset"
MODEL_NAME = "ArcFace"      # State-of-the-art model
DETECTOR_BACKEND = "retinaface"  # More accurate detector
DISTANCE_THRESHOLD = 0.6

# ==============================
# DATABASE SETUP
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
        pass

# ==============================
# START CAMERA
# ==============================
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

print("[INFO] DeepFace ArcFace Recognition Started")
print("[INFO] Press Q to quit")

while True:
    ret, frame = video.read()
    if not ret:
        break

    try:
        results = DeepFace.find(
            img_path=frame,
            db_path=DATASET_PATH,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            distance_metric="cosine"
        )

        if len(results) > 0 and len(results[0]) > 0:

            best_match = results[0].iloc[0]

            distance = best_match["distance"]

            if distance < DISTANCE_THRESHOLD:
                identity_path = best_match["identity"]
                name = identity_path.split(os.sep)[-2]

                mark_attendance(name)

                # Draw rectangle manually
                cv2.putText(frame, f"{name}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

    except Exception as e:
        pass

    cv2.imshow("DeepFace Attendance - ArcFace", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
conn.close()