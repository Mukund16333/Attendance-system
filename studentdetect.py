import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import math
import time
import os

# --- 1. SETTINGS & PATHS ---
VIDEO_SAMPLES = {
    "Mukund": "videos/mukund.mp4",
    "Mathi": "videos/mathi.mp4",
    "Dharaneesh": "videos/dharaneesh.mp4",
    "Dharani": "videos/dharani.mp4"
}
ESP32_URL = "http://10.43.224.119:81/stream"

# --- 2. PRE-PROCESS VIDEO SAMPLES (Learning the faces) ---
known_face_encodings = []
known_face_names = []

print("🔄 Loading student samples...")
for name, file_path in VIDEO_SAMPLES.items():
    if os.path.exists(file_path):
        v_cap = cv2.VideoCapture(file_path)
        success, img = v_cap.read() # Capture the first frame of the video
        if success:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✅ Loaded: {name}")
        v_cap.release()
    else:
        print(f"❌ File not found: {file_path}")

# --- 3. INITIALIZE MODELS ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(ESP32_URL)

# Fight Logic Variables
motion_history = []
ESCALATION_THRESHOLD = 0.55  # Sensitivity for full-body movement
ESCALATION_COUNT_LIMIT = 10
prev_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cap = cv2.VideoCapture(ESP32_URL)
        continue

    # Scale down for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # --- PART A: FACE RECOGNITION ---
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    current_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        current_names.append(name)

    # --- PART B: FULL-BODY POSE DETECTION ---
    # Process original frame for full landmark precision
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    motion_score = 0
    
    if results.pose_landmarks:
        # Drawing the 33 pinpoint landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        current_landmarks = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        if prev_landmarks:
            # Sum of Euclidean distances for all 33 body points
            total_dist = sum(math.hypot(c[0]-p[0], c[1]-p[1]) for c, p in zip(current_landmarks, prev_landmarks))
            motion_score = total_dist
        prev_landmarks = current_landmarks

    # Temporal Escalation Logic (Sliding Window)
    motion_history.append(motion_score)
    if len(motion_history) > 15: motion_history.pop(0)
    high_motion_frames = sum(1 for m in motion_history if m > ESCALATION_THRESHOLD)

    # --- PART C: VISUALS & ALERTS ---
    # Draw ID boxes
    for (top, right, bottom, left), name in zip(face_locations, current_names):
        top, right, bottom, left = top*2, right*2, bottom*2, left*2 # Scale up
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Fight Detected?
    if high_motion_frames >= ESCALATION_COUNT_LIMIT:
        involved = ", ".join(set(current_names)) # Unique names identified in frame
        cv2.putText(frame, "⚠️ FIGHT ESCALATION DETECTED!", (20, 80), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(frame, f"IDENTIFIED: {involved}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("CEDS - ESP32 Campus Surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()