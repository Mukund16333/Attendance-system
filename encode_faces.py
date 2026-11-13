import cv2
import face_recognition
import pickle
import os
import numpy as np

VIDEO_FOLDER = "videos"
FRAME_SKIP = 10  # Process every 10th frame (adjust if needed)

print("[INFO] Generating encodings from videos...")

for video_file in os.listdir(VIDEO_FOLDER):

    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
        continue

    person_name = os.path.splitext(video_file)[0]
    video_path = os.path.join(VIDEO_FOLDER, video_file)

    print(f"[INFO] Processing video: {video_file}")

    cap = cv2.VideoCapture(video_path)

    encodings = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames to speed up processing
        if frame_count % FRAME_SKIP != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            encodings.append(encoding)

        # Optional: stop after enough samples
        if len(encodings) >= 20:
            break

    cap.release()

    if len(encodings) == 0:
        print(f"[WARNING] No faces found in {video_file}")
        continue

    encodings = np.array(encodings)

    print(f"[INFO] Collected {len(encodings)} encodings for {person_name}")
    print(f"[INFO] Encoding shape: {encodings.shape}")

    with open(f"{person_name}_encoding.pkl", "wb") as f:
        pickle.dump(encodings, f)

    print(f"[SUCCESS] Saved {person_name}_encoding.pkl\n")

print("[INFO] Encoding generation complete.")