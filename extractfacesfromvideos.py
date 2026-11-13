import cv2
import os
from deepface import DeepFace

VIDEO_FOLDER = "videos"
DATASET_FOLDER = "dataset"

os.makedirs(DATASET_FOLDER, exist_ok=True)

print("[INFO] Starting face extraction...")

for video_file in os.listdir(VIDEO_FOLDER):

    if not video_file.endswith((".mp4", ".avi", ".mov")):
        continue

    person_name = os.path.splitext(video_file)[0]
    person_folder = os.path.join(DATASET_FOLDER, person_name)
    os.makedirs(person_folder, exist_ok=True)

    video_path = os.path.join(VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    print(f"[INFO] Processing {video_file}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Extract every 10th frame to reduce duplicates
        if frame_count % 10 != 0:
            continue

        try:
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="retinaface",
                enforce_detection=False
            )

            for face in detections:
                face_img = face["face"]
                face_img = (face_img * 255).astype("uint8")

                save_path = os.path.join(
                    person_folder,
                    f"{person_name}_{saved_count}.jpg"
                )

                cv2.imwrite(save_path, face_img)
                saved_count += 1

        except:
            pass

    cap.release()
    print(f"[INFO] Saved {saved_count} faces for {person_name}")

print("[INFO] Extraction complete.")