import os
import cv2
import pickle
from tqdm import tqdm

# Paths
VIDEO_DIR = "simone_bag_subset"
OUTPUT_DIR = "detected_bags"
os.makedirs(OUTPUT_DIR, exist_ok=True)
from ultralytics import YOLO

model = YOLO("best.pt")


def process_video(video_path, output_file):
    """Track masks for ±1 second around the middle frame and save the tracked video."""
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if num_frames == 0 or fps == 0:
        print(f"Skipping {video_path}: No frames found or invalid FPS.")
        return

    detections = {}
    frame_id = 0
    ret, frame = cap.read()
    while ret:
        results = model.predict(frame, conf=0.4, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
        detections[frame_id] = {
            "boxes": boxes,
            "classes": classes,
            "confs": confs,
        }
        ret, frame = cap.read()
        frame_id += 1

    # Save detections data as a .pkl file (NumPy format)
    with open(output_file, "wb") as f:
        pickle.dump(detections, f)


# Process all videos
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
for video_file in tqdm(video_files, desc="Processing Videos"):
    video_path = os.path.join(VIDEO_DIR, video_file)
    output_file = os.path.join(
        OUTPUT_DIR, f"{os.path.splitext(video_file)[0]}_detections.pkl"
    )

    if os.path.exists(output_file):
        print(f"Skipping {video_file}: Already processed.")
        continue

    process_video(video_path, output_file)
