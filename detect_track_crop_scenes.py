import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from tqdm import tqdm

YOLO_WEIGHTS_PATH = "handball_project2/handball_model/weights/best.pt"
VIDEO_PATH = "trim.mp4"
SEQUENCES_SAVE_DIR = "game_sequences"
IMG_SIZE = (640, 640)
SEQUENCE_LENGTH = 20
CONFIDENCE_THRESHOLD = 0.5
TRACK_HIGH_THRESH = 0.6
TRACK_LOW_THRESH = 0.1
NEW_TRACK_THRESH = 0.7
TRACK_BUFFER = 30
MATCH_THRESH = 0.8
FRAME_RATE = 30
PLAYER_CLASS_ID = 2
BALL_CLASS_ID = 0

os.makedirs(SEQUENCES_SAVE_DIR, exist_ok=True)

# --- Load YOLOv12 model ---
yolo_model = YOLO(YOLO_WEIGHTS_PATH)
print("YOLOv12 model loaded!")

# --- ByteTrack Setup ---
class ArgsStub:
    track_thresh = TRACK_HIGH_THRESH
    track_buffer = TRACK_BUFFER
    match_thresh = MATCH_THRESH
    mot20 = False
    low_thresh = TRACK_LOW_THRESH
    new_track_thresh = NEW_TRACK_THRESH

tracker = BYTETracker(args=ArgsStub(), frame_rate=FRAME_RATE)

def detect_objects(frame):
    results = yolo_model.predict(frame, imgsz=IMG_SIZE, conf=CONFIDENCE_THRESHOLD, device='cpu', verbose=False)
    player_detections = []
    ball_detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            class_id = int(class_id)
            detection = np.concatenate([box, [score, class_id]])

            if class_id == PLAYER_CLASS_ID:
                player_detections.append(detection)

    return (np.array(player_detections) if player_detections else np.empty((0, 6)))

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Could not open video {VIDEO_PATH}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

frame_idx = 0
memory = {}
frames = []
ball_memory = {}

for _ in tqdm(range(total_frames), desc="Processing video"):
    ret, frame = cap.read()
    if not ret:
        break

    frames.append(frame)
    player_detections = detect_objects(frame)

    if len(player_detections) > 0:
        detections_tensor = torch.from_numpy(player_detections).float()
        online_targets = tracker.update(detections_tensor, [frame.shape[0], frame.shape[1]], (frame.shape[0], frame.shape[1]))
        for target in online_targets:
            tid = target.track_id
            if tid not in memory:
                memory[tid] = []
            memory[tid].append((frame_idx, target.tlbr))

    frame_idx += 1

cap.release()

# --- Extract only top-motion player per window ---
print("Extracting most active player sequences (1 per segment)...")
sequence_counter = 0
all_metadata = []

def compute_motion_optical_flow(segment, frames):
    motion_magnitude = 0.0
    for i in range(1, len(segment)):
        frame_idx1, bbox1 = segment[i - 1]
        frame_idx2, bbox2 = segment[i]
        x1 = int((bbox1[0] + bbox2[0]) / 2)
        y1 = int((bbox1[1] + bbox2[1]) / 2)
        x2 = int((bbox1[2] + bbox2[2]) / 2)
        y2 = int((bbox1[3] + bbox2[3]) / 2)

        frame1 = cv2.cvtColor(frames[frame_idx1], cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frames[frame_idx2], cv2.COLOR_BGR2GRAY)

        roi1 = frame1[y1:y2, x1:x2]
        roi2 = frame2[y1:y2, x1:x2]

        if roi1.shape[0] == 0 or roi1.shape[1] == 0 or roi2.shape != roi1.shape:
            continue

        flow = cv2.calcOpticalFlowFarneback(roi1, roi2, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitude += np.sum(mag)
    return motion_magnitude

segments = []
for tid, track_data in memory.items():
    for i in range(0, len(track_data) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH):
        segment = track_data[i:i+SEQUENCE_LENGTH]
        segments.append((tid, segment))

segments_by_start = {}
for tid, segment in segments:
    start_frame = segment[0][0]
    motion = compute_motion_optical_flow(segment, frames)
    if start_frame not in segments_by_start or motion > segments_by_start[start_frame]["motion"]:
        segments_by_start[start_frame] = {
            "tid": tid,
            "segment": segment,
            "motion": motion
        }

for start_frame, data in segments_by_start.items():
    tid = data["tid"]
    segment = data["segment"]
    end_frame = segment[-1][0]
    video_name = f"track{tid}_f{start_frame}.avi"
    video_save_path = os.path.join(SEQUENCES_SAVE_DIR, video_name)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_save_path, fourcc, FRAME_RATE, (224, 224))

    for frame_id, bbox in segment:
        x1, y1, x2, y2 = map(int, bbox)
        crop = frames[frame_id][y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (224, 224))
        out.write(crop)

    out.release()
    sequence_counter += 1

    all_boxes = np.array([bbox for _, bbox in segment])
    avg_box = np.mean(all_boxes, axis=0).astype(int)

    all_metadata.append({
        "sequence_file": video_name,
        "track_id": tid,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "x_min": avg_box[0],
        "y_min": avg_box[1],
        "x_max": avg_box[2],
        "y_max": avg_box[3]
    })

df_meta = pd.DataFrame(all_metadata)
df_meta.to_csv("sequence_metadata.csv", index=False)

print(f"Saved {sequence_counter} top-motion sequences and updated metadata.")


