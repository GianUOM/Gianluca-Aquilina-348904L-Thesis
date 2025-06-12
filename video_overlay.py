import cv2
import pandas as pd
from collections import defaultdict

VIDEO_PATH = "trim.mp4"
CSV_PATH = "final_predictions_with_metadata.csv"
OUTPUT_PATH = "annotated_output.mp4"

# --- Load predictions with metadata ---
df = pd.read_csv(CSV_PATH)

frame_annotations = defaultdict(list)

for _, row in df.iterrows():
    for f in range(row.start_frame, row.end_frame + 1):
        bbox = (int(row.x_min), int(row.y_min), int(row.x_max), int(row.y_max))
        label = f"{row.predicted_action} ({row.confidence:.2f})"
        frame_annotations[f].append((bbox, label))

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_idx = 0

print("Overlaying predictions on video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx in frame_annotations:
        for (x1, y1, x2, y2), label in frame_annotations[frame_idx]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print(f"Annotated video saved to: {OUTPUT_PATH}")
