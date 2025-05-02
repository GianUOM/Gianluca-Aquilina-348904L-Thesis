import cv2
import pandas as pd
import os

# --- Config ---
VIDEO_PATH = "trim.mp4"
ACTIONS_CSV = "predicted_actions.csv"
METADATA_CSV = "sequence_metadata.csv"
OUTPUT_PATH = "highlight_jump_shots_filtered.mp4"
TARGET_ACTION = "jump-shot"
FRAME_PADDING = 30  # Add 1 second before/after at 30 FPS
MIN_CONFIDENCE = 0.99  # Only include strong predictions

# --- Load and filter predictions ---
df_actions = pd.read_csv(ACTIONS_CSV)
df_meta = pd.read_csv(METADATA_CSV)
df = pd.merge(df_actions, df_meta, on="sequence_file")

# Keep only high-confidence jump-shots
highlight_clips = df[(df["predicted_action"] == TARGET_ACTION) & (df["confidence"] >= MIN_CONFIDENCE)]

if highlight_clips.empty:
    print("⚠️ No confident jump-shots found.")
    exit()

# --- Load video info ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- Prepare output writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --- Write each clip with overlay ---
for _, row in highlight_clips.iterrows():
    start = max(0, int(row["start_frame"]) - FRAME_PADDING)
    end = min(total_frames - 1, int(row["end_frame"]) + FRAME_PADDING)
    confidence = row["confidence"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    for frame_id in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw overlay text
        label = f"Jump Shot ({confidence:.2f})"
        cv2.putText(frame, label, (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        out.write(frame)

# --- Finalize ---
cap.release()
out.release()
print(f"✅ Filtered jump-shot highlight saved to: {OUTPUT_PATH}")
