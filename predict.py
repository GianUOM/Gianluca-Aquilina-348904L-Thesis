import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

# --- Configuration ---
SEQUENCES_DIR = "game_sequences" 
MODEL_PATH = "LRCN_model__Date_Time_2025_04_29__21_17_47__Loss_0.39231064915657043__Accuracy_0.8394160866737366.h5"     
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  
CLASSES_LIST = ["jump-shot", "dribbling", "shot", "defence", "passing"]

# --- Load trained model ---
model = load_model(MODEL_PATH)
print("✅ Loaded LRCN model!")

# --- Frame extraction function ---
def frames_extraction(video_path):
    frames_list = []

    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255.0
        frames_list.append(normalized_frame)

    video_reader.release()
    return frames_list

# --- Run prediction for each .avi sequence ---
results = []

for file_name in tqdm(sorted(os.listdir(SEQUENCES_DIR)), desc="Predicting actions"):
    if not file_name.endswith(".avi"):
        continue

    video_path = os.path.join(SEQUENCES_DIR, file_name)
    frames = frames_extraction(video_path)

    if len(frames) != SEQUENCE_LENGTH:
        continue  

    input_sequence = np.expand_dims(np.array(frames), axis=0) 
    predictions = model.predict(input_sequence, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_class = CLASSES_LIST[predicted_index]
    confidence = float(predictions[0][predicted_index])

    results.append({
        "sequence_file": file_name,
        "predicted_action": predicted_class,
        "confidence": confidence
    })

df = pd.DataFrame(results)
df.to_csv("predicted_actions.csv", index=False)
print("✅ Prediction complete! Saved to 'predicted_actions.csv'")
