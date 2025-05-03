# Handball Action Recognition and Highlight Generation

This repository contains the core code and resources for a handball action recognition system, designed as part of a Final Year Project. The system performs detection, tracking, action classification, and highlight generation based on player actions, particularly in jump shot motion, in match/training footage.

## External Resources

Due to storage constraints, large files, like trained models for YOLOv12 and ByteTrack and videos, are hosted externally:

a. Google Drive (videos, models, outputs): [Click here to access video results and models.](https://drive.google.com/drive/folders/1GmdGgizF0ED77yhQNRhDGQRYJd70pLgg?usp=sharing)

b. Roboflow Custom Detection Dataset: [Click here for the dataset (VERSION 1).](https://universe.roboflow.com/uom-sgd5l/full-handball-dataset/dataset/1)

## Repository Structure

1. **Handball_Action_Recognition.ipynb** - Full end-to-end action recognition pipeline including preprocessing, training, and evaluation results
2. **handball_YOLO_detection.ipynb** - Code for running the YOLOv12-L model on handball match frames custom dataset to get the model's weights (best.pt)
3. **detect_track_crop_scenes.py** - Code for detecting, tracking and croppping player sequences from full match videos using the YOLOv12 cusotm weights and the tracker, ByteTrack.
4. **predict.py** - Loads the trained LCRN model from the *Handball_Action_Recognition.ipynb* file and runs action classification on cropped sequences
5. **highlights.py** - Generates a highlight reel by filtering jumps-hots with confidence >= 0.99.
6. **video_overlay.py** - Overlays predicted action action labels and confidence values on bounding boxes in full video.
7. **final_csv_with_metadata.py** - Merges predicted actions with corresponding sequence metadata
8. **final_predictions_with_metadata.csv** - Output with bounding boxes, predicted labels, confidence, and temporal metadata.
9. **predicted_actions.csv** - Outputs CSV of raw predictions from the LCRN Model
10. **sequence_metadata.csv** - Contains start and end frame and bounding box data for each cropped video
