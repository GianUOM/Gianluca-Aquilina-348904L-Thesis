import pandas as pd

PREDICTIONS_CSV = "predicted_actions.csv"
METADATA_CSV = "sequence_metadata.csv"
OUTPUT_CSV = "final_predictions_with_metadata.csv"

# --- Load CSVs ---
try:
    predictions_df = pd.read_csv(PREDICTIONS_CSV)
    metadata_df = pd.read_csv(METADATA_CSV)
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# --- Merge by sequence_file ---
merged_df = pd.merge(predictions_df, metadata_df, on="sequence_file", how="inner")

ordered_columns = [
    "sequence_file", "track_id", "predicted_action", "confidence",
    "start_frame", "end_frame", "x_min", "y_min", "x_max", "y_max"
]
merged_df = merged_df[ordered_columns]

# --- Save to CSV ---
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Final merged predictions with metadata saved to: {OUTPUT_CSV}")
