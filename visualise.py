import pandas as pd

csv_path = "./submission_smoothed-16.csv"
df = pd.read_csv(csv_path)
print(df.columns.tolist())
print(df.head())

# Build lookup:  ("01", 939) -> score
score_dict = {}

for _, row in df.iterrows():
    vid, frame = row["Id"].split("_")
    score_dict[(vid.zfill(2), int(frame))] = row["Predicted"]

import cv2
import os

base_dir = "./AvenueDatasetVisual/Avenue_Corrupted/Dataset/testing_videos"
output_dir = "testing_videos_scored"

os.makedirs(output_dir, exist_ok=True)

for vid in sorted(os.listdir(base_dir)):
    video_path = os.path.join(base_dir, vid)
    if not os.path.isdir(video_path):
        continue

    out_video_dir = os.path.join(output_dir, vid)
    os.makedirs(out_video_dir, exist_ok=True)

    for frame_file in sorted(os.listdir(video_path)):
        if not frame_file.endswith(".jpg"):
            continue

        # Extract frame number from filename
        # frame_00939.jpg -> 939
        frame_num = int(frame_file.split("_")[1].split(".")[0])

        key = (vid, frame_num)
        if key not in score_dict:
            continue  # some frames may not have scores

        score = score_dict[key]

        img_path = os.path.join(video_path, frame_file)
        img = cv2.imread(img_path)

        text = f"Score: {score:.3f}"
        color = (0, 255, 0) if score < 0.5 else (0, 0, 255)

        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA
        )

        out_path = os.path.join(out_video_dir, frame_file)
        cv2.imwrite(out_path, img)
