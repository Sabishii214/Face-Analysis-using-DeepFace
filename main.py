import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from collections import deque, Counter
from datetime import datetime
import time
import glob

# For optional ground truth metrics
from sklearn.metrics import accuracy_score

# === VIDEO FILES CONFIG ===
video_files = None  # None = webcam
frame_skip = 10
smoothing_window = 10

# Performance settings
DETECTOR_BACKEND = 'mtcnn'  # 'opencv', 'ssd', 'mtcnn', 'retinaface'

print("Initializing DeepFace models...")
print(f"Using detector: {DETECTOR_BACKEND}")

# ===== PROCESS VIDEO FILES =====
def get_video_list(video_files):
    if video_files is None:
        return [None]  # Webcam
    elif isinstance(video_files, str):
        if '*' in video_files:
            files = glob.glob(video_files)
            if not files:
                print(f"No files found matching pattern: {video_files}")
                return []
            return sorted(files)
        else:
            return [video_files]
    elif isinstance(video_files, list):
        return video_files
    else:
        return []

def process_video(video_path, video_index=1, total_videos=1):
    # Video name
    if video_path is None:
        date_prefix = datetime.now().strftime("%Y%m%d")
        video_name = f"Webcam_{date_prefix}"
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

    print("\n" + "=" * 70)
    print(f"Processing video {video_index}/{total_videos}: {video_name}")
    print("=" * 70)

    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Create folder for frames
    output_dir = os.path.join("Video_Frames", video_name)
    os.makedirs(output_dir, exist_ok=True)

    # Counters
    frame_count = 0
    processed_frames = 0
    total_faces_detected = 0

    # Smoothing windows
    emotion_window = deque(maxlen=smoothing_window)
    age_window = deque(maxlen=smoothing_window)
    race_window = deque(maxlen=smoothing_window)
    last_text = "Loading..."

    # Data storage
    data = {
        "Frame": [], "Predicted_Age": [], "True_Age": [],
        "Predicted_Gender": [], "True_Gender": [],
        "Predicted_Emotion": [], "True_Emotion": [],
        "Predicted_Race": [], "True_Race": [],
        "Age_Correct": [], "Gender_Correct": [],
        "Emotion_Correct": [], "Race_Correct": []
    }

    # Ground truth
    ground_truth = {}
    for f in range(10, 41):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "surprise", "Race": "white"}
    for f in range(70, 101):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "happy", "Race": "white"}
    for f in range(130, 151):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "disgust", "Race": "white"}
    for f in range(180, 211):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "sad", "Race": "white"}
    for f in range(230, 261):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "surprise", "Race": "white"}
    for f in range(290, 321):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "happy", "Race": "white"}
    for f in range(340, 371):
        ground_truth[f] = {"Age": 28, "Gender": "Man", "Emotion": "surprise", "Race": "white"}

    def analyze_face(frame):
        try:
            result = DeepFace.analyze(
                frame,
                actions=["age", "gender", "emotion", "race"],
                enforce_detection=False,
                detector_backend=DETECTOR_BACKEND,
                silent=True
            )
            r = result[0] if isinstance(result, list) else result
            if not r.get("region"):
                return None
            age = r.get("age", 0)
            gender = r.get("dominant_gender", "Unknown")
            emotion = r.get("dominant_emotion", "Unknown")
            race = r.get("dominant_race", "Unknown")
            confidence = r.get("emotion", {}).get(emotion, 0)/100.0 if r.get("emotion") else 0
            return age, gender, emotion, race, confidence
        except Exception:
            return None

    # Performance tracking
    inference_times = []
    confidence_scores = []

    start_time = time.time()
    print("Starting analysis... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_small = cv2.resize(frame, (640, 480))

        if frame_count % frame_skip == 0:
            result = analyze_face(frame_small)
            if result:
                age, gender, emotion, race, confidence = result
                processed_frames += 1
                total_faces_detected += 1
                inference_times.append(confidence)  # optional for report
                confidence_scores.append(confidence)

                # Smoothing
                emotion_window.append(emotion)
                age_window.append(age)
                race_window.append(race)
                smoothed_emotion = Counter(emotion_window).most_common(1)[0][0]
                smoothed_age = int(sum(age_window)/len(age_window))
                smoothed_race = Counter(race_window).most_common(1)[0][0]

                last_text = f"{gender}, {smoothed_emotion}, {smoothed_age}, {smoothed_race}"

                # Save frame
                frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)

                # Get ground truth
                gt = ground_truth.get(frame_count, {"Age": None, "Gender": None, "Emotion": None, "Race": None})

                # Append data
                data["Frame"].append(frame_count)
                data["Predicted_Age"].append(smoothed_age)
                data["True_Age"].append(gt["Age"])
                data["Predicted_Gender"].append(gender)
                data["True_Gender"].append(gt["Gender"])
                data["Predicted_Emotion"].append(smoothed_emotion)
                data["True_Emotion"].append(gt["Emotion"])
                data["Predicted_Race"].append(smoothed_race)
                data["True_Race"].append(gt["Race"])
                data["Age_Correct"].append(smoothed_age == gt["Age"])
                data["Gender_Correct"].append(gender == gt["Gender"])
                data["Emotion_Correct"].append(smoothed_emotion == gt["Emotion"])
                data["Race_Correct"].append(smoothed_race == gt["Race"])

        cv2.putText(frame, last_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("DeepFace Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopping analysis...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ===== SAVE CSV =====
    df = pd.DataFrame(data)
    for col in ["Age_Correct","Gender_Correct","Emotion_Correct","Race_Correct"]:
        df[col] = df[col].apply(lambda x: "True" if x else "False")
    csv_filename = f"{video_name}-results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Frames saved in: {output_dir}")
    print(f"CSV report saved: {csv_filename}")

    # ===== SAVE TXT REPORT (original format) =====
    total_time = time.time() - start_time
    avg_fps = processed_frames/total_time if total_time>0 else 0
    avg_conf = np.mean(confidence_scores) if confidence_scores else 0
    std_conf = np.std(confidence_scores) if confidence_scores else 0
    high_conf = sum(c>=0.7 for c in confidence_scores)
    med_conf = sum((c>=0.4 and c<0.7) for c in confidence_scores)
    low_conf = sum(c<0.4 for c in confidence_scores)

    emotion_counter = Counter(df["Predicted_Emotion"])
    gender_counter = Counter(df["Predicted_Gender"])
    race_counter = Counter(df["Predicted_Race"])
    ages = df["Predicted_Age"]

    report_filename = f"{video_name}-report.txt"
    with open(report_filename, 'w') as f:
        f.write("="*70+"\n")
        f.write(f"PERFORMANCE REPORT - {video_name}\n")
        f.write("="*70+"\n\n")

        f.write("SPEED METRICS:\n")
        f.write(f"   Total Frames: {frame_count}\n")
        f.write(f"   Processed Frames: {processed_frames}\n")
        f.write(f"   Total Time: {total_time:.2f}s\n")
        f.write(f"   Average FPS: {avg_fps:.2f}\n\n")

        f.write("DETECTION METRICS:\n")
        f.write(f"   Total Faces Detected: {total_faces_detected}\n")
        f.write(f"   Avg Faces/Processed Frame: {total_faces_detected/processed_frames if processed_frames else 0:.2f}\n")
        f.write(f"   Avg Confidence: {avg_conf:.3f} ({avg_conf*100:.1f}%)\n")
        f.write(f"   Confidence Std Dev: {std_conf:.3f}\n\n")

        f.write("CONFIDENCE DISTRIBUTION:\n")
        f.write(f"   High (>=0.7): {high_conf} ({high_conf/processed_frames*100:.1f}%)\n")
        f.write(f"   Medium (0.4-0.7): {med_conf} ({med_conf/processed_frames*100:.1f}%)\n")
        f.write(f"   Low (<0.4): {low_conf} ({low_conf/processed_frames*100:.1f}%)\n\n")

        f.write("TOP DETECTED EMOTIONS:\n")
        for emo, cnt in emotion_counter.most_common():
            f.write(f"   {emo:<12}: {cnt:5d} ({cnt/processed_frames*100:.1f}%)\n")
        f.write("\nGENDER DISTRIBUTION:\n")
        for g, cnt in gender_counter.items():
            f.write(f"   {g:<12}: {cnt:5d} ({cnt/processed_frames*100:.1f}%)\n")
        f.write("\nRACE DISTRIBUTION:\n")
        for r, cnt in race_counter.items():
            f.write(f"   {r:<12}: {cnt:5d} ({cnt/processed_frames*100:.1f}%)\n")
        f.write("\nAGE STATISTICS:\n")
        f.write(f"   Average Age: {np.mean(ages):.1f}\n")
        f.write(f"   Median Age: {np.median(ages):.1f}\n")
        f.write(f"   Min Age: {np.min(ages)}\n")
        f.write(f"   Max Age: {np.max(ages)}\n")
        f.write("="*70+"\n")
    print(f"TXT report saved: {report_filename}\n")

# ===== MAIN EXECUTION =====
video_list = get_video_list(video_files)
if not video_list:
    print("No videos to process!")
else:
    print(f"\n{'='*70}")
    print("DEEPFACE BATCH ANALYSIS")
    print(f"{'='*70}")
    for idx, video_path in enumerate(video_list, 1):
        process_video(video_path, idx, len(video_list))