import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from collections import deque, Counter
from datetime import datetime
import time
import glob

# video_files = "videos/*"      # All video files in videos folder
# Webcam = None
video_files = "D:/VS Code/Deepface/Faces/Facial.mp4"

frame_skip = 10
smoothing_window = 10

# Performance settings
DETECTOR_BACKEND = 'opencv'  # Faster: opencv, ssd, mtcnn | Slower but accurate: retinaface
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to accept prediction

print("Initializing DeepFace models...")
print(f"Using detector: {DETECTOR_BACKEND}")

# ===== PROCESS VIDEO FILES =====
def get_video_list(video_files):
    """Convert video_files input to a list of video paths"""
    if video_files is None:
        return [None]  # Webcam
    elif isinstance(video_files, str):
        if '*' in video_files:
            # Glob pattern
            files = glob.glob(video_files)
            if not files:
                print(f"No files found matching pattern: {video_files}")
                return []
            return sorted(files)
        else:
            # Single file
            return [video_files]
    elif isinstance(video_files, list):
        return video_files
    else:
        return []

def process_video(video_path, video_index=1, total_videos=1):
    """Process a single video and generate report"""

    # Video name for report
    if video_path is None:
        # Webcam
        date_prefix = datetime.now().strftime("%Y%m%d")
        video_name = f"Webcam_{date_prefix}"
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

    print("\n" + "=" * 70)
    print(f"Processing video {video_index}/{total_videos}: {video_name}")
    print("=" * 70)

    # Open video
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Initialize counters
    frame_count = 0
    processed_frames = 0
    total_faces_detected = 0

    # Smoothing windows
    emotion_window = deque(maxlen=smoothing_window)
    age_window = deque(maxlen=smoothing_window)
    race_window = deque(maxlen=smoothing_window)
    last_text = "Loading..."

    # Performance tracking
    inference_times = []
    confidence_scores = []
    start_time = time.time()

    # Detection counters
    emotion_counter = Counter()
    gender_counter = Counter()
    race_counter = Counter()
    age_list = []

    # Data storage
    data = {
        "Frame": [],
        "Age": [],
        "Gender": [],
        "Emotion": [],
        "Race": [],
        "Confidence": [],
        "InferenceTime": []
    }

    def analyze_face(frame):
        """
        Uses DeepFace library for facial analysis
        DeepFace analyzes: age, gender, emotion, and race
        """
        try:
            inference_start = time.time()

            result = DeepFace.analyze(
                frame,                              # Input image/frame
                actions=["age", "gender", "emotion", "race"],  # What to analyze
                enforce_detection=False,            # Don't crash if no face found
                detector_backend=DETECTOR_BACKEND,  # Face detection method
                silent=True                         # No verbose output
            )

            inference_time = (time.time() - inference_start) * 1000
            r = result[0] if isinstance(result, list) else result

            # Check if face was actually detected
            if not r.get("region"):
                return None

            # Extract DeepFace results
            age = r.get("age", 0)                          # Age prediction
            gender = r.get("dominant_gender", "Unknown")   # Male/Female
            emotion = r.get("dominant_emotion", "Unknown") # Happy, Sad, Angry, etc.
            race = r.get("dominant_race", "Unknown")       # Ethnicity prediction

            # Get confidence from emotion scores
            emotion_scores = r.get("emotion", {})
            confidence = emotion_scores.get(emotion, 0) / 100.0 if emotion_scores else 0

            return age, gender, emotion, race, confidence, inference_time
        except Exception:
            return None

    print("Starting analysis... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_small = cv2.resize(frame, (640, 480))

        # Analyze only every N frames
        if frame_count % frame_skip == 0:
            result = analyze_face(frame_small)

            if result:
                age, gender, emotion, race, confidence, inference_time = result

                # Apply confidence threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    processed_frames += 1
                    total_faces_detected += 1

                    # Track performance
                    inference_times.append(inference_time)
                    confidence_scores.append(confidence)

                    # Add to smoothing windows
                    emotion_window.append(emotion)
                    age_window.append(age)
                    race_window.append(race)

                    # Compute smoothed values
                    smoothed_emotion = Counter(emotion_window).most_common(1)[0][0]
                    smoothed_age = int(sum(age_window) / len(age_window))
                    smoothed_race = Counter(race_window).most_common(1)[0][0]

                    last_text = f"{gender}, {smoothed_emotion}, {smoothed_age}, {smoothed_race}"

                    # Update counters
                    emotion_counter[smoothed_emotion] += 1
                    gender_counter[gender] += 1
                    race_counter[smoothed_race] += 1
                    age_list.append(smoothed_age)

                    # Save data
                    data["Frame"].append(frame_count)
                    data["Age"].append(smoothed_age)
                    data["Gender"].append(gender)
                    data["Emotion"].append(smoothed_emotion)
                    data["Race"].append(smoothed_race)
                    data["Confidence"].append(confidence)
                    data["InferenceTime"].append(inference_time)

        # Draw last known result
        cv2.putText(frame, last_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("DeepFace Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopping analysis...")
            break

    cap.release()
    cv2.destroyAllWindows()

    # ===== GENERATE REPORT =====
    total_time = time.time() - start_time
    avg_fps = processed_frames / total_time if total_time > 0 else 0

    # Calculate statistics
    avg_inference = np.mean(inference_times) if inference_times else 0
    std_inference = np.std(inference_times) if inference_times else 0
    min_inference = np.min(inference_times) if inference_times else 0
    max_inference = np.max(inference_times) if inference_times else 0

    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    std_confidence = np.std(confidence_scores) if confidence_scores else 0

    high_conf = sum(1 for c in confidence_scores if c >= 0.7)
    med_conf = sum(1 for c in confidence_scores if 0.4 <= c < 0.7)
    low_conf = sum(1 for c in confidence_scores if c < 0.4)
    total_conf = len(confidence_scores)

    report_filename = f"{video_name}-report.txt"
    csv_filename = f"{video_name}-results.csv"

    # Write report
    with open(report_filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"PERFORMANCE REPORT - {video_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SPEED METRICS:\n")
        f.write(f"   Total Frames: {frame_count}\n")
        f.write(f"   Processed Frames: {processed_frames}\n")
        f.write(f"   Total Time: {total_time:.2f}s\n")
        f.write(f"   Average FPS: {avg_fps:.2f}\n")
        f.write(f"   Avg Inference: {avg_inference:.2f}ms +/- {std_inference:.2f}ms\n")
        f.write(f"   Min Inference: {min_inference:.2f}ms\n")
        f.write(f"   Max Inference: {max_inference:.2f}ms\n\n")

        f.write("DETECTION METRICS:\n")
        f.write(f"   Total Faces Detected: {total_faces_detected}\n")
        if processed_frames > 0:
            f.write(f"   Avg Faces/Processed Frame: {total_faces_detected/processed_frames:.2f}\n")
        if confidence_scores:
            f.write(f"   Avg Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)\n")
            f.write(f"   Confidence Std Dev: {std_confidence:.3f}\n\n")

        if confidence_scores:
            f.write("CONFIDENCE DISTRIBUTION:\n")
            f.write(f"   High (>=0.7): {high_conf} ({high_conf/total_conf*100:.1f}%)\n")
            f.write(f"   Medium (0.4-0.7): {med_conf} ({med_conf/total_conf*100:.1f}%)\n")
            f.write(f"   Low (<0.4): {low_conf} ({low_conf/total_conf*100:.1f}%)\n\n")

        f.write("TOP DETECTED EMOTIONS:\n")
        for emotion, count in emotion_counter.most_common(5):
            pct = count / processed_frames * 100 if processed_frames > 0 else 0
            f.write(f"   {emotion:15s}: {count:5d} ({pct:.1f}%)\n")

        f.write("\nGENDER DISTRIBUTION:\n")
        for gender, count in gender_counter.most_common():
            pct = count / processed_frames * 100 if processed_frames > 0 else 0
            f.write(f"   {gender:15s}: {count:5d} ({pct:.1f}%)\n")

        f.write("\nRACE DISTRIBUTION:\n")
        for race, count in race_counter.most_common():
            pct = count / processed_frames * 100 if processed_frames > 0 else 0
            f.write(f"   {race:15s}: {count:5d} ({pct:.1f}%)\n")

        if age_list:
            f.write("\nAGE STATISTICS:\n")
            f.write(f"   Average Age: {np.mean(age_list):.1f}\n")
            f.write(f"   Median Age: {np.median(age_list):.1f}\n")
            f.write(f"   Min Age: {np.min(age_list)}\n")
            f.write(f"   Max Age: {np.max(age_list)}\n")

        f.write("\n" + "=" * 70 + "\n")

    # Save CSV results
    pd.DataFrame(data).to_csv(csv_filename, index=False)

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
