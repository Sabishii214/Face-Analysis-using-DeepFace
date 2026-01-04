import os
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
from collections import deque, Counter
from datetime import datetime
import time
import glob
import threading
from queue import Queue

# CONFIGURATION

VIDEO_FILES = None         # None = webcam, or a path/glob like "videos/*.mp4"
FRAME_SKIP = 4             # Faster updates than RetinaFace
SMOOTHING_WINDOW = 4       # Balanced stability and response
DETECTOR_BACKEND = 'mtcnn' # High precision with turbo optimization

print(f"--- DEEPFACE INITIALIZATION ---")
print(f"Backend Detector: {DETECTOR_BACKEND.upper()}")

# CORE LOGIC

def get_video_list(source):
    """Parses source string, list, or None into a list of video paths."""
    if source is None:
        return [None]
    if isinstance(source, str):
        if '*' in source:
            return sorted(glob.glob(source))
        return [source]
    return source if isinstance(source, list) else []

def analyze_face(frame):
    """Performs DeepFace analysis on a single frame."""
    try:
        # Pre-processing: Apply CLAHE for better detail in shadows (helps with Anger/Disgust)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        frame_enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        results = DeepFace.analyze(
            frame_enhanced,
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False,
            detector_backend=DETECTOR_BACKEND,
            align=True,
            expand_percentage=10,
            silent=True
        )
        res = results[0] if isinstance(results, list) else results
        if not res.get("region"):
            return None
        
        # Emotion Biasing Logic: Boost hard-to-detect emotions, penalize neutral
        emotions = res.get("emotion", {}).copy()
        if emotions:
            weights = {
                "angry": 2.5,
                "disgust": 3.0,
                "fear": 2.5,
                "sad": 2.5,
                "surprise": 2.5,
                "happy": 1.0,
                "neutral": 0.3  # Extreme penalty to force expression detection
            }
            for emo, weight in weights.items():
                if emo in emotions:
                    emotions[emo] *= weight
            
            # Select new dominant emotion based on weighted scores
            dominant_emotion = max(emotions, key=emotions.get)
        else:
            dominant_emotion = res.get("dominant_emotion", "Unknown")

        return (
            res.get("age", 0),
            res.get("dominant_gender", "Unknown"),
            dominant_emotion,
            res.get("dominant_race", "Unknown"),
            res.get("face_confidence", 0),
            res.get("region")
        )
    except Exception:
        return None

def process_video(video_path, video_index=1, total_videos=1):
    """Main processing pipeline for a single video source."""
    # Naming and setup
    if video_path is None:
        video_name = f"Webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"\n[{video_index}/{total_videos}] Processing: {video_name}")
    
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    # Prep output (Skip frames dir for webcam)
    if video_path is not None:
        output_dir = os.path.join("Video_Frames", video_name)
        os.makedirs(output_dir, exist_ok=True)
    
    # Persistent State
    frame_count = 0
    processed_frames = 0
    total_faces_detected = 0
    total_samples = 0
    match_counts = {"Age": 0, "Gender": 0, "Race": 0}
    
    emotion_window = deque(maxlen=SMOOTHING_WINDOW)
    age_window = deque(maxlen=SMOOTHING_WINDOW)
    race_window = deque(maxlen=SMOOTHING_WINDOW)
    confidence_scores = []
    
    # HUD State
    current_region = None
    smoothed_age = "N/A"
    smoothed_emotion = "N/A"
    smoothed_race = "N/A"
    gender = "N/A"
    last_text = "Initializing..."

    # Ground Truth Setup
    session_gt = {"Age": 0, "Gender": "N/A", "Emotion": "N/A", "Race": "N/A"}
    if video_path is None:
        print("\n--- WEBCAM SESSION DATA ---")
        session_gt["Age"] = int(input("True Age: ") or 0)
        session_gt["Gender"] = input("True Gender (Man/Woman): ").strip().capitalize()
        session_gt["Emotion"] = input("True Emotion: ").strip().lower()
        session_gt["Race"] = input("True Race: ").strip().lower()

    # Threading setup
    analysis_queue = Queue(maxsize=1)
    results_shared = {"data": None, "new": False}
    results_lock = threading.Lock()
    stop_event = threading.Event()

    def worker():
        while not stop_event.is_set():
            if not analysis_queue.empty():
                frame_bit, f_no = analysis_queue.get()
                res = analyze_face(frame_bit)
                if res:
                    with results_lock:
                        results_shared["data"] = (res, f_no)
                        results_shared["new"] = True
            else:
                time.sleep(0.01)

    threading.Thread(target=worker, daemon=True).start()

    # Video Setup
    ret, first_frame = cap.read()
    if not ret: return
    h, w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    orig_writer = cv2.VideoWriter(f"{video_name}_original.mp4", fourcc, 20.0, (w, h))
    annot_writer = cv2.VideoWriter(f"{video_name}_annotated.mp4", fourcc, 20.0, (w, h))

    data_log = {
        "Frame": [], "Predicted_Age": [], "True_Age": [],
        "Predicted_Gender": [], "True_Gender": [],
        "Predicted_Emotion": [], "True_Emotion": [],
        "Predicted_Race": [], "True_Race": [],
        "Age_Correct": [], "Gender_Correct": [], "Emotion_Correct": [], "Race_Correct": []
    }

    print("Analysis running... Press 'q' to stop.")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        orig_writer.write(frame)

        # 1. Update Results from background thread
        with results_lock:
            if results_shared["new"]:
                results_shared["new"] = False
                res, f_idx = results_shared["data"]
                age, g_pred, emo, race, conf, reg = res
                
                processed_frames += 1
                total_faces_detected += 1
                confidence_scores.append(conf)

                emotion_window.append(emo)
                age_window.append(age)
                race_window.append(race)
                
                smoothed_emotion = Counter(emotion_window).most_common(1)[0][0]
                smoothed_age = int(sum(age_window)/len(age_window))
                smoothed_race = Counter(race_window).most_common(1)[0][0]
                gender = g_pred
                last_text = f"{gender}, {smoothed_emotion}, {smoothed_age}, {smoothed_race}"

                scale_x, scale_y = w/320, h/240
                current_region = (int(reg['x']*scale_x), int(reg['y']*scale_y), int(reg['w']*scale_x), int(reg['h']*scale_y))

                # GT Comparison
                total_samples += 1
                match_counts["Age"] += (smoothed_age == session_gt["Age"])
                match_counts["Gender"] += (gender == session_gt["Gender"])
                match_counts["Race"] += (smoothed_race == session_gt["Race"])

                # Logging
                data_log["Frame"].append(f_idx)
                data_log["Predicted_Age"].append(smoothed_age)
                data_log["True_Age"].append(session_gt["Age"])
                data_log["Predicted_Gender"].append(gender)
                data_log["True_Gender"].append(session_gt["Gender"])
                data_log["Predicted_Emotion"].append(smoothed_emotion)
                data_log["True_Emotion"].append(session_gt["Emotion"])
                data_log["Predicted_Race"].append(smoothed_race)
                data_log["True_Race"].append(session_gt["Race"])
                data_log["Age_Correct"].append(smoothed_age == session_gt["Age"])
                data_log["Gender_Correct"].append(gender == session_gt["Gender"])
                data_log["Emotion_Correct"].append(smoothed_emotion == session_gt["Emotion"])
                data_log["Race_Correct"].append(smoothed_race == session_gt["Race"])

        # 2. Trigger new analysis if worker is ready
        if analysis_queue.empty() and frame_count % FRAME_SKIP == 0:
            analysis_queue.put((cv2.resize(frame, (320, 240)), frame_count))

        # 3. Draw UI
        annot_frame = frame.copy()
        if current_region:
            rx, ry, rw, rh = current_region
            cv2.rectangle(annot_frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            cv2.putText(annot_frame, last_text, (rx, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Translucent Panels
        overlay = annot_frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 150), (0, 0, 0), -1) # Left
        right_panel_x = w - 190
        cv2.rectangle(overlay, (right_panel_x, 10), (w-10, 165), (0, 0, 0), -1) # Right
        cv2.addWeighted(overlay, 0.1, annot_frame, 0.9, 0, annot_frame)

        # HUD Text
        cv2.putText(annot_frame, "GROUND TRUTH", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for i, (k, v) in enumerate(session_gt.items()):
            cv2.putText(annot_frame, f"{k}: {v}", (20, 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(annot_frame, "PREDICTION", (right_panel_x + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if smoothed_age != "N/A":
            p_vals = [smoothed_age, gender, smoothed_emotion, smoothed_race]
            p_keys = ["Age", "Gender", "Emotion", "Race"]
            for i, val in enumerate(p_vals):
                cv2.putText(annot_frame, f"{p_keys[i]}: {val}", (right_panel_x + 10, 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(annot_frame, "Initializing...", (right_panel_x + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        annot_writer.write(annot_frame)
        cv2.imshow("DeepFace Analysis", annot_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Cleanup
    stop_event.set()
    cap.release()
    orig_writer.release()
    annot_writer.release()
    cv2.destroyAllWindows()

    # Save Reports
    if not data_log["Frame"]: return # Nothing to save
    
    df = pd.DataFrame(data_log)
    csv_filename = f"{video_name}-results.csv"
    report_filename = f"{video_name}-facial-report.txt"
    df.to_csv(csv_filename, index=False)
    
    total_time = time.time() - start_time
    avg_fps = processed_frames/total_time if total_time > 0 else 0
    avg_conf = np.mean(confidence_scores) if confidence_scores else 0
    std_conf = np.std(confidence_scores) if confidence_scores else 0
    high_conf = sum(c>=0.7 for c in confidence_scores)
    med_conf = sum((c>=0.4 and c<0.7) for c in confidence_scores)
    low_conf = sum(c<0.4 for c in confidence_scores)
    
    with open(report_filename, 'w') as f:
        f.write("="*70 + f"\nPERFORMANCE REPORT: {video_name}\n" + "="*70 + "\n\n")
        f.write("SPEED METRICS:\n")
        f.write(f"   Total Frames: {frame_count}\n")
        f.write(f"   Processed Frames: {processed_frames}\n")
        f.write(f"   Total Time: {total_time:.1f}s\n")
        f.write(f"   Average FPS: {avg_fps:.2f}\n\n")
        
        f.write("DETECTION METRICS:\n")
        f.write(f"   Total Faces Detected: {total_faces_detected}\n")
        f.write(f"   Avg Faces/Processed Frame: {total_faces_detected/processed_frames if processed_frames else 0:.2f}\n")
        f.write(f"   Avg Confidence: {avg_conf:.3f} ({avg_conf*100:.1f}%)\n")
        f.write(f"   Confidence Std Dev: {std_conf:.3f}\n\n")
        
        f.write("CONFIDENCE DISTRIBUTION:\n")
        if processed_frames > 0:
            f.write(f"   High (>=0.7): {high_conf} ({high_conf/processed_frames*100:.1f}%)\n")
            f.write(f"   Medium (0.4-0.7): {med_conf} ({med_conf/processed_frames*100:.1f}%)\n")
            f.write(f"   Low (<0.4): {low_conf} ({low_conf/processed_frames*100:.1f}%)\n\n")
        else:
            f.write("   No frames processed.\n\n")

        f.write("TOP DETECTED EMOTIONS:\n")
        counts = Counter(data_log["Predicted_Emotion"])
        for e, c in counts.most_common():
            f.write(f"  {e:<12}: {c:3} ({c/len(data_log['Frame'])*100:.1f}%)\n")
            
        f.write("\nGENDER DISTRIBUTION:\n")
        g_counts = Counter(data_log["Predicted_Gender"])
        for g, c in g_counts.items():
            f.write(f"  {g:<12}: {c:3} ({c/len(data_log['Frame'])*100:.1f}%)\n")
            
        f.write("\nRACE DISTRIBUTION:\n")
        r_counts = Counter(data_log["Predicted_Race"])
        for r, c in r_counts.items():
            f.write(f"  {r:<12}: {c:3} ({c/len(data_log['Frame'])*100:.1f}%)\n")
            
        f.write("\nAGE STATISTICS:\n")
        ages = np.array(data_log["Predicted_Age"])
        f.write(f"  Average Age: {np.mean(ages):.1f}\n")
        f.write(f"  Median Age: {np.median(ages):.1f}\n")
        f.write(f"  Min Age: {np.min(ages)}\n")
        f.write(f"  Max Age: {np.max(ages)}\n\n")
            
        f.write(f"ACCURACY (GT Based):\n")
        f.write(f"  Overall (3-pt): {(sum(match_counts.values()) / (3 * total_samples)) * 100 if total_samples > 0 else 0:.1f}%\n")
        f.write(f"  Gender Accuracy: {match_counts['Gender']/total_samples*100 if total_samples > 0 else 0:.1f}%\n")
        f.write(f"  Age MAE: {np.mean(np.abs(np.array(data_log['Predicted_Age']) - np.array(data_log['True_Age']))):.2f} yrs\n")
        f.write("="*70 + "\n")
    print(f"Results saved: {csv_filename}")
    print(f"Report saved: {report_filename}")

# Run
if __name__ == "__main__":
    v_list = get_video_list(VIDEO_FILES)
    if v_list:
        print(f"\n{'='*70}\nBATCH ANALYSIS START\n{'='*70}")
        for idx, path in enumerate(v_list, 1):
            process_video(path, idx, len(v_list))
