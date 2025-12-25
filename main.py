import os
import cv2
import pandas as pd
from deepface import DeepFace
from collections import deque, Counter

# ===== SETTINGS =====
video_path = None  # set to None for webcam
frame_skip = 10
smoothing_window = 10

# ===== VIDEO SOURCE =====
cap = cv2.VideoCapture(0 if video_path is None else video_path)
frame_count = 0

# Smoothing windows
emotion_window = deque(maxlen=smoothing_window)
age_window = deque(maxlen=smoothing_window)
race_window = deque(maxlen=smoothing_window)

last_text = "Loading..."

# Data storage
data = {
    "Frame": [],
    "Age": [],
    "Gender": [],
    "Emotion": [],
    "Race": []
}

def analyze_face(frame):
    
    try:
        result = DeepFace.analyze(frame, actions=("age", "gender", "emotion", "race"), enforce_detection=True)
        r = result[0] if isinstance(result, list) else result
        age = r.get("age", 0)
        gender = r.get("dominant_gender", "Unknown")
        emotion = r.get("dominant_emotion", "Unknown")
        race = r.get("dominant_race", "Unknown")
        return age, gender, emotion, race
    except Exception as e:
        print(f"Frame {frame_count} skipped: {e}")
        return None, None, None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_small = cv2.resize(frame, (640, 480))

    # Analyze only every N frames
    if frame_count % frame_skip == 0:
        age, gender, emotion, race = analyze_face(frame_small)

        if age is not None:
            # Add to smoothing windows
            emotion_window.append(emotion)
            age_window.append(age)
            race_window.append(race)

            # Compute smoothed values
            smoothed_emotion = emotion
            smoothed_age = int(sum(age_window) / len(age_window))
            smoothed_race = Counter(race_window).most_common(1)[0][0]

            last_text = f"{gender}, {smoothed_emotion}, {smoothed_age}, {smoothed_race}"

            # Save data
            data["Frame"].append(frame_count)
            data["Age"].append(smoothed_age)
            data["Gender"].append(gender)
            data["Emotion"].append(smoothed_emotion)
            data["Race"].append(smoothed_race)

    # Draw last known result
    cv2.putText(frame, last_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("DeepFace Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save results
df = pd.DataFrame(data)
df.to_csv("analysis_results.csv", index=False)
print("Saved results to analysis_results.csv")
