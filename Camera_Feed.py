import cv2
import os
import csv
import pickle
import face_recognition
from datetime import datetime

with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

video = cv2.VideoCapture(0)
CPI_dir = "People_Inside"
os.makedirs(CPI_dir, exist_ok=True)
threshold = 0.6

print("Face recognition started...")
print("Press SPACE to log entry, BACKSPACE to log exit, Q to quit.")

def ensure_csv_header(path):
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Entry Time", "Exit Time"])

def has_open_entry(csv_path, name):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] == name:
                if len(row) < 3 or row[2].strip() == "":
                    return True
    return False

def add_entry(csv_path, name, time_str):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, time_str, ""])

def close_most_recent_open_entry(csv_path, name, time_str):
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.reader(f))

    for i in range(len(rows)-1, 0, -1):
        row = rows[i]
        if len(row) >= 2 and row[0] == name:
            if len(row) < 3 or row[2].strip() == "":
                if len(row) < 3:
                    row.extend([""] * (3 - len(row)))
                rows[i][2] = time_str
                with open(csv_path, 'w', newline='') as fw:
                    writer = csv.writer(fw)
                    writer.writerows(rows)
                return True
    return False

while True:
    ret, frame = video.read()
    if not ret:
        print("❌ Camera read error.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)
    date = datetime.now().strftime("%d-%m-%Y")
    csv_path = os.path.join(CPI_dir, f"People_Inside_{date}.csv")
    ensure_csv_header(csv_path)
    key = cv2.waitKey(1) & 0xFF

    for (top, right, bottom, left), face_enc in zip(faces, encodings):
        probs = model.predict_proba([face_enc])[0]
        max_prob = max(probs)
        predicted_name = model.classes_[probs.argmax()] if max_prob > threshold else "Unknown"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top - 30), (right, top), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, predicted_name, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        current_time = datetime.now().strftime("%H:%M:%S")

        if predicted_name != "Unknown" and key == 32:  # SPACE pressed
            if has_open_entry(csv_path, predicted_name):
                print(f"⚠️ Entry already open for {predicted_name}; not adding another entry.")
            else:
                add_entry(csv_path, predicted_name, current_time)
                print(f"✅ Entry logged for {predicted_name} at {current_time}")

        elif predicted_name != "Unknown" and key == 8:  # BACKSPACE pressed
            updated = close_most_recent_open_entry(csv_path, predicted_name, current_time)
            if updated:
                print(f"👋 Exit logged for {predicted_name} at {current_time}")
            else:
                print(f"⚠️ No open entry found for {predicted_name} to exit.")

    cv2.imshow("CAMERA FEED", frame)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
