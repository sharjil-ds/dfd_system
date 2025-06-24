import cv2
import torch
import numpy as np
from PIL import Image
from collections import Counter

# Load model
model = torch.jit.load("efficientnet_b0_deepfake.pt")
model.eval()

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Preprocess face image
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    img = np.asarray(face_img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return torch.tensor(img).unsqueeze(0)

# Run prediction on a video
def predict_video(video_path, conf_threshold=0.7):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = rgb[y:y+h, x:x+w]
            if face.size == 0:
                continue

            face_tensor = preprocess_face(face)

            with torch.no_grad():
                output = model(face_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)

                if conf.item() >= conf_threshold:
                    predictions.append(pred.item())
            break  # Use only one face per frame

    cap.release()

    if len(predictions) == 0:
        print("❗ No confident predictions made.")
        return "Uncertain"

    # Majority vote
    result = Counter(predictions).most_common(1)[0][0]
    label = "Real" if result == 0 else "Fake"
    print(f"✅ Final Prediction: {label} ({len(predictions)} confident faces processed)")
    return label

# Example usage
if __name__ == "__main__":
    video_path = "01__exit_phone_room.mp4"  # Replace with uploaded video path
    result = predict_video(video_path)
