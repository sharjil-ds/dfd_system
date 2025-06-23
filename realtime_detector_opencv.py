import cv2
import torch
import numpy as np
from PIL import Image
from collections import deque
from statistics import mode

# Load model
model = torch.jit.load("efficientnet_b0_deepfake.pt")
model.eval()

# OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Rolling window for smoothing
prediction_window = deque(maxlen=20)  # ~1 second at 20 FPS

# Preprocess face
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    img = np.asarray(face_img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return torch.tensor(img).unsqueeze(0)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    display_label = "No Face"
    display_color = (200, 200, 200)

    for (x, y, w, h) in faces:
        face = rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face_tensor = preprocess_face(face)

        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

            confidence = conf.item()
            prediction = pred.item()

            # Store prediction if confident enough
            if confidence >= 0.7:
                prediction_window.append(prediction)

        break  # Only process one face per frame for speed

    # Majority vote
    if len(prediction_window) > 0:
        try:
            final_label = mode(prediction_window)
            if final_label == 0:
                display_label = "Real"
                display_color = (0, 255, 0)
            else:
                display_label = "Fake"
                display_color = (0, 0, 255)
        except:
            display_label = "Uncertain"
            display_color = (255, 255, 0)

    # Draw label on screen
    cv2.putText(frame, f"{display_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, display_color, 3)

    cv2.imshow("Real-Time Deepfake Detector (Smoothed)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
