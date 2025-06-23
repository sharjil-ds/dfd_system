import cv2
import torch
import numpy as np
from retinaface import RetinaFace
from PIL import Image

# Load TorchScript model
model = torch.jit.load("efficientnet_b0_deepfake.pt")
model.eval()

# RetinaFace detector (uses ONNX backend)
detector = RetinaFace(quality="normal")  # Uses ResNet50 ONNX model

# Preprocess face
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    img = np.asarray(face_img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return torch.tensor(img).unsqueeze(0)

# Webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RetinaFace expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.predict(rgb_frame)

    for face in faces:
        x1, y1, x2, y2 = [int(coord) for coord in face['bbox']]
        face_crop = rgb_frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        face_tensor = preprocess_face(face_crop)

        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

            if conf.item() < 0.7:
                label = "Uncertain"
                color = (255, 255, 0)
            else:
                label = "Real" if pred.item() == 0 else "Fake"
                color = (0, 255, 0) if pred.item() == 0 else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf.item():.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("RetinaFace Deepfake Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
