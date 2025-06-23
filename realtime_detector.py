import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# ===== Load TorchScript Model =====
model = torch.jit.load("efficientnet_b0_deepfake.pt")
model.eval()

# ===== Face Detector =====
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

# ===== Manual Preprocessing =====
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (224, 224))
    face_img = Image.fromarray(face_img)
    img = np.asarray(face_img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    return torch.tensor(img).unsqueeze(0)

# ===== Webcam Inference =====
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb_frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            face = rgb_frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_tensor = preprocess_face(face)

            with torch.no_grad():
                output = model(face_tensor)
                pred = torch.argmax(output, dim=1).item()
                label = "Real" if pred == 0 else "Fake"
                color = (0, 255, 0) if pred == 0 else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv
