import cv2
import torch
import numpy as np
from insightface.app import FaceAnalysis
from torchvision import transforms
import torch.nn.functional as F
from collections import deque

# --------------------- Setup model ---------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("efficientnet_b0_deepfake.pt", map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------- Setup detector ---------------------

def setup_detector():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

# --------------------- Predict from webcam ---------------------

def predict_webcam(threshold=0.5, window_size=15):
    cap = cv2.VideoCapture(0)
    detector = setup_detector()
    predictions = deque(maxlen=window_size)

    print("🔍 Starting webcam. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.get(frame)
        label = "No face"
        color = (128, 128, 128)

        if faces:
            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                resized = cv2.resize(face_crop, (224, 224))
                input_tensor = transform(resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    prob = F.softmax(output, dim=1)[0, 1].item()
                    predictions.append(prob)

                avg_score = np.mean(predictions)
                if avg_score >= threshold:
                    label = f"Real ({avg_score:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Fake ({avg_score:.2f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display label
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2, cv2.LINE_AA)

        cv2.imshow("Webcam Deepfake Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------- Run ---------------------

if __name__ == "__main__":
    predict_webcam()
