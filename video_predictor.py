import os
import cv2
import torch
import numpy as np
from insightface.app import FaceAnalysis
from torchvision import transforms
import torch.nn.functional as F
import warnings

# --------------------- Setup model ---------------------
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("efficientnet_b0_deepfake.pt", map_location=device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# --------------------- Setup detector ---------------------

def setup_detector():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))  # Match training face crop resolution
    return app

# --------------------- Predict video ---------------------

def predict_video(video_path, conf_thresh=0.3, real_threshold=0.80):
    cap = cv2.VideoCapture(video_path)
    detector = setup_detector()
    predictions = []
    frame_count = 0
    valid_frames = 0

    os.makedirs("debug_faces", exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) if fps > 0 else 1  # 1 frame per second

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        frame_count += 1
        faces = detector.get(frame)
        if not faces:
            continue

        # Pick face with best detection confidence
        face = max(faces, key=lambda f: f.det_score)
        if face.det_score < conf_thresh:
            continue

        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        resized = cv2.resize(face_crop, (224, 224))
        cv2.imwrite(f"debug_faces/face_{frame_count:06d}.jpg", resized)

        input_tensor = transform(resized).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob_real = F.softmax(output, dim=1)[0, 1].item()  # index 1 = real
            predictions.append(prob_real)
            valid_frames += 1
            print(f"[Frame {frame_count}] det_score={face.det_score:.2f} → Prob (Real): {prob_real:.4f}")

    cap.release()

    if not predictions:
        print("❌ No valid face predictions.")
        return "Unknown"

    avg_real = np.mean(predictions)
    label = "Real" if avg_real >= real_threshold else "Fake"

    print(f"\n✅ Final Prediction: {label}")
    print(f"→ Average Real Confidence: {avg_real:.4f}")
    print(f"→ Used {valid_frames} frames (1 per second).")
    return label

# --------------------- Entry point ---------------------

if __name__ == "__main__":
    video_path = "model_testing/demo_test_video.mp4"  # Replace with your test video path
    predict_video(video_path)
