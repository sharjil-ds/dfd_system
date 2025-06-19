import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

def setup_detector():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))  # Set detection resolution
    return app

def detect_and_crop_faces(image_path, detector):
    image = cv2.imread(image_path)
    if image is None:
        return []

    faces = detector.get(image)
    cropped_faces = []

    for i, face in enumerate(faces):
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        resized = cv2.resize(face_crop, (224, 224))
        cropped_faces.append(resized)

    return cropped_faces

def process_frames(input_dir, output_dir):
    detector = setup_detector()
    os.makedirs(output_dir, exist_ok=True)

    for video_folder in tqdm(os.listdir(input_dir), desc="Processing videos"):
        video_path = os.path.join(input_dir, video_folder)
        if not os.path.isdir(video_path):
            continue

        save_video_folder = os.path.join(output_dir, video_folder)
        os.makedirs(save_video_folder, exist_ok=True)

        for frame_name in os.listdir(video_path):
            frame_path = os.path.join(video_path, frame_name)
            cropped_faces = detect_and_crop_faces(frame_path, detector)

            for i, face in enumerate(cropped_faces[:1]):  # Only keep 1 face per frame
                save_path = os.path.join(save_video_folder, f"{frame_name[:-4]}_face_{i}.jpg")
                cv2.imwrite(save_path, face)

if __name__ == "__main__":
    INPUT_DIR = "extracted_frames/real"    # or fake
    OUTPUT_DIR = "cropped_faces/real"

    process_frames(INPUT_DIR, OUTPUT_DIR)
