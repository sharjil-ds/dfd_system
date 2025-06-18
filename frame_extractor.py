import os
import cv2

def extract_frames_from_video(video_path, output_dir, fps=1):
    """Extract frames from video at specified FPS"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, video_name)
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(save_path, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[DONE] Extracted {saved_idx} frames from {video_name}")


def process_videos(input_dir, output_dir, fps=1):
    """Process all .mp4 videos in a directory"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            extract_frames_from_video(video_path, output_dir, fps=fps)


if __name__ == "__main__":
    # Customize these paths
    INPUT_DIR = "dataset/fake"   # or "dataset/fake"
    OUTPUT_DIR = "extracted_frames/fake"
    FPS = 1  # Extract 1 frame per second

    process_videos(INPUT_DIR, OUTPUT_DIR, fps=FPS)
