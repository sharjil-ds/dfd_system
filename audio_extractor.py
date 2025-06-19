import os
import subprocess
import librosa
import numpy as np


def has_audio_stream(video_path):
    command = ["ffprobe", "-loglevel", "error", "-show_streams", "-select_streams", "a", video_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout.strip())


def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    if not has_audio_stream(video_path):
        print(f"[!] No audio stream found in {video_path}")
        return False

    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"[FFMPEG ERROR] Could not extract audio from {video_path}")
        print(result.stderr.decode())
        return False

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        print(f"[!] Audio file not created or empty: {audio_path}")
        return False

    return True


def extract_mfcc(audio_path, max_pad_len=200):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc
    except Exception as e:
        print(f"[!] Error extracting MFCC from {audio_path}: {e}")
        return None


def process_dataset(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(video_dir):
        if not file.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        video_path = os.path.join(video_dir, file)
        base_name = os.path.splitext(file)[0]

        # Temporary audio path
        temp_audio = os.path.join(output_dir, base_name + ".wav")

        if not extract_audio(video_path, temp_audio):
            continue  # Skip if audio failed

        mfcc = extract_mfcc(temp_audio)
        if mfcc is not None:
            npy_path = os.path.join(output_dir, base_name + ".npy")
            np.save(npy_path, mfcc)
            print(f"[✓] Saved MFCC: {npy_path}")
        else:
            print(f"[x] Skipped due to MFCC error: {base_name}")

        # Optional: clean up wav files
        if os.path.exists(temp_audio):
            os.remove(temp_audio)


# Run this script to process both sets
if __name__ == "__main__":
    process_dataset("dataset/real", "audio_features/real")
    process_dataset("dataset/fake", "audio_features/fake")
