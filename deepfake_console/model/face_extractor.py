# model/face_extractor.py

import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm


def extract_faces_from_video(input_video_path, output_folder, num_frames=60, mtcnn=None):
    """
    Extracts faces from the first `num_frames` of a video and saves them as cropped image files.
    Returns the folder path containing the cropped face frames.
    """
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"⚠ Skipping {os.path.basename(input_video_path)} (no frames).")
        return None

    # === Take first N continuous frames ===
    frame_indices = np.arange(0, min(num_frames, total_frames)).astype(int)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    face_count = 0

    print(f"🎬 Extracting faces from {os.path.basename(input_video_path)} ({len(frame_indices)} frames)")
    for idx in tqdm(frame_indices, desc="Frames processed", ncols=80):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        face = mtcnn(img)
        if face is not None:
            # Convert tensor -> uint8 BGR image
            face_img = ((face.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)

            # Save cropped frame
            save_path = os.path.join(output_folder, f"frame_{face_count:04d}.jpg")
            cv2.imwrite(save_path, face_bgr)
            face_count += 1

    cap.release()

    if face_count == 0:
        print(f"❌ No faces detected in {os.path.basename(input_video_path)}")
        return None

    print(f"✅ Saved {face_count} cropped face frames in: {output_folder}\n")
    return output_folder


def initialize_mtcnn():
    """Safely initializes MTCNN on GPU if available, otherwise CPU."""
    try:
        mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=True, device='cuda')
        print("✅ Using GPU for MTCNN.")
    except Exception:
        print("⚠ GPU not available, using CPU.")
        mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=True, device='cpu')
    return mtcnn


def extract_face_video(input_video_path, output_dir="temp_faces", num_frames=60):
    """
    Wrapper used by inference.py — extracts faces and returns path to folder of face frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(input_video_path))[0]
    output_folder = os.path.join(output_dir, f"{filename}_faces")

    mtcnn = initialize_mtcnn()
    face_folder = extract_faces_from_video(
        input_video_path, output_folder, num_frames=num_frames, mtcnn=mtcnn
    )

    return face_folder


