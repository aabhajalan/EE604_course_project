import tempfile      # to create temporary files
import os            # for file path operations
from model.deepfake2_model import predict_video  # function to predict using the model
from model.face_extractor import extract_face_video  # function to extract face frames

def analyze_video(video_file):
    checkpoint_path = "model/weights/best_resnext_lstm.pth"  # weight is saved here

    # uploaded video is saved temporarily for further processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        tmp_path = tmp.name

    # uses the video saved temporarily to extract face frames
    face_frames_folder = extract_face_video(tmp_path, num_frames=60)
    if face_frames_folder is None or not os.listdir(face_frames_folder):
        return {"error": "Face extraction failed — no faces detected."}

    # uses the model to predict on the extracted face frames
    result = predict_video(face_frames_folder, checkpoint_path, threshold=0.5)

    if "error" in result:
        return result

    return {
        "real": round(result["probs"]["real"] * 100, 2),
        "fake": round(result["probs"]["fake"] * 100, 2),
        "pred_class": result["pred_class"]
    }
