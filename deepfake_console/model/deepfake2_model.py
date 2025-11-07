# model/deepfake2_model.py

import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# ===============================
# CONFIGURATION (match training)
# ===============================
CLASSES = ['real', 'fake']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

FRAMES_PER_CLIP = 60     # trained with 60 frames
IMG_SIZE = 224
LSTM_HIDDEN = 512
LSTM_LAYERS = 2
BIDIR = True
DROPOUT = 0.175
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===============================
# TRANSFORMS (match training)
# ===============================
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ===============================
# MODEL (same as training)
# ===============================
class ResNeXtLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=512, lstm_layers=2, bidir=True, dropout=DROPOUT):
        super().__init__()
        self.backbone = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove classification head

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidir,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        head_in = lstm_hidden * (2 if bidir else 1)
        self.classifier = nn.Linear(head_in, num_classes)

    @torch.no_grad()
    def encode_frames(self, x_btchw):
        # x_btchw: [B*T, C, H, W]
        return self.backbone(x_btchw)

    def forward(self, clips):
        """
        clips: [B, T, C, H, W]
        returns: logits [B, num_classes]
        """
        B, T, C, H, W = clips.shape
        x = clips.view(B * T, C, H, W)
        with torch.no_grad():  # freeze backbone at inference (as in your training/inference)
            feats = self.encode_frames(x)  # [B*T, D]
        feats = feats.view(B, T, -1)       # [B, T, D]
        seq, _ = self.lstm(feats)          # [B, T, H]
        clip_repr = seq.mean(dim=1)        # temporal average
        logits = self.classifier(clip_repr)
        return logits

# ===============================
# LOADER HELPERS
# ===============================
def _load_first_k_frames_from_folder(folder_path: str | Path, k: int):
    """
    Load first k frames from a folder of images (face-cropped), pad by repeating the last.
    Folder is expected to contain frame_0000.jpg ... etc.
    """
    folder = Path(folder_path)
    # Accept common image extensions
    frame_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
    if len(frame_paths) == 0:
        return []

    # Take first k frames, pad if needed
    frame_paths = frame_paths[:k]
    while len(frame_paths) < k:
        frame_paths.append(frame_paths[-1])

    frames = []
    for p in frame_paths:
        img = Image.open(p).convert('RGB')
        frames.append(img)
    return frames

def _to_clip_tensor(frames_pil):
    """
    frames_pil: list of PIL images length == FRAMES_PER_CLIP
    returns: torch.FloatTensor [1, T, C, H, W] on DEVICE
    """
    frames = [eval_tf(f) for f in frames_pil]            # each [C, H, W]
    clip = torch.stack(frames, dim=0).unsqueeze(0)       # [1, T, C, H, W]
    return clip.to(DEVICE)

# ===============================
# CHECKPOINT LOAD
# ===============================
def load_model(checkpoint_path: str | Path):
    model = ResNeXtLSTM(
        num_classes=len(CLASSES),
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        bidir=BIDIR,
        dropout=DROPOUT,
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    # If you saved as state_dict only (torch.save(model.state_dict(), ...)):
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        # If you saved a dict like {"model": state_dict, ...}
        state_dict = ckpt.get("model", ckpt)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model

# ===============================
# PREDICT
# ===============================
@torch.no_grad()
def predict_video(frames_folder: str | Path, checkpoint_path: str | Path, threshold: float = 0.5):
    """
    frames_folder: path to folder containing face-cropped frames (e.g., frame_0000.jpg ...).
    checkpoint_path: .pth checkpoint saved from your training loop.
    threshold: probability cutoff for class 'fake' (>= threshold => 'fake').

    Returns:
        {
          "pred_class": "real" | "fake",
          "probs": {"real": float, "fake": float},
          "threshold": float
        }
    """
    # 1) Load frames from folder (first K continuous, pad last)
    frames_pil = _load_first_k_frames_from_folder(frames_folder, FRAMES_PER_CLIP)
    if len(frames_pil) == 0:
        return {"error": f"No frames found in {frames_folder}"}

    # 2) To tensor clip
    clip = _to_clip_tensor(frames_pil)  # [1, T, C, H, W]

    # 3) Load model and run
    model = load_model(checkpoint_path)
    logits = model(clip)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    p_real = float(probs[CLS2IDX['real']])
    p_fake = float(probs[CLS2IDX['fake']])

    pred_class = 'fake' if p_fake >= threshold else 'real'

    return {
        "pred_class": pred_class,
        "probs": {"real": p_real, "fake": p_fake},
        "threshold": float(threshold),
    }
