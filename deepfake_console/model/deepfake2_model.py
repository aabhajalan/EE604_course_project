import os                        # for file path operations
from pathlib import Path         # for handling file paths
import numpy as np               # for numerical operations
from PIL import Image            # for image processing

import torch                     # for deep learning operations
import torch.nn as nn            # for neural network modules
from torchvision import models, transforms   # for pre-trained models and data transformations

# this ensures that testing pipeline is similiar to training pipeline
# so that the model performs well during inference as it did during training
CLASSES = ['real', 'fake']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}

FRAMES_PER_CLIP = 60   
IMG_SIZE = 224
LSTM_HIDDEN = 512
LSTM_LAYERS = 2
BIDIR = True
DROPOUT = 0.175
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' #using GPU if available

# resized and converted tensor and normalized as per ImageNet standards
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# model definition
class ResNeXtLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=512, lstm_layers=2, bidir=True, dropout=DROPOUT):
        super().__init__()
        self.backbone = models.resnext50_32x4d(                  #starting with ResNeXt50 backbone
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove final classification layer leaving feature extractor
        # temporal sequence modeling with LSTM which learns features across frames
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidir,       # enables bidirectional LSTM
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        # after LSTM, a FC layer classifies the video clip as real or fake
        head_in = lstm_hidden * (2 if bidir else 1)
        self.classifier = nn.Linear(head_in, num_classes)

    @torch.no_grad()
    def encode_frames(self, x_btchw):   # x_btchw: [B*T, C, H, W]
        return self.backbone(x_btchw)
    # flattens batch and extract CNN features , reshape and pass through LSTM and classifier
    def forward(self, clips):
 
        B, T, C, H, W = clips.shape
        x = clips.view(B * T, C, H, W)
        with torch.no_grad():  # freeze backbone at inference (as in your training/inference)
            feats = self.encode_frames(x)  # [B*T, D]
        feats = feats.view(B, T, -1)       # [B, T, D]
        seq, _ = self.lstm(feats)          # [B, T, H]
        clip_repr = seq.mean(dim=1)        # temporal average
        logits = self.classifier(clip_repr)
        return logits

# keeps the input frames consistent during inference
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
# applies preprocessing transforms and converts to tensor
def _to_clip_tensor(frames_pil):
    """
    frames_pil: list of PIL images length == FRAMES_PER_CLIP
    returns: torch.FloatTensor [1, T, C, H, W] on DEVICE
    """
    frames = [eval_tf(f) for f in frames_pil]            # each [C, H, W]
    clip = torch.stack(frames, dim=0).unsqueeze(0)       # [1, T, C, H, W]
    return clip.to(DEVICE)

# loads the model from checkpoint
def load_model(checkpoint_path: str | Path):
    model = ResNeXtLSTM(
        num_classes=len(CLASSES),
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        bidir=BIDIR,
        dropout=DROPOUT,
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    # distinguishes between state_dict(raw) and full checkpoint(wrapped)
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        state_dict = ckpt.get("model", ckpt)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
#frames_folder: path to folder containing face-cropped frames (e.g., frame_0000.jpg ...).
#checkpoint_path: .pth checkpoint saved from your training loop.
#threshold: probability cutoff for class 'fake' (>= threshold => 'fake').
def predict_video(frames_folder: str | Path, checkpoint_path: str | Path, threshold: float = 0.5):
  
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

