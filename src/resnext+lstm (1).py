import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms

# ===============================
# CONFIGURATION
# ===============================
DATA_ROOT = r"C:\Users\diyus\OneDrive\Desktop\data\train_frames"
SAVE_PATH = os.path.join(DATA_ROOT, "best_resnext_lstm.pth")

CLASSES = ['real', 'fake']
FRAMES_PER_CLIP = 60
IMG_SIZE = 224
LSTM_HIDDEN = 512
LSTM_LAYERS = 2
BIDIR = True
DROPOUT = 0.175
BATCH_SIZE = 8          # ✅ Fixed batch size
EPOCHS = 20
LR = 1e-4
VAL_SPLIT = 0.2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n✅ Using device: {DEVICE}")

# ===============================
# DATASET
# ===============================
class FramesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        print(f"\n🔍 Scanning dataset folder: {root_dir}")
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for cls in CLASSES:
            class_dir = self.root_dir / cls
            if not class_dir.exists():
                print(f"⚠ Warning: Folder not found -> {class_dir}")
                continue
            for vid_folder in sorted(class_dir.iterdir()):
                if not vid_folder.is_dir():
                    continue
                frames = sorted(list(vid_folder.glob("*.jpg")))
                if len(frames) == 0:
                    print(f"⚠ No frames found in {vid_folder}")
                    continue
                self.samples.append((frames, cls))

        print(f"✅ Found {len(self.samples)} video folders total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, cls = self.samples[idx]
        label = CLASSES.index(cls)
        frame_paths = frames[:FRAMES_PER_CLIP]
        while len(frame_paths) < FRAMES_PER_CLIP:
            frame_paths.append(frame_paths[-1])

        imgs = []
        for f in frame_paths:
            img = Image.open(f).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        clip = torch.stack(imgs, dim=0)  # [T,C,H,W]
        return clip, label

# ===============================
# TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ===============================
# MODEL
# ===============================
class ResNeXtLSTM(nn.Module):
    def __init__(self, num_classes, lstm_hidden=512, lstm_layers=2, bidir=True, dropout=0.175):
        super().__init__()
        self.backbone = models.resnext50_32x4d(
            weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        )
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove classification head

        # LSTM head
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

    def forward(self, clips):
        B, T, C, H, W = clips.shape
        x = clips.view(B * T, C, H, W)
        with torch.no_grad():
            feats = self.backbone(x)  # [B*T, D]
        feats = feats.view(B, T, -1)
        seq, _ = self.lstm(feats)
        clip_repr = seq.mean(dim=1)
        logits = self.classifier(clip_repr)
        return logits

# ===============================
# TRAIN / VALIDATION
# ===============================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0

    print(f"\n🚀 Starting epoch {epoch + 1} (Train)...")
    for clips, labels in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
        clips, labels = clips.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    print(f"✅ Train Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    return avg_loss, acc

def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        print(f"\n🔎 Validating epoch {epoch + 1} ...")
        for clips, labels in tqdm(loader, desc="Validating"):
            clips, labels = clips.to(DEVICE), labels.to(DEVICE)
            outputs = model(clips)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total
    print(f"📊 Validation: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    return avg_loss, acc

# ===============================
# MAIN LOOP
# ===============================
def main():
    print("\n🏁 Initializing training...")
    dataset = FramesDataset(DATA_ROOT, transform=transform)
    if len(dataset) == 0:
        print("❌ No data found! Please check DATA_ROOT path or folder structure.")
        return

    # Split dataset
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"📂 Train samples: {train_size}, Validation samples: {val_size}")

    model = ResNeXtLSTM(num_classes=len(CLASSES),
                        lstm_hidden=LSTM_HIDDEN,
                        lstm_layers=LSTM_LAYERS,
                        bidir=BIDIR,
                        dropout=DROPOUT).to(DEVICE)

    # ✅ Fixed batch size (no auto batch finder)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"✅ Using fixed batch size: {BATCH_SIZE}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    best_acc = 0.0

    # Resume training if checkpoint exists
    if os.path.exists(SAVE_PATH):
        print(f"📦 Found existing checkpoint. Loading from {SAVE_PATH}...")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
        print("✅ Checkpoint loaded successfully.")

    print("\n🏋 Starting training loop...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾 Best model saved at {SAVE_PATH} (Val Acc: {val_acc:.4f})")

    print("\n🎉 Training complete!")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")
    print(f"📁 Model saved to: {SAVE_PATH}")

# ===============================
if __name__ == "__main__":
    main()
