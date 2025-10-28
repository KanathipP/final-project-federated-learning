import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mne
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

device = "cuda:0" if torch.cuda.is_available() else "cpu"

in_channel = 22
num_classes = 4 # classes (เช่น 5,6,7,8 -> 0..3)

from pathlib import Path
import importlib
import sys
MODEL_DIR = Path("../../../2-models")

def get_eegnet_pytorch_model():
    target_dir = MODEL_DIR.resolve()
    if str(target_dir) not in sys.path:
        sys.path.append(str(target_dir))

    import eegnet_pytorch 
    importlib.reload(eegnet_pytorch)

    return eegnet_pytorch.EEGNet

EEGNet = get_eegnet_pytorch_model()

class Net(EEGNet):
    def __init__(self):
        super().__init__(in_channel=in_channel,num_classes=num_classes)


# =========================
# Utils
# =========================
class StandardScaler3D(BaseEstimator, TransformerMixin):
    """สเกลแบบต่อ-channel บนข้อมูลรูป (N, T, C)"""

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.reshape(-1, X.shape[2]))
        return self

    def transform(self, X):
        return self.scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)


def remap_np(y: torch.Tensor) -> torch.Tensor:
    y_np = y.view(-1).cpu().numpy()
    uniq = np.unique(y_np)  # เช่น [5,6,7,8]
    lut = {u: i for i, u in enumerate(uniq)}  # {5:0, 6:1, 7:2, 8:3}
    y_new = np.vectorize(lut.get)(y_np)
    return torch.as_tensor(y_new, dtype=torch.long, device=y.device)


def _fix_time_len(x: np.ndarray, target_t: int) -> np.ndarray:
    """บังคับความยาวแกนเวลาให้เท่ากัน (crop/pad) ; x: (N, C, T0)"""
    N, C, T0 = x.shape
    if T0 == target_t:
        return x
    if T0 > target_t:
        return x[..., :target_t]
    out = np.zeros((N, C, target_t), dtype=x.dtype)
    out[..., :T0] = x
    return out


# =========================
# I/O
# =========================
def read_data(path: str):
    raw = mne.io.read_raw_gdf(
        path,
        preload=True,
        eog=["EOG-left", "EOG-central", "EOG-right"],
        verbose="ERROR",
    )
    # drop เฉพาะที่มีจริง
    for ch in ["EOG-left", "EOG-central", "EOG-right"]:
        if ch in raw.ch_names:
            raw.drop_channels([ch])
    raw.set_eeg_reference()

    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=[5, 6, 7, 8],
        baseline=None,
        preload=True,
        on_missing="warn",
        verbose="ERROR",
    )
    X = epochs.get_data()  # (N,C,T)
    y = epochs.events[:, -1]  # (N,)
    n = min(X.shape[0], len(y))
    return X[:n], y[:n]


# =========================
# Data pipeline
# =========================
def load_data():
    # 1) รวบรวมไฟล์
    feat_list, lab_list, grp_list = [], [], []
    folder_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "data")
    )
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {folder_path}")

    for i, filename in enumerate(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".gdf"):
            X, y = read_data(file_path)  # X: (N,C,T), y: (N,)
            if X.size == 0 or y.size == 0:
                continue
            X = _fix_time_len(X, signal_length)  # บังคับ T ให้เท่ากัน
            n = min(X.shape[0], len(y))
            feat_list.append(X[:n])
            lab_list.append(y[:n])
            grp_list.append(np.full(n, i, dtype=int))

    if not feat_list:
        raise RuntimeError(f"ไม่พบตัวอย่างจาก .gdf ใน {folder_path}")

    # 2) รวม และเตรียมสเกลบน (N,T,C)
    features = np.concatenate(feat_list, axis=0).astype(
        np.float32, copy=False
    )  # (N,C,T)
    labels = np.concatenate(lab_list, axis=0).astype(np.int64, copy=False)  # (N,)
    groups = np.concatenate(grp_list, axis=0).astype(np.int64, copy=False)  # (N,)

    features = np.moveaxis(features, 1, 2)  # (N, T, C)

    # 3) แบ่ง train/val: ใช้ GroupKFold ถ้ามีกลุ่ม >= 2, ไม่งั้น stratified split
    uniq_groups = np.unique(groups)
    if uniq_groups.size >= 2:
        gkf = GroupKFold(n_splits=min(5, uniq_groups.size))
        train_idx, val_idx = next(gkf.split(features, labels, groups))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(features, labels))

    trX, trY = features[train_idx], labels[train_idx]
    vaX, vaY = features[val_idx], labels[val_idx]

    # 4) สเกลต่อ-channel แล้วสลับกลับเป็น (N,C,T)
    scaler = StandardScaler3D()
    trX = scaler.fit_transform(trX)  # (N,T,C)
    vaX = scaler.transform(vaX)

    trX = np.moveaxis(trX, 1, 2)  # (N,C,T)
    vaX = np.moveaxis(vaX, 1, 2)

    # 5) เทนเซอร์ (N,1,C,T) + label เป็น long และรีแมป 5/6/7/8 -> 0..3
    trX = torch.from_numpy(trX).float().unsqueeze(1)  # (N,1,C,T)
    vaX = torch.from_numpy(vaX).float().unsqueeze(1)
    trY = torch.as_tensor(trY, dtype=torch.long)
    vaY = torch.as_tensor(vaY, dtype=torch.long)

    trY = remap_np(trY)
    vaY = remap_np(vaY)

    # 6) DataLoader
    train_dataset = TensorDataset(trX, trY)
    val_dataset = TensorDataset(vaX, vaY)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader


# =========================
# Train / Test
# =========================
def train(net, train_dataloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(int(epochs)):
        for images, labels in train_dataloader:  # images: (B,1,C,T)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)  # EEGNet ส่ง logits ออก
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / max(len(train_dataloader), 1)


@torch.no_grad()
def test(net, val_dataloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)
