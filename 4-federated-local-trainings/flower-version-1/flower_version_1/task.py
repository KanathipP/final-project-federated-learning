# flower_version_1/task.py

from __future__ import annotations

import json
import logging
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flower_version_1.motor_imaginary_data_cleaner as motor_imaginary_data_cleaner
from flower_version_1.model import EEGNet

# -------------------------------------------------------------------
# Config ทั่วไป
# -------------------------------------------------------------------

IN_CHANNELS = 22
NUM_CLASSES = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# JSON logger → stdout
# -------------------------------------------------------------------

logger = logging.getLogger("flwr_client")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)

logger.setLevel(logging.INFO)
logger.propagate = False


def log_progress(
    *,
    phase: str,
    status: str,
    server_round: Optional[int] = None,
    partition_id: Optional[int] = None,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None,
    batch: Optional[int] = None,
    num_batches: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    """พ่น log แบบ JSON ต่อบรรทัด ให้ backend ไปอ่านเอา"""
    payload: dict = {
        "component": "flwr-client",
        "phase": phase,    # "load_data" | "train" | "validate" | "test"
        "status": status,  # "start" | "running" | "end"
    }

    if server_round is not None:
        payload["server_round"] = int(server_round)
    if partition_id is not None:
        payload["partition_id"] = int(partition_id)
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if num_epochs is not None:
        payload["num_epochs"] = int(num_epochs)
    if batch is not None:
        payload["batch"] = int(batch)
    if num_batches is not None:
        payload["num_batches"] = int(num_batches)

    # progress % (ใน local epoch/batch)
    if (
        epoch is not None
        and num_epochs is not None
        and batch is not None
        and num_batches is not None
        and num_epochs > 0
        and num_batches > 0
    ):
        global_step = (epoch - 1) * num_batches + batch
        total_steps = num_epochs * num_batches
        payload["progress_pct"] = 100.0 * float(global_step) / float(total_steps)

    if extra:
        for k, v in extra.items():
            payload[k] = v

    logger.info(json.dumps(payload, ensure_ascii=False))


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------


class Net(EEGNet):
    def __init__(self) -> None:
        super().__init__(in_channel=IN_CHANNELS, num_classes=NUM_CLASSES)


# -------------------------------------------------------------------
# Data pipeline
# -------------------------------------------------------------------


def load_data(
    partition_id: int,
    server_round: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # เริ่มโหลดข้อมูล
    log_progress(
        phase="load_data",
        status="start",
        server_round=server_round,
        partition_id=partition_id,
        extra={"message": "loading_data_start"},
    )

    # step: เริ่มอ่าน/raw จาก pipeline
    log_progress(
        phase="load_data",
        status="running",
        server_round=server_round,
        partition_id=partition_id,
        extra={"step": "read_and_prepare_raw"},
    )

    (
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
    ) = motor_imaginary_data_cleaner.pipeline(partition_id)

    # step: หลัง pipeline เสร็จ กำลัง transform/split เสร็จแล้ว
    log_progress(
        phase="load_data",
        status="running",
        server_round=server_round,
        partition_id=partition_id,
        extra={"step": "transform_and_split_done"},
    )

    log_progress(
        phase="load_data",
        status="end",
        server_round=server_round,
        partition_id=partition_id,
        extra={
            "message": "loading_data_done",
            "train_examples": int(len(train_labels)),
            "val_examples": int(len(val_labels)),
            "test_examples": int(len(test_labels)),
        },
    )

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


# -------------------------------------------------------------------
# Train + Validate
# -------------------------------------------------------------------


def train(
    net: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    server_round: Optional[int] = None,
    partition_id: Optional[int] = None,
) -> Tuple[float, float, float]:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    num_epochs = int(epochs)
    num_batches = max(len(train_dataloader), 1)

    # เริ่ม train
    log_progress(
        phase="train",
        status="start",
        server_round=server_round,
        partition_id=partition_id,
        num_epochs=num_epochs,
        num_batches=num_batches,
    )

    net.train()
    running_loss = 0.0

    for epoch_idx in range(num_epochs):
        epoch = epoch_idx + 1
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            batch = batch_idx + 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            running_loss += loss_value

            # log progress ของ train ทุก batch
            log_progress(
                phase="train",
                status="running",
                server_round=server_round,
                partition_id=partition_id,
                epoch=epoch,
                num_epochs=num_epochs,
                batch=batch,
                num_batches=num_batches,
                extra={"loss": loss_value},
            )

    train_loss = running_loss / float(num_epochs * num_batches)

    log_progress(
        phase="train",
        status="end",
        server_round=server_round,
        partition_id=partition_id,
        num_epochs=num_epochs,
        num_batches=num_batches,
        extra={"train_loss": float(train_loss)},
    )

    # ---------------- validation ----------------
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    num_epochs_val = 1
    num_batches_val = max(len(val_dataloader), 1)

    log_progress(
        phase="validate",
        status="start",
        server_round=server_round,
        partition_id=partition_id,
        num_epochs=num_epochs_val,
        num_batches=num_batches_val,
        extra={"num_examples": int(len(val_dataloader.dataset))},
    )

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            batch = batch_idx + 1

            images, labels = images.to(device), labels.to(device)
            logits = net(images)
            batch_loss = criterion(logits, labels).item()
            total_loss += batch_loss * images.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += images.size(0)

            # log progress ของ validation
            log_progress(
                phase="validate",
                status="running",
                server_round=server_round,
                partition_id=partition_id,
                epoch=1,
                num_epochs=num_epochs_val,
                batch=batch,
                num_batches=num_batches_val,
                extra={"loss": float(batch_loss)},
