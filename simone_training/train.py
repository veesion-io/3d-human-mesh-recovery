import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import multiprocessing as mp
import queue
import threading
from torch.utils.tensorboard import SummaryWriter
import time
import sys
import torch
from torch import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn")

from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torch import nn

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from simone_training.dataset import TrackDataset
from simone_training.model import VideoClassifier

# Hyperparameters
nk = 58  # Number of keypoints
num_bag_classes = 13  # Example number of bag classes
keypoint_hidden_dim = 32
learning_rate = 1e-3
target_fps = 2.0
batch_size = 32
num_epochs = 200
save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)


def collate_fn(batch):
    return batch


import time


# Data Loading with Shared Queue
class DataLoaderProcess(mp.Process):
    def __init__(self, dataset, data_queue, stop_event):
        super().__init__()
        self.dataset = dataset
        self.data_queue = data_queue
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            for sample in self.dataset:
                self.data_queue.put(sample)
                if self.stop_event.is_set():
                    break


# Batch Constructor Thread
class BatchConstructor(threading.Thread):
    def __init__(self, data_queue, batch_queue, batch_size, stop_event):
        super().__init__()
        self.data_queue = data_queue
        self.batch_queue = batch_queue
        self.batch_size = batch_size
        self.stop_event = stop_event

    def run(self):
        poses_list, bag_features_list, video_indices, labels = [], [], [], []
        video_idx = 0
        while not self.stop_event.is_set():
            try:
                sample = self.data_queue.get(timeout=1)

                for data in sample:
                    if data is None:
                        continue
                    num_tracks = data["poses"].size(0)
                    if num_tracks > 0:
                        poses_list.append(data["poses"])
                        bag_features_list.append(data["bag_features"])
                        video_indices.extend([video_idx] * num_tracks)
                    labels.append(data["label"])
                    video_idx += 1

                if len(poses_list) >= self.batch_size:
                    poses_list = torch.cat(
                        [s["poses"] for s in self.batch], dim=0
                    ).cuda()
                    bag_features_list = torch.cat(
                        [s["bag_features"] for s in self.batch], dim=0
                    ).cuda()
                    video_indices = torch.tensor(
                        [s["video_index"] for s in self.batch], dtype=torch.long
                    ).cuda()
                    labels = torch.tensor(
                        [s["label"] for s in self.batch], dtype=torch.float32
                    ).cuda()

                    self.batch_queue.put(
                        (poses_list, bag_features_list, video_indices, labels)
                    )
                    poses_list, bag_features_list, video_indices, labels = (
                        [],
                        [],
                        [],
                        [],
                    )
                    video_idx = 0
            except queue.Empty:
                continue


# Training Function
class VideoClassifierTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size,
        learning_rate,
        num_epochs,
    ):
        self.model = model.cuda()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.writer = SummaryWriter()
        self.train_queue = mp.Queue(maxsize=64)
        self.val_queue = mp.Queue(maxsize=32)
        self.train_batch_queue = queue.Queue(maxsize=2)
        self.val_batch_queue = queue.Queue(maxsize=2)
        self.stop_event = mp.Event()

    def start_data_loaders(self):
        self.train_loader_procs = [
            DataLoaderProcess(self.train_dataset, self.train_queue, self.stop_event)
            for _ in range(32)
        ]
        self.val_loader_procs = [
            DataLoaderProcess(self.val_dataset, self.val_queue, self.stop_event)
            for _ in range(12)
        ]

        self.train_batch_thread = BatchConstructor(
            self.train_queue, self.train_batch_queue, self.batch_size, self.stop_event
        )
        self.val_batch_thread = BatchConstructor(
            self.val_queue, self.val_batch_queue, self.batch_size, self.stop_event
        )

        for proc in self.train_loader_procs:
            proc.start()
        for proc in self.val_loader_procs:
            proc.start()
        self.train_batch_thread.start()
        self.val_batch_thread.start()

    def stop_data_loaders(self):
        self.stop_event.set()
        for proc in self.train_loader_procs:
            proc.join()
        for proc in self.val_loader_procs:
            proc.join()
        self.train_batch_thread.join()
        self.val_batch_thread.join()

    def train_epoch(self, epoch):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        start_time = time.time()

        for _ in range(len(self.train_dataset) // self.batch_size):
            if self.train_batch_queue.empty():
                continue

            poses_list, bag_features_list, video_indices, labels = (
                self.train_batch_queue.get()
            )

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = self.model(
                    poses_list, bag_features_list, video_indices, len(labels)
                )
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.0).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            compute_time = time.time() - start_time
            speed = total / compute_time if compute_time > 0 else 0
            sys.stdout.write(
                f"\rEpoch {epoch + 1}, Training Speed: {speed:.2f} samples/sec"
            )
            sys.stdout.flush()

        print()
        print(
            loss.item(),
            outputs.data.cpu().numpy().tolist(),
            labels.data.cpu().numpy().astype(int).tolist(),
        )
        avg_train_loss = train_loss / max(1, total)
        train_accuracy = correct / max(1, total)
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        self.writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)
        print(
            f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

    def validate_epoch(self, epoch):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        start_time = time.time()
        with torch.no_grad():
            for _ in range(len(self.val_dataset) // self.batch_size):
                if self.val_batch_queue.empty():
                    continue

                poses_list, bag_features_list, video_indices, labels = (
                    self.val_batch_queue.get()
                )
                with torch.amp.autocast("cuda"):
                    outputs = self.model(
                        poses_list, bag_features_list, video_indices, len(labels)
                    )
                    loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                predictions = (outputs > 0.0).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                compute_time = time.time() - start_time
                speed = total / compute_time if compute_time > 0 else 0
                sys.stdout.write(
                    f"\rEpoch {epoch + 1}, Training Speed: {speed:.2f} samples/sec"
                )
                sys.stdout.flush()

        print(
            loss.item(),
            outputs.data.cpu().numpy().tolist(),
            labels.data.cpu().numpy().astype(int).tolist(),
        )
        avg_val_loss = val_loss / max(1, total)
        val_accuracy = correct / max(1, total)
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
        self.writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)
        print(
            f"Epoch {epoch + 1}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    def train(self):
        self.start_data_loaders()
        for epoch in range(self.num_epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
        self.stop_data_loaders()
        self.writer.close()


if __name__ == "__main__":
    train_dataset = TrackDataset(
        "simone_subset.json",
        7.0,
        target_fps=target_fps,
        mode="train",
    )
    val_dataset = TrackDataset(
        "simone_subset.json",
        7.0,
        target_fps=target_fps,
        mode="val",
    )

    model = VideoClassifier(
        nk=nk,
        num_bag_classes=num_bag_classes,
        keypoint_hidden_dim=keypoint_hidden_dim,
    ).cuda()
    trainer = VideoClassifierTrainer(
        model,
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
    )
    trainer.train()
