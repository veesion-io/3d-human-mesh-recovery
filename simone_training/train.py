import torch
from torch import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork")

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
learning_rate = 1e-4
target_fps = 2.0
batch_size = 32
num_epochs = 200
save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)


def collate_fn(batch):
    return batch


import time


def main():
    run_name = "run_" + str(time.time()).split(".")[0]
    writer = SummaryWriter(log_dir=f"tensorboard_logs/{run_name}")

    train_dataset = TrackDataset(
        "simone_subset.json",
        7.0,
        target_fps=target_fps,
        mode="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
    )
    val_dataset = TrackDataset(
        "simone_subset.json",
        7.0,
        target_fps=target_fps,
        mode="val",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=True,
        collate_fn=collate_fn,
        num_workers=12,
        pin_memory=True,
    )

    model = VideoClassifier(
        nk=nk,
        num_bag_classes=num_bag_classes,
        keypoint_hidden_dim=keypoint_hidden_dim,
    ).cuda()
    model = torch.compile(model)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        iterations = 0
        for batch in train_loader:
            poses_list, bag_features_list, video_indices, labels = [], [], [], []
            video_idx = 0
            for data in batch:
                if data is None:
                    continue
                num_tracks = data["poses"].size(0)
                if num_tracks > 0:
                    poses_list.append(data["poses"].cuda())
                    bag_features_list.append(data["bag_features"].cuda())
                    video_indices.extend([video_idx] * num_tracks)
                labels.append(data["label"])
                video_idx += 1

            if not poses_list:
                continue

            poses_list = torch.cat(poses_list, dim=0)
            bag_features_list = torch.cat(bag_features_list, dim=0)
            video_indices = torch.tensor(video_indices, dtype=torch.long).cuda()
            labels = torch.tensor(labels, dtype=torch.float32).cuda()

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(
                    poses_list, bag_features_list, video_indices, len(labels)
                )
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.0).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            iterations += 1
            print(
                loss.item(),
                outputs.data.cpu().numpy().tolist(),
                labels.data.cpu().numpy().astype(int).tolist(),
                video_indices.data.cpu().numpy().tolist(),
            )

        avg_train_loss = train_loss / iterations
        train_accuracy = correct / total if total > 0 else 0
        writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
        )

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        iterations = 0
        with torch.no_grad():
            for batch in val_loader:
                poses_list, bag_features_list, video_indices, labels = [], [], [], []
                video_idx = 0
                for data in batch:
                    if data is None:
                        continue
                    num_tracks = data["poses"].size(0)
                    if num_tracks > 0:
                        poses_list.append(data["poses"].cuda())
                        bag_features_list.append(data["bag_features"].cuda())
                        video_indices.extend([video_idx] * num_tracks)
                    labels.append(data["label"])
                    video_idx += 1

                if not poses_list:
                    continue

                poses_list = torch.cat(poses_list, dim=0)
                bag_features_list = torch.cat(bag_features_list, dim=0)
                video_indices = torch.tensor(video_indices, dtype=torch.long).cuda()
                labels = torch.tensor(labels, dtype=torch.float32).cuda()
                with torch.amp.autocast("cuda"):
                    outputs = model(
                        poses_list, bag_features_list, video_indices, len(labels)
                    )
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = (outputs > 0.0).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                iterations += 1
                print(
                    loss.item(),
                    outputs.data.cpu().numpy().tolist(),
                    labels.data.cpu().numpy().astype(int).tolist(),
                    video_indices.data.cpu().numpy().tolist(),
                )
        avg_val_loss = val_loss / iterations
        val_accuracy = correct / total if total > 0 else 0
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch + 1}.pth")

    writer.close()


if __name__ == "__main__":
    main()
