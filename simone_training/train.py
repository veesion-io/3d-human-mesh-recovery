import torch
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
keypoint_hidden_dim = 128
hand_feature_dim = 32
final_hidden_dim = 128
learning_rate = 1e-4
target_fps = 3.0
batch_size = 8
num_epochs = 200
save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="tensorboard_logs")

# Initialize dataset, dataloaders, model, optimizer, and loss function
train_dataset = TrackDataset(
    "simone_subset.json",
    7.0,
    target_fps=target_fps,
    hands_height=128,
    hands_width=128,
    mode="train",
)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x
)
val_dataset = TrackDataset(
    "simone_subset.json",
    7.0,
    target_fps=target_fps,
    hands_height=128,
    hands_width=128,
    mode="val",
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
)

model = VideoClassifier(
    nk=nk,
    keypoint_hidden_dim=keypoint_hidden_dim,
    hand_feature_dim=hand_feature_dim,
    final_hidden_dim=final_hidden_dim,
).cuda()

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    iterations = 0
    for batch in train_loader:
        poses_list, hands_list, video_indices, labels = [], [], [], []
        video_idx = 0
        for data in batch:
            if data is None:
                continue
            num_tracks = data["poses"].size(0)
            if num_tracks > 0:
                poses_list.append(data["poses"].cuda())
                hands_list.append(data["hands_regions"].cuda())
                video_indices.extend([video_idx] * num_tracks)
            labels.append(data["label"])
            video_idx += 1

        if not poses_list:
            continue

        poses_list = torch.cat(poses_list, dim=0)
        hands_list = torch.cat(hands_list, dim=0)
        video_indices = torch.tensor(video_indices, dtype=torch.long).cuda()
        labels = torch.tensor(labels, dtype=torch.float32).cuda()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):  # Mixed precision training
            outputs = model(poses_list, hands_list, video_indices, len(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        predictions = (outputs > 0.0).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        iterations += 1
        print(loss.item(), outputs, labels, video_indices)

    avg_train_loss = train_loss / iterations
    train_accuracy = correct / total if total > 0 else 0
    writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)
    writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}"
    )

    # Validation phase
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    iterations = 0
    with torch.no_grad():
        for batch in val_loader:
            poses_list, hands_list, video_indices, labels = [], [], [], []
            video_idx = 0
            for data in batch:
                if data is None:
                    continue
                num_tracks = data["poses"].size(0)
                if num_tracks > 0:
                    poses_list.append(data["poses"].cuda())
                    hands_list.append(data["hands_regions"].cuda())
                    video_indices.extend([video_idx] * num_tracks)
                labels.append(data["label"])
                video_idx += 1

            if not poses_list:
                continue

            poses_list = torch.cat(poses_list, dim=0)
            hands_list = torch.cat(hands_list, dim=0)
            video_indices = torch.tensor(video_indices, dtype=torch.long).cuda()
            labels = torch.tensor(labels, dtype=torch.float32).cuda()
            with torch.amp.autocast("cuda"):
                outputs = model(poses_list, hands_list, video_indices, len(labels))
                loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = (outputs > 0.0).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            iterations += 1
            print(loss.item(), outputs, labels, video_indices)

    avg_val_loss = val_loss / iterations
    val_accuracy = correct / total if total > 0 else 0
    writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
    writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)
    print(
        f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
    )

    # Save checkpoint
    torch.save(model.state_dict(), f"{save_path}/model_epoch_{epoch+1}.pth")

# Close TensorBoard writer
writer.close()
