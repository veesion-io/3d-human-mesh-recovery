import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from simone_training.dataset import TrackDataset
from simone_training.model import VideoClassifier
from torch import nn

# Hyperparameters
nk = 58  # Number of keypoints, one every 120 in 3d poses
keypoint_hidden_dim = 128
hand_feature_dim = 32
final_hidden_dim = 128
learning_rate = 1e-4
batch_size = 8
num_epochs = 20

# Initialize dataset, dataloader, model, optimizer, and loss function
train_dataset = TrackDataset(
    "simone_subset.json",
    7.0,
    tracks_path="results",
    target_fps=3.0,
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
    tracks_path="results",
    target_fps=3.0,
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
)
model = model.cuda()

optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        poses_list, hands_list, video_indices, labels = [], [], [], []

        for video_idx, data in enumerate(batch):
            if data is None:  # Skip videos with invalid metadata
                continue

            num_tracks = data["poses"].size(0)
            if num_tracks > 0:
                poses_list.append(data["poses"].cuda())
                hands_list.append(data["hands_regions"].cuda())
                video_indices.extend([video_idx] * num_tracks)
            labels.append(data["label"])

        # If no valid videos in batch, skip the batch
        if not poses_list:
            continue

        poses_list = torch.cat(poses_list, dim=0)
        hands_list = torch.cat(hands_list, dim=0)
        video_indices = torch.tensor(video_indices, dtype=torch.long).cuda()
        labels = torch.tensor(labels, dtype=torch.float32).cuda()

        optimizer.zero_grad()
        outputs = model(poses_list, hands_list, video_indices)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}"
        )
