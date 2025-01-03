import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from .dataset import TrackDataset
from .model import VideoClassifier
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

val_dataset = TrackDataset(split="val")
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
        video_tracks = [
            (data["poses"].cuda(), data["hands_regions"].cuda()) for data in batch
        ]
        labels = torch.tensor(
            [data["label"] for data in batch], dtype=torch.float32
        ).cuda()

        optimizer.zero_grad()
        outputs = torch.stack([model(tracks) for tracks in video_tracks])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            video_tracks = [
                (data["poses"].cuda(), data["hands_regions"].cuda()) for data in batch
            ]
            labels = torch.tensor(
                [data["label"] for data in batch], dtype=torch.float32
            ).cuda()

            outputs = torch.stack([model(tracks) for tracks in video_tracks])
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}"
    )
