import torch
import torch.nn as nn
import torch.nn.functional as F


class Keypoint3DTrajectoryEncoder(nn.Module):
    def __init__(self, nk, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=nk * 3, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )

    def forward(self, keypoint_trajectories):
        B, T, nk, _ = keypoint_trajectories.shape
        x = keypoint_trajectories.view(B, T, -1).permute(
            0, 2, 1
        )  # Flatten nk and 3 dimensions, transpose to (B, C, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # Transpose back to (B, T, C)
        _, h_n = self.gru(x)
        return h_n.squeeze(0)  # (B, hidden_dim)


class HandImageEncoder(nn.Module):
    def __init__(self, pretrained_model_name="resnet18", output_dim=256):
        super().__init__()
        self.feature_extractor = torch.hub.load(
            "pytorch/vision:v0.10.0", pretrained_model_name, pretrained=True
        )
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        self.fc = nn.Linear(
            512, output_dim
        )  # Adjust input size based on the pretrained model

    def forward(self, hand_images):
        B, T, _, H, W = hand_images.shape
        hand_images = hand_images.view(
            B * T * 2, H, W
        )  # Combine batch, time, and hand dimensions
        features = self.feature_extractor(hand_images)
        features = self.fc(features)
        features = features.view(B, T, 2, -1).max(dim=2)  # Average over two hands
        return features  # (B, T, output_dim)


class VideoClassifier(nn.Module):
    def __init__(self, nk, keypoint_hidden_dim, hand_feature_dim, final_hidden_dim):
        super().__init__()
        self.keypoint_encoder = Keypoint3DTrajectoryEncoder(nk, keypoint_hidden_dim)
        self.hand_encoder = HandImageEncoder(output_dim=hand_feature_dim)
        self.track_fc = nn.Sequential(
            nn.Linear(keypoint_hidden_dim + hand_feature_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Linear(final_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, video_tracks):
        track_predictions = []

        for keypoints, hands in video_tracks:
            keypoint_feat = self.keypoint_encoder(keypoints)
            hand_feat = self.hand_encoder(hands).max(dim=1)  # Temporal pooling

            combined_features = torch.cat([keypoint_feat, hand_feat], dim=-1)
            track_pred = self.track_fc(combined_features)
            track_predictions.append(track_pred)

        track_predictions = torch.stack(track_predictions)  # (num_tracks, 1)
        return track_predictions


# Example usage
# Assuming the inputs for a video are a list of tuples (keypoints_tensor, hand_images_tensor)
# where keypoints_tensor is of shape (T, nk, 3) and hand_images_tensor is of shape (T, 2, h, w)
