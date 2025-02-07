import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class KeypointBagEncoder(nn.Module):
    def __init__(self, nk, num_bag_classes, hidden_dim):
        super().__init__()
        self.hidden_dim = (nk * 3) + (2 * num_bag_classes)  # Ensure correct d_model

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim,  # Keep the hidden_dim for feedforward layer
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=4)

        # Learnable Positional Encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 512, self.hidden_dim) * 0.02
        )

    def forward(self, keypoint_trajectories, bag_features):
        B, T, nk, _ = keypoint_trajectories.shape

        # Normalize Keypoints
        keypoint_trajectories = (
            keypoint_trajectories - keypoint_trajectories.mean()
        ) / (keypoint_trajectories.std() + 1e-6)

        # Flatten keypoints and **concatenate** bag features
        x = torch.cat(
            [keypoint_trajectories.view(B, T, -1), bag_features], dim=-1
        )  # (B, T, nk*3 + bag_features)

        # Add positional encoding
        if self.positional_encoding.size(1) < T:
            raise ValueError(
                f"Positional encoding length ({self.positional_encoding.size(1)}) is less than input sequence length ({T})."
            )
        x = x + self.positional_encoding[:, :T, :]

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Aggregate features
        return x.mean(dim=1)  # (B, hidden_dim) now correct


class VideoClassifier(nn.Module):
    def __init__(self, nk, num_bag_classes, keypoint_hidden_dim):
        super().__init__()
        self.feature_dim = (nk * 3) + (2 * num_bag_classes)  # Ensure correct shape
        self.keypoint_bag_encoder = KeypointBagEncoder(
            nk, num_bag_classes, keypoint_hidden_dim
        )
        self.no_track_score = nn.Parameter(torch.tensor(-1.0))

        # Fix `track_fc` to match Transformer output dim
        self.track_fc = nn.Linear(self.feature_dim, 1)  # Correct input size

    def forward(self, poses_list, bag_features, video_indices, num_videos):
        keypoint_features = self.keypoint_bag_encoder(
            poses_list, bag_features
        )  # (B, feature_dim)
        track_logits = self.track_fc(keypoint_features).squeeze(-1)  # (B,)

        video_logits = torch.full(
            (num_videos,), -float("inf"), device=track_logits.device
        ).half()

        video_logits = torch.scatter_reduce(
            video_logits,
            dim=0,
            index=video_indices,
            src=track_logits,
            reduce="amax",
            include_self=False,
        )

        video_logits = torch.where(
            video_logits == -float("inf"), self.no_track_score, video_logits
        )

        return video_logits
