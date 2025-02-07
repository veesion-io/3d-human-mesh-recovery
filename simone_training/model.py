import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class KeypointBagEncoder(nn.Module):
    def __init__(self, nk, num_bag_classes, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Linear projection for bag features
        self.bag_projection = nn.Linear(2 * num_bag_classes, hidden_dim)

        # Convolution layers with residual connections
        self.conv1 = nn.Conv1d(
            in_channels=(nk * 3) + hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2
        )

        # Layer Normalization before Transformer
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=256, dropout=0.2
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=3)

        # Learnable Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

    def forward(self, keypoint_trajectories, bag_features):
        B, T, nk, _ = keypoint_trajectories.shape

        # Flatten keypoints and apply bag feature transformation
        keypoints = keypoint_trajectories.view(B, T, -1)  # (B, T, nk*3)
        bag_features = self.bag_projection(bag_features)  # (B, T, hidden_dim)
        x = torch.cat([keypoints, bag_features], dim=-1)  # (B, T, nk*3 + hidden_dim)

        # Convert to (B, C, T) format for convolutions
        x = x.permute(0, 2, 1)

        # Apply convolutions with residual connections
        x_res = x  # Store original input for residual connection
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x)) + x_res  # Residual connection
        x = F.leaky_relu(self.conv3(x))

        # Convert back to (B, T, C) and apply LayerNorm
        x = x.permute(0, 2, 1)
        x = self.norm1(x)

        # Add learnable positional encoding
        if self.positional_encoding.size(1) < T:
            raise ValueError(
                f"Positional encoding length ({self.positional_encoding.size(1)}) is less than input sequence length ({T})."
            )
        x = x + self.positional_encoding[:, :T, :]

        # Transformer Encoder
        x = x.permute(1, 0, 2)  # (T, B, C)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Back to (B, T, C)

        # Aggregate features
        return x.mean(dim=1)  # (B, hidden_dim)


class VideoClassifier(nn.Module):
    def __init__(self, nk, num_bag_classes, keypoint_hidden_dim):
        super().__init__()
        self.keypoint_bag_encoder = KeypointBagEncoder(
            nk, num_bag_classes, keypoint_hidden_dim
        )
        self.no_track_score = nn.Parameter(torch.tensor(-1.0))
        self.track_fc = nn.Linear(keypoint_hidden_dim * 2, 1)

    def forward(self, poses_list, bag_features, video_indices, num_videos):
        keypoint_features = self.keypoint_bag_encoder(poses_list, bag_features)
        track_logits = self.track_fc(keypoint_features).squeeze(-1)

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
