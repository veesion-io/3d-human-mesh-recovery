import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class KeypointBagEncoder(nn.Module):
    def __init__(self, nk, num_bag_classes, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Conv1d(
            in_channels=(nk * 3) + 2 * num_bag_classes,
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
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim * 2, nhead=8, dim_feedforward=256
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=3)

        self.positional_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim * 2))

    def forward(self, keypoint_trajectories, bag_features):
        B, T, nk, _ = keypoint_trajectories.shape

        x = keypoint_trajectories.view(B, T, -1)  # Flatten nk and 3 dimensions
        x = torch.cat([x, bag_features], dim=-1)  # Concatenate bag features
        x = x.permute(0, 2, 1)  # (B, C, T)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.permute(0, 2, 1)  # (B, T, C)

        x, _ = self.gru(x)  # (B, T, hidden_dim * 2)

        if self.positional_encoding.size(1) < T:
            raise ValueError(
                f"Positional encoding length ({self.positional_encoding.size(1)}) is less than input sequence length ({T})."
            )
        x = x + self.positional_encoding[:, :T, :]

        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        return x.mean(dim=1)


class VideoClassifier(nn.Module):
    def __init__(self, nk, num_bag_classes, keypoint_hidden_dim, final_hidden_dim):
        super().__init__()
        self.keypoint_bag_encoder = KeypointBagEncoder(
            nk, num_bag_classes, keypoint_hidden_dim
        )
        self.no_track_score = nn.Parameter(torch.tensor(-1.0))
        self.track_fc = nn.Linear(keypoint_hidden_dim * 2, 1)

    def forward(self, poses_list, bag_features, video_indices, num_videos):
        keypoint_features = self.keypoint_bag_encoder(poses_list, bag_features)
        print(poses_list[0], bag_features[0], keypoint_features[0])
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
