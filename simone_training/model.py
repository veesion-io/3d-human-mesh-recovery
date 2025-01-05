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
        self.conv3 = nn.Conv1d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, padding=2
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim * 2, nhead=8, num_encoder_layers=2, dim_feedforward=256
        )
        self.positional_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim * 2))

    def forward(self, keypoint_trajectories):
        B, T, nk, _ = keypoint_trajectories.shape
        x = keypoint_trajectories.view(B, T, -1).permute(0, 2, 1)  # (B, C, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)  # (B, T, C)

        # BiGRU
        _, h_n = self.gru(x)
        h_n = h_n.view(B, -1)  # Concatenate forward and backward hidden states

        # Transformer with positional encoding
        x = x + self.positional_encoding[:, :T, :]
        x = x.permute(1, 0, 2)  # (T, B, C)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Back to (B, T, C)

        return x.mean(dim=1)  # Aggregate temporal features


import torchvision


class HandImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.feature_extractor = torchvision.models.efficientnet_b1(
            weights="IMAGENET1K_V2"
        )
        self.feature_extractor.fc = nn.Identity()  # Remove classification layer
        self.fc = nn.Linear(
            512, output_dim
        )  # Adjust input size based on the pretrained model

    def forward(self, hand_images):
        B, T, _, H, W, C = hand_images.shape
        hand_images = hand_images.view(
            B * T * 2, H, W, C
        )  # Combine batch, time, and hand dimensions
        hand_images = hand_images.permute(0, 3, 1, 2)
        features = self.feature_extractor(hand_images.float())
        features = self.fc(features)
        features, _ = features.view(B, T, 2, -1).max(dim=2)  # Pooling over two hands
        features, _ = features.max(dim=1)  # Pooling over all frames hands
        return features  # (B, output_dim)


class VideoClassifier(nn.Module):
    def __init__(self, nk, keypoint_hidden_dim, hand_feature_dim, final_hidden_dim):
        super().__init__()
        self.keypoint_encoder = Keypoint3DTrajectoryEncoder(nk, keypoint_hidden_dim)
        self.hand_encoder = HandImageEncoder(output_dim=hand_feature_dim)
        self.track_fc = nn.Sequential(
            nn.Linear(keypoint_hidden_dim + hand_feature_dim, final_hidden_dim),
            nn.ReLU(),
            nn.Linear(final_hidden_dim, 1),
        )
        self.no_track_score = nn.Parameter(
            torch.tensor(0.5)
        )  # Learnable score for no-track cases
        self.video_fc = nn.Sigmoid()  # Final video-level classification

    def forward(self, poses_list, hands_list, video_indices):
        """
        Args:
            poses_list: Tensor of shape (N, T, nk, 3), all tracks concatenated
            hands_list: Tensor of shape (N, T, 2, h, w, c), all hand regions concatenated
            video_indices: Tensor of shape (N,), mapping each track to its video

        Returns:
            video_predictions: Tensor of shape (B,), video-level predictions
        """
        # Encode all tracks together
        keypoint_features = self.keypoint_encoder(
            poses_list
        )  # (N, keypoint_hidden_dim)
        hand_features = self.hand_encoder(hands_list)  # (N, hand_feature_dim)

        # Combine features and compute per-track logits
        track_features = torch.cat([keypoint_features, hand_features], dim=-1)
        track_logits = self.track_fc(track_features).squeeze(-1)  # (N,)

        # Aggregate track predictions back to videos
        num_videos = video_indices.max().item() + 1
        video_logits = torch.full(
            (num_videos,), -float("inf"), device=track_logits.device
        )  # Initialize logits

        video_logits = torch.scatter_reduce(
            video_logits,  # Destination tensor
            dim=0,
            index=video_indices,
            src=track_logits,
            reduce="amax",
            include_self=False,
        )

        # Replace -inf with learnable score for videos with no tracks
        video_logits = torch.where(
            video_logits == -float("inf"), self.no_track_score, video_logits
        )
        # Final video-level prediction
        video_predictions = self.video_fc(video_logits.unsqueeze(-1)).squeeze(
            -1
        )  # (B,)

        return video_predictions


# Example usage
# Assuming the inputs for a video are a list of tuples (keypoints_tensor, hand_images_tensor)
# where keypoints_tensor is of shape (T, nk, 3) and hand_images_tensor is of shape (T, 2, h, w)
