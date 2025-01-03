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
        )
        self.no_track_score = nn.Parameter(
            torch.tensor(0.5)
        )  # Learnable score for no-track cases
        self.video_fc = nn.Sigmoid()  # Final video-level classification

    def forward(self, poses_list, hands_list, video_indices):
        """
        Args:
            poses_list: Tensor of shape (N, T, nk, 3), all tracks concatenated
            hands_list: Tensor of shape (N, T, 2, h, w), all hand regions concatenated
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
        )  # Max-pooling init
        for i in range(len(track_logits)):
            video_logits[video_indices[i]] = torch.max(
                video_logits[video_indices[i]], track_logits[i]
            )

        # Replace -inf with learnable score for videos with no tracks
        video_logits[video_logits == -float("inf")] = self.no_track_score

        # Final video-level prediction
        video_predictions = self.video_fc(video_logits.unsqueeze(-1)).squeeze(
            -1
        )  # (B,)

        return video_predictions


# Example usage
# Assuming the inputs for a video are a list of tuples (keypoints_tensor, hand_images_tensor)
# where keypoints_tensor is of shape (T, nk, 3) and hand_images_tensor is of shape (T, 2, h, w)
