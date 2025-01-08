import torch
from torch.utils.data import Dataset
import numpy as np
import json
from glob import glob
import traceback
from functools import lru_cache


def compute_timestamp_intersection(window, timespan, normalize=True):
    """
    minimal intersection
    input :
        window : video start and end
        timespan : label start and end
    output:
        iou : intersection over union
    """
    timespans = [window, timespan]
    timespans = sorted(timespans, key=lambda x: x[0])

    intersection = max(min(timespans[0][1], timespans[1][1]) - timespans[1][0], 0)

    window_duration = window[1] - window[0]

    return intersection / window_duration


def find_window_label(video_meta_data, window):
    actions_timespans = video_meta_data["actions_timespans"]
    if "Dissimulation sac" not in actions_timespans:
        return False
    return any(
        compute_timestamp_intersection(timespan, window) > 0.5
        for timespan in actions_timespans["Dissimulation sac"]
    )


import os
from lib.models.smpl import SMPL
from lib.vis.traj import fit_to_ground_easy, traj_filter
from lib.vis.renderer import Renderer


def find_closest(sorted_list1, sorted_list2):
    sorted_list1 = np.array(sorted_list1)
    sorted_list2 = np.array(sorted_list2)

    # Compute absolute differences and find the indices of minimum differences
    indices = np.abs(sorted_list1[:, None] - sorted_list2).argmin(axis=1)

    # Use the indices to get the closest values from sorted_list2
    return sorted_list2[indices]


import cv2


class TrackDataset(Dataset):
    def __init__(
        self,
        meta_data_file,
        window_duration,
        tracks_path="tracks",
        images_path="results",
        target_fps=3.0,
        hands_height=128,
        hands_width=128,
        mode="train",
        max_num_tracks=3,
    ):
        super(TrackDataset, self).__init__()
        with open(meta_data_file, "r") as f:
            self.videos_meta_data = json.load(f)
        self.videos_names = sorted(list(self.videos_meta_data))
        np.random.seed(42)
        np.random.shuffle(self.videos_names)
        if mode == "train":
            self.videos_names = self.videos_names[: int(0.9 * len(self.videos_names))]
        else:
            self.videos_names = self.videos_names[int(0.9 * len(self.videos_names)) :]
        self.videos_meta_data = {k: self.videos_meta_data[k] for k in self.videos_names}
        self.window_duration = window_duration
        self.tracks_path = tracks_path
        self.images_path = images_path
        self.target_fps = target_fps
        self.hands_height = hands_height
        self.hands_width = hands_width
        self.max_num_tracks = max_num_tracks

    def __len__(self):
        return len(self.videos_meta_data)

    # from collections import defaultdict

    def load_video_tracks(
        self,
        video_name,
    ):
        video_barename = os.path.splitext(video_name)[0]
        tracks_info = np.load(
            os.path.join(self.tracks_path, f"{video_barename}.npy"), allow_pickle=True
        ).item()
        return tracks_info
        # formatted_tracks = {track_id : {"frames_ids": frames_ids} for track_id, frames_ids in tracks_frames_ids.items()}

    def crop_track(self, track_id, video_tracks, video_fps, target_fps, timespan):
        step = 1 / target_fps
        timestamps_to_select = np.arange(timespan[0], timespan[1] + step / 2, step)
        frames_ids_to_select = timestamps_to_select * video_fps
        adjusted_frames_ids = find_closest(
            frames_ids_to_select, video_tracks["frames_ids"][track_id]
        )
        return {
            "frames_ids": adjusted_frames_ids,
            "vertices": np.array(
                [
                    video_tracks["vertices"][frame_id][track_id][0]
                    for frame_id in adjusted_frames_ids
                ]
            ),
        }

    @lru_cache(maxsize=2)
    def load_frame(self, frame_path):
        return cv2.imread(frame_path)

    def load_hands_regions(
        self, track_id, video_name, video_tracks, cropped_track_info
    ):
        video_barename = os.path.splitext(video_name)[0]
        img_folder = os.path.join(
            self.images_path, video_barename.split(".")[0], "images"
        )
        imgfiles = sorted(glob(f"{img_folder}/*.jpg"))
        track_hands = []
        for frame_id in cropped_track_info["frames_ids"]:
            img = np.ascontiguousarray(
                self.load_frame(imgfiles[frame_id])[:, :, ::-1], dtype=np.uint8
            )
            person_hands, height = video_tracks["hands"][frame_id][track_id]
            frame_hands_regions = []
            for x, y in person_hands:
                dx = int(0.21 * height)
                dy = int(0.15 * height)
                # y = img.shape[1] - y
                x1, y1 = (int(x - 0.25 * dx), int(y - dy))
                x2, y2 = (int(x + 1.75 * dx), int(y + dy))
                hand_region = img[x1:x2, y1:y2]
                if 0 in hand_region.shape:
                    frame_hands_regions.append(
                        (
                            127 * np.ones((self.hands_width, self.hands_height, 3))
                        ).astype(np.uint8)
                    )
                else:
                    frame_hands_regions.append(
                        cv2.resize(hand_region, (self.hands_width, self.hands_height))
                    )
            track_hands.append(frame_hands_regions)
        return track_hands

    def window_intersection(self, video_fps, frames_ids, window):
        track_timestamps = np.array(frames_ids) / video_fps
        intersecting_timestamps = track_timestamps[track_timestamps >= window[0]]
        intersecting_timestamps = intersecting_timestamps[
            intersecting_timestamps < window[1]
        ]
        intersecting_duration = len(intersecting_timestamps) / video_fps
        return intersecting_duration / (window[1] - window[0])

    def __getitem__(self, index):
        video_name = self.videos_names[index]
        video_meta_data = self.videos_meta_data[video_name]
        start_time = np.random.uniform(
            0, max(0, video_meta_data["duration"] - self.window_duration)
        )
        end_time = min(video_meta_data["duration"], start_time + self.window_duration)
        video_info = read_video_info(os.path.join("simone_bag_subset", video_name))
        # Check if metadata is missing or invalid
        try:
            video_tracks = self.load_video_tracks(video_name)
        except FileNotFoundError:
            # traceback.print_exc()
            return None
        tracks_data = []
        tracks_intersections = {}
        for track_id, frames_ids in video_tracks["frames_ids"].items():
            tracks_intersections[track_id] = self.window_intersection(
                video_info["fps"], frames_ids, [start_time, end_time]
            )
        tracks_intersections = sorted(tracks_intersections.items(), key=lambda x: x[1])[
            ::-1
        ]
        tracks_intersections = tracks_intersections[: self.max_num_tracks]
        selected_tracks = [
            track_id
            for track_id, intersection in tracks_intersections
            if intersection > 0.4
        ]
        for track_id in selected_tracks:
            cropped_track_info = self.crop_track(
                track_id,
                video_tracks,
                video_info["fps"],
                self.target_fps,
                [start_time, end_time],
            )
            hands_regions = self.load_hands_regions(
                track_id, video_name, video_tracks, cropped_track_info
            )
            tracks_data.append(
                (
                    torch.from_numpy(cropped_track_info["vertices"]),
                    torch.from_numpy(np.array(hands_regions)),
                )
            )
        label = find_window_label(video_meta_data, [start_time, end_time])
        if len(tracks_data) == 0:
            # print(
            #     "No track",
            #     video_name,
            #     start_time,
            #     video_info["fps"],
            #     video_tracks["frames_ids"],
            # )
            return {
                "poses": torch.empty(0),
                "hands_regions": torch.empty(0),
                "label": label,
            }
        formatted_data = {
            "poses": torch.stack([x[0] for x in tracks_data]),
            "hands_regions": torch.stack([x[1] for x in tracks_data]),
            "label": label,
        }
        if self.mode == "train":
            for track_num in range(len(formatted_data["hands_regions"])):
                if np.random.choice(2):
                    formatted_data["hands_regions"][track_num] = torch.flip(
                        formatted_data["hands_regions"][track_num], (3,)
                    )
                formatted_data["poses"][track_num] = random_horizontal_rotation_3d(
                    formatted_data["poses"][track_num]
                )
                if np.random.choice(2):
                    formatted_data["poses"][track_num] = random_horizontal_flip_3d(
                        formatted_data["poses"][track_num], axis=0
                    )
                if np.random.choice(2):
                    formatted_data["poses"][track_num] = random_horizontal_flip_3d(
                        formatted_data["poses"][track_num], axis=2
                    )
        # os.makedirs("inputs", exist_ok=True)
        # np.save(
        #     f"inputs/{os.path.splitext(video_name)[0]}_{start_time}.npy", formatted_data
        # )
        # np.save("hands.npy", formatted_data["hands_regions"].numpy())
        # print(video_name, start_time)
        # dvsdv
        return formatted_data


import av

import torch


def random_horizontal_rotation_3d(vertices, max_angle_degrees=180):
    """
    Apply a random rotation to 3D vertices for each frame along the horizontal axis (Y-axis).

    Args:
        vertices (torch.Tensor): A tensor of shape (T, nk, 3), where T is the number of frames,
                                 nk is the number of keypoints, and 3 represents (x, y, z).
        max_angle_degrees (float): The maximum angle for rotation in degrees.

    Returns:
        torch.Tensor: Rotated vertices with the same shape as input.
    """
    # Convert max_angle to radians
    max_angle_radians = torch.deg2rad(torch.tensor(max_angle_degrees))

    # Generate a random angle in the range [-max_angle, max_angle]
    angle = torch.empty(1).uniform_(-max_angle_radians, max_angle_radians)
    print(angle)
    # Compute the rotation matrix for Y-axis
    rotation_matrix = torch.tensor(
        [
            [torch.cos(angle).item(), 0, torch.sin(angle).item()],
            [0, 1, 0],
            [-torch.sin(angle).item(), 0, torch.cos(angle).item()],
        ],
        dtype=vertices.dtype,
        device=vertices.device,
    )

    # Reshape vertices to (T * nk, 3), apply rotation, and reshape back to (T, nk, 3)
    T, nk, _ = vertices.shape
    vertices_flat = vertices.view(-1, 3)  # (T * nk, 3)
    rotated_vertices_flat = torch.matmul(vertices_flat, rotation_matrix.T)
    rotated_vertices = rotated_vertices_flat.view(T, nk, 3)

    return rotated_vertices


def random_horizontal_flip_3d(vertices, axis):
    """
    Apply a random horizontal flip to 3D vertices for each frame.
    The flip mirrors the vertices across the Y-axis without centering.

    Args:
        vertices (torch.Tensor): A tensor of shape (T, nk, 3), where T is the number of frames,
                                 nk is the number of keypoints, and 3 represents (x, y, z).
        flip_prob (float): Probability of applying the flip (default: 0.5).

    Returns:
        torch.Tensor: Flipped vertices with the same shape as input.
    """
    # Compute the bounding box for the X and Z dimensions
    x_min, _ = torch.min(
        vertices[..., axis], dim=1, keepdim=True
    )  # Min across keypoints (nk)
    x_max, _ = torch.max(vertices[..., axis], dim=1, keepdim=True)

    # Flip X and Z coordinates around their respective midpoints
    x_mid = (x_max + x_min) / 2

    flipped_vertices = vertices.clone()
    flipped_vertices[..., axis] = 2 * x_mid - vertices[..., axis]  # Flip X

    return flipped_vertices


def read_video_info(video_path: str):
    try:
        with av.open(video_path) as video:
            stream = video.streams.video[0]
            fps = float(stream.average_rate)

            if stream.duration is None:
                duration = video.duration / av.time_base
            else:
                duration = float(stream.duration * stream.time_base)

            width = stream.width
            height = stream.height

        return {"fps": fps, "duration": duration, "width": width, "height": height}
    except av.error.InvalidDataError:
        return {"fps": None, "duration": None, "width": None, "height": None}
