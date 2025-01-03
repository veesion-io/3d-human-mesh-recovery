import torch
from torch.utils.data import Dataset
import numpy as np
import json
from glob import glob


def compute_timestamp_intersection(window, timespan):
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
    if "Dissimulation Sac" not in actions_timespans:
        return False
    return any(
        compute_timestamp_intersection(timespan, window) > 0.5
        for timespan in actions_timespans["Dissimulation Sac"]
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
        tracks_path="results",
        target_fps=3.0,
        hands_height=128,
        hands_width=128,
        mode="train",
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
        self.target_fps = target_fps
        self.hands_height = hands_height
        self.hands_width = hands_width

    def __len__(self):
        return len(self.videos_meta_data)

    def load_video_tracks(
        self,
        video_name,
        height,
        width,
        bin_size=-1,
        floor_scale=2,
        max_faces_per_bin=30000,
    ):
        video_barename = os.path.splitext(video_name)[0]
        img_folder = os.path.join(self.tracks_path, video_barename, "images")
        hps_folder = os.path.join(self.tracks_path, video_barename, "hps")
        imgfiles = sorted(glob(f"{img_folder}/*.jpg"))
        hps_files = sorted(glob(f"{hps_folder}/*.npy"))
        camera_file = os.path.join(self.tracks_path, video_barename, "camera.npy")
        smpl = SMPL()
        colors = np.loadtxt("data/colors.txt") / 255
        colors = torch.from_numpy(colors).float()

        max_track = len(hps_files)
        track_verts = {i: [] for i in range(len(imgfiles))}
        track_joints = {i: [] for i in range(len(imgfiles))}
        track_tid = {i: [] for i in range(len(imgfiles))}

        ##### TRAM + VIMO #####
        pred_cam = np.load(camera_file, allow_pickle=True).item()
        img_focal = pred_cam["img_focal"].item()
        pred_cam_R = torch.tensor(pred_cam["pred_cam_R"])
        pred_cam_T = torch.tensor(pred_cam["pred_cam_T"])
        locations = []
        tracks_info = {}
        for i in range(max_track):
            hps_file = hps_files[i]

            pred_smpl = np.load(hps_file, allow_pickle=True).item()
            pred_rotmat = pred_smpl["pred_rotmat"]
            pred_shape = pred_smpl["pred_shape"]
            pred_trans = pred_smpl["pred_trans"]
            frame = pred_smpl["frame"]

            mean_shape = pred_shape.mean(dim=0, keepdim=True)
            pred_shape = mean_shape.repeat(len(pred_shape), 1)

            pred = smpl(
                body_pose=pred_rotmat[:, 1:],
                global_orient=pred_rotmat[:, [0]],
                betas=pred_shape,
                transl=pred_trans.squeeze(),
                pose2rot=False,
                default_smpl=True,
            )
            pred_vert = pred.vertices
            pred_j3d = pred.joints[:, :24]

            cam_r = pred_cam_R[frame]
            cam_t = pred_cam_T[frame]

            pred_vert_w = (
                torch.einsum("bij,bnj->bni", cam_r, pred_vert) + cam_t[:, None]
            )
            pred_j3d_w = torch.einsum("bij,bnj->bni", cam_r, pred_j3d) + cam_t[:, None]
            pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w, pred_j3d_w)
            locations.append(pred_j3d_w.mean(1))

            for j, f in enumerate(frame.tolist()):
                track_tid[f].append(i)
                track_verts[f].append(pred_vert_w[j])
                track_joints[f].append(pred_j3d_w[j])
            tracks_info[i] = {"vertices": pred_vert_w, "frames_ids": frame}
        ##### Fit to Ground #####
        grounding_verts = []
        grounding_joints = []
        for t in range(len(imgfiles)):
            try:
                grounding_verts.append(torch.stack(track_verts[t]))
                grounding_joints.append(torch.stack(track_joints[t]))
            except Exception:
                continue

        grounding_verts = torch.cat(grounding_verts)
        grounding_joints = torch.cat(grounding_joints)

        R, offset = fit_to_ground_easy(grounding_verts, grounding_joints)
        offset = torch.tensor([0, offset, 0])
        locations = torch.cat(locations)
        locations = torch.einsum("ij,bj->bi", R, locations) - offset
        cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
        sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * floor_scale

        ##### Viewing Camera #####
        pred_cam = np.load(camera_file, allow_pickle=True).item()
        pred_cam_R = torch.tensor(pred_cam["pred_cam_R"])
        pred_cam_T = torch.tensor(pred_cam["pred_cam_T"])

        cam_R = torch.einsum("ij,bjk->bik", R, pred_cam_R)
        cam_T = torch.einsum("ij,bj->bi", R, pred_cam_T) - offset
        cam_R = cam_R.mT
        cam_T = -torch.einsum("bij,bj->bi", cam_R, cam_T)

        cam_R = cam_R.to("cuda")
        cam_T = cam_T.to("cuda")
        renderer = Renderer(
            width,
            height,
            img_focal,
            "cuda",
            smpl.faces,
            bin_size=bin_size,
            max_faces_per_bin=max_faces_per_bin,
        )
        renderer.set_ground(scale, cx.item(), cz.item())
        cameras, _ = renderer.create_camera_from_cv(cam_R[[0]], cam_T[[0]])
        return cameras, tracks_info

    def crop_track(self, track_info, video_fps, target_fps, timespan):
        step = 1 / target_fps
        timestamps_to_select = np.arange(timespan[0], timespan[1] + step / 2, step)
        frames_ids_to_select = timestamps_to_select * video_fps
        adjusted_frames_ids = find_closest(
            frames_ids_to_select, track_info["frames_ids"]
        )
        return {
            "frames_ids": adjusted_frames_ids,
            "vertices": track_info["vertices"][adjusted_frames_ids],
        }

    def load_hands_regions(self, video_name, video_camera, track_info):
        video_barename = os.path.splitext(video_name)[0]
        img_folder = os.path.join(self.tracks_path, video_barename, "images")
        imgfiles = sorted(glob(f"{img_folder}/*.jpg"))
        track_hands = []
        for frame_id, frame_vertices in zip(
            track_info["frames_ids"], track_info["vertices"]
        ):
            img = cv2.imread(imgfiles[frame_id])
            hands_points = frame_vertices[0, [2500, 5000]]
            screen_points = video_camera.transform_points_screen(
                hands_points, image_size=img.shape[:2]
            )

            # Extract screen coordinates
            x_coords = screen_points[..., 0]
            y_coords = screen_points[..., 1]

            # Convert to image indices (integer pixel indices)
            person_hands = (
                torch.stack([y_coords, x_coords], dim=-1).long().data.cpu().numpy()
            )
            extremal_points = frame_vertices[0, [0, 5000]].to("cuda")
            # Project points to the screen space
            screen_points = video_camera.transform_points_screen(
                extremal_points, image_size=img.shape[:2]
            )

            # Extract screen coordinates
            x_coords = screen_points[..., 0]
            y_coords = screen_points[..., 1]
            points = torch.stack([y_coords, x_coords], dim=-1).long().data.cpu().numpy()
            height = np.sum((points[0, :] - points[1, :]) ** 2) ** 0.5
            frame_hands_regions = []
            for y, x in person_hands:
                dx = int(0.15 * height)
                dy = int(0.21 * height)
                # y = img.shape[1] - y
                x1, y1 = (int(x - dx), int(y - 0.25 * dy))
                x2, y2 = (int(x + dx), int(y + 1.75 * dy))
                frame_hands_regions.append(img[y1:y2, x1:x2])
            track_hands.append(
                cv2.resize(frame_hands_regions, (self.hands_width, self.hands_height))
            )
        return track_hands

    def track_in_window(self, video_fps, track_info, window):
        track_timespan = [
            track_info["frames_ids"][0] / video_fps,
            track_info["frames_ids"][1] / video_fps,
        ]
        return compute_timestamp_intersection(window, track_timespan) > 0.5

    def __getitem__(self, index):
        video_name = self.videos_names[index]
        video_meta_data = self.videos_meta_data[video_name]
        start_time = np.random.uniform(
            0, max(0, video_meta_data["duration"] - self.window_duration)
        )
        end_time = min(video_meta_data["duration"], start_time + self.window_duration)
        # Check if metadata is missing or invalid
        try:
            video_camera, video_tracks = self.load_video_tracks(
                video_name, video_meta_data["height"], video_meta_data["width"]
            )
        except:
            raise FileNotFoundError(
                f"Metadata for video {video_name} is missing or invalid."
            )

        tracks_data = []
        for track_id, track_info in video_tracks.items():
            if not self.track_in_window(
                video_meta_data["fps"], track_info, [start_time, end_time]
            ):
                continue
            cropped_track_info = self.crop_track(
                track_info,
                video_meta_data["fps"],
                self.target_fps,
                [start_time, end_time],
            )
            hands_regions = self.load_hands_regions(video_camera, cropped_track_info)
            tracks_data.append(
                (
                    torch.from_numpy(cropped_track_info["vertices"][:, ::120]),
                    torch.from_numpy(hands_regions),
                )
            )
        label = find_window_label(video_meta_data, [start_time, end_time])
        if len(tracks_data) == 0:
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
        return formatted_data
