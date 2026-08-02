from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

AgentInput = Any


class AbstractFeatureBuilder:
    @abstractmethod
    def get_unique_name(self) -> str:
        pass

    @abstractmethod
    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        pass


@dataclass
class NavsimV2FeatureConfig:
    video_height: int = 384
    video_width: int = 640
    crop_top_bottom: int = 28
    num_history_frames: int = 5

    def __post_init__(self) -> None:
        self.video_height = int(self.video_height)
        self.video_width = int(self.video_width)
        self.crop_top_bottom = int(self.crop_top_bottom)
        self.num_history_frames = int(self.num_history_frames)

    @property
    def video_size(self) -> Tuple[int, int]:
        return (self.video_height, self.video_width)

def _resize_rgb_to_tensor(image: np.ndarray, video_size: Tuple[int, int], crop_top_bottom: int) -> torch.Tensor:
    if image is None:
        raise RuntimeError("Expected a loaded NAVSIM camera image, got None.")
    crop = int(crop_top_bottom)
    if crop > 0 and image.shape[0] > 2 * crop:
        image = image[crop:-crop]
    target_h, target_w = video_size
    resized = Image.fromarray(image).resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class NavsimV2FeatureBuilder(AbstractFeatureBuilder):
    """Emits front-camera history and ego-status tensors."""

    def __init__(self, config: NavsimV2FeatureConfig):
        self._config = config

    def get_unique_name(self) -> str:
        return "navsim_v2_feature"

    def _front_frames(self, agent_input: AgentInput) -> List[torch.Tensor]:
        frames = [
            _resize_rgb_to_tensor(cameras.cam_f0.image, self._config.video_size, self._config.crop_top_bottom)
            for cameras in agent_input.cameras
            if cameras.cam_f0.image is not None
        ]
        if not frames:
            raise RuntimeError("NAVSIM v2 adapter did not receive any front-camera history frames.")
        return frames

    def _front_projection_metadata(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        camera = agent_input.cameras[-1].cam_f0
        target_h, target_w = self._config.video_size
        if camera.image is None:
            raise RuntimeError("NAVSIM v2 adapter expected a loaded cam_f0 image.")
        if (
            camera.sensor2lidar_rotation is None
            or camera.sensor2lidar_translation is None
            or camera.intrinsics is None
        ):
            return {
                "lidar2img": torch.eye(4, dtype=torch.float32),
                "img_shape": torch.as_tensor([target_h, target_w, 3], dtype=torch.float32),
            }

        lidar2cam_r = np.linalg.inv(np.asarray(camera.sensor2lidar_rotation, dtype=np.float32))
        lidar2cam_t = np.asarray(camera.sensor2lidar_translation, dtype=np.float32) @ lidar2cam_r.T
        lidar2cam_rt = np.eye(4, dtype=np.float32)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t

        intrinsic = np.asarray(camera.intrinsics, dtype=np.float32)
        viewpad = np.eye(4, dtype=np.float32)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        lidar2img = viewpad @ lidar2cam_rt.T

        crop = int(self._config.crop_top_bottom)
        if crop > 0 and camera.image.shape[0] > 2 * crop:
            cropped_h = camera.image.shape[0] - 2 * crop
        else:
            cropped_h = camera.image.shape[0]
        image_transform = np.eye(4, dtype=np.float32)
        image_transform[0, 0] = float(target_w) / float(camera.image.shape[1])
        image_transform[1, 1] = float(target_h) / float(cropped_h)
        if cropped_h != camera.image.shape[0]:
            image_transform[1, 2] = -float(crop) * image_transform[1, 1]

        return {
            "lidar2img": torch.as_tensor(image_transform @ lidar2img, dtype=torch.float32),
            "img_shape": torch.as_tensor([target_h, target_w, 3], dtype=torch.float32),
        }

    @staticmethod
    def _ego_status_tensor(agent_input: AgentInput) -> torch.Tensor:
        ego_features: List[torch.Tensor] = []
        for ego_status in agent_input.ego_statuses:
            ego_features.append(
                torch.cat(
                    [
                        torch.as_tensor(ego_status.ego_pose, dtype=torch.float32),
                        torch.as_tensor(ego_status.ego_velocity, dtype=torch.float32),
                        torch.as_tensor(ego_status.ego_acceleration, dtype=torch.float32),
                        torch.as_tensor(ego_status.driving_command, dtype=torch.float32),
                    ],
                    dim=-1,
                )
            )
        if not ego_features:
            raise RuntimeError("NAVSIM v2 adapter did not receive any ego-status history.")
        return torch.stack(ego_features, dim=0)

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        front_frames = self._front_frames(agent_input)
        expected_history = int(self._config.num_history_frames)
        if len(front_frames) != expected_history:
            raise RuntimeError(
                f"NAVSIM history frame mismatch: got {len(front_frames)}, expected {expected_history}."
            )
        ego_status = self._ego_status_tensor(agent_input)
        if int(ego_status.shape[0]) != expected_history:
            raise RuntimeError(
                f"NAVSIM ego-status history mismatch: got {int(ego_status.shape[0])}, expected {expected_history}."
            )
        features: Dict[str, torch.Tensor] = {
            "video_front": torch.stack(front_frames, dim=0),
            "ego_status": ego_status,
        }
        features.update(self._front_projection_metadata(agent_input))
        return features
