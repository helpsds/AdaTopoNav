import numpy as np
import os
import pickle
import yaml
from typing import Any, Dict, List, Optional, Tuple
import tqdm
import io
import lmdb

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Add missing PIL Image import
from PIL import Image

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.gnm.depth_anything_3.utils.io.input_processor import InputProcessor
from vint_train.data.data_utils import get_data_path, resize_and_aspect_crop


class ViNT_Dataset_DA3(ViNT_Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        # Remove image_size since DA3 handles its own sizing
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        min_action_distance: int,
        max_action_distance: int,
        negative_mining: bool,
        len_traj_pred: int,
        learn_angle: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        goals_per_obs: int = 1,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
        # DA3 specific parameters
        da3_process_res: int = 504,
        da3_process_res_method: str = "upper_bound_resize",
        da3_num_workers: int = 1,  # Set to 1 for DataLoader compatibility
        load_original_images: bool = True,
    ):
        """
        ViNT dataset class that uses Depth Anything 3's InputProcessor for image preprocessing.
        
        This class inherits from ViNT_Dataset but overrides the image loading and preprocessing
        to use DA3's sophisticated preprocessing pipeline instead of simple resize and crop.
        """
        # Store DA3 parameters
        self.da3_process_res = da3_process_res
        self.da3_process_res_method = da3_process_res_method
        self.da3_num_workers = da3_num_workers
        self.load_original_images = load_original_images
        
        # Initialize InputProcessor
        self.input_processor = InputProcessor()
        
        # Call parent constructor with dummy image_size (won't be used)
        super().__init__(
            data_folder=data_folder,
            data_split_folder=data_split_folder,
            dataset_name=dataset_name,
            image_size=(224, 224),  # Dummy value, not used
            waypoint_spacing=waypoint_spacing,
            min_dist_cat=min_dist_cat,
            max_dist_cat=max_dist_cat,
            min_action_distance=min_action_distance,
            max_action_distance=max_action_distance,
            negative_mining=negative_mining,
            len_traj_pred=len_traj_pred,
            learn_angle=learn_angle,
            context_size=context_size,
            context_type=context_type,
            end_slack=end_slack,
            goals_per_obs=goals_per_obs,
            normalize=normalize,
            obs_type=obs_type,
            goal_type=goal_type,
        )

    def _load_original_image(self, trajectory_name, time):
        """Load original image without DA3 preprocessing for visualization purposes."""
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            
            # Load PIL image and apply standard resize/crop for visualization
            pil_image = Image.open(image_bytes).convert("RGB")
            # Use standard visualization size (160, 120) from data_utils
            original_tensor = resize_and_aspect_crop(pil_image, (160, 120))
            return original_tensor
            
        except Exception as e:
            print(f"Failed to load original image {image_path}: {e}")
            # Return dummy tensor with visualization size
            return torch.zeros(3, 120, 160)

    def _load_image_da3(self, trajectory_name, time):
        """Load and preprocess a single image using Depth Anything 3's InputProcessor."""
        image_path = get_data_path(self.data_folder, trajectory_name, time)

        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = bytes(image_buffer)
            image_bytes = io.BytesIO(image_bytes)
            
            # Load PIL image
            pil_image = Image.open(image_bytes).convert("RGB")
            
            # Use DA3 InputProcessor to preprocess
            # InputProcessor expects a list, so we wrap the single image
            batch_tensor, _, _ = self.input_processor(
                image=[pil_image],  # Only one image parameter needed
                process_res=self.da3_process_res,
                process_res_method=self.da3_process_res_method,
                num_workers=self.da3_num_workers,
                sequential=True  # Always sequential for single images in DataLoader
            )
            
            # Return single image tensor (3, H, W)
            return batch_tensor.squeeze(0).squeeze(0)  # Remove batch and sequence dims
            
        except Exception as e:
            print(f"Failed to load and preprocess image {image_path}: {e}")
            # Return dummy tensor with expected DA3 output size
            # DA3 typically outputs sizes divisible by 14, so estimate based on process_res
            h = (self.da3_process_res // 14) * 14
            w = (self.da3_process_res // 14) * 14
            return torch.zeros(3, h, w)

    def _load_image(self, trajectory_name, time):
        """Override the parent _load_image method to use DA3 preprocessing."""
        return self._load_image_da3(trajectory_name, time)

    def __getitem__(self, i: int):
        """
        Override __getitem__ to also return original images for visualization.
        Returns additional original_obs_image and original_goal_image tensors.
        """
        # Call parent __getitem__ to get processed images and other data
        parent_result = super().__getitem__(i)
        
        # Extract the original index data
        f_curr, curr_time, max_goal_dist = self.index_to_data[i]
        f_goal, goal_time, goal_is_negative = self._sample_goal(f_curr, curr_time, max_goal_dist)

        if self.load_original_images:
            original_obs_image = self._load_original_image(f_curr, curr_time)
            original_goal_image = self._load_original_image(f_goal, goal_time)
        else:
            original_obs_image = torch.empty(0)
            original_goal_image = torch.empty(0)
        
        # Return: processed_data + original_images
        return parent_result + (original_obs_image, original_goal_image)
