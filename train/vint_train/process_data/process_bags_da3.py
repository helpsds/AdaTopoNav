# #!/usr/bin/env python3
# """
# Preprocess ROS bag data using Depth Anything 3's InputProcessor for GNM training.

# This script processes ROS bag files and applies the same preprocessing that will be
# used during GNM training, ensuring consistency between preprocessing and training.
# """

# import os
# import sys
# import argparse
# import yaml
# import rosbag
# from PIL import Image
# import numpy as np
# import torch
# from typing import List, Dict, Any, Optional

# # Add the current directory to Python path to import local modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from process_data.process_data_utils import process_images, process_tartan_img, process_locobot_img, process_scand_img
# from models.gnm.depth_anything_3.utils.io.input_processor import InputProcessor


# def get_image_processor(dataset_type: str):
#     """Get the appropriate image processor function based on dataset type."""
#     if dataset_type == "tartan_drive":
#         return process_tartan_img
#     elif dataset_type == "locobot":
#         return process_locobot_img
#     elif dataset_type == "scand":
#         return process_scand_img
#     else:
#         raise ValueError(f"Unknown dataset type: {dataset_type}")


# def preprocess_with_da3(
#     images: List[Image.Image],
#     input_processor: InputProcessor,
#     process_res: int = 504,
#     process_res_method: str = "upper_bound_resize",
#     num_workers: int = 4,
# ) -> torch.Tensor:
#     """
#     Preprocess a list of PIL images using Depth Anything 3's InputProcessor.
    
#     Args:
#         images: List of PIL images to preprocess
#         input_processor: Initialized InputProcessor instance
#         process_res: Target resolution for preprocessing
#         process_res_method: Preprocessing method
#         num_workers: Number of parallel workers
        
#     Returns:
#         Preprocessed tensor of shape (N, 3, H, W)
#     """
#     # Convert PIL images to numpy arrays for InputProcessor compatibility
#     image_arrays = [np.array(img) for img in images]
    
#     # Use InputProcessor to handle all preprocessing
#     batch_tensor, _, _ = input_processor(
#         image=image_arrays,
#         process_res=process_res,
#         process_res_method=process_res_method,
#         num_workers=num_workers,
#         sequential=(num_workers <= 1)
#     )
    
#     # InputProcessor returns (1, N, 3, H, W), we need (N, 3, H, W)
#     return batch_tensor.squeeze(0)


# def process_single_bag(
#     bag_path: str,
#     config: Dict[str, Any],
#     output_dir: str,
#     input_processor: InputProcessor,
# ) -> None:
#     """
#     Process a single ROS bag file using Depth Anything 3 preprocessing.
    
#     Args:
#         bag_path: Path to the ROS bag file
#         config: Configuration dictionary from YAML file
#         output_dir: Directory to save processed data
#         input_processor: Initialized InputProcessor instance
#     """
#     print(f"Processing bag: {bag_path}")
    
#     bag_name = os.path.splitext(os.path.basename(bag_path))[0]
#     bag_output_dir = os.path.join(output_dir, bag_name)
#     os.makedirs(bag_output_dir, exist_ok=True)
    
#     try:
#         bag = rosbag.Bag(bag_path, 'r')
#     except Exception as e:
#         print(f"Error opening bag {bag_path}: {e}")
#         return
    
#     # Get configuration parameters
#     dataset_type = config.get('dataset', 'tartan_drive')
#     image_topic = config.get('image_topic', '/camera/color/image_raw')
#     process_res = config.get('process_res', 504)
#     process_res_method = config.get('process_res_method', 'upper_bound_resize')
#     num_workers = config.get('num_workers', 4)
    
#     # Get the appropriate image processor
#     img_processor_func = get_image_processor(dataset_type)
    
#     # Extract all images from the bag
#     image_msgs = []
#     for topic, msg, t in bag.read_messages(topics=[image_topic]):
#         image_msgs.append(msg)
    
#     print(f"Extracted {len(image_msgs)} images from bag")
    
#     if len(image_msgs) == 0:
#         print(f"No images found in bag {bag_path}")
#         bag.close()
#         return
    
#     # Process images to PIL format
#     pil_images = process_images(image_msgs, img_processor_func)
#     print(f"Converted {len(pil_images)} images to PIL format")
    
#     # Apply Depth Anything 3 preprocessing
#     processed_tensor = preprocess_with_da3(
#         images=pil_images,
#         input_processor=input_processor,
#         process_res=process_res,
#         process_res_method=process_res_method,
#         num_workers=num_workers
#     )
    
#     print(f"Preprocessed tensor shape: {processed_tensor.shape}")
    
#     # Save the preprocessed data
#     output_file = os.path.join(bag_output_dir, 'preprocessed_images.pt')
#     torch.save(processed_tensor, output_file)
#     print(f"Saved preprocessed data to {output_file}")
    
#     # Also save metadata
#     metadata = {
#         'bag_path': bag_path,
#         'num_images': len(pil_images),
#         'tensor_shape': processed_tensor.shape,
#         'process_res': process_res,
#         'process_res_method': process_res_method,
#         'dataset_type': dataset_type
#     }
#     metadata_file = os.path.join(bag_output_dir, 'metadata.yaml')
#     with open(metadata_file, 'w') as f:
#         yaml.dump(metadata, f)
#     print(f"Saved metadata to {metadata_file}")
    
#     bag.close()


# def main():
#     parser = argparse.ArgumentParser(description='Preprocess ROS bag data with Depth Anything 3 for GNM training')
#     parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
#     parser.add_argument('--input_dir', type=str, required=True, help='Directory containing ROS bag files')
#     parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed data')
#     parser.add_argument('--bag_file', type=str, help='Process a specific bag file instead of entire directory')
    
#     args = parser.parse_args()
    
#     # Load configuration
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Initialize InputProcessor
#     input_processor = InputProcessor()
    
#     if args.bag_file:
#         # Process a single bag file
#         process_single_bag(args.bag_file, config, args.output_dir, input_processor)
#     else:
#         # Process all bag files in the input directory
#         bag_files = [f for f in os.listdir(args.input_dir) if f.endswith('.bag')]
#         print(f"Found {len(bag_files)} bag files to process")
        
#         for bag_file in bag_files:
#             bag_path = os.path.join(args.input_dir, bag_file)
#             process_single_bag(bag_path, config, args.output_dir, input_processor)
    
#     print("Preprocessing completed!")


# if __name__ == '__main__':
#     main()