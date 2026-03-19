# AdaTopoNav: Adaptive Topological Navigation Framework

[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-green.svg)](http://wiki.ros.org/noetic)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AdaTopoNav is a highly robust, end-to-end visual navigation framework designed for complex, occlusion-heavy indoor environments. It bridges the gap between Visual Foundation Models (VFMs) and physical robot deployment by integrating an online adaptive topological mapper and a dynamic V-LOS (Visual Line-of-Sight) target adjuster.

![System Architecture](system_architecture.jpg)
*(Please place your architecture diagram image named `system_architecture.jpg` in the root directory)*

---

## ⚙️ Installation & Dependencies

### Prerequisites
* Ubuntu 20.04 + ROS Noetic
* Python 3.8+
* NVIDIA GPU with CUDA support

### Setup Workspace
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[你的用户名]/AdaTopoNav.git
cd AdaTopoNav

# Install Python dependencies
pip install torch torchvision networkx diffusers pyyaml

# Terminal 1: Start ROS core
roscore

Note: The system automatically downloads the DINOv2 weights via torch.hub upon first run. Pre-trained weights for the Nomad diffusion model should be placed in the ../model_weights/ directory (Check config/models.yaml for details).

🚀 Operation Workflow
The complete workflow consists of three stages: Environment Setup, Online Mapping, and Autonomous Navigation.

Step 1: Launch Simulation / Physical Robot
Before running the mapping or navigation nodes, ensure your ROS master is running and the robot's sensors (Camera and Odometry) are publishing to the standard topics.

Bash
# Terminal 1: Start ROS core
roscore

# Terminal 2: Launch your robot's simulation environment or physical chassis drivers
# (Replace the command below with your specific robot's launch file)
# Example for a typical simulation:
roslaunch my_robot_gazebo empty_world.launch
Ensure the following topics are active:

RGB Camera: /camera/color/image_raw

Odometry: /odom

Velocity Command: /cmd_vel

Step 2: Online Topological Mapping (Exploration)
Teleoperate your robot to explore the environment. Run the online_mapper.py to construct the topological graph dynamically. The mapper uses a "Spatial Re-entry Lock" to prevent redundant nodes.

Terminal 3:

Bash
# Start the online mapper and specify a map name
python3 online_mapper.py --map my_custom_map
Action: Drive the robot around your environment manually (e.g., using a joystick or teleop_twist_keyboard).

Save: Press Ctrl+C in the terminal to stop mapping. The system will automatically save the topological graph (nodes, edges, .pt feature vectors, and images) to ~/visualnav-transformer/deployment/topomaps/my_custom_map/.

Step 3: Autonomous Navigation (Deployment)
To navigate to a specific goal, you need to run both the Global Planner (for V-LOS routing) and the Diffusion Navigator (for local collision avoidance).

Terminal 4: Start the Global Planner
The global planner loads the saved map and publishes dynamic lookahead targets.

Bash
# Load the map and set the goal node (e.g., node 25)
# If --goal is set to -1, it defaults to the last node generated during mapping.
python3 global_planner.py --dir my_custom_map --goal 25
Terminal 5: Start the Diffusion Navigator
The navigator subscribes to the dynamic target images and outputs collision-free velocity commands using kinematic velocity modulation.

Bash
# Start the Nomad diffusion policy
python3 navigate_dynamic.py --model nomad --dir my_custom_map
📂 Code Structure
online_mapper.py: Subscribes to /odom and /camera/color/image_raw. Extracts DINOv2 features and builds the graph using dynamic distance sampling.

global_planner.py: Loads the graph, performs Dijkstra routing, detects geometric occlusions via feature cosine similarity, and publishes dynamically adjusted target images (/topoplan/target_image).

Maps_dynamic.py: Subscribes to the target images, runs the diffusion policy, and outputs smooth cmd_vel commands using continuous kinematic velocity modulation (tank-style spin-in-place recovery).

