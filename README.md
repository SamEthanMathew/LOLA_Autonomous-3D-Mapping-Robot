# LOLA - Autonomous 3D-Mapping Robot

**Built in one week for Build18 2026**

![Build18](https://img.shields.io/badge/program-Build18-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%204-red)

> We built an autonomous 3D-mapping robot in one week.  
> Not a simulation. Not a demo script.  
> A real rover that drives and reconstructs the room around it.

---

## Overview

**LOLA** is a $150 mecanum-drive RC rover running on a Raspberry Pi 4 that autonomously explores and reconstructs 3D environments. The system combines LiDAR-based SLAM with RGB-only 3D reconstruction to create detailed spatial maps, all without relying on expensive depth cameras.

### Key Features

- ✅ **Real-time autonomous navigation** using LiDAR SLAM
- ✅ **3D reconstruction** from RGB video only (no depth camera needed)
- ✅ **Point-to-Line ICP (PL-ICP)** for faster and more accurate SLAM convergence
- ✅ **Cost-effective** - total build cost under $150
- ✅ **Accessible workflow** - heavy compute offloaded to Google Colab
- ✅ **Open-source** - designed for easy replication

---

## Technical Approach

### Hardware
- **Platform**: Mecanum-drive RC rover chassis
- **Compute**: Raspberry Pi 4
- **Sensor**: RPLiDAR A1/A2 for SLAM
- **Camera**: Standard RGB camera for 3D reconstruction
- **Total Cost**: ~$150

## 3D Printing

We 3D-printed several custom mounts and structural parts for LOLA using:

- **Printer**: BambuLab X1
- **Material**: PLA filament
- **Supports**: Not required (all parts were designed to print cleanly without supports)

These printed components helped with sensor mounting and securing electronics to the chassis.


## Software Architecture

```
┌─────────────────────────────────────┐
│         Raspberry Pi 4              │
│  • Motor Control                    │
│  • LiDAR SLAM (PL-ICP)              │
│  • Path Planning (RRT)              │
│  • Autonomous Navigation            │
└──────────────┬──────────────────────┘
               │
               │ Video Stream / Map Data
               ▼
┌─────────────────────────────────────┐
│    Offboard Compute (Colab/Laptop)  │
│  • CUT3R 3D Reconstruction          │
│  • Point Cloud Processing           │
└─────────────────────────────────────┘
```

### SLAM Pipeline

Our LiDAR-based SLAM implementation uses:

1. **Point-to-Line ICP (PL-ICP)** instead of traditional point-to-point ICP
   - Faster convergence
   - More accurate pose estimation
   - Better handling of sparse scan data

2. **SVD-based closed-form solve** for transformation estimation
   - Efficient computation
   - Numerically stable

3. **KD-tree for nearest-neighbor matching**
   - O(log n) query time
   - Efficient scan alignment

4. **Occupancy grid mapping** with dynamic updates
   - Real-time map visualization
   - Export to PNG for path planning

### 3D Reconstruction

We use **CUT3R** (Camera-Unified Transformer for 3D Reconstruction):
- Reconstructs 3D scenes using **only RGB video** + camera intrinsics/extrinsics
- No expensive depth cameras or structured light sensors needed
- Modified pipeline for easy iteration and team accessibility
- Runs on Google Colab for reproducibility

---

## Repository Structure

```
Build18-Contribution/
├── LiDARSensor.py              # SLAM implementation (PL-ICP, mapping, visualization)
├── path_planner.py             # RRT-based path planning on occupancy grids
├── example_use_path.py         # Example script for autonomous navigation
├── Vidto3Dmodel_Reconstruction.ipynb  # CUT3R reconstruction notebook (Colab)
├── drive/
│   └── manual_motor_control.py # Motor control interface
├── map_exports/                # Generated SLAM maps
└── path_outputs/               # Planned paths (JSON + visualization)
```

---

## Bill of Materials (BOM)

> Note: This BOM does **not** include the Raspberry Pi 4 and miscellaneous wiring/fasteners, which were not included in cost.

| # | Item | Qty | Unit Cost (USD) | Total (USD) | Product Link |
|---:|------|---:|----------------:|------------:|--------------|
| 1 | Motor Driver | 1 | 10.59 | 10.59 | [L298N Motor Driver](https://www.amazon.com/dp/B0C5JCF5RS?ref=ppx_yo2ov_dt_b_fed_asin_title) |
| 2 | Robot Car Chassis | 1 | 39.21 | 39.21 | [Mecanum Robot Chassis](https://www.seeedstudio.com/Robot-car-Kit-RC-Smart-Car-Chassis-p-4226.html) |
| 3 | LiDAR Sensor | 1 | 104.94 | 104.94 | [RPLIDAR A1M8](https://a.co/d/cjdxPnk) |
| 4 | Pi4 Power Supply | 1 | ~ | ~ | Used a power bank |
| 5 | 9V Battery | 2 | ~ | ~ | 9V Alkaline Batteries |
|  | **Total** |  |  | **154.74** |  |



## Installation & Setup

### On Raspberry Pi 4

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/lola-robot.git
   cd lola-robot/Build18-Contribution
   ```

2. **Install dependencies**
   ```bash
   pip install numpy scipy pygame opencv-python rplidar-roboticia numba
   ```

3. **Connect LiDAR sensor**
   - Plug RPLiDAR into USB port
   - Verify with `ls /dev/ttyUSB*`

### For 3D Reconstruction (Colab)

1. Open `Vidto3Dmodel_Reconstruction.ipynb` in Google Colab
2. Upload your RGB video from the rover
3. Follow the notebook steps to generate 3D point clouds

---

## Usage

### 1. Run SLAM and Generate Map

```bash
python LiDARSensor.py
```

This will:
- Initialize the LiDAR sensor
- Start real-time SLAM with visualization
- Save occupancy grid maps to `map_exports/`

**Controls** (during SLAM):
- `S` - Save current map
- `R` - Reset map
- `ESC` - Exit

### 2. Plan Exploration Path

```bash
python path_planner.py map_exports/slam_map_YYYYMMDD-HHMMSS.png
```

This generates:
- `path_outputs/path_coords_*.json` - Waypoint coordinates
- `path_outputs/path_map_*.png` - Visualization of planned path

### 3. Execute Autonomous Navigation

```bash
python example_use_path.py
```

Loads the planned path and sends motor commands to follow waypoints autonomously.

### 4. 3D Reconstruction (Offboard)

1. Record RGB video while rover explores
2. Upload to Colab
3. Run `Vidto3Dmodel_Reconstruction.ipynb`
4. Download the reconstructed 3D model

---

## Results

- **SLAM Performance**: Real-time mapping at 10Hz with consistent loop closure
- **Path Planning**: RRT generates collision-free paths in <2 seconds
- **3D Reconstruction**: High-quality depth visualization showing room structure, furniture, and wall details
- **Autonomy**: Successfully navigates rooms without human intervention

---

## Team

Built with ❤️ by:
- **[Sam](https://www.linkedin.com/in/sam-mathew-1a9778254/)**
- **[Darren](https://www.linkedin.com/in/darrpinto/)**
- **[Tanay](https://www.linkedin.com/in/tanay-mishra-a86b25274/)**
- **[Jayden](https://www.linkedin.com/in/jayden-chen-3038a22a7/)**
- **[Jintong](https://www.linkedin.com/in/jintong-wang-096645289/)**

---

## Acknowledgments

Huge thanks to:
- The **Build18** organizing team
- Carnegie Mellon **ECE Department**
- All mentors who guided us throughout the sprint
- Open-source projects: CUT3R, RPLiDAR drivers, NumPy, SciPy
---

## Citation

If you use CUT3R in your work, please cite:

```bibtex
@inproceedings{cut3r,
  author    = {Qianqian Wang* and Yifei Zhang* and Aleksander Holynski and Alexei A. Efros and Angjoo Kanazawa},
  title     = {Continuous 3D Perception Model with Persistent State},
  booktitle = {CVPR},
  year      = {2025}
}
```

## License

This project is open-source and available under the MIT License.

---


## References

- [CUT3R: Camera-Unified Transformer for 3D Reconstruction](https://github.com/nianticlabs/cutr)
- [Point-to-Line ICP](https://ieeexplore.ieee.org/document/6906555)
- [RPLiDAR Documentation](https://www.slamtec.com/en/Lidar/A1)

---

**Built for Build18 2026** - One week. One robot. Real results.

