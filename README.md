# SAM-6D: 6DoF Object Pose Estimation for Lynxmotion AL5D

This repository contains the implementation of SAM-6D with FastSAM for 6DoF object pose estimation, specifically optimized for the Lynxmotion AL5D robotic arm. The method enables accurate object pose estimation using limited computational resources, making advanced computer vision techniques accessible for educational and research robotics.

![Lynxmotion AL5D robot arm](images/robot_arm.png)

## Overview

SAM-6D is a novel approach that combines segment anything model (SAM) with a two-step pose estimation pipeline for efficient 6DoF object pose estimation. Our implementation specifically adapts this model for the Lynxmotion AL5D robotic arm, a cost-effective platform commonly used in educational settings.

Key features:
- FastSAM for efficient object segmentation with reduced computational requirements
- Two-stage point matching for accurate pose estimation
- Optimized for limited hardware resources
- Suitable for real-world robotic manipulation tasks

## Directory Structure

```
SAM-6D/
├── Data/                            # Dataset and example files
├── Instance_Segmentation_Model/     # SAM-based segmentation model
├── Pose_Estimation_Model/           # Two-stage pose estimation module
├── Render/                          # 3D model rendering for templates
├── demo.sh                          # Demo script for running the pipeline
├── environment.yaml                 # Conda environment specification
├── prepare.sh                       # Setup script for environment preparation
└── README.md                        # Documentation (this file)
```

## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+
- Conda package manager
- Blender 3.3.1+ (for rendering templates)

### Setting up the Environment

1. Clone the repository:
```bash
git clone https://github.com/joeltooni/acv_project.git
cd acv_project
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate sam6d
```

3. Run the preparation script to download pretrained models:
```bash
bash prepare.sh
```

## Usage

### Running the Complete Pipeline

The `demo.sh` script demonstrates the complete pipeline from template rendering to pose estimation:

```bash
bash demo.sh
```

This script:
1. Renders 3D object model templates using Blender
2. Runs the instance segmentation model (SAM or FastSAM)
3. Performs pose estimation using the segmented object

### Customizing for Your Own Objects

To use the system with your own objects:

1. Prepare a 3D model of your object in PLY format
2. Update the paths in `demo.sh` to point to your model and desired output directory
3. Capture RGB and depth images of the scene with your object
4. Ensure camera intrinsic parameters are available in JSON format
5. Run the modified script

## Technical Details

### Instance Segmentation Model (ISM)

The ISM employs either SAM or FastSAM to generate object proposals. Each proposal is evaluated using a comprehensive matching score that considers:

- Semantic matching: Measures similarity between proposals and template renderings
- Appearance matching: Evaluates visual similarity with templates
- Geometric matching: Calculates IoU between proposal bounding box and projected object

### Pose Estimation Model (PEM)

The PEM uses a two-stage point matching approach:

1. Coarse Point Matching: Establishes sparse correspondence between proposal and object model
2. Fine Point Matching: Refines the estimated pose using denser point sets

### Performance

Our implementation achieves significant performance improvements while maintaining accuracy:

| Method | Segmentation (s) | Pose Estimation (s) | Total (s) |
|--------|------------------|---------------------|-----------|
| SAM-6D with FastSAM | 0.45 | 0.98 | 1.43 |
| SAM-6D with SAM | 2.80 | 1.57 | 4.37 |

## Applications

This implementation is particularly suited for:
- Educational robotics labs
- Research environments with limited computational resources
- Pick-and-place tasks with affordable robotic arms
- Object manipulation in cluttered environments

## Future Work

Future development will focus on:
- Improving depth fusion through plenoptic data
- Further optimizing latency through model quantization and pruning
- Implementing adaptive learning capabilities for new object classes

## Citation

If you use this code in your research, please cite our work:

```
@inproceedings{adebayo2025sam6d,
  title={6DoF Object pose estimation using SAM-6D with FastSAM for Lynxmotion AL5D},
  author={Adebayo, Oluwatooni and Tuyisenge, Floride and Nwovu, Sunday},
  booktitle={Proceedings of Computer Vision Conference},
  year={2025}
}
```

## Acknowledgments

This work builds upon the SAM-6D framework by Lin et al. and the FastSAM implementation by Zhao et al.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact:
- Oluwatooni Adebayo - jadebayo@andrew.cmu.edu
- Floride Tuyisenge - ftuyisen@andrew.cmu.edu
- Sunday Nwovu - snwovu@andrew.cmu.edu
