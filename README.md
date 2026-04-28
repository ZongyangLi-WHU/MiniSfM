# MiniSfM: A Lightweight C++ Incremental Structure-from-Motion System

**MiniSfM** is a lightweight C++ incremental Structure-from-Motion prototype built for understanding the core geometry pipeline behind COLMAP and classical photogrammetry-based 3D reconstruction.

This project is not intended to replace COLMAP. Instead, it implements a compact SfM workflow from feature extraction to incremental mapping and bundle adjustment, with an emphasis on multi-view geometry, data association, and C++ engineering practice.

---

## Overview

The system takes a small set of calibrated or approximately calibrated images as input and reconstructs a sparse 3D point cloud through an incremental SfM pipeline.

Current pipeline:

```text
Input images
    ↓
SIFT feature extraction
    ↓
FLANN matching + Lowe Ratio Test
    ↓
Global match graph construction
    ↓
Two-view bootstrapping
Essential Matrix + recoverPose
    ↓
Initial triangulation
    ↓
Next-Best-View selection
based on 2D-3D correspondences
    ↓
PnP-RANSAC camera registration
    ↓
Incremental triangulation
    ↓
Local Bundle Adjustment
Ceres reprojection error optimization
    ↓
Reprojection error analysis and outlier cleaning
    ↓
PLY / COLMAP-style TXT export
```
## Motivation

I developed this project to better understand the geometry foundation of modern 3D reconstruction systems such as COLMAP, 3D Gaussian Splatting, and NeRF-style pipelines.

While tools like COLMAP are powerful and mature, their internal pipeline can feel like a black box. MiniSfM is a learning-oriented implementation that explicitly exposes the key steps:

feature extraction and matching
two-view initialization
triangulation
2D-3D track management
PnP-based incremental registration
bundle adjustment
reprojection error analysis
sparse point cloud export

This project also serves as a geometry-side complement to my later experiments with 3DGS and open-vocabulary 3D scene understanding.

## Features
**1. Feature Extraction and Matching**
Uses OpenCV SIFT for keypoint and descriptor extraction.
Uses FLANN-based matching for descriptor search.
Applies Lowe Ratio Test to remove ambiguous matches.
Encapsulates per-image features in a FeatureData structure.

```cpp
struct FeatureData {
    int image_id;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> point3d_idx;
};
```
The `point3d_idx` field records whether each 2D feature has already been associated with a global 3D point.

**2. Global Match Graph**
MiniSfM builds a global image-pair match graph before incremental reconstruction.

For each image pair, the system computes feature matches and stores sufficiently strong connections in a symmetric graph:
```cpp
match_graph_[image_i][image_j] = matches
match_graph_[image_j][image_i] = reversed_matches
```
This graph is later used for:

- selecting the initial image pair
- finding the next best view
- collecting 2D-3D correspondences
- identifying covisible cameras for local BA

**3. Two-View Bootstrapping**

The system initializes reconstruction by selecting the image pair with the largest number of feature matches.

For the selected pair:

- Estimate the Essential Matrix with RANSAC.
- Recover relative rotation and translation using recoverPose.
- Set the first camera as the world reference.
- Triangulate initial sparse 3D points.
- Build initial 2D-3D track relationships.

**4. Incremental Camera Registration**

After initialization, new images are added incrementally.

For each pending image, MiniSfM selects the next best view by counting how many valid 2D-3D correspondences it has with already registered images.

The selected frame is localized using PnP-RANSAC:

```txt
known 3D points + corresponding 2D keypoints
        ↓
solvePnPRansac
        ↓
new camera pose
```
This design follows the core idea of incremental SfM: register a new camera using existing map points, then grow the map with new triangulated points.

**5. Track Management**

Each 3D point stores a multi-view observation track:

```cpp
struct Point3D {
    cv::Point3d pt;
    std::map<int, int> track;  // image_id -> keypoint index
};
```

This allows the system to answer:

```txt
Which images observe this 3D point?
Which keypoint in each image corresponds to it?
```

This track structure is used by both PnP registration and Bundle Adjustment.

**6. Incremental Triangulation**

After a new camera is registered, MiniSfM triangulates new points between the new frame and existing registered frames.

To reduce degenerate triangulation, the system checks the camera baseline before triangulating:

```txt
C = -R^T t
baseline = || C_new - C_old ||
```

If the baseline is too small, triangulation is skipped to avoid unstable depth estimation caused by near-pure rotation or insufficient parallax.

**7. Local Bundle Adjustment**

MiniSfM uses Ceres Solver to perform reprojection-error-based Bundle Adjustment.

The reprojection error is defined as:

```txt
3D point
  → camera coordinate system
  → perspective projection
  → pixel coordinate
  → residual with observed 2D keypoint
```

The system optimizes:

- camera pose parameters
- 3D point coordinates

and keeps camera intrinsics fixed.

For incremental reconstruction, the system performs a local BA strategy:

1. Activate the newly registered camera.
2. Select several highly covisible registered cameras.
3. Optimize active cameras and map points.
4. Freeze historical cameras outside the active window.

This is a simplified local optimization strategy designed for learning and small-scale experiments.

**8. Reprojection Error Analysis and Outlier Cleaning**

After local BA, MiniSfM evaluates reprojection error for all visible 3D point observations.

It reports:

- total evaluated observations
- mean reprojection error
- maximum reprojection error
- number of points removed as outliers
- final number of retained 3D points

Points with reprojection error above a threshold are removed from the sparse map.

This module helps connect the geometric optimization result with a quantitative quality measure.

**9. Export**

MiniSfM currently supports two export formats:

PLY point cloud
final_incremental_cloud.ply

This can be opened in tools such as MeshLab or CloudCompare.

```txt
COLMAP-style text files
cameras.txt
images.txt
points3D.txt
```

The export module writes camera intrinsics, camera poses, 2D observations, 3D point coordinates, and track information in a COLMAP-like text format.

This part is mainly used for understanding how SfM outputs can be organized for downstream 3D reconstruction pipelines.

## Project Structure
```txt
MiniSfM/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── feature.h
│   ├── geometry.h
│   ├── optimization.h
│   └── sfm_system.h
├── src/
│   ├── feature.cpp
│   ├── geometry.cpp
│   ├── main.cpp
│   └── sfm_system.cpp
└── data/
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
    ├── final_incremental_cloud.ply
    └── colmap_workspace/
```

## Dependencies
- C++17
- OpenCV
- Eigen3
- Ceres Solver

Recommended environment:
```txt
Ubuntu / Linux
OpenCV 4.x
Ceres Solver
Eigen3
CMake >= 3.10
```
## Build
```bah
mkdir build
cd build
cmake ..
make -j4
```
## Run

The current demo reads three images from the data/ folder:

```txt
data/img1.jpg
data/img2.jpg
data/img3.jpg
```

**Run:**

```bah
cd build
./sfm_test
```

The program will:

- load and resize the images to 800 × 600
- extract SIFT features
- build the global match graph
- bootstrap the initial two-view map
- register the remaining view incrementally
- run local bundle adjustment
- clean high-error points
- export sparse point cloud and COLMAP-style files

## Output

Typical outputs include:
```txt
data/final_incremental_cloud.ply
data/colmap_workspace/sparse/0/cameras.txt
data/colmap_workspace/sparse/0/images.txt
data/colmap_workspace/sparse/0/points3D.txt
```

## Current Limitations

This project is a learning-oriented SfM prototype. It still has several limitations:

- The demo currently uses a small image set.
- Camera intrinsics are manually specified and simplified.
- No full camera calibration module is included.
- Distortion parameters are not modeled.
- Loop closure is not implemented.
- Large-scale track merging and robust global reconstruction are not yet implemented.
- The COLMAP-style export is intended for understanding data organization and may still require further adaptation for direct use in downstream pipelines.

## Future Work

Planned improvements include:

- support for more input images
- camera calibration or EXIF-based intrinsic estimation
- stronger RANSAC inlier filtering before triangulation
- improved track merging across multiple views
- global Bundle Adjustment after full registration
- better sparse point visualization
- stricter COLMAP compatibility checking
- integration with 3DGS / NeRF preprocessing workflows

## Positioning

MiniSfM is mainly a geometry learning and engineering practice project.

It demonstrates my understanding of:

- multi-view geometry
- SfM pipeline design
- 2D-3D data association
- PnP registration
- triangulation
- Bundle Adjustment
- C++ modular system implementation

It is also intended to support my broader interest in computational photogrammetry, 3D reconstruction, and the intersection of surveying, remote sensing, and computer vision.
