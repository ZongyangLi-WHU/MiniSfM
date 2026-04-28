# MiniSfM: A Lightweight C++ Incremental Structure-from-Motion System

本项目是一个完全基于 C++ 从零构建的轻量级增量式三维重建系统。项目深入参考了 COLMAP 的核心架构设计，专注于底层多视图几何逻辑与 C++ 内存性能优化。

## 🚀 Core Features (核心特性)

* **Frontend Data Association**: 基于 OpenCV 实现特征提取与匹配，手写 Lowe's Ratio Test 剔除误匹配 。
* **Incremental Registration**: 设计高效的 2D-3D Track 映射数据结构，基于 PnP 算法与三角化实现鲁棒的增量式点云生长 。
* **Bundle Adjustment (BA)**: 深度集成 Ceres Solver，构建重投影误差代价函数，支持全局光束法平差。
* **Local BA (Sliding Window)**: 利用共视图 (Covisibility Graph) 提取局部子地图进行非线性优化，冻结历史相机参数，极大提升大规模数据处理效率 。
* **Error Analysis & Cleaning**: 独立的重投影误差量化评估模块，自动计算 RMSE 并剔除严重漂移的离群点。
* **3DGS / NeRF Ready**: 完美打通前沿生态，支持将相机内参、位姿和 3D 点云严格按照 COLMAP 格式导出 (`cameras.txt`, `images.txt`, `points3D.txt`)，可直接作为 3D Gaussian Splatting 的训练输入。

## 🛠️ Dependencies (依赖项)

* OpenCV 4.x
* Eigen 3
* Ceres Solver (SuiteSparse backend recommended)

## 🎯 Build & Run

```bash
mkdir build && cd build
cmake ..
make -j4
./sfm_test
