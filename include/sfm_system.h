#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>


#include "feature.h"  // 引入 FeatureData, FeatureExtractor, FeatureMatcher
#include "geometry.h" // 引入 Point3D, PoseEstimator

class SfMSystem {
public:
    // 构造函数：传入相机内参 K
    SfMSystem(const cv::Mat& K);

    // 外部接口：添加图像并触发特征提取
    void addImage(const cv::Mat& image, int frame_id);

    // 外部接口：启动全自动重建流程
    void runReconstruction();

    // 外部接口：保存最终点云 (可直接调用 PoseEstimator::savePLY)
    void exportPointcloud(const std::string& filename) const;
    void exportToCOLMAP(const std::string& export_dir) const;
private:
    // --- 核心状态数据 (State Variables) ---
    cv::Mat K_;                                    
    std::map<int, FeatureData> all_features_;       // 存储所有帧特征
    std::map<int, cv::Mat> camera_R_;               // 存储已注册相机的 R
    std::map<int, cv::Mat> camera_t_;               // 存储已注册相机的 t
    std::vector<Point3D> global_points_;            // 全局 3D 点云地图

    // --- 调度器状态 (Scheduler State) ---
    int current_frame_id_;                          // 当前正在处理的帧 ID
    std::set<int> registered_frames_;               // 已加入地图的帧 ID 集合
    std::set<int> pending_frames_;                  // 待处理的帧 ID 集合
    
    // 全局匹配图：[图A][图B] -> 匹配点对列表
    std::map<int, std::map<int, std::vector<cv::DMatch>>> match_graph_; 

    // --- 内部流水线模块 (Pipeline Modules) ---
    
    // 0. 特征预处理
    void extractFeatures(const cv::Mat& image, int frame_id);
    void buildGlobalMatchGraph();

    // 1. 初始化模块 (Phase 1)
    bool bootstrapMap();

    // 2. 增量模块 (Phase 2)
    // 【核心调度】：根据共视关系寻找下一张最适合加入重建的照片
    int findNextBestView();
    // 【增量算子】：执行 PnP 和新点的三角化
    bool trackAndMapIncremental(int frame_id);

    // 3. 优化模块 (Phase 3)
    // 局部 BA：只优化当前帧及其关联的局部邻域
    void runLocalBundleAdjustment(int current_frame_id);

};