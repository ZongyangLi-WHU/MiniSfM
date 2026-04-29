#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>

#include "feature.h"
#include "geometry.h"
#include "viewer.h"

class SfMSystem {
public:
    // 构造函数：传入相机内参 K
    SfMSystem(const cv::Mat& K);

    // 外部接口：添加图像并触发特征提取
    void addImage(const cv::Mat& image, int frame_id);

    // 新增：添加图像时同时记录真实文件名
    void addImage(const cv::Mat& image, int frame_id, const std::string& image_name);

    // 外部接口：启动全自动重建流程
    void runReconstruction();

    // 外部接口：保存最终点云
    void exportPointcloud(const std::string& filename) const;
    void exportToCOLMAP(const std::string& export_dir) const;

    // 新增：给系统绑定一个 Viewer
    void setViewer(std::shared_ptr<Viewer> viewer);
    

private:
    // --- 核心状态数据 ---
    cv::Mat K_;
    std::map<int, FeatureData> all_features_;
    std::map<int, cv::Mat> camera_R_;
    std::map<int, cv::Mat> camera_t_;
    std::vector<Point3D> global_points_;

    // --- 调度器状态 ---
    int current_frame_id_;
    std::set<int> registered_frames_;
    std::set<int> pending_frames_;

    // 全局匹配图：[图A][图B] -> 匹配点对列表
    std::map<int, std::map<int, std::vector<cv::DMatch>>> match_graph_;

    // --- Viewer ---
    std::shared_ptr<Viewer> viewer_;
    std::map<int, std::string> image_names_;

private:
    // 0. 特征预处理
    void extractFeatures(const cv::Mat& image, int frame_id);
    void buildGlobalMatchGraph();
    // 用 Essential Matrix RANSAC 对描述子匹配结果做几何一致性验证
    std::vector<cv::DMatch> filterMatchesByEssential(
        const FeatureData& data1,
        const FeatureData& data2,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& K
    );

    // 1. 初始化模块
    bool bootstrapMap();

    // 2. 增量模块
    int findNextBestView();
    bool trackAndMapIncremental(int frame_id);

    // 3. 优化模块
    void runLocalBundleAdjustment(int current_frame_id);

    // 新增：把当前地图状态通知 Viewer
    void notifyViewer();

    
};