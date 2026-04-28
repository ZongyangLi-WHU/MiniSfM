#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "feature.h" // 需要用到之前的 FeatureData

struct Point3D{
    cv::Point3d pt;
    std::map<int,int>track;     // 记录观测记录 观测到它的图像集合。Key: 图像ID, Value: 特征点索引 用来解决global_points中的三维点 “”哪张图片中的哪个三维点看到我的问题“”
};

class PoseEstimator {
public:
    // 核心功能：输入匹配数据和相机内参 K，输出相机的旋转 R 和平移 t
    static void estimate(const FeatureData& data1, 
                         const FeatureData& data2,
                         const std::vector<cv::DMatch>& matches,
                         const cv::Mat& K,
                         cv::Mat& R, 
                         cv::Mat& t);
    static void triangulate(FeatureData& data1,
                            FeatureData& data2,
                            const std::vector<cv::DMatch>& matches,
                            const cv::Mat& K,
                            const cv::Mat& R1, const cv::Mat& t1, // 相机1的外参
                            const cv::Mat& R2, const cv::Mat& t2, // 相机2的外参
                            std::vector<Point3D>& global_points);
                            // 新增：保存点云为 PLY 文件
    static void savePLY(const std::string& filename, const std::vector<Point3D>& points);

    static bool estimatePnP( const std::vector<cv::Point3f>& object_3d,
                             const std::vector<cv::Point2f>& object_2d,
                             const cv::Mat &k,
                             cv::Mat& R,
                             cv::Mat& t);
                             
    // 更新：全局与局部 Bundle Adjustment 联合优化接口
    static void optimize(std::map<int, cv::Mat>& camera_R,        
                         std::map<int, cv::Mat>& camera_t,        
                         std::vector<Point3D>& global_points,     
                         const std::map<int, FeatureData>& all_features, 
                         const cv::Mat& K,
                         const std::set<int>& active_cameras = std::set<int>()); // 新增活跃相机集合
                         
    // 误差分析与点云清洗模块
    static void analyzeAndCleanErrors(const std::map<int, cv::Mat>& camera_R,
                                      const std::map<int, cv::Mat>& camera_t,
                                      std::vector<Point3D>& global_points,
                                      const std::map<int, FeatureData>& all_features,
                                      const cv::Mat& K,
                                      double error_threshold = 2.0); // 默认剔除误差大于 2 个像素的坏点
};

/*
1. cv::KeyPoint：我是谁，我在哪？

在我们的代码中，提取出的点存放在 std::vector<cv::KeyPoint> keypoints 中。cv::KeyPoint 不是一个简单的坐标点 (x,y)，它是一个包含了很多物理信息的结构体。

它的核心成员变量包括：

    pt (Point2f): 这是特征点在图像上的 2D 坐标 (x,y)。这是我们后续计算三维重建最需要的数据。

    size: 特征点邻域的直径。SIFT 是在不同的尺度空间（图像放大或缩小）中寻找特征点的，这个值记录了它是在哪个尺度下被发现的。

    angle: 特征点的方向。SIFT 会计算点周围像素的梯度方向，这使得图片旋转后，算法依然能认出这个点（旋转不变性）。

    response: 响应程度，代表这个点作为特征点的“强度”或“质量”。

2. cv::DMatch：牵线的红娘

这是最容易让人绕晕的结构体。匹配完成后，我们得到了 std::vector<cv::DMatch> matches。一个 DMatch 对象并不是包含了特征点本身，而是记录了两个特征点在各自数组中的“索引（Index）”。

它的核心成员变量有三个：

    queryIdx (查询图像索引): 图 1（data1）特征点数组中的下标。

    trainIdx (训练图像/目标图像索引): 图 2（data2）特征点数组中的下标。

    distance: 这两个特征点描述子之间的距离（差异度）。值越小，说明这两个点长得越像。

如何使用它？ 假设 match_1 是我们得到的一个好匹配。如果我们想知道图 1 中的哪个点匹配上了图 2 中的哪个点，我们需要这样写：

    图 1 的点坐标：data1.keypoints[match_1.queryIdx].pt

    图 2 的点坐标：data2.keypoints[match_1.trainIdx].pt

这就是 DMatch 作为“桥梁”的作用。
*/