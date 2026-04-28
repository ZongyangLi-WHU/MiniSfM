/*Phase1 前半部分： 双目图片的特征提取与匹配*/
// 在动手写算法之前，首先要思考数据如何流转。在SfM系统中，后续我们会处理几十上百张图片，如果把所有的特征点和匹配关系都乱糟糟地塞进main函数，内存管理和代码可读性都会大大降低
// 首先要将数据载体和算法逻辑解耦

/*
整体逻辑与架构
（1）数据中心 Data carrier:每张图片被读取后，我们需要提取它的特征点（keyPoints，包含像素坐标（u，v）和描述子（Description，用于区分不同特征点的向量），这部分数据需要被统一打包
（2）提取器（extractor）：这是一个功能模块，负责输入一张图像，输出打包好的特征数据
（3）匹配器(Marcher)：这是另一个功能模块，负责输入两组特征的描述子，输出它们之间的匹配关系。
*/
#pragma once
#include <vector>
#include<map>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

// 1. 数据结构体：用于打包单张图像的特征信息
struct FeatureData {
    int image_id;                        // 给每张图片设置一个ID
    std::vector<cv::KeyPoint> keypoints; // 特征点 (包含坐标、响应值等)
    cv::Mat descriptors;                 // 描述子矩阵
    std::vector<int>point3d_idx;         // 记录每个特征点对应的3D点在global_points数组中的索引 如果这个特征点暂时未被三角化，则初始化为-1
};

// 2. 特征提取类
class FeatureExtractor {
public:
    // 构造函数，这里我们可以预留参数，比如最大特征点数量
    FeatureExtractor(); 

    // 核心功能函数：输入单张图像，返回提取到的特征数据
    FeatureData extract(const cv::Mat& image);

private:
    // 内部持有一个 OpenCV 的特征提取器指针 (我们准备使用 SIFT)
    cv::Ptr<cv::SIFT> detector_;
};

// 3. 特征匹配类
class FeatureMatcher {
public:
    FeatureMatcher();

    // 核心功能函数：输入两组特征数据，返回初始的匹配结果
    std::vector<cv::DMatch> match(const FeatureData& data1, const FeatureData& data2);

private:
    // 内部持有一个 OpenCV 的特征匹配器指针
    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

