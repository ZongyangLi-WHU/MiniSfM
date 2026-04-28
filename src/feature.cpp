#include"feature.h"
#include<iostream>

using namespace std;

FeatureExtractor::FeatureExtractor(){
    // 实例化 SIFT 提取器，并赋值给成员变量 detector_
    // OpenCV 中创建 SIFT 的静态方法是 cv::SIFT::create()
    detector_ = cv::SIFT::create();
    
}
// 2. 核心提取函数实现
FeatureData FeatureExtractor::extract(const cv::Mat& image) {
    FeatureData result;

    // 安全检查：如果传入的图像为空，直接输出警告并返回空的 result
    if (image.empty()) {
        std::cerr << "Warning: Input image is empty!" << std::endl;
        return result;
    }
    
    // 调用 SIFT 提取特征点和描述子
    // 使用 detector_->detectAndCompute() 函数。
    // 这个函数需要 4 个参数：
    // 参数1：输入图像
    // 参数2：掩码 (Mask)，我们不需要，所以传入 cv::noArray() 即可
    // 参数3：用于存放输出关键点的 std::vector<cv::KeyPoint> 
    // 参数4：用于存放输出描述子的 cv::Mat 
    // 把提取到的结果直接存入 result 对应的成员变量中。
    // 直接传入 result 结构体内部的成员变量
    detector_->detectAndCompute(image, cv::noArray(), result.keypoints, result.descriptors);
    

    // 打印一下提取了多少个点，方便我们调试
    std::cout << "Extracted " << result.keypoints.size() << " keypoints." << std::endl;
    
    result.point3d_idx.assign(result.keypoints.size(), -1);  // 初始化point3d_idx数组，其大小和提取到的特征点数量一致，并全都初始化为-1

    return result;
}

// ---------------------------------------------------------
// FeatureMatcher 类的实现
// ---------------------------------------------------------

FeatureMatcher::FeatureMatcher() {
    // 实例化 FLANN 匹配器，专门用于加速高维浮点向量的匹配
    matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

std::vector<cv::DMatch> FeatureMatcher::match(const FeatureData& data1, const FeatureData& data2) {
    std::vector<cv::DMatch> good_matches; // 用于存放筛选后的好匹配

    if (data1.descriptors.empty() || data2.descriptors.empty()) {
        std::cerr << "Warning: Descriptors are empty! Cannot match." << std::endl;
        return good_matches;
    }

    // 1. 使用 KNN 匹配，设定 k=2，即每个点找 2 个最佳匹配
    // knn_matches 是一个嵌套的 vector，因为每个原始点对应 2 个 DMatch 结果
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher_->knnMatch(data1.descriptors, data2.descriptors, knn_matches, 2);

    // 2. 比例测试阈值 (通常在 0.7 到 0.8 之间，0.75 是个经典的折中值)
    const float ratio_thresh = 0.75f;

    // 3. 遍历所有的 knn 匹配结果，进行筛选
    for (size_t i = 0; i < knn_matches.size(); i++) {
        // 安全检查：确保找到了至少两个点
        if (knn_matches[i].size() >= 2) {
            // 利用 Lowe's Ratio Test 筛选
            // 第一近邻的距离是 knn_matches[i][0].distance
            // 第二近邻的距离是 knn_matches[i][1].distance
            // 如果第一近邻距离 < ratio_thresh * 第二近邻距离
            // 就把第一近邻 knn_matches[i][0] 加入到 good_matches 容器中 (使用 push_back)
            if(knn_matches[i][0].distance<ratio_thresh*knn_matches[i][1].distance){
                good_matches.push_back(knn_matches[i][0]);
            }
            
        }
    }

    std::cout << "Found " << knn_matches.size() << " raw matches." << std::endl;
    std::cout << "Kept " << good_matches.size() << " good matches after Ratio Test." << std::endl;
    return good_matches;
}