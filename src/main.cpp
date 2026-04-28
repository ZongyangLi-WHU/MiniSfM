#include "sfm_system.h"
#include <iostream>

int main() {
    // 1. 设置近似内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 400, 0, 800, 300, 0, 0, 1);
    
    // 2. 初始化核心系统
    SfMSystem system(K);

    // 3.读取图像数据 (假设你 data 目录下有 img1.jpg, img2.jpg, img3.jpg...)
    for (int i = 1; i <= 3; ++i) { 
        std::string path = "../data/img" + std::to_string(i) + ".jpg";
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cerr << "Cannot read " << path << std::endl;
            continue;
        }
        // 为了速度，统一缩放
        cv::resize(img, img, cv::Size(800, 600)); 
        system.addImage(img, i);
    }

    // 4. 重建
    system.runReconstruction();

    // 5. 导出最后的成果
    system.exportPointcloud("../data/final_incremental_cloud.ply");
    system.exportToCOLMAP("../data/colmap_workspace/sparse/0");
    
    return 0;
}