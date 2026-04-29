#include "sfm_system.h"
#include "viewer.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <string>
#include <thread>
#include <memory>

namespace fs = std::filesystem;

// 判断是不是图片文件
bool isImageFile(const fs::path& path) {
    std::string ext = path.extension().string();

    // 统一转小写，兼容 .JPG / .jpg / .PNG / .png
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

int main() {
    // 1. 设置近似内参
    // 注意：后面 resize 到 800 x 600，所以 cx=400, cy=300 与图像尺寸对应
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        800, 0, 400,
        0, 800, 300,
        0, 0, 1
    );

    // 2. 初始化 SfM 系统
    SfMSystem system(K);

    // 3. 创建 Viewer，并绑定到 SfMSystem
    auto viewer = std::make_shared<Viewer>();
    system.setViewer(viewer);

    // 4. 启动 Pangolin Viewer 线程
    //
    // 因为 Pangolin 的 run() 是一个窗口循环。
    // 如果不开线程，程序会卡在 Viewer 里，SfM 重建流程无法继续执行。
    std::thread viewer_thread([viewer]() {
        viewer->run();
    });

    // 5. 自动扫描 data 文件夹
    std::string data_dir = "../data";
    std::vector<fs::path> image_paths;

    if (!fs::exists(data_dir)) {
        std::cerr << "Error: data directory does not exist: "
                  << data_dir << std::endl;

        viewer->requestFinish();
        if (viewer_thread.joinable()) {
            viewer_thread.join();
        }

        return -1;
    }

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) {
            image_paths.push_back(entry.path());
        }
    }

    // 6. 文件名排序，保证 IMG_2037, IMG_2038, IMG_2039... 按顺序读入
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.size() < 2) {
        std::cerr << "Error: Need at least 2 images for SfM reconstruction."
                  << std::endl;

        viewer->requestFinish();
        if (viewer_thread.joinable()) {
            viewer_thread.join();
        }

        return -1;
    }

    std::cout << "Found " << image_paths.size() << " images." << std::endl;

    // 7. 逐张读取图片并加入系统
    int frame_id = 0;

    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path.string());

        if (img.empty()) {
            std::cerr << "Cannot read image: " << path << std::endl;
            continue;
        }

        // 统一缩放到 800 x 600，和 K 保持一致
        cv::resize(img, img, cv::Size(800, 600));

        std::cout << "Adding image: " << path.filename().string()
                  << " as frame " << frame_id << std::endl;

        system.addImage(img, frame_id, path.filename().string());

        frame_id++;
    }

    // 8. 启动重建
    system.runReconstruction();

    // 9. 导出结果
    system.exportPointcloud("../data/final_incremental_cloud.ply");
    system.exportToCOLMAP("../data/colmap_workspace/sparse/0");

    std::cout << "\n[Main] Reconstruction finished." << std::endl;
    std::cout << "[Main] You can inspect the final map in Pangolin." << std::endl;
    std::cout << "[Main] Close the Pangolin window to exit." << std::endl;

    // 10. 等待用户关闭 Pangolin 窗口
    //
    // 这里不主动 requestFinish，是为了让你能看最终点云和相机轨迹。
    if (viewer_thread.joinable()) {
        viewer_thread.join();
    }

    return 0;
}