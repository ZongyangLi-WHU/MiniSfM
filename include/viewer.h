// viewer.h
#pragma once

#include <map>
#include <vector>
#include <mutex>
#include <atomic>
#include <opencv2/opencv.hpp>

#include "geometry.h"

class Viewer {
public:
    Viewer();

    // SfMSystem 每次地图更新后调用这个函数
    void updateMap(const std::map<int, cv::Mat>& camera_R,
                   const std::map<int, cv::Mat>& camera_t,
                   const std::vector<Point3D>& global_points,
                   int current_frame_id);

    // Pangolin 主循环
    void run();

    // 主程序结束时请求关闭 Viewer
    void requestFinish();

private:
    struct CameraPose {
        int id;
        cv::Mat R;
        cv::Mat t;
    };

    std::mutex data_mutex_;

    std::map<int, CameraPose> cameras_;
    std::vector<cv::Point3d> points_;

    int current_frame_id_;
    std::atomic<bool> finish_requested_;

private:
    cv::Point3d cameraCenter(const cv::Mat& R, const cv::Mat& t) const;

    cv::Point3d cameraPointToWorld(const cv::Mat& R,
                                   const cv::Mat& t,
                                   const cv::Point3d& p_cam) const;

    void drawWorldAxis() const;
    void drawMapPoints(const std::vector<cv::Point3d>& points) const;
    void drawCameras(const std::map<int, CameraPose>& cameras,
                     int current_frame_id) const;
    void drawSingleCamera(const CameraPose& cam, bool is_current) const;
};