#include "viewer.h"

#include <pangolin/pangolin.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cmath>

Viewer::Viewer()
    : current_frame_id_(-1),
      finish_requested_(false) {}

void Viewer::updateMap(const std::map<int, cv::Mat>& camera_R,
                       const std::map<int, cv::Mat>& camera_t,
                       const std::vector<Point3D>& global_points,
                       int current_frame_id) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    cameras_.clear();
    points_.clear();

    for (const auto& pair : camera_R) {
        int id = pair.first;

        if (camera_t.find(id) == camera_t.end()) {
            continue;
        }

        CameraPose cam;
        cam.id = id;
        cam.R = camera_R.at(id).clone();
        cam.t = camera_t.at(id).clone();

        cameras_[id] = cam;
    }

    points_.reserve(global_points.size());
    for (const auto& p : global_points) {
        points_.push_back(p.pt);
    }

    current_frame_id_ = current_frame_id;

    std::cout << "[Viewer] Updated map: "
              << cameras_.size() << " cameras, "
              << points_.size() << " points." << std::endl;
}

void Viewer::requestFinish() {
    finish_requested_ = true;
}

cv::Point3d Viewer::cameraCenter(const cv::Mat& R, const cv::Mat& t) const {
    cv::Mat C = -R.t() * t;
    return cv::Point3d(C.at<double>(0), C.at<double>(1), C.at<double>(2));
}

cv::Point3d Viewer::cameraPointToWorld(const cv::Mat& R,
                                       const cv::Mat& t,
                                       const cv::Point3d& p_cam) const {
    cv::Mat p = (cv::Mat_<double>(3, 1) << p_cam.x, p_cam.y, p_cam.z);
    cv::Mat C = -R.t() * t;
    cv::Mat p_world = R.t() * p + C;

    return cv::Point3d(p_world.at<double>(0),
                       p_world.at<double>(1),
                       p_world.at<double>(2));
}

void Viewer::drawWorldAxis() const {
    glLineWidth(2.0);

    glBegin(GL_LINES);

    // X axis
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);

    // Y axis
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);

    // Z axis
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);

    glEnd();
}

void Viewer::drawMapPoints(const std::vector<cv::Point3d>& points) const {
    glPointSize(2.0f);
    glBegin(GL_POINTS);

    glColor3f(0.2f, 0.8f, 0.2f);

    for (const auto& p : points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        
        if (std::abs(p.x) > 1000 ||
        std::abs(p.y) > 1000 ||
        std::abs(p.z) > 1000) {
        continue;
    }
        glVertex3d(p.x, p.y, p.z);
    }

    glEnd();
}

void Viewer::drawSingleCamera(const CameraPose& cam, bool is_current) const {
    const double w = is_current ? 0.12 : 0.08;
    const double h = w * 0.75;
    const double z = w * 1.5;

    cv::Point3d C  = cameraCenter(cam.R, cam.t);
    cv::Point3d p1 = cameraPointToWorld(cam.R, cam.t, cv::Point3d( w,  h, z));
    cv::Point3d p2 = cameraPointToWorld(cam.R, cam.t, cv::Point3d( w, -h, z));
    cv::Point3d p3 = cameraPointToWorld(cam.R, cam.t, cv::Point3d(-w, -h, z));
    cv::Point3d p4 = cameraPointToWorld(cam.R, cam.t, cv::Point3d(-w,  h, z));

    glLineWidth(is_current ? 3.0f : 1.5f);

    if (is_current) {
        glColor3f(1.0f, 0.1f, 0.1f);
    } else {
        glColor3f(0.1f, 0.4f, 1.0f);
    }

    glBegin(GL_LINES);

    // camera center to image plane corners
    glVertex3d(C.x, C.y, C.z); glVertex3d(p1.x, p1.y, p1.z);
    glVertex3d(C.x, C.y, C.z); glVertex3d(p2.x, p2.y, p2.z);
    glVertex3d(C.x, C.y, C.z); glVertex3d(p3.x, p3.y, p3.z);
    glVertex3d(C.x, C.y, C.z); glVertex3d(p4.x, p4.y, p4.z);

    // image plane rectangle
    glVertex3d(p1.x, p1.y, p1.z); glVertex3d(p2.x, p2.y, p2.z);
    glVertex3d(p2.x, p2.y, p2.z); glVertex3d(p3.x, p3.y, p3.z);
    glVertex3d(p3.x, p3.y, p3.z); glVertex3d(p4.x, p4.y, p4.z);
    glVertex3d(p4.x, p4.y, p4.z); glVertex3d(p1.x, p1.y, p1.z);

    glEnd();
}

void Viewer::drawCameras(const std::map<int, CameraPose>& cameras,
                         int current_frame_id) const {
    if (cameras.empty()) {
        return;
    }

    // 1. 绘制每个相机视锥
    for (const auto& pair : cameras) {
        bool is_current = (pair.first == current_frame_id);
        drawSingleCamera(pair.second, is_current);
    }

    // 2. 绘制相机中心轨迹线
    glLineWidth(2.0f);
    glColor3f(1.0f, 1.0f, 0.0f);

    glBegin(GL_LINE_STRIP);
    for (const auto& pair : cameras) {
        cv::Point3d C = cameraCenter(pair.second.R, pair.second.t);
        glVertex3d(C.x, C.y, C.z);
    }
    glEnd();
}

void Viewer::run() {
    pangolin::CreateWindowAndBind("MiniSfM Pangolin Viewer", 1024, 768);

    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10,
                                  0,  0,   0,
                                  pangolin::AxisY)
    );

    pangolin::Handler3D handler(s_cam);

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(&handler);

    while (!pangolin::ShouldQuit() && !finish_requested_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        std::map<int, CameraPose> cameras_snapshot;
        std::vector<cv::Point3d> points_snapshot;
        int current_frame_snapshot = -1;

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            cameras_snapshot = cameras_;
            points_snapshot = points_;
            current_frame_snapshot = current_frame_id_;
        }

        drawWorldAxis();
        drawMapPoints(points_snapshot);
        drawCameras(cameras_snapshot, current_frame_snapshot);

        pangolin::FinishFrame();

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
}