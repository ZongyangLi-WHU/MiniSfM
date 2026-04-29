#include "geometry.h"
#include <iostream>
#include<optimization.h>
#include<fstream>
#include <ceres/ceres.h>
#include "optimization.h"

void PoseEstimator::estimate(const FeatureData& data1, 
                             const FeatureData& data2,
                             const std::vector<cv::DMatch>& matches,
                             const cv::Mat& K,
                             cv::Mat& R, 
                             cv::Mat& t) {
    
    // 1. 将 DMatch 结构体转换为 OpenCV 计算所需的 Point2f 格式
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (size_t i = 0; i < matches.size(); i++) {
        // 利用 matches[i].queryIdx 获取图 1 的点坐标，压入 points1
        // 利用 matches[i].trainIdx 获取图 2 的点坐标，压入 points2
        // 提示： DMatch 的用法，坐标存在 keypoints[idx].pt 中
        points1.push_back(data1.keypoints[matches[i].queryIdx].pt);
        points2.push_back(data2.keypoints[matches[i].trainIdx].pt);
    }

    std::cout << "Extracted " << points1.size() << " matched point pairs for Pose Estimation." << std::endl;

    // 2. 求解本质矩阵 E (配合 RANSAC 剔除最后的误匹配)
    // findEssentialMat 默认使用 RANSAC 算法
    cv::Mat mask; // 这是一个掩码，用来标记哪些点是真正的内点 (Inliers)
    cv::Mat E = cv::findEssentialMat(points1, points2, K, cv::RANSAC, 0.999, 1.0, mask);
    
    std::cout << "Essential Matrix E computed successfully." << std::endl;

    // 3. 从本质矩阵 E 中恢复相机的旋转 R 和平移 t
    // recoverPose 会利用 mask 进一步验证点在相机前方，并输出 R 和 t
    int inlier_cnt = cv::recoverPose(E, points1, points2, K, R, t, mask);

    std::cout << "Recovered Pose with " << inlier_cnt << " pure inliers." << std::endl;
    
    // 简单打印一下结果
    std::cout << "Rotation Matrix R:\n" << R << std::endl;
    std::cout << "Translation Vector t:\n" << t << std::endl;
}
// ---------------------------------------------------------
// PoseEstimator::triangulate 实现
// ---------------------------------------------------------
void PoseEstimator::triangulate(FeatureData& data1,
                                FeatureData& data2,
                                const std::vector<cv::DMatch>& matches,
                                const cv::Mat& K,
                                const cv::Mat& R1, const cv::Mat& t1, // 相机1的外参
                                const cv::Mat& R2, const cv::Mat& t2, // 相机2的外参
                                std::vector<Point3D>& global_points) {
    
// 1. 准备投影矩阵 P1 和 P2
    cv::Mat T1;
    cv::hconcat(R1, t1, T1);
    cv::Mat P1 = K * T1; 

    cv::Mat T2;
    cv::hconcat(R2, t2, T2); 
    cv::Mat P2 = K * T2;

    // 2. 提取匹配好的 2D 点对
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(data1.keypoints[m.queryIdx].pt);
        pts2.push_back(data2.keypoints[m.trainIdx].pt);
    }

    // 3. 执行三角化
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    points4D.convertTo(points4D, CV_64F);

    for (int i = 0; i < points4D.cols; i++) {
        double x = points4D.at<double>(0, i);
        double y = points4D.at<double>(1, i);
        double z = points4D.at<double>(2, i);
        double w = points4D.at<double>(3, i);

        if (std::abs(w) < 1e-8) {
            continue;
        }

        cv::Point3d X(x / w, y / w, z / w);

        if (!std::isfinite(X.x) || !std::isfinite(X.y) || !std::isfinite(X.z)) {
            continue;
        }

        // 检查点是否在两个相机前方
        cv::Mat X_mat = (cv::Mat_<double>(3, 1) << X.x, X.y, X.z);
        cv::Mat X_cam1 = R1 * X_mat + t1;
        cv::Mat X_cam2 = R2 * X_mat + t2;

        if (X_cam1.at<double>(2) <= 0 || X_cam2.at<double>(2) <= 0) {
            continue;
        }

        Point3D new_pt;
        new_pt.pt = X;

        int idx1 = matches[i].queryIdx;
        int idx2 = matches[i].trainIdx;

        new_pt.track[data1.image_id] = idx1;
        new_pt.track[data2.image_id] = idx2;

        int new_pt_index = static_cast<int>(global_points.size());
        global_points.push_back(new_pt);

        data1.point3d_idx[idx1] = new_pt_index;
        data2.point3d_idx[idx2] = new_pt_index;
    }

    std::cout << "Triangulated " << global_points.size() << " 3D points." << std::endl;
}

// ---------------------------------------------------------
// PoseEstimator::savePLY 实现
// ---------------------------------------------------------
void PoseEstimator::savePLY(const std::string& filename, const std::vector<Point3D>& points) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error: Cannot open file " << filename << " for writing!" << std::endl;
        return;
    }

    // 写入 PLY 文件头
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << points.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "end_header\n";

    // 遍历 points 向量，将每个点的 x, y, z 坐标写入文件
    // 用空格分隔，每写完一个点换行 ( out << ... << "\n"; )

    for(int i = 0; i<points.size(); i++){
        out<<points[i].pt.x<<" "<<points[i].pt.y<<" "<<points[i].pt.z<<"\n";
    }
    
    out.close();
    std::cout << "Saved point cloud to " << filename << std::endl;
}

// ---------------------------------------------------------
// PoseEstimator::estimatePnP 实现
// ---------------------------------------------------------
bool PoseEstimator::estimatePnP(const std::vector<cv::Point3f>& object_3d,
                                const std::vector<cv::Point2f>& object_2d,
                                const cv::Mat& K,
                                cv::Mat& R,
                                cv::Mat& t,
                                std::vector<int>* inliers_out) {
    
    // PnP 理论上至少需要 4 个点；工程上后面 trackAndMapIncremental 还会要求更多
    if (object_3d.size() < 4) {
        std::cerr << "Error: Not enough points for PnP estimation. Need at least 4, got "
                  << object_3d.size() << std::endl;
        return false;
    }

    cv::Mat rvec;
    std::vector<int> inliers;

    bool success = cv::solvePnPRansac(
        object_3d,
        object_2d,
        K,
        cv::noArray(),
        rvec,
        t,
        false,
        100,
        8.0,
        0.99,
        inliers
    );

    if (!success) {
        std::cerr << "PnP estimation failed!" << std::endl;
        return false;
    }

    if (inliers.empty()) {
        std::cerr << "PnP estimation failed: no RANSAC inliers." << std::endl;
        return false;
    }

    cv::Rodrigues(rvec, R);

    if (inliers_out) {
        *inliers_out = inliers;
    }

    std::cout << "PnP solved successfully! Inliers: "
              << inliers.size() << " / " << object_3d.size()
              << std::endl;

    return true;
}

void PoseEstimator::optimize(std::map<int, cv::Mat>& camera_R,
                             std::map<int, cv::Mat>& camera_t,
                             std::vector<Point3D>& global_points,
                             const std::map<int, FeatureData>& all_features,
                             const cv::Mat& K,
                             const std::set<int>& active_cameras) {
    
    ceres::Problem problem;


    // 1. 准备内参数据 (固定不变，只作为已知参数传入)
    double camera_intrinsics[3] = { K.at<double>(0, 0),   // f
                                    K.at<double>(0, 2),   // cx
                                    K.at<double>(1, 2) }; // cy

    // 2. 准备相机外参数据 (将 3x3 矩阵降维为 1D double 数组)

    // 使用 std::vector<double> 存储 6 维数组 (3旋转 + 3平移)
    std::map<int, std::vector<double>> camera_params;
    for (const auto& pair : camera_R) {
        int img_id = pair.first;
        cv::Mat rvec;
        // 把 3x3 旋转矩阵 R 转换为 3x1 旋转向量 rvec
        cv::Rodrigues(camera_R[img_id], rvec); 

        camera_params[img_id] = std::vector<double>(6);
        // 注意：OpenCV 解算出的通常是 64 位浮点数 double，安全起见这里转换为 double
        camera_params[img_id][0] = rvec.at<double>(0);
        camera_params[img_id][1] = rvec.at<double>(1);
        camera_params[img_id][2] = rvec.at<double>(2);
        camera_params[img_id][3] = camera_t[img_id].at<double>(0);
        camera_params[img_id][4] = camera_t[img_id].at<double>(1);
        camera_params[img_id][5] = camera_t[img_id].at<double>(2);
    }

    // ---------------------------------------------------------
    // 3. 准备 3D 点数据 (将 Point3d 提取为连续的 double 数组)
    // ---------------------------------------------------------
    std::vector<std::vector<double>> point_params(global_points.size(), std::vector<double>(3));
    for (size_t i = 0; i < global_points.size(); ++i) {
        point_params[i][0] = global_points[i].pt.x;
        point_params[i][1] = global_points[i].pt.y;
        point_params[i][2] = global_points[i].pt.z;
    }

    // ---------------------------------------------------------
    // 4. 构建优化图 (最核心的一步：把观测数据和参数块绑在一起)
    // ---------------------------------------------------------
    int residual_count = 0;
    for (size_t i = 0; i < global_points.size(); ++i) {
        // 遍历这一个 3D 点在各个图片上的 2D 观测
        for (const auto& track_record : global_points[i].track) {
            int img_id = track_record.first;
            int kp_idx = track_record.second;

            // 向前端借数据：拿到真实的 2D 像素观测坐标 (u, v)
            double obs_x = all_features.at(img_id).keypoints[kp_idx].pt.x;
            double obs_y = all_features.at(img_id).keypoints[kp_idx].pt.y;

            // 使用我们写的代价函数工厂方法
            ceres::CostFunction* cost_function = ReprojectionError::Create(obs_x, obs_y);

            // 向图中添加一个误差节点
            problem.AddResidualBlock(cost_function,
                                     new ceres::HuberLoss(1.0), // 鲁棒核函数：降低极个别严重误匹配点的影响
                                     camera_params[img_id].data(),
                                     point_params[i].data(),
                                     camera_intrinsics);
            
            // 锁定内参，不让优化器修改它
            problem.SetParameterBlockConstant(camera_intrinsics);

           
            // Local BA 的参数冻结逻辑
            // 如果 active_cameras 不为空，说明我们正在执行局部 BA
            // 如果当前 img_id 不在这个活跃集合里，强制要求 Ceres 把它当成常量
            if (!active_cameras.empty() && active_cameras.find(img_id) == active_cameras.end()) {
                problem.SetParameterBlockConstant(camera_params[img_id].data());
            }

            residual_count++;
        }
    }

    std::cout << "Built Ceres problem with " << residual_count << " residuals." << std::endl;

    // ---------------------------------------------------------
    // 5. 点燃优化器，开始执行求解
    // ---------------------------------------------------------
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR; // 专为 Bundle Adjustment 优化的稀疏矩阵求解器
    options.minimizer_progress_to_stdout = true;     // 在终端打印优化进度条
    options.max_num_iterations = 50;                 // 最大迭代次数

    ceres::Solver::Summary summary;
    std::cout << "Starting Bundle Adjustment..." << std::endl;
    ceres::Solve(options, &problem, &summary);
    
    // 打印优化前后的误差对比
    std::cout << summary.BriefReport() << std::endl;

    // ---------------------------------------------------------
    // 6. 数据升维：把优化好的结果“写回”给 OpenCV 和我们的点云数组
    // ---------------------------------------------------------
    // 写回 3D 点
    for (size_t i = 0; i < global_points.size(); ++i) {
        global_points[i].pt.x = point_params[i][0];
        global_points[i].pt.y = point_params[i][1];
        global_points[i].pt.z = point_params[i][2];
    }

    // 写回相机位姿
    for (auto& pair : camera_R) {
        int img_id = pair.first;
        cv::Mat rvec = (cv::Mat_<double>(3, 1) << camera_params[img_id][0], 
                                                  camera_params[img_id][1], 
                                                  camera_params[img_id][2]);
        // 把优化后的 3x1 旋转向量重新还原成 3x3 旋转矩阵
        cv::Rodrigues(rvec, camera_R[img_id]);

        camera_t[img_id].at<double>(0) = camera_params[img_id][3];
        camera_t[img_id].at<double>(1) = camera_params[img_id][4];
        camera_t[img_id].at<double>(2) = camera_params[img_id][5];
    }
}

// 误差分析与点云清洗模块
void PoseEstimator::analyzeAndCleanErrors(const std::map<int, cv::Mat>& camera_R,
                                          const std::map<int, cv::Mat>& camera_t,
                                          std::vector<Point3D>& global_points,
                                          std::map<int, FeatureData>& all_features,
                                          const cv::Mat& K,
                                          double error_threshold) {
    
    std::cout << "\n[Error Analysis] === Running Reprojection Error Analysis ===" << std::endl;

    double total_error = 0.0;
    int total_observations = 0;
    double max_error = 0.0;
    int bad_points_count = 0;

    // 用于保存清洗后的高质量点云
    std::vector<Point3D> cleaned_points;

    // 遍历每一个全局 3D 点
    for (size_t i = 0; i < global_points.size(); ++i) {
        const Point3D& pt3d = global_points[i];
        bool is_bad_point = false;
        double point_total_error = 0.0;

        // 遍历看到这个 3D 点的所有相机观测记录
        for (const auto& track_record : pt3d.track) {
            int img_id = track_record.first;
            int kp_idx = track_record.second;

            // 1. 获取相机的 R 和 t
            const cv::Mat& R = camera_R.at(img_id);
            const cv::Mat& t = camera_t.at(img_id);

            // 2. 将 3D 点转换到相机坐标系: P_cam = R * P_world + t
            cv::Mat P_world = (cv::Mat_<double>(3, 1) << pt3d.pt.x, pt3d.pt.y, pt3d.pt.z);
            cv::Mat P_cam = R * P_world + t;

            // 3. 透视除法与内参投影 (投影到像素平面)
            double x_cam = P_cam.at<double>(0, 0) / P_cam.at<double>(2, 0);
            double y_cam = P_cam.at<double>(1, 0) / P_cam.at<double>(2, 0);
            
            double u_proj = K.at<double>(0, 0) * x_cam + K.at<double>(0, 2);
            double v_proj = K.at<double>(1, 1) * y_cam + K.at<double>(1, 2);

            // 4. 获取前端提取的真实观测坐标
            double u_obs = all_features.at(img_id).keypoints[kp_idx].pt.x;
            double v_obs = all_features.at(img_id).keypoints[kp_idx].pt.y;

            // 5. 计算欧式距离误差 (L2 Norm)
            double error = std::sqrt(std::pow(u_proj - u_obs, 2) + std::pow(v_proj - v_obs, 2));

            // 更新统计数据
            total_error += error;
            total_observations++;
            if (error > max_error) max_error = error;

            // 如果该点在任意一个视角的重投影误差超过了设定的阈值，标记为坏点
            if (error > error_threshold) {
                is_bad_point = true;
            }
        }

        // 核心清洗逻辑：只有当它不是坏点时，才保留到新数组中
        if (!is_bad_point) {
            cleaned_points.push_back(pt3d);
        } else {
            bad_points_count++;
        }
    }

    // 覆盖旧的点云数据
    global_points = cleaned_points;

    // 重要：清洗点云后，global_points 的下标会重新排列。
    // 因此必须重建 all_features[*].point3d_idx，避免后续 PnP 使用旧索引。

    // 1. 先把所有 2D 特征点对应的 3D 索引全部清空
    for (auto& pair : all_features) {
        FeatureData& feature = pair.second;
        std::fill(feature.point3d_idx.begin(),
                feature.point3d_idx.end(),
                -1);
    }

    // 2. 再根据清洗后的 global_points 的 track 重新写回 point3d_idx
    for (size_t new_idx = 0; new_idx < global_points.size(); ++new_idx) {
        const Point3D& pt = global_points[new_idx];

        for (const auto& obs : pt.track) {
            int image_id = obs.first;
            int kp_idx = obs.second;

            auto it = all_features.find(image_id);
            if (it == all_features.end()) {
                continue;
            }

            FeatureData& feature = it->second;

            if (kp_idx >= 0 && kp_idx < static_cast<int>(feature.point3d_idx.size())) {
                feature.point3d_idx[kp_idx] = static_cast<int>(new_idx);
            }
        }
    }

    // 打印量化评估报告
    double mean_error = total_observations > 0 ? (total_error / total_observations) : 0.0;
    std::cout << "  - Total Observations Eval: " << total_observations << std::endl;
    std::cout << "  - Mean Reprojection Error: " << mean_error << " pixels" << std::endl;
    std::cout << "  - Max Reprojection Error:  " << max_error << " pixels" << std::endl;
    std::cout << "  - Outlier Points Cleaned:  " << bad_points_count << " points (Error > " << error_threshold << "px)" << std::endl;
    std::cout << "  - Final High-Quality Map:  " << global_points.size() << " 3D points" << std::endl;
    std::cout << "[Error Analysis] === Analysis Complete ===\n" << std::endl;
}