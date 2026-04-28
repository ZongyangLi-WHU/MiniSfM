#include "sfm_system.h"
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <filesystem> // C++17 标准库，用于创建文件夹
#include <iostream>

// 构造函数
SfMSystem::SfMSystem(const cv::Mat& K) : current_frame_id_(-1) {
    K_ = K.clone();
    std::cout << "[SfMSystem] System initialized with camera intrinsics." << std::endl;
}

// 阶段 1：数据摄入与预处理
void SfMSystem::addImage(const cv::Mat& image, int frame_id) {
    extractFeatures(image, frame_id);
    pending_frames_.insert(frame_id);
    std::cout << "[SfMSystem] Frame " << frame_id << " added to pending queue." << std::endl;
}

void SfMSystem::extractFeatures(const cv::Mat& image, int frame_id) {
    FeatureExtractor extractor;
    FeatureData data = extractor.extract(image);
    
    // 必须给 image_id 赋值，底层 triangulate 强依赖此属性构建 track
    data.image_id = frame_id; 
    
    all_features_[frame_id] = data;
}

// 构建全局的连接图
void SfMSystem::buildGlobalMatchGraph() {
    std::cout << "[SfMSystem] Building Global Match Graph..." << std::endl;
    FeatureMatcher matcher; 

    for (auto it1 = pending_frames_.begin(); it1 != pending_frames_.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != pending_frames_.end(); ++it2) {
            int id1 = *it1;
            int id2 = *it2;

            std::vector<cv::DMatch> good_matches = matcher.match(all_features_[id1], all_features_[id2]);
            // 只有两张图之间的特征点数量大于15 才会将这两张图加入到全局的连接图中
            if (good_matches.size() >= 15) {
                match_graph_[id1][id2] = good_matches;
            // 构建对称的全局匹配图，保证“查询视角”的物理逻辑正确，从而构建一个对称的无向图
                std::vector<cv::DMatch> reverse_matches;
                reverse_matches.reserve(good_matches.size()); 
                for (const auto& m : good_matches) {
                    reverse_matches.emplace_back(m.trainIdx, m.queryIdx, m.distance); // 采用emplace_back而不是push_back 在于二者一个是移动一个是复制
                }
                match_graph_[id2][id1] = reverse_matches;  // 显式构建无向图 
            }
        }
    }
    std::cout << "[SfMSystem] Global Match Graph built successfully." << std::endl;
}


// 系统中央调度器 主函数中调用 

void SfMSystem::runReconstruction() {
    if (pending_frames_.size() < 2) {
        std::cerr << "[Error] Not enough frames to start reconstruction." << std::endl;
        return;
    }

    buildGlobalMatchGraph();  // 全局连接图

    std::cout << "\n[SfMSystem] --- Phase 1: Bootstrapping ---" << std::endl;
    if (!bootstrapMap()) {
        std::cerr << "[Error] Bootstrapping failed. Reconstruction aborted." << std::endl;
        return;
    }

    std::cout << "\n[SfMSystem] --- Phase 2: Incremental Mapping ---" << std::endl;
    while (!pending_frames_.empty()) {
        int next_id = findNextBestView();
        
        if (next_id == -1) {
            std::cout << "[Warning] No more overlapping frames found. Stopping incremental mapping." << std::endl;
            break; 
        }

        current_frame_id_ = next_id;
        std::cout << "\n[SfMSystem] Selected Next Best View: Frame " << current_frame_id_ << std::endl;

        if (trackAndMapIncremental(current_frame_id_)) {

            runLocalBundleAdjustment(current_frame_id_);
            pending_frames_.erase(current_frame_id_);
            registered_frames_.insert(current_frame_id_);
        } else {
            std::cout << "[Warning] Frame " << current_frame_id_ << " failed to register. Discarding." << std::endl;
            pending_frames_.erase(current_frame_id_);
        }
    }

    std::cout << "\n[SfMSystem] Reconstruction pipeline finished." << std::endl;
    std::cout << "Total registered frames: " << registered_frames_.size() << std::endl;
    std::cout << "Total 3D points generated: " << global_points_.size() << std::endl;
}


// 双目初始化 (Bootstrapping) 

bool SfMSystem::bootstrapMap() {
    int best_id1 = -1;
    int best_id2 = -1;
    int max_matches = 0;

    // 1. 寻找最佳初始化基线 简单的最优选择 即选择两张图特征匹配点最多的两张初始视图
    for (const auto& pair1 : match_graph_) {
        int id1 = pair1.first;
        for (const auto& pair2 : pair1.second) {
            int id2 = pair2.first;
            int match_count = pair2.second.size();
            if (match_count > max_matches) {
                max_matches = match_count;
                best_id1 = id1;
                best_id2 = id2;
            }
        }
    }

    if (best_id1 == -1 || best_id2 == -1 || max_matches < 100) {
        std::cerr << "[Error] Cannot find a valid image pair for bootstrapping." << std::endl;
        return false;
    }

    std::cout << "[SfMSystem] Bootstrapping with Frame " << best_id1 
              << " and Frame " << best_id2 << " (" << max_matches << " matches)" << std::endl;

    const auto& matches = match_graph_[best_id1][best_id2];

    // 直接传入 FeatureData
    cv::Mat R, t;
    PoseEstimator::estimate(all_features_[best_id1], all_features_[best_id2], matches, K_, R, t);
    
    // 检查 estimate 是否成功恢复出位姿
    if (R.empty() || t.empty()) {
        std::cerr << "[Error] Essential matrix estimation failed." << std::endl;
        return false;
    }

    // 2. 世界坐标系锚定
    camera_R_[best_id1] = cv::Mat::eye(3, 3, CV_64F);
    camera_t_[best_id1] = cv::Mat::zeros(3, 1, CV_64F);
    camera_R_[best_id2] = R.clone();
    camera_t_[best_id2] = t.clone();

    // 注意：这里传入的是 all_features_ 的非常量引用
    PoseEstimator::triangulate(all_features_[best_id1], all_features_[best_id2],
                               matches, K_,
                               camera_R_[best_id1], camera_t_[best_id1],
                               camera_R_[best_id2], camera_t_[best_id2],
                               global_points_);

    // 3. 状态流转
    pending_frames_.erase(best_id1);
    pending_frames_.erase(best_id2);
    registered_frames_.insert(best_id1);
    registered_frames_.insert(best_id2);

    return true;
}


// 最佳下一视图调度 (Next Best View Selection)
int SfMSystem::findNextBestView() {
    int best_frame_id = -1;
    int max_2d3d_matches = 0;

    // 1. 遍历所有还在排队等待处理的相片
    for (int pending_id : pending_frames_) {
        int current_2d3d_matches = 0;

        // 2. 检查这张待处理的照片，与所有"已经成功拼进地图"的照片的共视关系
        for (int registered_id : registered_frames_) {
            
            // 如果这两张照片之间在初始化时根本没有匹配点，直接跳过，节省算力 
            /*
            在 C++ STL（标准模板库）中，if (map.find(key) == map.end()) 是一句极其经典的固定语法，
            它的唯一作用是：极其高效地查询这个字典（map）里，到底有没有某个特定的“键（Key）”，而不触发任何内存分配。
            */
            if (match_graph_[pending_id].find(registered_id) == match_graph_[pending_id].end()) {
                continue;
            }

            // 获取这两张照片的匹配点对
            const std::vector<cv::DMatch>& matches = match_graph_[pending_id][registered_id];

            // 3. 遍历每一个匹配对，寻找有效的 2D-3D 对应关系
            for (const auto& match : matches) {
                // match.queryIdx 是 pending_id 图像的特征点索引
                // match.trainIdx 是 registered_id 图像的特征点索引
                
                // 【核心逻辑】：去已注册的图像 (registered_id) 中查一下，
                // 这个特征点是不是在之前的流程中，已经被成功三角化并生成了 3D 点？
                int pt3d_idx = all_features_[registered_id].point3d_idx[match.trainIdx];
                
                if (pt3d_idx != -1) {
                    // 如果不为 -1，说明找到了一个极其珍贵的 2D-3D 数据对！
                    current_2d3d_matches++;
                }
            }
        }

        // 4. 记录拥有最多 2D-3D 匹配的照片
        if (current_2d3d_matches > max_2d3d_matches) {
            max_2d3d_matches = current_2d3d_matches;
            best_frame_id = pending_id;
        }
    }

    // 5. 鲁棒性与断链保护阈值
    // PnP 理论上只需要 4 个点，但考虑到 RANSAC 的稳定性，工程上通常要求至少 15~20 个共视点
    if (max_2d3d_matches < 15) {
        return -1; // 返回 -1 通知主循环：已经没有重叠度足够高的照片了，安全停止建图
    }

    return best_frame_id;
}

// 执行 PnP 追踪与地图生长 (Track and Map Incremental)

bool SfMSystem::trackAndMapIncremental(int frame_id) {
    std::vector<cv::Point3f> object_3d;
    std::vector<cv::Point2f> object_2d;
    
    // 用于去重：同一个 3D 点可能在历史图 1 和图 2 中都被看到，
    // 防止同一个 3D 点被重复塞进 PnP 数组里
    std::set<int> added_pt3d_idxs; 


    // 步骤 1：收集 PnP 所需的 2D-3D 锚点对
    for (int registered_id : registered_frames_) {
        // 如果和这张老照片没有匹配，跳过
        if (match_graph_[frame_id].find(registered_id) == match_graph_[frame_id].end()) {
            continue;
        }

        const auto& matches = match_graph_[frame_id][registered_id];
        for (const auto& match : matches) {
            // 去老照片 (registered_id) 里查一下，这个点是不是 3D 点
            int pt3d_idx = all_features_[registered_id].point3d_idx[match.trainIdx];

            if (pt3d_idx != -1) {
                // 如果这个 3D 点还没被加进目前的 PnP 运算池里
                if (added_pt3d_idxs.find(pt3d_idx) == added_pt3d_idxs.end()) {
                    object_3d.push_back(global_points_[pt3d_idx].pt);
                    object_2d.push_back(all_features_[frame_id].keypoints[match.queryIdx].pt);
                    added_pt3d_idxs.insert(pt3d_idx);
                    
                    // 同步更新：告诉这个全局 3D 点，"新相机也看到你了"
                    global_points_[pt3d_idx].track[frame_id] = match.queryIdx;
                    // 同步更新：告诉新相机，"你的这个像素点对应的 3D 点索引是它"
                    all_features_[frame_id].point3d_idx[match.queryIdx] = pt3d_idx;
                }
            }
        }
    }

    // 鲁棒性校验：如果没有足够的锚点，直接判定注册失败
    if (object_3d.size() < 15) {
        std::cerr << "[Warning] Not enough 3D-2D matches for PnP. Got: " << object_3d.size() << std::endl;
        return false;
    }


    // 执行 PnP，算出新相机位姿
    cv::Mat R, t;
    bool pnp_success = PoseEstimator::estimatePnP(object_3d, object_2d, K_, R, t);
    
    if (!pnp_success || R.empty() || t.empty()) {
        std::cerr << "[Warning] PnP RANSAC failed for frame " << frame_id << std::endl;
        return false;
    }

    // 成功！将新相机的绝对位姿记入系统史册
    camera_R_[frame_id] = R.clone();
    camera_t_[frame_id] = t.clone();
    std::cout << "[SfMSystem] PnP solved. Frame " << frame_id << " localized using " 
              << object_3d.size() << " points." << std::endl;

    // 步骤 3：地图生长 (三角化全新的未知点)
    
    int new_points_count = 0;
    
    for (int registered_id : registered_frames_) {
        if (match_graph_[frame_id].find(registered_id) == match_graph_[frame_id].end()) {
            continue;
        }

        // 1. 根据 C = -R^T * t 计算两台相机在世界坐标系下的绝对光心坐标
        // 注意：OpenCV 中矩阵的转置是 .t()，矩阵乘法直接用 *
        cv::Mat C_new = -camera_R_[frame_id].t() * camera_t_[frame_id];
        cv::Mat C_old = -camera_R_[registered_id].t() * camera_t_[registered_id];

        // 2. 计算基线长度 (平移距离的 L2 范数)
        double baseline = cv::norm(C_new - C_old);

        // 3. 核心校验：拒绝短基线/纯旋转退化
        // 这里的阈值 0.05 是一个经验比例。因为我们在 Phase 1 初始化时，
        // cv::recoverPose 已经自动将系统的初始平移尺度归一化到了 1.0 左右。
        // 如果基线距离不到初始尺度的 5%，说明相机几乎没动，强行三角化会导致深度误差呈指数级爆炸。

        if (baseline < 0.05) { 
            std::cout << "[SfMSystem] Baseline too short (" << baseline << ") between frame " 
                      << frame_id << " and " << registered_id << ". Skipping triangulation to prevent outliers." << std::endl;
            continue; // 直接跳过这两张图的三角化，去评估下一张已注册的图片
        }

        const auto& matches = match_graph_[frame_id][registered_id];
        std::vector<cv::DMatch> unmapped_matches;

        // 挑选出纯粹的 2D-2D 对应关系（即双方都没被三角化过的点）
        for (const auto& match : matches) {
            if (all_features_[frame_id].point3d_idx[match.queryIdx] == -1 &&
                all_features_[registered_id].point3d_idx[match.trainIdx] == -1) {
                unmapped_matches.push_back(match);
            }
        }

        if (unmapped_matches.size() < 15) continue;

        // 直接复用我们写好的 Triangulate 接口
        size_t points_before = global_points_.size();
        PoseEstimator::triangulate(all_features_[frame_id], all_features_[registered_id],
                                   unmapped_matches, K_,
                                   camera_R_[frame_id], camera_t_[frame_id],
                                   camera_R_[registered_id], camera_t_[registered_id],
                                   global_points_);
        
        new_points_count += (global_points_.size() - points_before);
    }

    std::cout << "[SfMSystem] Map grown. Triangulated " << new_points_count << " new points." << std::endl;
    return true;
}


// 导出最终点云 (Export Point Cloud)
void SfMSystem::exportPointcloud(const std::string& filename) const {
    // 鲁棒性校验
    if (global_points_.empty()) {
        std::cerr << "[Warning] Point cloud is empty. Nothing to export." << std::endl;
        return;
    }
    PoseEstimator::savePLY(filename, global_points_);
    
    std::cout << "[SfMSystem] Exported " << global_points_.size() 
              << " 3D points to " << filename << std::endl;
}

// 光束法平差优化 (Bundle Adjustment)
void SfMSystem::runLocalBundleAdjustment(int current_frame_id) {
    std::cout << "\n[SfMSystem] === Triggering Local Bundle Adjustment ===" << std::endl;

    // 1. 构建活跃相机窗口 (Active Window)
    std::set<int> active_cameras;
    active_cameras.insert(current_frame_id); // 当前刚加入的相机必须优化

    // 2. 利用匹配图寻找与当前帧“共视”的历史相机
    // 使用 std::multimap 自动按共视点数量从小到大排序
    std::multimap<int, int> covisibility_ranking; 
    
    for (int registered_id : registered_frames_) {
        if (registered_id == current_frame_id) continue;
        
        // 查询匹配图，看这两张图之间有没有连线
        if (match_graph_[current_frame_id].find(registered_id) != match_graph_[current_frame_id].end()) {
            int match_count = match_graph_[current_frame_id][registered_id].size();
            covisibility_ranking.insert({match_count, registered_id});
        }
    }

    // 3. 挑选共视最强的最近 N 台相机加入活跃窗口 (这里 N 设为 5)
    int local_window_size = 5;
    int count = 0;
    // 使用反向迭代器 (rbegin) 从匹配数量最多的相机开始挑选
    for (auto it = covisibility_ranking.rbegin(); it != covisibility_ranking.rend(); ++it) {
        if (count >= local_window_size) break;
        active_cameras.insert(it->second);
        count++;
    }

    std::cout << "[SfMSystem] Local Window covers " << active_cameras.size() << " active cameras." << std::endl;

    // 4. 将活跃名单传给底层 Ceres 优化器
    PoseEstimator::optimize(camera_R_, camera_t_, global_points_, all_features_, K_, active_cameras);

    // 5. 误差量化评估与劣质点清洗 (保持不变)
    PoseEstimator::analyzeAndCleanErrors(camera_R_, camera_t_, global_points_, all_features_, K_, 2.0);

    std::cout << "[SfMSystem] === Local BA Finished ===" << std::endl;
}

// 导出 COLMAP 标准格式 (Export to COLMAP / 3DGS)
void SfMSystem::exportToCOLMAP(const std::string& export_dir) const {
    // 1. 创建导出目录 (如果不存在的话)
    std::filesystem::create_directories(export_dir);

    // 1. 导出 cameras.txt
    std::ofstream cam_out(export_dir + "/cameras.txt");
    cam_out << "# Camera list with one line of data per camera:\n";
    cam_out << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    
    // 提取内参
    double fx = K_.at<double>(0, 0);
    double fy = K_.at<double>(1, 1);
    double cx = K_.at<double>(0, 2);
    double cy = K_.at<double>(1, 2);
    // 假设所有图片共享同一个相机 (CAMERA_ID = 1)，并且使用最基础的 PINHOLE 模型
    // 注意：这里的宽和高假设是 800x600，如果是其他尺寸请确保和 main.cpp 中的 resize 一致
    cam_out << "1 PINHOLE 800 600 " << fx << " " << fy << " " << cx << " " << cy << "\n";
    cam_out.close();

    // 2. 导出 images.txt
    std::ofstream img_out(export_dir + "/images.txt");
    img_out << "# Image list with two lines of data per image:\n";
    img_out << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    img_out << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";

    for (const auto& pair : camera_R_) {
        int img_id = pair.first;
        cv::Mat R = pair.second;
        cv::Mat t = camera_t_.at(img_id);

        // 核心数学：将 OpenCV 的 3x3 旋转矩阵 R 转换为 Eigen 的四元数 Quaternion
        Eigen::Matrix3d eigen_R;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                eigen_R(i, j) = R.at<double>(i, j);
            }
        }
        Eigen::Quaterniond q(eigen_R);

        std::string img_name = "img" + std::to_string(img_id) + ".jpg";

        // 第一行：写入四元数位姿和平移
        img_out << img_id << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
                << t.at<double>(0) << " " << t.at<double>(1) << " " << t.at<double>(2) << " "
                << "1 " << img_name << "\n";

        // 第二行：写入所有的 2D 特征点观测，以及它们对应的 3D 点 ID
        const auto& feature_data = all_features_.at(img_id);
        for (size_t i = 0; i < feature_data.keypoints.size(); ++i) {
            double x = feature_data.keypoints[i].pt.x;
            double y = feature_data.keypoints[i].pt.y;
            int pt3d_id = feature_data.point3d_idx[i];
            
            // 如果这个 2D 点没有三角化成 3D 点，COLMAP 规定 ID 必须写 -1
            img_out << x << " " << y << " " << pt3d_id << " ";
        }
        img_out << "\n";
    }
    img_out.close();

    // 3. 导出 points3D.txt

    std::ofstream pt_out(export_dir + "/points3D.txt");
    pt_out << "# 3D point list with one line of data per point:\n";
    pt_out << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";

    for (size_t i = 0; i < global_points_.size(); ++i) {
        const auto& p = global_points_[i];
        
        // 我们没有提取颜色，所以 RGB 默认全写成红色 (255 0 0)，误差写 0.0
        pt_out << i << " " << p.pt.x << " " << p.pt.y << " " << p.pt.z << " 255 0 0 0.0 ";
        
        // 写入最核心的 Track 数据 (这个 3D 点在哪些图片的哪个索引上出现过)
        for (const auto& track_pair : p.track) {
            pt_out << track_pair.first << " " << track_pair.second << " ";
        }
        pt_out << "\n";
    }
    pt_out.close();

    std::cout << "[SfMSystem] Successfully exported COLMAP format to directory: " << export_dir << std::endl;
}