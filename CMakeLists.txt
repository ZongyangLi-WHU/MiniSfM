cmake_minimum_required(VERSION 3.10)
project(MiniSfM)

# 设置 C++ 标准为 C++17 (现代 C++ 开发的基础)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==========================================
# 查找依赖包
# ==========================================
# 查找 OpenCV (要求包含 core, highgui, imgproc, features2d 等模块)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 查找 Ceres Solver ---
find_package(Ceres REQUIRED)
message(STATUS "Ceres Solver found: ${Ceres_DIR}")

# 查找 Eigen3
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 include path: ${EIGEN3_INCLUDE_DIRS}")

find_package(Pangolin REQUIRED)


add_compile_definitions(GL_GLEXT_PROTOTYPES)

# ==========================================
# 设置头文件包含路径
# ==========================================
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

# ==========================================
# 编译可执行文件并链接库
# ==========================================
# 将 src/main.cpp 编译为名为 sfm_test 的可执行文件
add_executable(
    sfm_test src/main.cpp 
    src/feature.cpp
    src/geometry.cpp
    src/sfm_system.cpp
    src/viewer.cpp )


target_link_libraries(sfm_test
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Pangolin_LIBRARIES}
    OpenGL::GL
    pthread
)