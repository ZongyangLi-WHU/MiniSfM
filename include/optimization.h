/*
Bundle Adjustment(BA 光束法平差)
这一步首先需要构建代价函数（Cost Function） 重投影误差
我们会把目前所有的相机位姿数据和3D坐标数据，作为待优化的变量，全部放入Ceres里。让优化器在多维空间中找到一个极小值，把所有相机位姿和3D点“拉“到最完美、误差最小的位置。

重投影误差，这个过程就是将一个真实的三维世界点X 通过相机的外参（旋转矩阵R和平移向量t）变换到摄像机坐标系，然后再通过摄像机内参矩阵K投射到二维像素平面上
在Ceres Solver中 我们需要把这个过程写成一个仿函数
（1）自动求导：Ceres 需要对我们写的投影公式求偏导数（计算梯度，从而知道如何调整参数才能让误差变得更小），而为了让Ceres能够自动求导，我们的函数必须写成模板函数template<typename T>
（2）旋转向量必须代替旋转矩阵：在优化过程中，3×3的旋转矩阵有9各参数，但一个旋转矩阵只有三个自由度，且必须满足正交约束，如果直接优化9各参数，容易破坏原有的矩阵性质。因此在Ceres中，相机的旋转矩阵
统一使用3×1的旋转向量来表示。

这一步：在optimization.h内编写代价函数

*/

#pragma once
#include<ceres/ceres.h>
#include<ceres/rotation.h>

// 重投影误差的代价函数
struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y)
    :  observed_x(observed_x),observed_y(observed_y){}

    /*

    在 C++ 中，operator() 叫做“函数调用运算符重载”。
    当我们把这个东西写在一个 struct（结构体）或者 class（类）里面时，这个结构体就变成了一个“仿函数 (Functor)”。

    它的意思是：你可以像调用普通函数一样，去调用这个结构体对象。
    Ceres 的底层代码强制规定了，你提交给它的“代价函数”必须是一个仿函数，并且带有特定的参数签名。

    */
   
// 核心重投影逻辑 必须写成模板函数
template<typename T>
bool operator()(    const T* const camera,     // 相机外参: 6维 (3维旋转向量 + 3维平移向量)
                    const T* const point,      // 3D点坐标: 3维 (X, Y, Z)
                    const T* const intrinsics, // 相机内参: 3维 (焦距f, cx, cy)
                    T* residuals) const{
    
    T p[3];
    // 是Ceres Solver底层提供的一个及其高效的数学核心计算函数 它的核心作用是：将一个三维空间点，按照指定的旋转向量 进行一次纯粹的旋转计算
    // 其利用三维几何中非常有名的 罗德里格斯旋转公式 其读取长度为3的旋转向量 即这里camera 数组中的前三个元素 
    // 这个向量的方向是 三维空间中旋转轴的方向 代表了三维空间中旋转轴的角度
    /*
    该函数 以这3个极其精简的数字 就直接将输入的3D point 旋转到了新的方向 并把结果存在p中 并且这个函数已经被Goole工程师设计为严密的模板函数
    当你把 Ceres 的求导数据类型（ceres::Jet）传进去时，它不仅能算出旋转后的坐标，还能自动顺藤摸瓜，推导出这整个旋转过程的微积分偏导数（计算梯度），彻底省去了我们手算偏导数的噩梦。
    
    */
    ceres::AngleAxisRotatePoint(camera, point, p);
   
    // 加上平移向量 t
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // 2. 透视除法：投影到归一化图像平面
    // 防止除以 0 的极小概率崩溃
    T xp = p[0] / (p[2] + T(1e-8)); 
    T yp = p[1] / (p[2] + T(1e-8));
 
// 3. 应用相机内参矩阵 K，转换到像素坐标系
    const T& focal = intrinsics[0];
    const T& cx = intrinsics[1];
    const T& cy = intrinsics[2];
        
    T predicted_x = focal * xp + cx;
    T predicted_y = focal * yp + cy;

    // 4. 计算残差：预测的像素坐标 - 实际观测的像素坐标
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);    
    
    return true;


    }
    // 辅助工厂函数，方便后续创建 CostFunction
    static ceres::CostFunction* Create(const double observed_x,
                                       const double observed_y) {
        // 参数维度说明：<误差类型, 残差维度(2), camera维度(6), point维度(3), intrinsics维度(3)>
        return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3, 3>(
            new ReprojectionError(observed_x, observed_y)));
    }
/*
static（静态）意味着这个 Create 函数属于 ReprojectionError 这个结构体“本身”，而不是属于它的某个具体实例对象。
这就意味着，在后续的 main.cpp 中，我们不需要繁琐地先创建一个对象，而是直接用一行代码就能凭空“制造”出一个代价函数交给 Ceres：
ceres::CostFunction* cost_function = ReprojectionError::Create(x, y);
这让主循环中添加成千上万个残差块的代码变得极其干净整洁。
*/

    double observed_x;
    double observed_y;


};