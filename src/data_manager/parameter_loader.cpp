#include "data_manager/parameter_loader.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int yaml_write(const std::string& filepath, const parameter_loader_t& params);
int yaml_load(const std::string& filepath, parameter_loader_t& params);
int para_load(std::string filepath);
cv::Mat eigenToCvMat3d(const Eigen::Matrix3d& eigenMat);
cv::Mat eigenToCvMat4d(const Eigen::Matrix4d& eigenMat);
void readCameraExtrinsics(const std::string& filename);
void writeMatricesToFile(const std::string& filename);
void readCameraParametersFromYaml(const std::string& filename);

parameter_loader_t params;

// ================================================================
// 参数入口
// ================================================================
int para_load(std::string filepath)
{
    // 可调试用写入
    // yaml_write(filepath, params);
    // return 0;
    yaml_load(filepath, params);
    return 0;
}

// ================================================================
// 写入 YAML 参数文件
// ================================================================
int yaml_write(const std::string& filepath, const parameter_loader_t& params) {
    cv::FileStorage fs(filepath, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "错误：无法打开文件 " << filepath << " 用于写入！" << std::endl;
        return -1;
    }

    // 推理参数
    fs << "model_path_xml" << params.model_path_xml;
    fs << "model_path_bin" << params.model_path_bin;
    fs << "conf_threshold" << params.conf_threshold;
    fs << "rect_cut" << params.rect_cut;

    // 标定参数
    fs << "boardSize_h" << params.boardSize_h;
    fs << "boardSize_w" << params.boardSize_w;
    fs << "squareSize" << params.squareSize;
    fs << "img_count" << params.img_count;
    fs << "sample_period" << params.sample_period;
    fs << "calib_yaml_path" << params.calib_yaml_path;
    fs << "camera_extrinsics_path" << params.camera_extrinsics_path;

    // 相机参数
    fs << "cam_gain" << params.cam_gain;
    fs << "cam_exptime" << params.cam_exptime;
    fs << "framerate" << params.framerate;

    // 其他
    fs << "imshow_en" << params.imshow_en;
    fs << "debug_info" << params.debug_info;
    fs << "armor_small_h" << params.armor_small_h;
    fs << "armor_small_w" << params.armor_small_w;
    fs << "armor_large_h" << params.armor_large_h;
    fs << "armor_large_w" << params.armor_large_w;

    fs.release();

    // 写外参
    writeMatricesToFile("../config/camera_extrinsics.yaml");

    std::cout << "参数已保存到 " << filepath << std::endl;
    return 0;
}

// ================================================================
// 读取 YAML 参数文件
// ================================================================
int yaml_load(const std::string& filepath, parameter_loader_t& params) {
    cv::FileStorage fs(filepath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "错误：无法打开文件 " << filepath << " 用于读取！" << std::endl;
        return -1;
    }

    // 推理参数
    fs["model_path_xml"] >> params.model_path_xml;
    fs["model_path_bin"] >> params.model_path_bin;
    fs["conf_threshold"] >> params.conf_threshold;
    fs["rect_cut"] >> params.rect_cut;

    // 标定参数
    fs["boardSize_h"] >> params.boardSize_h;
    fs["boardSize_w"] >> params.boardSize_w;
    fs["squareSize"] >> params.squareSize;
    fs["img_count"] >> params.img_count;
    fs["sample_period"] >> params.sample_period;
    fs["calib_yaml_path"] >> params.calib_yaml_path;
    fs["camera_extrinsics_path"] >> params.camera_extrinsics_path;

    // 相机参数
    fs["cam_gain"] >> params.cam_gain;
    fs["cam_exptime"] >> params.cam_exptime;
    fs["framerate"] >> params.framerate;
    // 其他参数
    fs["imshow_en"] >> params.imshow_en;
    fs["debug_info"] >> params.debug_info;
    fs["armor_small_h"] >> params.armor_small_h;
    fs["armor_small_w"] >> params.armor_small_w;
    fs["armor_large_h"] >> params.armor_large_h;
    fs["armor_large_w"] >> params.armor_large_w;

    // 相机文件
    readCameraParametersFromYaml(params.calib_yaml_path);
    readCameraExtrinsics(params.camera_extrinsics_path);

    // 新增参数
    fs["shoot_speed"] >> params.shoot_speed;
    fs["is_reverse"] >> params.is_camreverse;
    fs["is_filter"] >> params.is_filted;
    fs["is_imshow"] >> params.is_imshow;
    fs["is_pose_out"] >> params.is_pose_out;
    fs["is_target_out"] >> params.is_target_out;
    fs["sentry_head_type"] >> params.sentry_head_type;

    fs.release();

    std::cout << "参数已从 " << filepath << " 加载" << std::endl;
    std::cout << "trans_pnp2head\n" << params.trans_pnp2head << std::endl;
    std::cout << "rotate_pnp2hea\n" << params.rotate_pnp2hea << std::endl;
    std::cout << "intrinsic_matrix\n" << params.intrinsic_matrix << std::endl;
    std::cout << "distortion_coeffs\n" << params.distortion_coeffs << std::endl;
    std::cout << "shoot speed = " << params.shoot_speed << std::endl;
    std::cout << "is cam reverse = " << params.is_camreverse << std::endl;
    std::cout << "head type = " << params.sentry_head_type << std::endl;

    return 0;
}

// ================================================================
// 加载相机内参
// ================================================================
void readCameraParametersFromYaml(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "❌ 无法打开内参文件: " << filename << std::endl;
        return;
    }

    cv::Mat camMat, distMat;
    fs["camera_matrix"] >> camMat;
    fs["distortion_coefficients"] >> distMat;
    fs.release();

    // ✅ clone 深拷贝防止悬空
    params.intrinsic_matrix = camMat.clone();
    params.distortion_coeffs = distMat.clone();
}

// ================================================================
// 写相机外参
// ================================================================
void writeMatricesToFile(const std::string& filename) {
    Eigen::Matrix3d rotate_pnp2hea = params.rotate_pnp2hea;
    Eigen::Matrix4d trans_pnp2head = params.trans_pnp2head;

    cv::Mat cvRotate = eigenToCvMat3d(rotate_pnp2hea);
    cv::Mat cvTrans = eigenToCvMat4d(trans_pnp2head);

    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "rotate_pnp2hea" << cvRotate;
    fs << "trans_pnp2head" << cvTrans;
    fs.release();
}

// ================================================================
// 读取相机外参
// ================================================================
void readCameraExtrinsics(const std::string& filename) {
    std::cout << "path: !: " << filename << std::endl;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "❌ 无法打开外参文件: " << filename << std::endl;
        return;
    }

    cv::Mat cvRotate, cvTrans;
    fs["rotate_pnp2hea"] >> cvRotate;
    fs["trans_pnp2head"] >> cvTrans;
    fs.release();

    // ✅ clone 避免悬空引用
    cvRotate = cvRotate.clone();
    cvTrans  = cvTrans.clone();

    std::cout << "matrix:\n" << cvTrans << std::endl;

    Eigen::Matrix3d rotate_pnp2hea;
    Eigen::Matrix4d trans_pnp2head;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            rotate_pnp2hea(i, j) = cvRotate.at<double>(i, j);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            trans_pnp2head(i, j) = cvTrans.at<double>(i, j);

    params.rotate_pnp2hea = rotate_pnp2hea;
    params.trans_pnp2head = trans_pnp2head;
}

// ================================================================
// Eigen 转 cv::Mat
// ================================================================
cv::Mat eigenToCvMat3d(const Eigen::Matrix3d& eigenMat) {
    cv::Mat cvMat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cvMat.at<double>(i, j) = eigenMat(i, j);
    return cvMat;
}

cv::Mat eigenToCvMat4d(const Eigen::Matrix4d& eigenMat) {
    cv::Mat cvMat(4, 4, CV_64F);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            cvMat.at<double>(i, j) = eigenMat(i, j);
    return cvMat;
}
