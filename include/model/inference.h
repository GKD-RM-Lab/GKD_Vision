#ifndef GKD_CV_2026_ARMORNEWYOLO_H
#define GKD_CV_2026_ARMORNEWYOLO_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <sys/types.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <openvino/openvino.hpp>
#include "timer/timer.hpp"
#include "data_manager/parameter_loader.h"
#include <algorithm>
#include <thread>
#include <sstream>
#include "video/hikdriver.h"
#include "send_control/socket_interface.hpp"
#include "structure/stamp.hpp"
#include "timer/timer.hpp"

#define NMS_THRESHOLD   0.10f  // NMS参数（建议调到0.45）
// #define CONF_THRESHOLD_D 0.35f // 置信度参数， 这个置信度是初始值， 实际的置信度是从config中加载的
#define CONF_REMAIN     0.0f   // 保留帧权重比例
#define IMG_SIZE        640    // 推理图像大小
#define ANCHOR          3      // anchor 数量
#define DETECT_MODE     0      // ARMOR 0, WIN 1, BOARD 2
#define DEVICE          "CPU"  // 设备选择
#define VIDEO           0// 是否展示推理视频

#if DETECT_MODE == 0
    #define KPT_NUM 4
    #define CLS_NUM 14
    // #define MODEL_PATH "../models/RMyolov7-best-fp32/rmyolo-v7-best.xml","../models/RMyolov7-best-fp32/rmyolo-v7-best.bin"
#elif DETECT_MODE == 1
    #define KPT_NUM 5
    #define CLS_NUM 4
    #define MODEL_PATH ""
#elif DETECT_MODE == 2
    #define KPT_NUM 0
    #define CLS_NUM 4
    #define MODEL_PATH ""
#endif


class yolo_kpt {
public:
    yolo_kpt();

    struct Object {
        cv::Rect_<float> rect;
        int label;
        float prob;
        std::vector<cv::Point2f> kpt;
        // pnp数据
        int pnp_is_calculated = 0;  // -1无解，0未计算，1计算完成
        int kpt_lost_index = -1;    // 角点缺失索引：0左上、1左下、2右下、3右上
        cv::Mat pnp_tvec;
        cv::Mat pnp_rvec;
    };
    

    cv::Mat letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd);
    std::vector<cv::Point2f> scale_box_kpt(std::vector<cv::Point2f> points,
                                           std::vector<float> &padd,
                                           float raw_w, float raw_h,
                                           int idx);
    cv::Rect scale_box(cv::Rect box, std::vector<float> &padd,
                       float raw_w, float raw_h);

    void drawPred(int classId, float conf, const cv::Rect& box, const std::vector<cv::Point2f>& keypoints,
                        cv::Mat& frame, const std::vector<std::string>& classes);

    static void generate_proposals(int stride, const float *feat,
                                   std::vector<Object> &objects);
    
    std::vector<float> pre_process(cv::Mat& src_img, ov::Tensor& dst_tensor);
    
    std::vector<yolo_kpt::Object> post_process(const float *result_p8, const float *result_p16, const float *result_p32z, std::vector<float>& padd, cv::Mat& src_img);

    std::string label2string(int num);

    void removePointsOutOfRect(std::vector<cv::Point2f>& kpt, const cv::Rect2f& rect);

    int findMissingCorner(const std::vector<cv::Point2f>& pts);
    
    int pnp_kpt_preprocess(std::vector<yolo_kpt::Object>& result);

    cv::Mat visual_label(cv::Mat inputImage, std::vector<yolo_kpt::Object> result);
    
    std::vector<yolo_kpt::Object> enemy_check(std::vector<yolo_kpt::Object>& object_result);
    
    void image_show(cv::Mat src_img, std::vector<yolo_kpt::Object> result, yolo_kpt& model);

    void send2frame(std::vector<yolo_kpt::Object>& enemy_result, cv::Mat& src_img , std::mutex& mutex_out, bool& flag_out, std::shared_ptr<rm::Frame>& frame_out);

    void async_infer(std::mutex& mutex_in, bool& flag_in, std::shared_ptr<rm::Frame>& frame_in, 
                    std::mutex& mutex_out, bool& flag_out, std::shared_ptr<rm::Frame>& frame_out);
private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    std::vector<ov::InferRequest> infer_request;
    std::array<ov::Tensor, 2> input_tensors;

#if DETECT_MODE == 0
    const std::vector<std::string> class_names = {
        "Hero1", "ENG2", "INF3", "INF4", "ZijianQin", "TianyiJiang", "SEN7", "R1", "R2", "R3", "R4", "R5", "RO", "RS"
    };
#elif DETECT_MODE == 1
    const std::vector<std::string> class_names = {
             "RR", "RW", "BR", "BW"
    };
#elif DETECT_MODE == 2
     const std::vector<std::string> class_names = {
            "RA", "RD", "BA", "BD"
    };
#endif

    static float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    Timer timer1, timer2, timer3, timer4;
};


#endif