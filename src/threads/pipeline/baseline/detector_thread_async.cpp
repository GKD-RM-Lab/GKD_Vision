#include "threads/pipeline.h"
#include <unistd.h>
#include <iostream>
#include "video/hikdriver.h"
#include "model/inference.h"

#include "timer/timer.hpp"
#include "send_control/socket_interface.hpp"

#include "data_manager/parameter_loader.h"

void Pipeline::detector_baseline_thread(
    std::mutex& mutex_in, bool& flag_in, std::shared_ptr<rm::Frame>& frame_in, std::mutex& mutex_out, bool& flag_out, std::shared_ptr<rm::Frame>& frame_out) {
    auto param = Param::get_instance();
    auto garage = Garage::get_instance();

    std::string yolo_type = (*param)["Model"]["YoloArmor"]["Type"];

    int    infer_width       = (*param)["Model"]["YoloArmor"][yolo_type]["InferWidth"];
    int    infer_height      = (*param)["Model"]["YoloArmor"][yolo_type]["InferHeight"];
    int    class_num         = (*param)["Model"]["YoloArmor"][yolo_type]["ClassNum"];
    int    locate_num        = (*param)["Model"]["YoloArmor"][yolo_type]["LocateNum"];
    int    color_num         = (*param)["Model"]["YoloArmor"][yolo_type]["ColorNum"];
    int    bboxes_num        = (*param)["Model"]["YoloArmor"][yolo_type]["BboxesNum"];
    double confidence_thresh = (*param)["Model"]["YoloArmor"][yolo_type]["ConfThresh"];
    double nms_thresh        = (*param)["Model"]["YoloArmor"][yolo_type]["NMSThresh"];

    size_t yolo_struct_size = sizeof(float) * static_cast<size_t>(locate_num + 1 + color_num + class_num);
    std::mutex mutex;
    TimePoint tp0, tp1, tp2;
    cv::Mat inputImage, label_image;
    yolo_kpt model;
    std::vector<yolo_kpt::Object> result;
    Timer timer, timer1, timer2, timer3;

    model.async_infer();
    
}