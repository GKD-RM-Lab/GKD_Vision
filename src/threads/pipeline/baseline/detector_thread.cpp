#include "threads/pipeline.h"
#include <unistd.h>
#include <iostream>
#include "video/hikdriver.h"
#include "model/inference.h"
#include <thread>
#include <chrono>

void Pipeline::detector_baseline_thread(
    std::mutex& mutex_in, bool& flag_in, std::shared_ptr<rm::Frame>& frame_in, std::mutex& mutex_out, bool& flag_out, std::shared_ptr<rm::Frame>& frame_out) {
    (void)mutex_in;
    (void)flag_in;
    (void)frame_in;
    yolo_kpt model;
    
    model.async_infer([this, &mutex_out, &flag_out, &frame_out](const std::vector<yolo_kpt::Object>& detections,
                                                               const cv::Mat& frame_image) {
        rm::Frame frame;

        for (const auto& det : detections) {
            rm::YoloRect rect;

            if (det.kpt.size() < 4) {
                rect.four_points = det.kpt;
            } else {
                rect.four_points.push_back(det.kpt[2]);
                rect.four_points.push_back(det.kpt[1]);
                rect.four_points.push_back(det.kpt[3]);
                rect.four_points.push_back(det.kpt[0]);
            }

            rect.box = cv::Rect(
                cvRound(det.rect.x),
                cvRound(det.rect.y),
                cvRound(det.rect.width),
                cvRound(det.rect.height)
            );

            rect.confidence = det.prob;

            if (det.label < 7) {
                rect.color_id = rm::ARMOR_COLOR_BLUE;
                rect.class_id = det.label;
            } else {
                rect.color_id = rm::ARMOR_COLOR_RED;
                rect.class_id = det.label - 7;
            }

            frame.yolo_list.push_back(rect);
        }

        frame.height = frame_image.rows;
        frame.width = frame_image.cols;
        frame.camera_id = 0;
        if (!frame.image) {
            frame.image = std::make_shared<cv::Mat>();
        }
        frame_image.copyTo(*frame.image);
        frame.time_point = std::chrono::high_resolution_clock::now();

        std::unique_lock<std::mutex> lock_out(mutex_out);
        while (flag_out) {
            lock_out.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            lock_out.lock();
        }
        frame_out = std::make_shared<rm::Frame>(frame);
        flag_out = true;
        lock_out.unlock();
        tracker_in_cv_.notify_one();
    });
}
