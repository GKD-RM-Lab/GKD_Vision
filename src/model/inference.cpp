#include "model/inference.h"

std::mutex result_mtx;
std::vector<yolo_kpt::Object> object_result;

yolo_kpt::yolo_kpt() {
    CONF_THRESHOLD = params.conf_threshold;

    model = core.read_model(params.model_path_xml, params.model_path_bin);
    num_cores = std::thread::hardware_concurrency();

    ov::AnyMap config;
    if (+DEVICE == +"CPU") {
        config = {
            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
            ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY),
            ov::inference_num_threads(num_cores)
        };
        std::cout << "CPU : NUM_THREADS =" << num_cores << std::endl;
    }
    else if (+DEVICE == +"GPU") {
        config = {
            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT),
            ov::streams::num(2)
        };
        std::cout << "GPU, streams = 2" << std::endl;
    }
    compiled_model = core.compile_model(model, DEVICE, config);

    infer_request = {compiled_model.create_infer_request(), 
                                                   compiled_model.create_infer_request()};
                                                   

    for(int i = 0; i < 2; i++) {
        input_tensor = infer_request[i].get_input_tensor(0);
    }
}

cv::Mat yolo_kpt::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd) {
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;
    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));
    padd.push_back(padd_w);
    padd.push_back(padd_h);
    padd.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
}

cv::Rect yolo_kpt::scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h) {
    cv::Rect scaled_box;
    scaled_box.width = box.width / padd[2];
    scaled_box.height = box.height / padd[2];
    scaled_box.x = std::max(std::min((float) ((box.x - padd[0]) / padd[2]), (float) (raw_w - 1)), 0.f);
    scaled_box.y = std::max(std::min((float) ((box.y - padd[1]) / padd[2]), (float) (raw_h - 1)), 0.f);
    return scaled_box;
}

std::vector<cv::Point2f>
yolo_kpt::scale_box_kpt(std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx) {
    std::vector<cv::Point2f> scaled_points;
    for (int i = 0; i < KPT_NUM; i++) {
        points[idx * KPT_NUM + i].x = std::max(
                std::min((points[idx * KPT_NUM + i].x - padd[0]) / padd[2], (float) (raw_w - 1)), 0.f);
        points[idx * KPT_NUM + i].y = std::max(
                std::min((points[idx * KPT_NUM + i].y - padd[1]) / padd[2], (float) (raw_h - 1)), 0.f);
        scaled_points.push_back(points[idx * KPT_NUM + i]);

    }
    return scaled_points;
}

/*
* 绘制预测结果， 这些参数是目标检测模型标准输出格式
*/
void yolo_kpt::drawPred(int classId, float conf, const cv::Rect& box, const std::vector<cv::Point2f>& keypoints,
                        cv::Mat& frame, const std::vector<std::string>& classes) {

    cv::rectangle(frame, box, cv::Scalar(255, 255, 255), 1);

    cv::Point2f center(box.x + box.width / 2.0f, box.y + box.height / 2.0f);
    std::array<bool, 5> valid = {false};

    for (size_t i = 0; i < std::min((size_t)5, keypoints.size()); ++i) {
        if (i != 2 && keypoints[i].x > 0 && keypoints[i].y > 0) {
            valid[i] = true;
        }
    }

    if (valid[0] && valid[1] && valid[3] && valid[4]) {
        center = (keypoints[0] + keypoints[1] + keypoints[3] + keypoints[4]) * 0.25f;
    } else if (valid[0] && valid[3] && (!valid[1] || !valid[4])) {
        center = (keypoints[0] + keypoints[3]) * 0.5f;
    } else if (valid[1] && valid[4] && (!valid[0] || !valid[3])) {
        center = (keypoints[1] + keypoints[4]) * 0.5f;
    }

    cv::circle(frame, center, 2, cv::Scalar(255, 255, 255), 2);

    if (DETECT_MODE == 1) {
        for (int i = 0; i < KPT_NUM && i < (int)keypoints.size(); ++i) {
            cv::Scalar color = (i == 2) ? cv::Scalar(163, 164, 163) : cv::Scalar(0, 255, 0);
            int radius = (i == 2) ? 4 : 3;
            cv::circle(frame, keypoints[i], radius, color, radius);
        }
    }

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty() && classId < (int)classes.size()) {
        label = classes[classId] + ": " + label;
    }

    int baseLine = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    int y_text = std::max(box.y, textSize.height);

    cv::rectangle(frame,
                  cv::Point(box.x, y_text - textSize.height - 3),
                  cv::Point(box.x + textSize.width + 5, y_text + baseLine),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(box.x + 2, y_text),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
}

/*
*每个tensor包含的数据是[x,y,w,h,conf,cls1pro,cls2pro,...clsnpro,kpt1.x,kpt1.y,kpt1.conf,kpt2...kptm.conf]
*/
void yolo_kpt::generate_proposals(int stride, const float *feat, std::vector<Object> &objects) {
    int feat_w = IMG_SIZE / stride;
    int feat_h = IMG_SIZE / stride;
    float anchors[18] = {11, 10, 19, 15, 28, 22, 39, 34, 64, 48, 92, 76, 132, 110, 197, 119, 265, 162};
    int anchor_group = 0;
    if (stride == 8) anchor_group = 0;
    if (stride == 16) anchor_group = 1;
    if (stride == 32) anchor_group = 2;

    for (int anchor = 0; anchor < ANCHOR; anchor++) {
        for (int i = 0; i < feat_h; i++) {
            for (int j = 0; j < feat_w; j++) {
                float box_prob = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                      i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                      j * (5 + CLS_NUM + KPT_NUM * 3) + 4];
                box_prob = sigmoid(box_prob);
                if (box_prob < CONF_THRESHOLD) continue;

                [[maybe_unused]] float kptx[5], kpty[5], kptp[5];

                float x = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 0];
                float y = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 1];
                float w = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 2];
                float h = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                               j * (5 + CLS_NUM + KPT_NUM * 3) + 3];

                if (KPT_NUM != 0) {
                    for (int k = 0; k < KPT_NUM; k++) {
                        kptx[k] = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       j * (5 + CLS_NUM + KPT_NUM * 3) + 5 + CLS_NUM + k * 3];
                        kpty[k] = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       j * (5 + CLS_NUM + KPT_NUM * 3) + 5 + CLS_NUM + k * 3 + 1];
                        kptp[k] = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                       j * (5 + CLS_NUM + KPT_NUM * 3) + 5 + CLS_NUM + k * 3 + 2];
                        kptx[k] = (kptx[k] * 2 - 0.5 + j) * stride;
                        kpty[k] = (kpty[k] * 2 - 0.5 + i) * stride;
                    }
                }

                double max_prob = 0;
                int idx = 0;
                for (int k = 5; k < CLS_NUM + 5; k++) {
                    double tp = feat[anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                     i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                                     j * (5 + CLS_NUM + KPT_NUM * 3) + k];
                    tp = sigmoid(tp);
                    if (tp > max_prob) max_prob = tp, idx = k;
                }

                float cof = std::min<float>(box_prob * max_prob, 1.0f);
                if (cof < CONF_THRESHOLD) continue;

                x = (sigmoid(x) * 2 - 0.5 + j) * stride;
                y = (sigmoid(y) * 2 - 0.5 + i) * stride;
                w = pow(sigmoid(w) * 2, 2) * anchors[anchor_group * 6 + anchor * 2];
                h = pow(sigmoid(h) * 2, 2) * anchors[anchor_group * 6 + anchor * 2 + 1];

                float r_x = x - w / 2;
                float r_y = y - h / 2;

                Object obj;
                obj.rect.x = r_x;
                obj.rect.y = r_y;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = idx - 5;
                obj.prob = cof;

                if (KPT_NUM != 0) {
                    for (std::vector<cv::Point_<float>>::size_type k = 0; k < KPT_NUM; k++) {
                        if (k != 2 && kptx[k] > r_x && kptx[k] < r_x + w && kpty[k] > r_y && kpty[k] < r_y + h) {
                            obj.kpt.push_back(cv::Point2f(kptx[k], kpty[k]));
                        } else if (k == 2) {
                            obj.kpt.push_back(cv::Point2f(kptx[k], kpty[k]));
                        } else {
                            obj.kpt.push_back(cv::Point2f(0, 0));
                        }
                    }
                }
                objects.push_back(obj);
            }
        }
    }
}


std::vector<float> yolo_kpt::pre_process(cv::Mat& src_img) {
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;
    cv::Mat img;
    std::vector<float> padd;

    cv::Mat boxed = letter_box(src_img, img_h, img_w, padd);
    cv::cvtColor(boxed, img, cv::COLOR_BGR2RGB);
    auto datal = input_tensor.data<float>();
    
    for (int h = 0; h < img_h; h++) {
        for (int w = 0; w < img_w; w++) {
            for (int c = 0; c < 3; c++) {
                int out_index = c * img_h * img_w + h * img_w + w;
                datal[out_index] = float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
            }
        }
    }
    return padd;
}

std::vector<yolo_kpt::Object> yolo_kpt::post_process(const float *result_p8, const float *result_p16, const float *result_p32, std::vector<float>& padd, cv::Mat& src_img) {
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    std::vector<Object> objects16;
    std::vector<Object> objects32;
    generate_proposals(8, result_p8, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    generate_proposals(16, result_p16, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals(32, result_p32, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < proposals.size(); i++) {
        classIds.push_back(proposals[i].label);
        confidences.push_back(proposals[i].prob);
        boxes.push_back(proposals[i].rect);
        for (auto ii: proposals[i].kpt) {
            points.push_back(ii);
        }
    }
    std::vector<int> picked;
    std::vector<float> picked_useless;
    std::vector<Object> object_result;

    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    for (size_t i = 0; i < picked.size(); i++) {
        cv::Rect scaled_box = scale_box(boxes[picked[i]], padd, src_img.cols, src_img.rows);
        std::vector<cv::Point2f> scaled_point;
        if (KPT_NUM != 0)
            scaled_point = scale_box_kpt(points, padd, src_img.cols, src_img.rows, picked[i]);
        Object obj;
        obj.rect = scaled_box;
        obj.label = classIds[picked[i]];
        obj.prob = confidences[picked[i]];
        if (KPT_NUM != 0)
            obj.kpt = scaled_point;
        object_result.push_back(obj);
    
    #if VIDEO == 1
        if (DETECT_MODE == 1 && classIds[picked[i]] == 0)  {
             drawPred(classIds[picked[i]], confidences[picked[i]], scaled_box, scaled_point, src_img,
                     class_names);
        }
    #endif
    }
    #if VIDEO == 1
        // cv::imshow("Inference frame", src_img);
        // cv::waitKey(1);
    #endif
    return object_result;
}

std::vector<yolo_kpt::Object> enemy_check(std::vector<yolo_kpt::Object>& object_result) {
    std::vector<yolo_kpt::Object> enemy;

    for (auto& obj : object_result) {
        bool red = obj.label >= 7;
        if (red != socket_interface.pkg.red) {
            enemy.push_back(obj);
        }
    }

    return enemy;
}

void yolo_kpt::removePointsOutOfRect(std::vector<cv::Point2f>& kpt, const cv::Rect2f& rect)
{
    kpt.erase(
        std::remove_if(kpt.begin(), kpt.end(),
            [&rect](const cv::Point2f& p) {
                return !rect.contains(p);
            }
        ),
        kpt.end()
    );
}

cv::Mat yolo_kpt::visual_label(cv::Mat inputImage, std::vector<yolo_kpt::Object> result) {
    if(result.size() > 0)
    {
        for(size_t j=0; j<result.size(); j++)
        {
            for(size_t i=0; i<result[j].kpt.size(); i++)
            {
                cv::circle(inputImage, result[j].kpt[i], 3, cv::Scalar(0,255,0), 3);
                char text[10];
                std::sprintf(text, "%ld", i);
                cv::putText(inputImage, text, cv::Point(result[j].kpt[i].x, result[j].kpt[i].y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 2);
            }

            cv::rectangle(inputImage, result[j].rect, cv::Scalar(255,0,0), 5);

            if(result[j].kpt.size() == 4)
            {
                cv::line(inputImage, result[j].kpt[0], result[j].kpt[1], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[1], result[j].kpt[2], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[2], result[j].kpt[3], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[3], result[j].kpt[0], cv::Scalar(0,255,0), 5);
                char text[50];
                std::sprintf(text, "%s - P%.2f", label2string(result[j].label).c_str(), result[j].prob);
                cv::putText(inputImage, text, cv::Point(result[j].kpt[3].x, result[j].kpt[3].y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 3);
                //pnp结果
                if(result[j].pnp_is_calculated == 1)
                {
                    char text[50];
                    // std::cout << result[j].pnp_tvec << std::endl;
                    std::sprintf(text, "x%.2fy%.2fz%.2f", result[j].pnp_tvec.at<double>(0)
                    , result[j].pnp_tvec.at<double>(1), result[j].pnp_tvec.at<double>(2));
                    cv::putText(inputImage, text, cv::Point(result[j].kpt[3].x + 10, result[j].kpt[3].y + 30)
                    , cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255), 3);
                }
            }

            if(result[j].kpt.size() == 3)
            {
                cv::line(inputImage, result[j].kpt[0], result[j].kpt[1], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[1], result[j].kpt[2], cv::Scalar(0,255,0), 5);
                cv::line(inputImage, result[j].kpt[2], result[j].kpt[0], cv::Scalar(0,255,0), 5);
                char text[50];
                std::sprintf(text, "%s - %d", label2string(result[j].label).c_str(), result[j].kpt_lost_index);
                cv::putText(inputImage, text, cv::Point(result[j].kpt[2].x, result[j].kpt[2].y)
                , cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0,0,255), 3);
                if(result[j].pnp_is_calculated == 1)
                {
                    char text[50];
                    std::sprintf(text, "x%.2fy%.2fz%.2f", result[j].pnp_tvec.at<double>(0)
                    , result[j].pnp_tvec.at<double>(1), result[j].pnp_tvec.at<double>(2));
                    cv::putText(inputImage, text, cv::Point(result[j].kpt[2].x + 10, result[j].kpt[2].y + 30)
                    , cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,255), 3);
                }
            }

        }
    }
    return inputImage;
}

std::string yolo_kpt::label2string(int num) {
    return class_names[num];
}

int yolo_kpt::findMissingCorner(const std::vector<cv::Point2f>& trianglePoints) {
    if (trianglePoints.size() != 3)
        return -1;  

    double d01 = cv::norm(trianglePoints[0] - trianglePoints[1]);
    double d12 = cv::norm(trianglePoints[1] - trianglePoints[2]);
    double d20 = cv::norm(trianglePoints[2] - trianglePoints[0]);

    int gapIndex = 0;
    double maxGap = d01;
    if (d12 > maxGap) { maxGap = d12; gapIndex = 1; }
    if (d20 > maxGap) { maxGap = d20; gapIndex = 2; }

    if (gapIndex == 0) {
        return 1;
    }
    else if (gapIndex == 1) {
        return 2;
    }
    else {
        if (d01 < d12)
            return 3;
        else
            return 0;
    }
}

int yolo_kpt::pnp_kpt_preprocess(std::vector<yolo_kpt::Object>& result) {
    for(size_t j=0; j<result.size(); j++) {

        removePointsOutOfRect(result[j].kpt, result[j].rect);

        if(result[j].kpt.size() == 4) {
            result[j].pnp_is_calculated = 0;
        }

        if(result[j].kpt.size() == 3) {
            result[j].kpt_lost_index = findMissingCorner(result[j].kpt);
            result[j].pnp_is_calculated = 0;
        }
        
        if(result[j].kpt.size() < 3) {
            result[j].pnp_is_calculated = -1;   
        }

    }
    return 0;
}

void image_show(cv::Mat src_img, std::vector<yolo_kpt::Object> result, yolo_kpt& model) {
    cv::Mat label_image;
    if(params.is_imshow) {
        src_img.copyTo(label_image);
        label_image = model.visual_label(label_image, result);
        cv::imshow("cam", label_image);
    }
}
void yolo_kpt::async_infer() {
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;

    cv::Mat frame_one;
    HIKframemtx.lock();
    HIKimage.copyTo(frame_one);
    HIKframemtx.unlock();
       
    if(params.is_camreverse){
        cv::flip(frame_one, frame_one, -1);
    }

    std::vector<float> padd_one;
    std::vector<float> padd_two;
    int total_time = 0;

    padd_one = pre_process(frame_one);
    infer_request[0].set_input_tensor(input_tensor);
    infer_request[0].start_async();

    while (true) {
        cv::Mat next_frame;

        HIKframemtx.lock();
        HIKimage.copyTo(next_frame);
        HIKframemtx.unlock();

        if(params.is_camreverse){
            cv::flip(next_frame, next_frame, -1);
        }

        if(next_frame.empty()) continue;

        padd_two = pre_process(next_frame);

        infer_request[1].set_input_tensor(input_tensor);
        infer_request[1].start_async();

        infer_request[0].wait();
        auto output_tensor_p8 = infer_request[0].get_output_tensor(0);
        const float *result_p8 = output_tensor_p8.data<const float>();
        auto output_tensor_p16 = infer_request[0].get_output_tensor(1);
        const float *result_p16 = output_tensor_p16.data<const float>();
        auto output_tensor_p32 = infer_request[0].get_output_tensor(2);
        const float *result_p32 = output_tensor_p32.data<const float>();

        std::vector<Object> object_result = post_process(result_p8, result_p16, result_p32, padd_one, frame_one);
        std::vector<Object> enemy_result = enemy_check(object_result);

        pnp_kpt_preprocess(enemy_result);
        image_show(frame_one, enemy_result, *this);
        if(cv::waitKey(1)=='q') break;
        // cv::waitKey(1); 这里我不太确定 ？

        

        frame_one = next_frame;
        std::swap(padd_one, padd_two);
        std::swap(infer_request[0], infer_request[1]);
    }
}