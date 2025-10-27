//inference.cpp

// cv::Mat yolo_kpt::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd) {
//     int in_w = src.cols, in_h = src.rows;
//     float r = std::min(float(w) / in_w, float(h) / in_h);
//     int new_w = int(std::round(in_w * r));
//     int new_h = int(std::round(in_h * r));

//     int dw = w - new_w;   // total width padding
//     int dh = h - new_h;   // total height padding
//     int left   = dw / 2;
//     int right  = dw - left;
//     int top    = dh / 2;
//     int bottom = dh - top;

//     cv::Mat resize_img;
//     cv::resize(src, resize_img, cv::Size(new_w, new_h));
//     cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right,
//                        cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

//     padd.clear();
//     padd.push_back(float(left));   // padd[0] = left
//     padd.push_back(float(top));    // padd[1] = top
//     padd.push_back(r);             // padd[2] = scale

//     return resize_img;
// }
