#include "data_manager/base.h"
#include "data_manager/param.h"
#include "threads/pipeline.h"
#include "threads/control.h"
#include "garage/garage.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include "visionlib.h"
#include "video/hikdriver.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "data_manager/parameter_loader.h"

std::mutex hang_up_mutex;
std::condition_variable hang_up_cv;

int main(int argc, char** argv) {
    auto param = Param::get_instance();
    auto pipeline = Pipeline::get_instance();
    auto garage = Garage::get_instance();
    auto control = Control::get_instance();

    int option;

    para_load("/etc/visionlib/forward_config/infantry/config.yaml");

    // if(argc > 1) std::cout << "!!!!!!!!!!" << argv[1] << std::endl;
    while ((option = getopt(argc, argv, "hsv")) != -1) {
        switch (option) {
            case 's':
                Data::imshow_flag = true;
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [-h] [-s] " << std::endl;
                break;
            case 'v':
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!imshow enable" << std::endl;
                params.is_imshow = 1;
        }
    }
    

    /*相机读取线程*/
    std::thread cameraThread(HIKcamtask);
    cv::Mat inputImage;
    // rm::message_init("autoaim");
    init_debug();
    init_attack();
    //DEBUG 暂时关闭serial
    // if (Data::serial_flag) init_serial();
    //DEBUG 暂时关闭control
    control->autoaim();

    //3v3没能量机关，直接baseline
    pipeline->autoaim_baseline();

    // #if defined(TJURM_INFANTRY) || defined(TJURM_BALANCE)
    // pipeline->autoaim_combine();
    // std::cout << "set INFANTRY" << std::endl;  
    // #endif

    // #if defined(TJURM_SENTRY) || defined(TJURM_DRONSE) || defined(TJURM_HERO)
    // pipeline->autoaim_baseline();
    // std::cout << "set SENTRY" << std::endl;  
    // #endif

    while(Data::manu_fire) {
        std::cin.get();
        Data::auto_fire = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        Data::auto_fire = false;
    }

    // Keep the main thread alive to allow vision processing to continue
    // Use condition variable to hang up the main thread until signal
    std::unique_lock<std::mutex> lock(hang_up_mutex);
    hang_up_cv.wait(lock);
    
    // Before exiting, signal the camera thread to stop and wait for it to finish
    g_camera_thread_running = false;
    
    // Wait for a reasonable amount of time for the camera thread to finish
    if (cameraThread.joinable()) {
        cameraThread.join();
    }

    std::cout << "end" << std::endl;
    return 0;
}