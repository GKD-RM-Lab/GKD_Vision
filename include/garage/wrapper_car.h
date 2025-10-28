#ifndef RM2026_GARAGE_WRAPPER_CAR_H_
#define RM2026_GARAGE_WRAPPER_CAR_H_

#include "garage/interface.h"
#include "debug/error_calculator.h" //新增debug函数，用于计算yolo观测和trackqueue预测的误差

class WrapperCar : public ObjInterface {

public:
    WrapperCar(rm::ArmorID id);
    ~WrapperCar() {};
    void push(const rm::Target& target, TimePoint t) override;
    void update() override;
    bool getTarget(Eigen::Vector4d& pose, const double fly_delay, const double rotate_delay, const double shoot_delay) override;
    rm::ArmorSize getArmorSize() override;
    void getState(std::vector<std::string>& lines) override;
    void setState(int state) override {}

    //--- 新增debug函数 ---
    error_states getErrorStates() { return error_calculator_.error_states_; }; //获取误差统计结果
    void printErrorStates() { error_calculator_.Print_error_states(); }; //打印误差统计结果
    void Print_error(error_result err);

public:
    rm::TrackQueueV4 track_queue_;
    rm::AntitopV3* antitop_2_;
    rm::AntitopV3* antitop_4_;

    bool   flag_big_armor_ = false;         // 是否为大装甲板
    int    armor_size_count_ = 0;           // 装甲板尺寸计数
    int    curr_armor_num_ = 0;             // 当前一次更新内观测的装甲板数量

    bool flag_antitop_ = false;
    bool flag_center_ = false;

    double antitop_to_track_ = 0.6;
    double track_to_antitop_ = 1.0;
    double armor_to_center_ = 0.7;
    double center_to_armor_ = 0.6;

    //--- 新增debug成员 ---
    // 存储历史观测结果
    struct HistoryPrediction {
        TimePoint t;//当前时间点
        //Eigen::Matrix<double,4,1> view_pose_t;//该时刻的观测位姿
        Eigen::Matrix<double,4,1> pose_dt;//云台延迟预测位姿(t + dt时间后的位姿)
        double delay;//延迟时间
    };
    ErrorCalculator error_calculator_; //误差计算器
    std::vector<error_result> error_results_; //存储所有误差结果
    error_states error_stats_; //误差结果统计
    std::vector<HistoryPrediction> history_predictions_; //存储历史预测结果

};

#endif