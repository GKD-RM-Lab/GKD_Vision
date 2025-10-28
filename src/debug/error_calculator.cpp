#include <iostream>
#include <Eigen/Dense>
#include <utils/timer.h>
#include "utils/print.h"
#include "debug/error_calculator.h"
#include <memory>
#include <cmath>

using namespace rm;
using namespace std;

//安全计算角度差
double ErrorCalculator::getSafeAngle(const double angle1,const double angle2){
    double angle = angle1 - angle2;
    while(angle > M_PI) angle -= 2 * M_PI;
    while(angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

//根据误差结果计算误差分析
error_states ErrorCalculator::getErrorStats(const std::vector<error_result>& error){
    error_states_ = error_states(); //重置统计结果
    if(error.size() == 0) return error_states_;//没有数据直接返回

    error_states_.total_count = error.size();
    double all_pos = 0,all_angle = 0;

    //筛选有效误差并计算所有的误差和
    std::vector<error_result> valid_error;
    for (const auto& e : error) {
        if (e.is_error_valid) {
            valid_error.push_back(e);
            all_pos += e.pos_error;
            all_angle += e.angle_error;
        }
    }

    error_states_.valid_count = valid_error.size();//有效误差数量

    error_states_.avg_pos_error = all_pos / valid_error.size();
    error_states_.avg_angle_error = all_angle / valid_error.size();
    error_states_.max_pos_error = -1;
    error_states_.min_pos_error = 10000000.0;
    double std_all_pos = 0,std_all_angle = 0;
    for (auto i:error)
    {
        if( (i.pos_error > error_states_.max_pos_error) && i.is_error_valid){
            error_states_.max_pos_error = i.pos_error;
        }
        if( (i.pos_error < error_states_.min_pos_error) && i.is_error_valid){
            error_states_.min_pos_error = i.pos_error;
        }
        std_all_pos += pow(error_states_.avg_pos_error - i.pos_error,2);
        std_all_angle += pow(error_states_.avg_angle_error - i.angle_error,2);
    }
    error_states_.std_pos_error = sqrt(std_all_pos / valid_error.size() );
    error_states_.std_angle_error = sqrt(std_all_angle / valid_error.size() );
    return error_states_;
}

double ErrorCalculator::getDoubleOfS(const TimePoint& start, const TimePoint& end) {
    return getDuration_s(start, end).count();
}

double ErrorCalculator::getDistance(const Eigen::Matrix<double, 3, 1>& this_pose, const Eigen::Matrix<double, 3, 1>& last_pose){
    double dx = this_pose(0) - last_pose(0);
    double dy = this_pose(1) - last_pose(1);
    double dz = this_pose(2) - last_pose(2);
    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    return d;
}

double ErrorCalculator::getDistance(const Eigen::Matrix<double, 4, 1>& this_pose, const Eigen::Matrix<double, 4, 1>& last_pose) {
    double dx = this_pose(0) - last_pose(0);
    double dy = this_pose(1) - last_pose(1);
    double dz = this_pose(2) - last_pose(2);
    double d = std::sqrt(dx * dx + dy * dy + dz * dz);
    return d;
}

//打印误差信息
// void ErrorCalculator::Print_error(){
// std::cout<<"==== Error Result ===="<<std::endl;
// std::cout<<"pos_error: "<<error_result_.pos_error<<std::endl;
// std::cout<<"angle_error: "<<error_result_.angle_error<<std::endl;
// std::cout<<"combine_error: "<<error_result_.combin_error<<std::endl;
// std::cout<<"delay_time: "<<error_result_.delay_time <<std::endl;
// std::cout<<"last_predict_pose[x,y,z,angle]: ["<<error_result_.last_predic_pose(0)<<","
// <<error_result_.last_predic_pose(1)<<","
// <<error_result_.last_predic_pose(2)<<","
// <<error_result_.last_predic_pose(3) <<"]"<<std::endl;
// std::cout<<"now_pose[x,y,z,angle]: ["<<error_result_.now_pose(0)<<","
// <<error_result_.now_pose(1)<<","
// <<error_result_.now_pose(2)<<","
// <<error_result_.now_pose(3) <<"]"<<std::endl;
// return;
// }

void ErrorCalculator::Print_error_states(){
    // std::cout<<"==== Error States ===="<<std::endl;
    // std::cout<<"avg_pos_error [m]: "<<error_states_.avg_pos_error<<std::endl;
    // std::cout<<"avg_angle_error [rad]: "<<error_states_.avg_angle_error<<std::endl;
    // std::cout<<"std_pos_error [m]: "<<error_states_.std_pos_error<<std::endl;
    // std::cout<<"std_angle_error [rad]: "<<error_states_.std_angle_error<<std::endl;
    // std::cout<<"max_pos_error [m]: "<<error_states_.max_pos_error<<std::endl;
    // std::cout<<"min_pos_error [m]: "<<error_states_.min_pos_error<<std::endl;
    // std::cout<<"valid_count: "<<error_states_.valid_count<<std::endl;
    // std::cout<<"total_count: "<<error_states_.total_count<<std::endl;
    // return;
}

//3D误差暂时不可用，error_result_.now_pose = view报错，两边矩阵维数不相等
error_result ErrorCalculator::calculerror3D(const Eigen::Matrix<double, 3, 1> prediction , const Eigen::Matrix<double , 3, 1 > view,TimePoint predict_time,TimePoint view_time){
    error_result error_result_n; //重置误差结果
    error_result_n.now_pose_3 = view;
    error_result_n.last_predic_pose_3 = prediction;
    error_result_n.now_pose_4 << 0,0,0,0;
    error_result_n.last_predic_pose_4 << 0,0,0,0;
    error_result_n.pos_error = getDistance(prediction,view);
    double time_diff = getDoubleOfS(view_time,predict_time);//计算预测和观测时间差
    //时间差过大，误差无效
    if(time_diff > 0.5){ 
        error_result_n.is_error_valid = false;
        return error_result_n;
    } 
    error_result_n.delay_time = time_diff;
    error_result_n.is_error_valid = true;//3D误差只要位置误差有效即可
    return error_result_n;
}

error_result ErrorCalculator::calculerror4D(const Eigen::Matrix<double, 4, 1> prediction , const Eigen::Matrix<double , 4, 1 > view,TimePoint predict_time,TimePoint view_time){
    error_result error_result_n;
    //error_result_ = error_result(); //重置误差结果
    error_result_n.now_pose_4 = view;
    error_result_n.last_predic_pose_4 = prediction;
    error_result_n.now_pose_3 << 0,0,0;
    error_result_n.last_predic_pose_3 << 0,0,0;
    error_result_n.pos_error = getDistance(prediction,view);
    error_result_n.angle_error = getSafeAngle(prediction(3),view(3));
    double time_diff = getDoubleOfS(view_time,predict_time);//计算预测和观测时间差
    //时间差过大，误差无效
    if(time_diff > 0.5){ 
        error_result_n.is_error_valid = false;
        return error_result_n;
    } 
    error_result_n.delay_time = time_diff;
    error_result_n.is_error_valid = true;//4D误差只要位置和角度误差有效即可
    return error_result_n;
}

//打印误差信息
void ErrorCalculator::Print_error(error_result err){
    // std::cout<<"==== Error Result ===="<<std::endl;
    // std::cout<<"pos_error [m]: "<<err.pos_error<<std::endl;
    // std::cout<<"angle_error [rad]: "<<err.angle_error<<std::endl;
    // std::cout<<"delay_time [s]: "<<err.delay_time <<std::endl;
    // std::cout<<"last_predict_pose_4D[x,y,z,angle]: ["<<err.last_predic_pose_4(0)<<","
    // <<err.last_predic_pose_4(1)<<","
    // <<err.last_predic_pose_4(2)<<","
    // <<err.last_predic_pose_4(3) <<"]"<<std::endl;
    // std::cout<<"now_pose[x,y,z,angle]: ["<<err.now_pose_4(0)<<","
    // <<err.now_pose_4(1)<<","
    // <<err.now_pose_4(2)<<","
    // <<err.now_pose_4(3) <<"]"<<std::endl;
    return;
}