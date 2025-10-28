#include "garage/wrapper_car.h"
// --- debug新增 ---
#define DEBUG_ERROR_STATS 1 //1:开启误差统计，0：关闭误差统计
#define DEBUG_PRINT_ERROR 1 //1:开启误差打印，0：关闭误差打印 
// ----------------
using namespace rm;
using namespace std;

WrapperCar::WrapperCar(ArmorID id) : ObjInterface(id) {
    this->id_ = id;
    auto param = Param::get_instance();

    double track_count = (*param)["Kalman"]["TrackQueue"]["Count"];
    double track_dist = (*param)["Kalman"]["TrackQueue"]["Distance"];
    double track_delay = (*param)["Kalman"]["TrackQueue"]["Delay"];

    double antitop_min_r = (*param)["Kalman"]["Antitop"]["MinR"];
    double antitop_max_r = (*param)["Kalman"]["Antitop"]["MaxR"];

    int antitop_fire_update = (*param)["Kalman"]["Antitop"]["FireUpdate"];
    double antitop_fire_delay = (*param)["Kalman"]["Antitop"]["FireDelay"];
    double antitop_fire_angle = (*param)["Kalman"]["Antitop"]["FireAngle"]["Armor"];
    double antitop_fire_angle_big_ = (*param)["Kalman"]["Antitop"]["FireAngle"]["CenterBig"];
    double antitop_fire_angle_small_ = (*param)["Kalman"]["Antitop"]["FireAngle"]["CenterSmall"];

    vector<double> antitopQ = (*param)["Kalman"]["Antitop"]["Q"];
    vector<double> antitopR = (*param)["Kalman"]["Antitop"]["R"];
    vector<double> antitopCenterQ = (*param)["Kalman"]["Antitop"]["CenterQ"];
    vector<double> antitopCenterR = (*param)["Kalman"]["Antitop"]["CenterR"];

    vector<double> antitopOmegaQ = (*param)["Kalman"]["Antitop"]["OmegaQ"];
    vector<double> antitopOmegaR = (*param)["Kalman"]["Antitop"]["OmegaR"];
    vector<double> antitopBalanceOmegaQ = (*param)["Kalman"]["Antitop"]["BalanceOmegaQ"];
    vector<double> antitopBalanceOmegaR = (*param)["Kalman"]["Antitop"]["BalanceOmegaR"];

    vector<double> trackqueueQ = (*param)["Kalman"]["TrackQueue"]["CarQ"];
    vector<double> trackqueueR = (*param)["Kalman"]["TrackQueue"]["CarR"];

    track_to_antitop_ = (*param)["Kalman"]["Switch"]["TrackToAntitop"];
    antitop_to_track_ = (*param)["Kalman"]["Switch"]["AntitopToTrack"];

    armor_to_center_ = (*param)["Kalman"]["Switch"]["ArmorToCenter"];
    center_to_armor_ = (*param)["Kalman"]["Switch"]["CenterToArmor"];

    track_queue_ = TrackQueueV4(track_count, track_dist, track_delay);
    antitop_4_ = new AntitopV3(antitop_min_r, antitop_max_r, 4);
    antitop_2_ = new AntitopV3(antitop_min_r, antitop_max_r, 2);

    track_queue_.setMatrixQ(
        trackqueueQ[0], trackqueueQ[1], trackqueueQ[2], trackqueueQ[3], trackqueueQ[4], trackqueueQ[5],
        trackqueueQ[6], trackqueueQ[7]);
    track_queue_.setMatrixR(trackqueueR[0], trackqueueR[1], trackqueueR[2]);

    antitop_4_->setMatrixQ(
        antitopQ[0], antitopQ[1], antitopQ[2], antitopQ[3], antitopQ[4], 
        antitopQ[5], antitopQ[6], antitopQ[7], antitopQ[8]);
    antitop_4_->setMatrixR(antitopR[0], antitopR[1], antitopR[2], antitopR[3]);

    antitop_4_->setCenterMatrixQ(antitopCenterQ[0], antitopCenterQ[1], antitopCenterQ[2], antitopCenterQ[3]);
    antitop_4_->setCenterMatrixR(antitopCenterR[0], antitopCenterR[1]);

    antitop_4_->setOmegaMatrixQ(antitopOmegaQ[0], antitopOmegaQ[1], antitopOmegaQ[2]);
    antitop_4_->setOmegaMatrixR(antitopOmegaQ[0]);
    
    if (id_ == rm::ARMOR_ID_HERO)
        antitop_4_->setFireValue(antitop_fire_update, antitop_fire_delay, antitop_fire_angle, antitop_fire_angle_big_);
    else
        antitop_4_->setFireValue(antitop_fire_update, antitop_fire_delay, antitop_fire_angle, antitop_fire_angle_small_);


    antitop_2_->setMatrixQ(
        antitopQ[0], antitopQ[1], antitopQ[2], antitopQ[3], antitopQ[4], 
        antitopQ[5], antitopQ[6], antitopQ[7], antitopQ[8]);
    antitop_2_->setMatrixR(antitopR[0], antitopR[1], antitopR[2], antitopR[3]);

    antitop_2_->setCenterMatrixQ(antitopCenterQ[0], antitopCenterQ[1], antitopCenterQ[2], antitopCenterQ[3]);
    antitop_2_->setCenterMatrixR(antitopCenterR[0], antitopCenterR[1]);

    antitop_2_->setOmegaMatrixQ(antitopBalanceOmegaQ[0], antitopBalanceOmegaQ[1], antitopBalanceOmegaQ[2]);
    antitop_2_->setOmegaMatrixR(antitopBalanceOmegaR[0]);
    
    antitop_2_->setFireValue(antitop_fire_update, antitop_fire_delay, antitop_fire_angle, antitop_fire_angle_big_);

}

void WrapperCar::push(const Target& target, TimePoint t) {
    Eigen::Vector4d pose(
        target.pose_world[0], target.pose_world[1], target.pose_world[2], target.armor_yaw_world
    );


    /*-------debug新增-------------------------------*///记录推入数据前的状态
    //rm::TQstateV4* last_states_ = track_queue_.get_last_state_();
    //bool is_first_frame = (last_states_== nullptr || last_states_->count == 0);
    //转换为毫米再传入track_queue_的push,因为push内部还要再进行一次坐标转换，从mm -> m
    Eigen::Vector4d pose_in(
        target.pose_world[0] * 1000, target.pose_world[1] * 1000, target.pose_world[2] * 1000, target.armor_yaw_world
    );
    /*--------------------------------------------*/

    track_queue_.push(pose_in, t);

    curr_armor_num_++;
    if (target.armor_size == ARMOR_SIZE_BIG_ARMOR) {
        armor_size_count_ += 1;
    }

    //--------------- debug新增 ---------------------
    for(const auto& pred : history_predictions_){
        if(history_predictions_.size() == 0) break;
        auto err = error_calculator_.calculerror4D(pred.pose_dt , pose , t ,pred.t );//计算误差
        if(err.is_error_valid){
            error_results_.push_back(err);//存储误差结果
        //打印误差结果
        //Print_error(err);&& error_results_.size() % 10 == 0
            if(DEBUG_PRINT_ERROR ){
                //error_calculator_.Print_error();
                std::cout << "[DEBUG] Attempting to print error result. Index:" 
                << error_results_.size()<< ", is_valid: " << std::boolalpha 
                << err.is_error_valid << std::endl; // 先打印索引和有效性
                Print_error(err);
            }
        }
    }

    //定期更新误差统计结果
    if(error_results_.size() % 10 == 0){
        error_stats_ = error_calculator_.getErrorStats(error_results_);//每10帧计算一次误差统计结果
    }

    //打印误差统计结果
    if(DEBUG_ERROR_STATS && error_stats_.total_count > 0 && error_stats_.total_count % 50 == 0){
        printErrorStates();
    }
    //只保留最近1000个误差结果
    for(auto it = error_results_.begin(); it != error_results_.end(); ){
        if(error_results_.size() > 1000){
            it = error_results_.erase(it);
        } else {
            break;
        }
    }
//--------------------------------------------------

    
}

void WrapperCar::update() {
    //GKDTODO (figureout what is WrapperCar::update and why use it)

    track_queue_.update();

    if (curr_armor_num_ > 1) {
        armor_size_count_  -= 1;
    }
    curr_armor_num_ = 0;

    Eigen::Vector4d pose;
    TimePoint t;
    // if (!track_queue_.getPose(pose, t)) return;

    rm::AntitopV3* antitop = nullptr;
    if (id_ == rm::ARMOR_ID_INFANTRY_3 || id_ == rm::ARMOR_ID_INFANTRY_4 || id_ == rm::ARMOR_ID_INFANTRY_5) {

        if (armor_size_count_ > 0) {
            antitop = antitop_2_;
            // rm::message("antitop armor", 2);
        }
        else {
            antitop = antitop_4_;
            // rm::message("antitop armor", 4);
        }
    } else {
        antitop = antitop_4_;
        // rm::message("antitop armor", 4);
    }

    // antitop->push(pose, t);
}


inline void fuckWorld(
    Eigen::Matrix<double, 4, 4>& matrix_trans,
    const double yaw,
    const double pitch
) {
    Eigen::Matrix<double, 4, 4> rotate_yaw, rotate_pitch;

    rotate_yaw << cos(yaw), -sin(yaw), 0, 0,
                  sin(yaw),  cos(yaw), 0, 0,
                         0,         0, 1, 0,
                         0,         0, 0, 1;

    rotate_pitch << cos(pitch), 0, -sin(pitch), 0,
                             0, 1,           0, 0,
                    sin(pitch), 0,  cos(pitch), 0,
                             0, 0,           0, 1;

    matrix_trans = rotate_yaw * rotate_pitch;
}

bool WrapperCar::getTarget(Eigen::Vector4d& pose_rotate, const double fly_delay, const double rotate_delay, const double shoot_delay) {
    rm::AntitopV3* antitop = nullptr;
    Eigen::Vector4d pose_rotate_abs;
    Eigen::Matrix4d trans_head2world;

    if (id_ == rm::ARMOR_ID_INFANTRY_3 || id_ == rm::ARMOR_ID_INFANTRY_4 || id_ == rm::ARMOR_ID_INFANTRY_5) {

        if (armor_size_count_ > 0) {
            antitop = antitop_2_;
            // rm::message("antitop armor", 2);
        }
        else {
            antitop = antitop_4_;
            // rm::message("antitop armor", 4);
        }
    } else {
        antitop = antitop_4_;
        // rm::message("antitop armor", 4);
    }
    //射击延迟预测值就是计算子弹到达目标时目标的位置，单位是米
    Eigen::Vector4d pose_shoot = track_queue_.getPose(fly_delay + shoot_delay);//世界坐标系
    //云台延迟计算的就是当完成瞄准时，目标的位置，单位是米
    pose_rotate_abs = track_queue_.getPose(fly_delay + rotate_delay);//世界坐标系
    //这里转换为云台坐标系
    rm::tf_trans_head2world(trans_head2world, -Data::yaw, 0.);
    pose_rotate = trans_head2world * pose_rotate_abs;

    // --------------- debug新增 ---------------------
    TimePoint now = getTime();
    history_predictions_.push_back({now,pose_rotate_abs,fly_delay + rotate_delay});
    // 仅保留最近100个观测结果
    while(history_predictions_.size() > 100) {
        history_predictions_.erase(history_predictions_.begin());
    }
    //------------------------------------------------


    if(false)
    {
        std::cout << "yaw\t" << Data::yaw << std::endl; 
        std::cout << "pitch\t" << Data::pitch << std::endl;
        std::cout << "matrix world->gimbal\n" << trans_head2world << std::endl;
        std::cout << "pose gimbal?\n" << pose_rotate << std::endl;
    }


    Data::target_omega = antitop->getOmega();
    // rm::message("target omg", Data::target_omega);


    if(abs(Data::target_omega) > track_to_antitop_) flag_antitop_ = true;
    else if(abs(Data::target_omega) < antitop_to_track_) flag_antitop_ = false; 

    if(abs(Data::target_omega) > armor_to_center_) flag_center_ = true;
    else if(abs(Data::target_omega) < center_to_armor_) flag_center_ = false;

    if (flag_antitop_) {
        if(flag_center_) {
            // rm::message("mode", 'C');
            pose_shoot = antitop->getCenter(fly_delay + shoot_delay);
            pose_rotate = antitop->getCenter(fly_delay + rotate_delay);
            return antitop->getFireCenter(pose_shoot);
        } else {
            // rm::message("mode", 'A');
            pose_shoot = antitop->getPose(fly_delay + shoot_delay);
            pose_rotate = antitop->getPose(fly_delay + rotate_delay);

            return antitop->getFireArmor(pose_shoot);
        }      
    } else {
        // rm::message("mode", 'T');
        return true;
    }
    
}

rm::ArmorSize WrapperCar::getArmorSize() {
    double abs_big, abs_small;
    switch (id_) {
        case rm::ARMOR_ID_SENTRY:
            size_ = rm::ARMOR_SIZE_SMALL_ARMOR;
            return rm::ARMOR_SIZE_SMALL_ARMOR;
            
        case rm::ARMOR_ID_HERO:
            size_ = rm::ARMOR_SIZE_BIG_ARMOR;
            return rm::ARMOR_SIZE_BIG_ARMOR;

        case rm::ARMOR_ID_INFANTRY_3:
        case rm::ARMOR_ID_INFANTRY_4:
        case rm::ARMOR_ID_INFANTRY_5:

            if (armor_size_count_ > 0) {
                size_ = rm::ARMOR_SIZE_BIG_ARMOR;
                return rm::ARMOR_SIZE_BIG_ARMOR;
            } else {
                size_ = rm::ARMOR_SIZE_SMALL_ARMOR;
                return rm::ARMOR_SIZE_SMALL_ARMOR;
            }

        default:
            size_ = rm::ARMOR_SIZE_UNKNOWN;
            return rm::ARMOR_SIZE_UNKNOWN;
    }
}

void WrapperCar::getState(std::vector<std::string>& lines) {
    lines.clear();

    if (id_ == rm::ARMOR_ID_INFANTRY_3 || id_ == rm::ARMOR_ID_INFANTRY_4 || id_ == rm::ARMOR_ID_INFANTRY_5) {
        if (armor_size_count_ > 0) antitop_2_->getStateStr(lines);
        else antitop_4_->getStateStr(lines);
    } else antitop_4_->getStateStr(lines);
    track_queue_.getStateStr(lines);
}


//打印误差信息
void WrapperCar::Print_error(error_result err){
    std::cout<<"==== Error Result ===="<<std::endl;
    std::cout<<"pos_error [m]: "<<err.pos_error<<std::endl;
    std::cout<<"angle_error [rad]: "<<err.angle_error<<std::endl;
    std::cout<<"delay_time [s]: "<<err.delay_time <<std::endl;
    std::cout<<"last_predict_pose[x,y,z,angle]: ["<<err.last_predic_pose_4(0)<<","
                                                    <<err.last_predic_pose_4(1)<<","
                                                    <<err.last_predic_pose_4(2)<<","
                                                    <<err.last_predic_pose_4(3) <<"]"<<std::endl;
    std::cout<<"now_pose[x,y,z,angle]: ["<<err.now_pose_4(0)<<","
                                                    <<err.now_pose_4(1)<<","
                                                    <<err.now_pose_4(2)<<","
                                                    <<err.now_pose_4(3) <<"]"<<std::endl;
    std::cout<<"history_predictioj_size: "<<history_predictions_.size()<<std::endl; 
    
    return;
}