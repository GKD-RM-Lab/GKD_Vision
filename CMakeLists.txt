cmake_minimum_required(VERSION 3.15)
project(GKD_Vision LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ==============================================================
# 0. 基础设置
# ==============================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -w)

message(STATUS "===============================================")
message(STATUS "🚀 Building project: ${PROJECT_NAME}")
message(STATUS "===============================================")

# ==============================================================
# 1. OpenCV
# ==============================================================
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "❌ 未找到 OpenCV，请先安装 OpenCV！")
endif()
message(STATUS "✅ Using OpenCV version: ${OpenCV_VERSION}")

# ==============================================================
# 2. OpenVINO
# ==============================================================
set(OpenVINODeveloperPackage_DIR "/usr/lib/cmake/openvino2024.6.0")
find_package(OpenVINO REQUIRED)
message(STATUS "✅ Using OpenVINO version: ${openvino_VERSION}")

# ==============================================================
# 3. Eigen3
# ==============================================================
find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIRS})
else()
    include_directories(/usr/include/eigen3)
endif()

# ==============================================================
# 4. 海康相机 SDK
# ==============================================================
set(CAMERA_LIB_DIR "${CMAKE_SOURCE_DIR}/lib/64")
link_directories(${CAMERA_LIB_DIR})

# ==============================================================
# 5. 包含头文件目录
# ==============================================================
include_directories(
    ${CMAKE_SOURCE_DIR}/include              # 项目公共头文件
    ${CMAKE_SOURCE_DIR}/lib/include          # 第三方库头文件
    ${CMAKE_SOURCE_DIR}/lib/nlohmann         # JSON 库头文件
    ${CMAKE_SOURCE_DIR}/lib                  # 如有直接放在 lib 下的头文件
    ${OpenCV_INCLUDE_DIRS}                   # OpenCV 头文件
    # ${OpenVINO_INCLUDE_DIRS}               # 通常 find_package(OpenVINO) 已经处理
)

# ==============================================================
# 6. 源文件
# ==============================================================
file(GLOB_RECURSE SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
set(MAIN_FILE ${CMAKE_SOURCE_DIR}/main.cpp)

# ==============================================================
# 7. 生成可执行文件
# ==============================================================
add_executable(GKD_Vision ${MAIN_FILE} ${SOURCE_FILES})

# ==============================================================
# 8. 链接依赖库
# ==============================================================
target_link_libraries(GKD_Vision
    ${OpenCV_LIBS}
    openvino::runtime

    # 海康相机驱动相关库
    MvCameraControl
    MVGigEVisionSDK
    FormatConversion
    GCBase_gcc421_v3_0
    GenApi_gcc421_v3_0
    Log_gcc421_v3_0
    MediaProcess
    MVRender
    MvUsb3vTL
    NodeMapData_gcc421_v3_0
    XmlParser_gcc421_v3_0

    pthread
    X11
)

# ==============================================================
# 9. RPATH 设置（运行时能找到 lib/64）
# ==============================================================
set_target_properties(GKD_Vision PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib/64"
    INSTALL_RPATH "\$ORIGIN/../lib/64"
)

# ==============================================================
# 10. 兵种宏定义
# ==============================================================
set(ARMY "infantry" CACHE STRING "Robot type: infantry/hero/sentry_l/sentry_r/balance/dronse")
if(ARMY STREQUAL "infantry")
    target_compile_definitions(GKD_Vision PRIVATE TJURM_INFANTRY)
elseif(ARMY STREQUAL "hero")
    target_compile_definitions(GKD_Vision PRIVATE TJURM_HERO)
elseif(ARMY STREQUAL "sentry_l" OR ARMY STREQUAL "sentry_r")
    target_compile_definitions(GKD_Vision PRIVATE TJURM_SENTRY)
elseif(ARMY STREQUAL "balance")
    target_compile_definitions(GKD_Vision PRIVATE TJURM_BALANCE)
elseif(ARMY STREQUAL "dronse")
    target_compile_definitions(GKD_Vision PRIVATE TJURM_DRONSE)
endif()

message(STATUS "✅ Current army type: ${ARMY}")
message(STATUS "===============================================")
message(STATUS "✅ CMake configuration complete!")
message(STATUS "===============================================")
