cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()


#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#common commands for building c++ executables and libraries
rosbuild_add_library(${PROJECT_NAME} src/scene.cpp src/segmentation.cpp
  src/tracked_object.cpp src/sequence.cpp src/node_potential.cpp
  src/hand_segmenter_view_controller.cpp)

rosbuild_add_executable(view_data src/view_data.cpp)
target_link_libraries(view_data ${PROJECT_NAME})

#rosbuild_add_executable(baseline_segmenter src/baseline_segmenter.cpp)
#target_link_libraries(baseline_segmenter ${PROJECT_NAME})

rosbuild_add_executable(hand_segmenter src/hand_segmenter.cpp)
target_link_libraries(hand_segmenter ${PROJECT_NAME})

rosbuild_add_executable(graphcuts_segmenter src/graphcuts_segmenter.cpp)
target_link_libraries(graphcuts_segmenter ${PROJECT_NAME})

rosbuild_add_executable(evaluator src/evaluator.cpp)
target_link_libraries(evaluator ${PROJECT_NAME}) 

rosbuild_add_executable(bilateral_segmenter src/bilateral_segmenter.cpp)
target_link_libraries(bilateral_segmenter ${PROJECT_NAME}) 

rosbuild_add_executable(nographcuts_segmenter src/nographcuts_segmenter.cpp)
target_link_libraries(nographcuts_segmenter ${PROJECT_NAME}) 

rosbuild_add_executable(ssvm_learning src/ssvm_learning.cpp)
target_link_libraries(ssvm_learning ${PROJECT_NAME})

rosbuild_add_executable(ssvm_learning2 src/ssvm_learning2.cpp)
target_link_libraries(ssvm_learning2 ${PROJECT_NAME})

rosbuild_add_executable(sn_segmenter src/sn_segmenter.cpp)
target_link_libraries(sn_segmenter ${PROJECT_NAME}) 

rosbuild_add_executable(dist src/dist.cpp)
target_link_libraries(dist ${PROJECT_NAME}) 