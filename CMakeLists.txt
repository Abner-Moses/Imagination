cmake_minimum_required(VERSION 3.10)
project(metric_terrain_mapping LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(BUILD_TWO_VIEW "Build calibrated stereo and synthetic demo" ON)
option(BUILD_RGBD "Build the RGB-D command-line tool" ON)
option(BUILD_TESTING "Build regression tests" OFF)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Shared RGB-D geometry, terrain analysis, and file outputs.
add_library(metric_mapping
    src/rgbd/config.cpp
    src/rgbd/geometry.cpp
    src/terrain/grid.cpp
    src/terrain/point_cloud.cpp
    src/landing/analysis.cpp
    src/landing/surface.cpp
    src/landing/selection.cpp
    src/landing/fuzzy.cpp
    src/output/cloud_files.cpp
    src/output/raster.cpp
    src/output/rgbd_metadata.cpp
    src/output/landing_images.cpp
    src/output/landing_json.cpp
    src/output/overview.cpp)
target_include_directories(metric_mapping PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(metric_mapping SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(metric_mapping PUBLIC ${OpenCV_LIBS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(metric_mapping PRIVATE -Wall -Wextra)
endif()

if(BUILD_TWO_VIEW)
    # OpenCV 5 split stereo/geometry from calibration. Avoid unused calibration
    # and object-detection modules; imported targets carry their dependencies.
    if(TARGET opencv_features AND TARGET opencv_stereo AND TARGET opencv_geometry)
        set(STEREO_LIBS opencv_features opencv_stereo opencv_geometry)
    elseif(TARGET opencv_features2d AND TARGET opencv_calib3d)
        set(STEREO_LIBS opencv_features2d opencv_calib3d)
    else()
        message(FATAL_ERROR "Stereo requires OpenCV features and stereo geometry modules")
    endif()
    add_library(metric_stereo
        src/stereo/reconstruction.cpp
        src/stereo/features.cpp
        src/stereo/alignment.cpp
        src/stereo/dense_depth.cpp)
    target_link_libraries(metric_stereo PUBLIC metric_mapping ${STEREO_LIBS})
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(metric_stereo PRIVATE -Wall -Wextra)
    endif()
    add_executable(two_view
        apps/two_view.cpp
        src/app/stereo_run.cpp
        src/app/demo_run.cpp
        src/demo/synthetic_scene.cpp
        src/output/stereo_metadata.cpp)
    target_link_libraries(two_view PRIVATE metric_stereo)
endif()

if(BUILD_RGBD)
    add_executable(metric_mapper apps/metric_mapper.cpp src/app/rgbd_run.cpp)
    target_link_libraries(metric_mapper PRIVATE metric_mapping)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(metric_mapping_tests
        tests/test_main.cpp
        tests/rgbd_tests.cpp
        tests/terrain_tests.cpp
        tests/config_tests.cpp
        tests/output_tests.cpp)
    if(BUILD_TWO_VIEW)
        target_sources(metric_mapping_tests PRIVATE tests/stereo_tests.cpp)
        target_link_libraries(metric_mapping_tests PRIVATE metric_stereo)
        target_compile_definitions(metric_mapping_tests PRIVATE HAVE_TWO_VIEW)
    else()
        target_link_libraries(metric_mapping_tests PRIVATE metric_mapping)
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(metric_mapping_tests PRIVATE -Wall -Wextra)
    endif()
    add_test(NAME metric_mapping_tests COMMAND metric_mapping_tests)
    if(BUILD_TWO_VIEW AND UNIX)
        add_test(NAME demo_json_roundtrip COMMAND ${CMAKE_COMMAND}
            -DTOOL=$<TARGET_FILE:two_view>
            -DCHECKER=$<TARGET_FILE:metric_mapping_tests>
            -DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}
            -DWORK=${CMAKE_CURRENT_BINARY_DIR}/demo-json-test
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/demo_json.cmake)
    endif()
endif()
