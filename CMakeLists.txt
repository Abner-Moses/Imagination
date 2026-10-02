cmake_minimum_required(VERSION 3.21)
project(Imagination LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Switch off features you do not need on the flight computer.
option(BUILD_TWO_VIEW "Build calibrated stereo and synthetic demo" ON)
option(BUILD_RGBD "Build the RGB-D command" ON)
option(BUILD_MOTION "Build sparse temporal motion and benchmark" ON)
option(BUILD_MOTION_CAMERA "Build the webcam command (requires videoio)" OFF)
option(BUILD_VISUAL_FEATURES "Build analytic camera pre-extraction" ON)
option(BUILD_TESTING "Build regression tests" OFF)
# OpenCV 5 renamed modules such as features2d/calib3d to features/calib.
# Load the package once, then link only the modules this build needs.
find_package(OpenCV REQUIRED)
set(IMAGINATION_OPENCV_TARGETS opencv_core opencv_imgproc opencv_imgcodecs)
if(BUILD_MOTION OR BUILD_VISUAL_FEATURES)
    list(APPEND IMAGINATION_OPENCV_TARGETS opencv_video)
endif()
if(BUILD_MOTION_CAMERA)
    list(APPEND IMAGINATION_OPENCV_TARGETS opencv_videoio)
endif()
if(BUILD_TWO_VIEW OR BUILD_VISUAL_FEATURES)
    if(TARGET opencv_features)
        list(APPEND IMAGINATION_OPENCV_TARGETS opencv_features opencv_calib)
    else()
        list(APPEND IMAGINATION_OPENCV_TARGETS opencv_features2d opencv_calib3d)
    endif()
endif()
foreach(target IN LISTS IMAGINATION_OPENCV_TARGETS)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "Required OpenCV module target is missing: ${target}")
    endif()
endforeach()

# Each file owns one job; optional algorithms are excluded at build time.
add_library(metric_mapping
    IMF_HTransformer/imf/cpp/config.cpp
    IMF_HTransformer/imf/cpp/geometry.cpp
    IMF_HTransformer/imf/cpp/terrain.cpp
    IMF_HTransformer/imf/cpp/output.cpp)
add_library(Imagination::core ALIAS metric_mapping)
target_include_directories(metric_mapping PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/IMF_HTransformer/imf/cpp" "${CMAKE_CURRENT_BINARY_DIR}/include")
target_include_directories(metric_mapping SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(metric_mapping PUBLIC ${IMAGINATION_OPENCV_TARGETS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(metric_mapping PRIVATE -Wall -Wextra)
elseif(MSVC)
    target_compile_options(metric_mapping PRIVATE /W4)
endif()

if(BUILD_RGBD)
    target_compile_definitions(metric_mapping PUBLIC HAVE_RGBD)
endif()
if(BUILD_MOTION)
    target_sources(metric_mapping PRIVATE IMF_HTransformer/imf/cpp/motion.cpp)
    target_compile_definitions(metric_mapping PUBLIC HAVE_MOTION HAVE_VISUAL_MOTION)
    add_library(metric_motion ALIAS metric_mapping)
endif()
if(BUILD_MOTION_CAMERA)
    if(NOT BUILD_MOTION)
        message(FATAL_ERROR "BUILD_MOTION_CAMERA requires BUILD_MOTION")
    endif()
    target_compile_definitions(metric_mapping PUBLIC HAVE_MOTION_CAMERA)
endif()
if(BUILD_TWO_VIEW OR BUILD_VISUAL_FEATURES)
    # Required feature modules were resolved as components above.
endif()
if(BUILD_TWO_VIEW)
    target_compile_definitions(metric_mapping PUBLIC HAVE_TWO_VIEW)
    add_library(metric_stereo ALIAS metric_mapping)
endif()
if(BUILD_VISUAL_FEATURES)
    target_sources(metric_mapping PRIVATE IMF_HTransformer/imf/cpp/vision.cpp)
    target_compile_definitions(metric_mapping PUBLIC HAVE_VISUAL_FEATURES)
    add_library(metric_visual ALIAS metric_mapping)
endif()

add_executable(imagination tools/imagination_cli.cpp)
target_link_libraries(imagination PRIVATE metric_mapping)

# Existing executable names remain usable, without compiling five copies.
function(keep_command name enabled)
    if(${enabled})
        if(WIN32)
            set(alias_operation copy_if_different)
        else()
            set(alias_operation create_symlink)
        endif()
        add_custom_target(${name} ALL
            COMMAND ${CMAKE_COMMAND} -E ${alias_operation}
                $<TARGET_FILE:imagination> "${CMAKE_CURRENT_BINARY_DIR}/${name}${CMAKE_EXECUTABLE_SUFFIX}"
            DEPENDS imagination VERBATIM)
    else()
        file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/${name}${CMAKE_EXECUTABLE_SUFFIX}")
    endif()
endfunction()
keep_command(two_view BUILD_TWO_VIEW)
keep_command(metric_mapper BUILD_RGBD)
keep_command(motion_benchmark BUILD_MOTION)
keep_command(motion_camera BUILD_MOTION_CAMERA)
keep_command(optical_flow BUILD_MOTION_CAMERA)
keep_command(visual_features BUILD_VISUAL_FEATURES)

# Compatibility for applications that still include the previous header paths.
# These tiny forwarding headers are generated, not a second source tree to edit.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/metric_mapping")
foreach(header types config geometry point_cloud terrain ultrasonic io visualization
        two_view motion_geometry optical_flow visual_features)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/metric_mapping/${header}.hpp"
        "#pragma once\n#include \"imagination.hpp\"\n")
endforeach()

include(CTest)
if(BUILD_TESTING)
    add_executable(metric_mapping_tests tests/cpp/test_imf.cpp)
    target_link_libraries(metric_mapping_tests PRIVATE metric_mapping)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(metric_mapping_tests PRIVATE -Wall -Wextra)
    elseif(MSVC)
        target_compile_options(metric_mapping_tests PRIVATE /W4)
    endif()
    add_test(NAME metric_mapping_tests COMMAND metric_mapping_tests)
    if(BUILD_MOTION)
        add_test(NAME motion_synthetic_demo COMMAND imagination motion_benchmark --frames 60 --triangulate)
    endif()
    if(BUILD_VISUAL_FEATURES)
        add_test(NAME analytical_still COMMAND imagination pre_extract
            "${CMAKE_CURRENT_SOURCE_DIR}/data/raw/assets/sample_01.png" "${CMAKE_CURRENT_BINARY_DIR}/analytical-still-test"
            --config "${CMAKE_CURRENT_SOURCE_DIR}/data/configs/pre_extraction.yaml" --data --debug)
        if(BUILD_MOTION)
            add_test(NAME analytical_sequence COMMAND imagination pre_extract ignored
                "${CMAKE_CURRENT_BINARY_DIR}/analytical-sequence-test" --synthetic --frames 60)
        endif()
    endif()
    if(BUILD_TWO_VIEW AND UNIX)
        set(demo_work "${CMAKE_CURRENT_BINARY_DIR}/demo-json-test")
        file(MAKE_DIRECTORY "${demo_work}")
        set(quoted_image "${demo_work}/a\"quoted.png")
        file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/raw/assets/sample_01.png" DESTINATION "${demo_work}")
        file(RENAME "${demo_work}/sample_01.png" "${quoted_image}")
        add_test(NAME demo_json_generate COMMAND imagination two_view --demo
            "${quoted_image}" "${CMAKE_CURRENT_SOURCE_DIR}/data/raw/assets/sample_02.png")
        set_tests_properties(demo_json_generate PROPERTIES
            WORKING_DIRECTORY "${demo_work}" FIXTURES_SETUP demo_json)
        add_test(NAME demo_json_roundtrip COMMAND metric_mapping_tests --check-demo-json
            "${demo_work}/demo_output/metadata.json" "${quoted_image}")
        set_tests_properties(demo_json_roundtrip PROPERTIES FIXTURES_REQUIRED demo_json)
    endif()
endif()

# Copy canonical YAML examples for legacy clients expecting them beside build products.
foreach(name reconstruction ultrasonic motion_camera pre_extraction)
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/configs/${name}.yaml")
        configure_file("data/configs/${name}.yaml" "${CMAKE_CURRENT_BINARY_DIR}/configs/${name}.yaml" COPYONLY)
    endif()
endforeach()
