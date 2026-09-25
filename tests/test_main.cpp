#include "test_support.hpp"
#include <iostream>

using namespace metric_mapping::tests;

int main(int argc, char** argv)
{
    if (argc == 4 && std::string(argv[1]) == "--check-demo-json") {
        try {
            cv::FileStorage metadata(argv[2], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            require(metadata.isOpened(), "Demo metadata must be valid JSON");
            require(static_cast<std::string>(metadata["input_image_1"]) == argv[3],
                    "Escaped demo path must round-trip exactly");
            require(static_cast<int>(metadata["raw_points"]) == 262144 &&
                    static_cast<int>(metadata["measured_cells"]) == 262144 &&
                    static_cast<int>(metadata["idw_interpolated_cells"]) == 0,
                    "Demo must preserve every original lattice measurement");
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    std::vector<TestCase> tests;
    for (const auto& group : {rgbdTests(), terrainTests(), configTests(), outputTests()})
        tests.insert(tests.end(), group.begin(), group.end());
#ifdef HAVE_TWO_VIEW
    const auto stereo = stereoTests();
    tests.insert(tests.end(), stereo.begin(), stereo.end());
#endif

    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS: " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << test.first << ": " << error.what()
                      << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << tests.size() << " tests passed\n";
    return 0;
}
