#include "../src/app/rgbd_app.hpp"
#include <iostream>
#include <exception>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " configs/reconstruction.yaml\n";
        return 1;
    }
    try {
        metric_mapping::app::runRgbd(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
