
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include "../traits/OPOE_traits.hpp" // Using your project's traits

// A struct to hold the quantization grid data, using only Eigen types.
template<typename R = traits::DataType::PolynomialField>
struct QuantizationGrid {
    unsigned int N;
    unsigned int d;

    // Storing all coordinates in one Eigen matrix
    traits::DataType::StoringMatrix coordinates; // e.g., Eigen::MatrixXd

    // Using Eigen vectors for weights and distortions
    traits::DataType::StoringVector weights;
    traits::DataType::StoringVector local_l2_distortions;
    traits::DataType::StoringVector local_l1_distortions;

    // Global distortions
    R quadratic_distortion;
    R l1_distortion;
};

// Function to read a quantization grid from a file into the Eigen-only structure
template<typename R = traits::DataType::PolynomialField>
QuantizationGrid<R> readQuantizationGrid(int N, int d, const std::filesystem::path& base_dir) {
    // Construct the filename
    std::string filename = std::to_string(N) + "_" + std::to_string(d) + "_nopti";
    std::filesystem::path file_path = base_dir / filename;

    // Open the file
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_path.string());
    }

    QuantizationGrid<R> grid;
    grid.N = N;
    grid.d = d;

    // Pre-allocate memory for the Eigen matrix and vectors
    grid.coordinates.resize(N, d);
    grid.weights.resize(N);
    grid.local_l2_distortions.resize(N);
    grid.local_l1_distortions.resize(N);

    std::string line;
    int line_num = 0;

    // Read the first N rows and populate the matrix/vectors directly
    while (line_num < N && std::getline(file, line)) {
        std::stringstream ss(line);

        // Read weight into the Eigen vector
        if (!(ss >> grid.weights(line_num))) {
            throw std::runtime_error("Invalid data format for weight at line " + std::to_string(line_num + 1));
        }

        // Read coordinates into the corresponding row of the matrix
        for (int i = 0; i < d; ++i) {
            if (!(ss >> grid.coordinates(line_num, i))) {
                throw std::runtime_error("Invalid data format for coordinates at line " + std::to_string(line_num + 1));
            }
        }

        // Read distortions into Eigen vectors
        if (!(ss >> grid.local_l2_distortions(line_num) >> grid.local_l1_distortions(line_num))) {
            throw std::runtime_error("Invalid data format for distortions at line " + std::to_string(line_num + 1));
        }

        line_num++;
    }

    // Read the last row (N+1) for global distortions
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        double val;
        // Skip the first d+1 zeros
        for(int i = 0; i < d + 1; ++i) {
            if (!(ss >> val)) {
                 throw std::runtime_error("Invalid data format in the last row");
            }
        }
        if (!(ss >> grid.quadratic_distortion >> grid.l1_distortion)) {
            throw std::runtime_error("Invalid data format in the last row for distortions");
        }
    } else {
        throw std::runtime_error("File does not have enough rows for N = " + std::to_string(N));
    }

    return grid;
}

