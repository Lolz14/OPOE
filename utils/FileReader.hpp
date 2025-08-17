
/**
 * @file FileReader.hpp
 * @brief Provides functionality to read quantization grid data from a file into Eigen-based structures.
 *
 * This header defines the QuantizationGrid struct and the readQuantizationGrid function template,
 * which are used to load quantization grid data (including coordinates, weights, and distortions)
 * from a file into Eigen matrices and vectors for efficient numerical processing.
 *
 * Dependencies:
 * - OPOE_traits.hpp: For type definitions and concepts.
 */

#ifndef FILE_READER_HPP
#define FILE_READER_HPP
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include "../traits/OPOE_traits.hpp" 

/**
 * @struct QuantizationGrid
 * @brief Structure to hold quantization grid data using Eigen types.
 *
 * @tparam R The type used for global distortion values (default: traits::DataType::PolynomialField).
 *
 * Members:
 * - N: Number of grid points.
 * - d: Dimension of each grid point.
 * - coordinates: Eigen matrix storing all grid point coordinates (size N x d).
 * - weights: Eigen vector storing the weight for each grid point (size N).
 * - local_l2_distortions: Eigen vector of local L2 distortions for each grid point (size N).
 * - local_l1_distortions: Eigen vector of local L1 distortions for each grid point (size N).
 * - quadratic_distortion: Global quadratic (L2) distortion.
 * - l1_distortion: Global L1 distortion.
 */
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


/**
 * @brief Reads a quantization grid from a file into a QuantizationGrid structure using Eigen types.
 *
 * The function expects the file to have N rows of data, each containing:
 *   - weight (scalar)
 *   - d coordinates (space-separated)
 *   - local L2 distortion (scalar)
 *   - local L1 distortion (scalar)
 * followed by a final row containing global distortion values.
 *
 * @tparam R The type used for global distortion values (default: traits::DataType::PolynomialField).
 * @param N Number of grid points.
 * @param d Dimension of each grid point.
 * @param base_dir Directory containing the grid file.
 * @return QuantizationGrid<R> Structure populated with data from the file.
 * @throws std::runtime_error If the file cannot be opened or if the data format is invalid.
 */
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

#endif // FILE_READER_HPP

