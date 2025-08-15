/*!
 * @file Utils.hpp
 * @brief Utility functions and templates for polynomial operations, random sampling, and vector/matrix computations.
 *
 * This header provides a collection of utility templates and functions, including:
 * - Compile-time exponentiation of polynomials via the `Pow` struct template, with specializations for exponents 0 and 1.
 * - Computation of the Pochhammer symbol (falling and rising factorial) for generic numeric types.
 * - A flexible `sampler` function for generating Eigen vectors or matrices filled with random samples from a specified distribution.
 * - An `argmin` function to find the column of a matrix closest (in L2 norm) to a target vector, returning the index, distance, and all distances.
 * - A `calculate_overall_distortion` function to compute the average distortion (root mean squared error) between randomly generated samples and their closest centroids.
 *
 * The utilities are designed to work with Eigen types and generic numeric types, and leverage modern C++ features such as variadic templates, `if constexpr`, and concepts.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cmath>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <utility>       
#include <type_traits>
#include "../traits/OPOE_traits.hpp" 
#include <unsupported/Eigen/KroneckerProduct>

namespace Utils
{

/**
 * @section Polynomial Utilities
 * @brief Provides utilities for polynomial operations, including compile-time exponentiation.
 */

using IndexMap = std::unordered_map<long, int>;

// Forward declaration of Polynomial class
template <unsigned int N, typename R = traits::DataType::PolynomialField> 
class Polynomial;

/**
 * @brief Type alias for a map that associates a long key (m, n) to an integer index.
 *
 * This map is used to efficiently store and retrieve indices based on polynomial degree and exponent.
 */

/**
 * @brief Template structure to compute powers of a Polynomial at compile time.
 *
 * @tparam RDegree Degree of the polynomial.
 * @tparam R Type of the polynomial's coefficients.
 * @tparam Exp Exponent to raise the polynomial to.
 */
template <unsigned int RDegree, typename R = traits::DataType::PolynomialField, unsigned int Exp = 2u> 
struct Pow
{
    /**
     * @brief Computes the polynomial raised to the power of Exp.
     * @param p The input polynomial.
     * @return Result of p^Exp.
     */
    auto inline operator()(Polynomial<RDegree, R> const &p) const
    {
        Pow<RDegree, R, Exp - 1u> next;
        return next(p) * p;
    }
};

/**
 * @brief Specialization of Pow for exponent equal to 1.
 */
template <unsigned int RDegree, typename R> 
struct Pow<RDegree, R, 1u>
{
    /**
     * @brief Returns the polynomial unchanged (p^1 = p).
     * @param p The input polynomial.
     * @return The same polynomial.
     */
    Polynomial<RDegree, R> inline operator()(Polynomial<RDegree, R> const &p) const
        {
            return p;
        }
};

/**
 * @brief Specialization of Pow for exponent equal to 0.
 */
template <unsigned int RDegree, typename R> 
struct Pow<RDegree, R, 0u>
{
     /**
     * @brief Returns the constant polynomial 1 (p^0 = 1).
     * @param p The input polynomial.
     * @return Constant polynomial with value 1.
     */
    Polynomial<0u, R>
    inline operator()(Polynomial<RDegree, R> const &p)
    {
        return Polynomial<0u, R>{{R(1)}};
    }
};

/**
 * @brief Computes the Pochhammer symbol (falling or rising factorial).
 *
 * @tparam R Numeric type.
 * @tparam Type Enum value specifying falling or rising factorial.
 * @param m Base value.
 * @param n Number of terms in the product.
 * @return The result of the Pochhammer symbol computation.
 */
template <typename R = traits::DataType::PolynomialField, traits::PochammerType Type>
requires (Type == traits::PochammerType::Falling || Type == traits::PochammerType::Rising)
inline R Pochhammer(R m, R n) {
    if constexpr (Type == traits::PochammerType::Falling) {
        R result = 1;
        for (R i = 0; i < n; ++i) {
            result *= (m - i);
        }
        return result;
    } else if constexpr (Type == traits::PochammerType::Rising) {
        R result = 1;
        for (R i = 0; i < n; ++i) {
            result *= (m + i);
        }
        return result;
    }
}


/**
 * @section Random Generation Utilities
 * @brief Provides utilities for generating random samples using Eigen types.
 *
 * This section includes a generic `sampler` function that can generate Eigen vectors or matrices filled with random samples from a specified distribution.
 * The function supports both single-dimensional (vector) and two-dimensional (matrix) outputs, leveraging variadic templates for flexibility.
 * It also includes an `argmin` function to find the column of a matrix closest (in L2 norm) to a target vector, and a function to compute average distortion between random samples and centroids.
 */

/**
 * @brief Generates an Eigen vector or matrix filled with random samples.
 *
 * @tparam T Scalar type (e.g., float, double).
 * @tparam DistributionType Type of random number distribution.
 * @tparam DimArgs Variadic template for dimensions (1 for vector, 2 for matrix).
 * @param generator Mersenne Twister random number generator.
 * @param distribution Distribution object used to generate samples.
 * @param dims_args Dimensions of the output (size or rows, cols).
 * @return An Eigen vector or matrix filled with samples.
 */

template<typename T = traits::DataType::PolynomialField, typename DistributionType, typename... DimArgs>
auto sampler(
    std::mt19937& generator,          // Mersenne Twister engine (passed by reference)
    DistributionType& distribution,   // Generic distribution (passed by reference)
    DimArgs... dims_args              // Variadic template for dimension arguments
) {
    // Determine the number of dimension arguments at compile time
    constexpr size_t number_of_dimensions = sizeof...(dims_args);

    if constexpr (number_of_dimensions == 1) {
        // Case 1: Single dimension argument, implies a Vector
        auto dimensions_tuple = std::make_tuple(dims_args...);
        // Extract the vector size from the tuple
        typename std::tuple_element<0, std::tuple<std::decay_t<DimArgs>...>>::type vector_size = std::get<0>(dimensions_tuple);

        // Use Eigen's NullaryExpr to fill the vector with random samples
        return traits::DataType::StoringVector::NullaryExpr(vector_size, [&]() -> T {
            return static_cast<T>(distribution(generator)); // Generate one random number per element
        });

    } else if constexpr (number_of_dimensions == 2) {
        // Case 2: Two dimension arguments, implies a Matrix
        auto dimensions_tuple = std::make_tuple(dims_args...);
        // Extract matrix rows and columns from the tuple
        typename std::tuple_element<0, std::tuple<std::decay_t<DimArgs>...>>::type matrix_rows = std::get<0>(dimensions_tuple);
        typename std::tuple_element<1, std::tuple<std::decay_t<DimArgs>...>>::type matrix_cols = std::get<1>(dimensions_tuple);

        // Use Eigen's NullaryExpr to fill the matrix with random samples
        return traits::DataType::StoringMatrix::NullaryExpr(matrix_rows, matrix_cols, [&]() -> T {
            return static_cast<T>(distribution(generator)); // Generate one random number per element
        });

    } else {
        // Compile-time error for an unsupported number of dimension arguments
        static_assert(number_of_dimensions == 1 || number_of_dimensions == 2,
            "sample_any_distribution_eigen expects 1 argument for a Vector (size) "
            "or 2 arguments for a Matrix (rows, cols).");
        // This part should never be reached due to static_assert
    }
}


/**
 * @brief Finds the column of a matrix closest (in L2 norm) to a target vector.
 *
 * @tparam T Numeric type for distance.
 * @param matrix Matrix where each column represents a point.
 * @param target_vector Target vector to compare against.
 * @return Tuple of (index of closest column, minimum L2 distance, vector of all distances).
 */

template<typename T = traits::DataType::PolynomialField>
std::tuple<int, T, traits::DataType::StoringVector> argmin(
    const traits::DataType::StoringMatrix& matrix,
    const traits::DataType::StoringVector& target_vector) {

    // --- Input Validations ---
    if (matrix.cols() == 0) {
        std::cerr << "Error: Matrix has no columns." << std::endl;
        return {-1, std::numeric_limits<T>::infinity(), traits::DataType::StoringVector()};
    }
    if (matrix.rows() != target_vector.size()) {
        std::cerr << "Error: Number of rows in matrix (" << matrix.rows()
                  << ") does not match size of target vector (" << target_vector.size()
                  << ")." << std::endl;
        return {-1, std::numeric_limits<T>::infinity(), traits::DataType::StoringVector()};
    }

    // Compute the L2 distances between each column and the target vector
    traits::DataType::StoringVector l2_distances = (matrix.colwise() - target_vector).colwise().norm();

    // Find the minimum L2 distance and its index
    Eigen::Index min_col_index;
    T min_distance_value = l2_distances.minCoeff(&min_col_index);

    return {static_cast<int>(min_col_index), min_distance_value, l2_distances};
}



/**
 * @brief Computes average distortion (root mean squared error) between random samples and nearest centroids.
 *
 * @tparam T Numeric type.
 * @tparam Dim Dimensionality of the samples and centroids.
 * @param current_centroids Matrix of centroids (each column is a centroid).
 * @param num_test_samples Number of random test samples to generate.
 * @param generator Random number generator.
 * @param distribution Distribution used to generate test samples.
 * @return Root mean squared error between samples and nearest centroid.
 */

template<typename T = traits::DataType::PolynomialField, int Dim>
T calculate_overall_distortion(
    const traits::DataType::StoringMatrix& current_centroids,
    int num_test_samples, // Number of random samples to generate
    std::mt19937& generator,
    std::normal_distribution<T>& distribution) {

    // Validate input
    if (current_centroids.cols() == 0 || num_test_samples == 0) {
        if (current_centroids.cols() == 0) std::cerr << "Distortion calc error: No centroids." << std::endl;
        return std::numeric_limits<T>::infinity();
    }
    if (current_centroids.rows() != Dim) {
        std::cerr << "Distortion calc error: Centroid dimension mismatch." << std::endl;
        return std::numeric_limits<T>::infinity();
    }

    T total_squared_error = static_cast<T>(0.0);
    int valid_samples_for_distortion = 0;

    // Generate random samples and accumulate squared error to closest centroid
    for (int i = 0; i < num_test_samples; ++i) {
        // Generate a single, fresh test sample
        traits::DataType::StoringVector test_sample =
            sampler<T>(generator, distribution, Dim);

        // Find the closest centroid and its L2 distance
        auto closest_info_for_test_sample = argmin<T>(current_centroids, test_sample);

        if (std::get<0>(closest_info_for_test_sample) != -1) { // Valid centroid found
            T min_l2_distance = std::get<1>(closest_info_for_test_sample);
            total_squared_error += (min_l2_distance * min_l2_distance); // Accumulate squared L2 distance
            valid_samples_for_distortion++;
        }
    }

    // Compute RMSE if there are valid samples
    if (valid_samples_for_distortion > 0) {
        return std::sqrt(total_squared_error / static_cast<T>(valid_samples_for_distortion));
    }

    return std::numeric_limits<T>::infinity(); // Return infinity if no valid samples
}


/**
 * @section Matrix Utilities
 * @brief Provides utilities for matrix operations, including enumeration of polynomial basis indices and Kronecker products.
 */
using StoringMatrix = traits::DataType::StoringMatrix;
using StoringVector = traits::DataType::StoringVector;
/**
 * @brief Enumerates all pairs (m, n) such that m + n <= N.
 * 
 * This function generates a vector of pairs representing the indices of the basis functions
 * for a polynomial basis of degree N. Each pair corresponds to the indices of the monomials
 * in the polynomial expansion.
 *
 * @param N The maximum degree of the polynomial basis.
 * @return A vector of pairs (m, n) where m + n <= N.
 */
static std::vector<std::pair<int,int>> enumerate_basis(int N) {
    std::vector<std::pair<int,int>> E;
    E.reserve((N+1)*(N+2)/2);
    for (int m = 0; m <= N; ++m) {
        for (int n = 0; n <= N - m; ++n) {
            E.emplace_back(m, n);
        }
    }
    return E;
}



// Derivative matrices on monomial basis: {1, x, x^2, ...}
template<typename T = traits::DataType::PolynomialField>
[[nodiscard]] inline StoringMatrix build_Dmono(int N) {
    StoringMatrix D = StoringMatrix::Zero(N, N);
    if (N > 1) {
        D.diagonal(1) = StoringVector::LinSpaced(N - 1, T(1), T(N - 1));
    }
    return D;
}

template<typename T = traits::DataType::PolynomialField>
[[nodiscard]] inline StoringMatrix build_D2mono(int N) {
    StoringMatrix D2 = StoringMatrix::Zero(N, N);
    if (N > 2) {
        StoringVector j = StoringVector::LinSpaced(N - 2, T(2), T(N - 1));
        D2.diagonal(2) = j.array() * (j.array() - T(1));
    }
    return D2;
}

// Polynomial coeffs -> multiplication matrix in v (dense)
template<typename T = traits::DataType::PolynomialField>
[[nodiscard]] inline StoringMatrix poly_to_M(const StoringVector& coeffs, int Nv) {
    StoringMatrix R = StoringMatrix::Zero(Nv, Nv);
    const int L = std::min<int>(coeffs.size(), Nv);
    for (int r = 0; r < L; ++r) {
        const T c = coeffs[r];
        if (c != T(0)) {
            R.diagonal(-r).array() += c;
        }
    }
    return R;
}

// Build full rectangular G (dense)
template<typename T = traits::DataType::PolynomialField>
[[nodiscard]] inline StoringMatrix build_G_full(
    const StoringMatrix& H,
    const StoringVector& bx,
    const StoringVector& axx,
    const StoringVector& bv,
    const StoringVector& axv,
    const StoringVector& avv,
    int Nv
) {
    const int Nx = H.cols();

    // Derivatives in H basis
    const StoringMatrix Dmono  = build_Dmono<T>(Nx);
    const StoringMatrix D2mono = build_D2mono<T>(Nx);

    auto dec = H.colPivHouseholderQr();
    const StoringMatrix D_H  = dec.solve(Dmono  * H);
    const StoringMatrix D_H2 = dec.solve(D2mono * H);

    // v-side operators
    const StoringMatrix Bx   = poly_to_M<T>(bx,  Nv);
    const StoringMatrix Bxx  = poly_to_M<T>(axx, Nv);
    const StoringMatrix Bv   = poly_to_M<T>(bv,  Nv);
    const StoringMatrix Bxv  = poly_to_M<T>(axv, Nv);
    const StoringMatrix Bvv  = poly_to_M<T>(avv, Nv);

    const StoringMatrix Dvmono  = build_Dmono<T>(Nv);
    const StoringMatrix D2vmono = build_D2mono<T>(Nv);

    StoringMatrix G_full = StoringMatrix::Zero(Nv * Nx, Nv * Nx);

    // Operators
    G_full.noalias() += Eigen::kroneckerProduct(Bx, D_H).eval();
    G_full.noalias() += 0.5 * Eigen::kroneckerProduct(Bxx, D_H2).eval();
    G_full.noalias() += Eigen::kroneckerProduct((Bv * Dvmono).eval(), StoringMatrix::Identity(Nx, Nx)).eval();
    G_full.noalias() += Eigen::kroneckerProduct((Bxv * Dvmono).eval(), D_H).eval();
    G_full.noalias() += 0.5 * Eigen::kroneckerProduct((Bvv * D2vmono).eval(), StoringMatrix::Identity(Nx, Nx)).eval();

    return G_full;
}

// Project G_full onto triangular basis
[[nodiscard]] inline StoringMatrix project_to_triangular(
    const StoringMatrix& G_full,
    const std::vector<std::pair<int,int>>& E_tri,
    int Nx
) {
    const int M = static_cast<int>(E_tri.size());
    StoringMatrix G_tri = StoringMatrix::Zero(M, M);

    // Flatten indices (v-major: idx = v*Nx + x)
    Eigen::VectorXi flat(M);
    for (int k = 0; k < M; ++k) {
        flat[k] = E_tri[k].first * Nx + E_tri[k].second;
    }

    // Gather into G_tri
    for (int c = 0; c < M; ++c) {
        const int ic = flat[c];
        for (int r = 0; r < M; ++r) {
            G_tri(r, c) = G_full(flat[r], ic);
        }
    }
    return G_tri;
}



} // namespace Utils

#endif // UTILS_HPP











