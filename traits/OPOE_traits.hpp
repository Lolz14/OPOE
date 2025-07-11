/*!
 * @file OPOE_traits.hpp
 * @brief Defines core type traits, enumerations, and data structures for the OPOE library.
 *
 * This header provides essential type definitions and enumerations used throughout the OPOE library,
 * including matrix and vector types based on Eigen, polynomial field types, and various enums for
 * evaluation, integration, and quantization methods.
 */

#ifndef OPOE_TRAITS_HPP
#define OPOE_TRAITS_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tuple>
#include <vector>
#include <array>
#include <variant>
#include <type_traits>
#include <cmath>
#include <string>
#include <complex>

namespace traits
/*!
 * @namespace traits
 * @brief Contains all type traits, type aliases, and enumerations used across the OPOE library.
 */
{

// Forward Declaration of Polynomial
class Polynomial;

/*!
 * @struct DataType
 * @brief Central container of type aliases for commonly used matrix/vector structures in OPOE.
 *
 * These types are based on Eigen and are designed for dynamic sizing and numerical efficiency,
 * targeting polynomial manipulation and high-dimensional vector/matrix operations.
 */
struct DataType
{
public:
    using PolynomialField = double;  ///< Scalar field used for all polynomial operations (default: double).

    using StoringMatrix = Eigen::Matrix<PolynomialField, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; ///< Dynamic-size matrix type.
    
    using StoringVector = Eigen::Matrix<PolynomialField, Eigen::Dynamic, 1>; ///< Dynamic-size column vector type.

    using ComplexStoringVector = Eigen::Matrix<std::complex<PolynomialField>, Eigen::Dynamic, 1>; ///< Dynamic-size complex column vector type.
    
    using StoringArray  = Eigen::Array<PolynomialField, Eigen::Dynamic, 1>; ///< Dynamic-size array for element-wise operations.
    
    using ComplexArray  = Eigen::Array<std::complex<PolynomialField>, Eigen::Dynamic, 1>; ///< Array of complex numbers for advanced operations.

    using SparseStoringMatrix = Eigen::SparseMatrix<PolynomialField>; ///< Sparse matrix type for memory-efficient storage of large, sparse data.
    
    using Triplet = Eigen::Triplet<PolynomialField>; ///< Triplet structure for building sparse matrices.
    
    using Triplets = std::vector<Triplet>; ///< Collection of triplets used for sparse matrix assembly.
};

/*!
 * @brief Converts an integer to a type at compile time.
 * @tparam N The unsigned integer to wrap in a type.
 * 
 * This is a meta-programming utility useful for compile-time dispatch based on integer constants.
 */
template <unsigned int N> 
using IntToType = std::integral_constant<unsigned int, N>;

/*!
 * @enum EvalMethod
 * @brief Enumeration of available methods for evaluating polynomials.
 */
enum class EvalMethod
{
    Horner, ///< Use Horner's method: efficient nested multiplication.
    Direct  ///< Use direct evaluation (less efficient, straightforward).
};

/*!
 * @enum PochammerType
 * @brief Types of Pochhammer symbol (factorial-like product).
 */
enum class PochammerType
{
    Rising,  ///< Rising factorial: \( m (m+1) (m+2) \dots (m+n-1) \)
    Falling  ///< Falling factorial: \( m (m-1) (m-2) \dots (m-n+1) \)
};

/*!
 * @enum QuadratureMethod
 * @brief Advanced numerical integration techniques.
 */
enum class QuadratureMethod
{
    TanhSinh, ///< Tanh-Sinh quadrature for handling endpoint singularities.
    QAGI      ///< Infinite interval integration using QAGI (from QUADPACK).
};

/*!
 * @enum IntegrationMethod
 * @brief Strategies for computing integrals of polynomials.
 */
enum class IntegrationMethod
{
    Direct,      ///< Straightforward analytical or numerical integration.
    Simplified   ///< Approximated or optimized integration strategy.
};

/*!
 * @enum OptionType
 * @brief Financial option types.
 */
enum class OptionType
{
    Call, ///< Call option: right to buy.
    Put   ///< Put option: right to sell.
};

/*!
 * @enum QuantizationProcedure
 * @brief Enumerates different quantization strategies.
 */
enum class QuantizationProcedure
{
    Newton, ///< Quantization using Newton-based optimization.
    CLVQ    ///< Competitive Learning Vector Quantization.
};



/*!
 * @enum class PricingMethod
 * @brief Enumerates different methods for option pricing.
 */
enum class PricingMethod
{
    MonteCarlo, ///< Monte Carlo simulation for option pricing.
    PolynomialExpansion, ///< Ackerer's polynomial expansion method.
    FourierTransform ///< For option pricing using Fourier methods.
};

} // namespace traits

#endif // OPOE_TRAITS_HPP
