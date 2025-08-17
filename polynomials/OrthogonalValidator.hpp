/**
 * @file OrthogonalValidator.hpp
 * @brief Provides utilities for validating domains and parameters of orthogonal polynomials.
 *
 * This header defines concepts, structures, and classes to facilitate the validation of
 * parameters and domains for various families of orthogonal polynomials (e.g., Jacobi, Laguerre, Hermite, Gegenbauer).
 * 
 * 
 * It includes:
 *   - Function type alias for recurrence coefficients.
 *   - DomainInterval for interval validation (with floating-point support).
 *   - Exception types for domain violations.
 *   - Parameter base structure and specific parameter types for common polynomial families.
 *   - PolynomialDomainValidator class template for parameter validation and access.
 *
 */
#ifndef ORTHOGONAL_VALIDATOR_HPP
#define ORTHOGONAL_VALIDATOR_HPP
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <cmath>
#include <concepts>

namespace polynomials{

/**
 * @brief A callable function taking and returning a scalar value.
 * 
 * @tparam R Scalar type.
 */
template <typename R>
using Function = std::function<R(R)>;

/**
 * @brief Structure for defining recurrence coefficients of orthogonal polynomials.
 * 
 * The recurrence takes the form:
 *     P_{k+1}(x) = (x - alpha_k(x)) * P_k(x) - beta_k(x) * P_{k-1}(x)
 * 
 * @tparam R Scalar type.
 */
template <typename R>
struct RecurrenceCoefficients {
    Function<R> alpha_k;
    Function<R> beta_k;
    Function<R> beta_0;
};

/**
 * @brief Concept restricting T to arithmetic types.
 */
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

/**
 * @brief Represents a closed numeric interval [lower, upper].
 * 
 * @tparam T Type of the interval endpoints.
 */
template<typename T>
struct DomainInterval {
    T lower; ///< Lower bound of the interval
    T upper; ///< Upper bound of the interval

    /**
     * @brief Checks if a value lies within the interval.
     * 
     * @param x The value to check.
     * @return true if x is in [lower, upper], false otherwise.
     */
    constexpr bool contains(T x) const noexcept {
        return x >= lower && x <= upper;
    }
    
    /**
     * @brief Approximate check for floating-point values using epsilon.
     * 
     * @tparam U Type, must be floating-point.
     * @param x Value to test.
     * @param epsilon Tolerance (default: machine epsilon).
     * @return true if x lies approximately within [lower, upper].
     */    template<typename U = T>
    requires std::is_floating_point_v<U>
    bool contains_approx(U x, U epsilon = std::numeric_limits<U>::epsilon()) const noexcept {
        return x >= (lower - epsilon) && x <= (upper + epsilon);
    }
};

/**
 * @brief Base class for all domain-related exceptions.
 */
class DomainError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * @brief Exception for parameter validation errors.
 */
class ParameterDomainError : public DomainError {
    using DomainError::DomainError;
};

/**
 * @brief Exception for evaluation input errors.
 */
class EvaluationDomainError : public DomainError {
    using DomainError::DomainError;
};

/**
 * @brief A generic parameter with value range validation.
 * 
 * @tparam R Scalar type.
 */
template<typename R>
struct Parameter {
    R value;                ///< The parameter's current value.
    std::pair<R, R> range;  ///< The valid domain for the parameter.
    const char* name;       ///< Human-readable name for error messages.

    /**
     * @brief Constructs a named parameter with range.
     * 
     * @param val Initial value.
     * @param range Valid (exclusive-inclusive) interval.
     * @param name Name for display/debugging.
     */
    Parameter(R val, std::pair<R, R> range, const char* name)
        : value(val), range(range), name(name) {}

    /**
     * @brief Validates if the value lies in (range.first, range.second].
     * 
     * @return true if valid, false otherwise.
     */
    bool isValid() const {
        return value > range.first && value <= range.second;
    }

    /**
     * @brief Returns the domain constraint as a human-readable string.
     */
    std::string getDomain() const {
        return std::to_string(range.first) + " < " + name + " <= " + std::to_string(range.second);
    }
};

// -----------------------------------------------------------------------------
// Specific parameter type definitions for common orthogonal polynomials
// -----------------------------------------------------------------------------

/// @brief α parameter for Jacobi polynomials: α > -1
template<typename R>
struct JacobiAlpha : Parameter<R> {
    JacobiAlpha(R val) : Parameter<R>{val, {-1, std::numeric_limits<R>::infinity()}, "alpha"} {}
};

/// @brief β parameter for Jacobi polynomials: β > -1
template<typename R>
struct JacobiBeta : Parameter<R> {
    JacobiBeta(R val) : Parameter<R>{val, {-1, std::numeric_limits<R>::infinity()}, "beta"} {}
};

/// @brief α parameter for Laguerre polynomials: α > -1
template<typename R>
struct LaguerreAlpha : Parameter<R> {
    LaguerreAlpha(R val) : Parameter<R>{val, {-1, std::numeric_limits<R>::infinity()}, "alpha"} {}
};

/// @brief θ (scale) parameter for Generalized Laguerre: θ ≥ 0
template<typename R>
struct LaguerreTheta : Parameter<R> {
    LaguerreTheta(R val) : Parameter<R>{val, {0, std::numeric_limits<R>::infinity()}, "theta"} {}
};

/// @brief μ parameter for Hermite polynomials: unbounded
template<typename R>
struct HermiteMu : Parameter<R> {
    HermiteMu(R val) : Parameter<R>{val, {-std::numeric_limits<R>::infinity(), std::numeric_limits<R>::infinity()}, "mu"} {}
};

/// @brief σ (scale) parameter for Hermite: σ > 0
template<typename R>
struct HermiteSigma : Parameter<R> {
    HermiteSigma(R val) : Parameter<R>{val, {0.0, std::numeric_limits<R>::infinity()}, "sigma"} {}
};

/// @brief μ parameter for Gegenbauer polynomials: μ > -0.5
template<typename R>
struct GegenbauerLambda : Parameter<R> {
    GegenbauerLambda(R val) : Parameter<R>{val, {-0.5, std::numeric_limits<R>::infinity()}, "mu"} {}
};


// -----------------------------------------------------------------------------
// Validator for parameter tuples
// -----------------------------------------------------------------------------

/**
 * @brief Utility to validate multiple parameters for a polynomial family.
 * 
 * @tparam R Scalar type
 * @tparam Params List of parameter types (must be derived from Parameter<R>)
 */
template<typename R, typename... Params>
class PolynomialDomainValidator {
private:
    std::tuple<Params...> parameters_;

public:
    /**
     * @brief Constructs the validator and performs initial validation.
     * 
     * @param params Variadic list of parameter values.
     */
    explicit PolynomialDomainValidator(Params... params)
        : parameters_(std::make_tuple(params...)) {
        validateParameters();
    }

    /**
     * @brief Throws if any parameter is out of domain.
     */
    void validateParameters() const;

    /**
     * @brief Gets a reference to a specific parameter by type.
     * 
     * @tparam P Parameter type to extract.
     * @return const P& Reference to the parameter.
     */    template<typename P>
    const P& getParameter() const {
        return std::get<P>(parameters_);
    }

    /**
     * @brief Gets the scalar value of a parameter.
     * 
     * @tparam P Parameter type to extract.
     * @return Value of the parameter.
     */
    template<typename P>
    auto getValue() const {
        return std::get<P>(parameters_).value;
    }

    /**
     * @brief Prints parameters to standard output (for debugging).
     */
    void debugParameters() const;

private:
    /**
     * @brief Builds a string message listing invalid parameters.
     * 
     * @return std::string Message for exception.
     */
    std::string buildErrorMessage() const;
};



}

#endif // ORTHOGONAL_VALIDATOR_HPP