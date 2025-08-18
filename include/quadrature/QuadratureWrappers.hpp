
/**
 * @file QuadratureWrappers.hpp
 * @brief Provides wrapper classes for different quadrature (numerical integration) backends.
 *
 * This header defines wrapper classes for Boost and GSL quadrature rules, allowing them to be used
 * through a common interface. The wrappers manage the underlying rule objects
 * and provide methods for integration and cloning.
 * 
 * Dependencies:
 * - QuadratureRule.hpp: Base class for quadrature rules.
 * - QuadratureRuleAbstract.hpp: Abstract interface for quadrature rules.
 *
 */
#ifndef QUADRATURE_WRAPPERS_HPP
#define QUADRATURE_WRAPPERS_HPP

#include <memory>
#include "QuadratureRule.hpp"
#include "QuadratureRuleAbstract.hpp"

namespace quadrature {
/**
 * @class BoostQuadratureWrapper
 * @tparam R The floating-point type used for integration (e.g., double).
 * @brief Wrapper for the Boost Tanh-Sinh quadrature rule.
 *
 * This class wraps a `BoostTanhSinhQuadrature` instance and exposes it through the `IQuadratureRule` interface.
 * It supports construction from an existing rule or by specifying the desired relative error.
 *
 * - `integrate`: Performs numerical integration using the wrapped Boost rule.
 * - `clone`: Creates a deep copy of the wrapper and its underlying rule.
 *
 * 
 */
template<typename R = traits::DataType::PolynomialField>
class BoostQuadratureWrapper final : public IQuadratureRule<R> {
    BoostTanhSinhQuadrature<R> rule_;
public:
    // Constructor taking the specific rule instance
    explicit BoostQuadratureWrapper(const BoostTanhSinhQuadrature<R>& rule)
        : rule_(rule) {}

    // Constructor taking parameters to create the rule internally
    explicit BoostQuadratureWrapper(R relative_error = std::sqrt(std::numeric_limits<R>::epsilon()))
        : rule_(relative_error) {}

    R integrate(
        const std::function<R(R)>& integrand,
        R lower_bound,
        R upper_bound) const override {
        return rule_.integrate(integrand, lower_bound, upper_bound);
    }

    std::unique_ptr<IQuadratureRule<R>> clone() const override {
        // Create a new wrapper containing a copy of the rule
        return std::make_unique<BoostQuadratureWrapper<R>>(rule_);
    }
};

/**
 * @class GSLQuadratureWrapper
 * @tparam R The floating-point type used for integration (e.g., double).
 * @brief Wrapper for the GSL adaptive quadrature rule.
 *
 * This class wraps a `GSLQuadrature` instance and exposes it through the `IQuadratureRule` interface.
 * It supports construction from an existing rule or by specifying the desired absolute and relative errors.
 *
 * - `integrate`: Performs numerical integration using the wrapped GSL rule.
 * - `clone`: Creates a deep copy of the wrapper and its underlying rule.
 */
template<typename R = traits::DataType::PolynomialField>
class GSLQuadratureWrapper final : public IQuadratureRule<R> {
     GSLQuadrature<R> rule_;
 public:
    // Constructor taking the specific rule instance
     explicit GSLQuadratureWrapper(const GSLQuadrature<R>& rule)
         : rule_(rule) {} // Note: GSLQuadrature might not be easily copyable if workspace is member

    // Constructor taking parameters to create the rule internally
    explicit GSLQuadratureWrapper(
        R absolute_error = 1e-9,
        R relative_error = 1e-9,
        size_t workspace_limit = 1000)
        : rule_(absolute_error, relative_error, workspace_limit) {}


     R integrate(
         const std::function<R(R)>& integrand,
         R lower_bound,
         R upper_bound) const override {
         return rule_.integrate(integrand, lower_bound, upper_bound);
     }

     std::unique_ptr<IQuadratureRule<R>> clone() const override {
          return std::make_unique<GSLQuadratureWrapper<R>>(
              rule_); 

     }
};

} // namespace quadrature
#endif // QUADRATURE_WRAPPERS_HPP
