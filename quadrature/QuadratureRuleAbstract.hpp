/**
 * @file QuadratureRuleAbstract.hpp
 * @brief Defines the abstract interface for numerical quadrature rules.
 *
 * This file declares the IQuadratureRule interface, which provides a common
 * abstraction for implementing numerical integration rules.
 * The interface supports integration of single-variable functions over a
 * specified interval and enables polymorphic copying of rule objects.
 * 
 *
 */
#ifndef I_QUADRATURE_RULE_HPP
#define I_QUADRATURE_RULE_HPP

#include <functional>
#include <memory> 

namespace quadrature {
template<typename NumericType>
class IQuadratureRule {
public:
    virtual ~IQuadratureRule() = default;

    /**
     * @brief Integrates the given function over the specified interval.
     * @param integrand A callable function object taking NumericType, returning NumericType.
     * @param lower_bound The lower integration limit.
     * @param upper_bound The upper integration limit.
     * @return The approximate value of the definite integral.
     */
    virtual NumericType integrate(
        const std::function<NumericType(NumericType)>& integrand,
        NumericType lower_bound,
        NumericType upper_bound) const = 0;

    /**
     * @brief Creates a copy of the underlying rule object.
     * Needed for value semantics of the QuadratureRuleHolder.
     * @return A std::unique_ptr to the new IQuadratureRule object.
     */
    virtual std::unique_ptr<IQuadratureRule<NumericType>> clone() const = 0;
};
} // namespace quadrature
#endif // I_QUADRATURE_RULE_HPP
