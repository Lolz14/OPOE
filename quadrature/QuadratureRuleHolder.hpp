
/**
 * @file QuadratureRuleHolder.hpp
 * @brief Defines the QuadratureRuleHolder class, a type-erased holder for various quadrature rule implementations.
 *
 * This header provides the QuadratureRuleHolder template class, which acts as a runtime-polymorphic wrapper
 * for different quadrature rule implementations (e.g., Boost Tanh-Sinh, GSL QAGI). The holder enables
 * selection and usage of a quadrature rule via a common interface, supporting both default and parameterized
 * construction. Deep copy and move semantics are implemented, and integration is performed via a uniform method.
 *
 */
#ifndef QUADRATURE_RULE_HOLDER_HPP
#define QUADRATURE_RULE_HOLDER_HPP


#include <memory>
#include <stdexcept>
#include <utility> // For std::move
#include "QuadratureWrappers.hpp" // Include concrete wrappers
#include "../traits/OPOE_traits.hpp" // Include abstract interface

namespace quadrature {
using QuadratureType = traits::QuadratureMethod; 

/**
 * @brief A holder class for different numerical quadrature rules.
 * 
 * This class encapsulates various quadrature rules through runtime polymorphism.
 * It enables selection and usage of a rule at runtime, while maintaining a consistent interface.
 * 
 * @tparam R The floating-point type used for integration (e.g., float, double).
 */
template<typename R>
class QuadratureRuleHolder {
private:
    std::unique_ptr<IQuadratureRule<R>> p_rule_;

public:
/**
     * @brief Constructor selecting rule based on enum. Uses default parameters for rules.
     * 
     * @param type The enum value specifying which quadrature rule to use.
     * @throws std::invalid_argument If an unsupported quadrature type is specified.
     */
    explicit QuadratureRuleHolder(QuadratureType type = QuadratureType::TanhSinh) {
        switch (type) {
            case QuadratureType::TanhSinh:
                // Create the Boost wrapper using its default parameters
                p_rule_ = std::make_unique<BoostQuadratureWrapper<R>>();
                break;
            case QuadratureType::QAGI:
                 // Create the GSL wrapper using its default parameters
                p_rule_ = std::make_unique<GSLQuadratureWrapper<R>>();
                break;            
            default:
                throw std::invalid_argument("Unsupported QuadratureType specified.");
        }
    }

    /**
     * @brief Constructor for Boost quadrature with a custom tolerance.
     * 
     * @param type Must be QuadratureType::TanhSinh.
     * @param boost_tolerance Desired error tolerance for Boost quadrature.
     * @throws std::invalid_argument If the type is not TanhSinh.
     */
    QuadratureRuleHolder(QuadratureType type, R boost_tolerance) {
         if (type != QuadratureType::TanhSinh)
            throw std::invalid_argument("Tolerance parameter constructor only valid for BOOST_TANH_SINH");
         p_rule_ = std::make_unique<BoostQuadratureWrapper<R>>(boost_tolerance);
    }
    /**
     * @brief Constructor for GSL quadrature with custom absolute/relative tolerances and workspace size.
     * 
     * @param type Must be QuadratureType::QAGI.
     * @param gsl_abs_tol Absolute error tolerance for GSL quadrature.
     * @param gsl_rel_tol Relative error tolerance for GSL quadrature.
     * @param gsl_ws_size Workspace size for the GSL integrator.
     * @throws std::invalid_argument If the type is not QAGI.
     */
    QuadratureRuleHolder(QuadratureType type, double gsl_abs_tol, double gsl_rel_tol, size_t gsl_ws_size) {
         if (type != QuadratureType::QAGI)
             throw std::invalid_argument("GSL parameter constructor only valid for GSL_ADAPTIVE");
         p_rule_ = std::make_unique<GSLQuadratureWrapper<R>>(gsl_abs_tol, gsl_rel_tol, gsl_ws_size);
    }


    /**
     * @brief Copy constructor. Performs a deep copy using the clone interface.
     * 
     * @param other The other QuadratureRuleHolder to copy from.
     */   

    QuadratureRuleHolder(const QuadratureRuleHolder& other)
        : p_rule_(other.p_rule_ ? other.p_rule_->clone() : nullptr) {}

    /**
     * @brief Copy assignment operator. Performs a deep copy using the clone interface.
     * 
     * @param other The other QuadratureRuleHolder to assign from.
     * @return Reference to this object.
     */    
    QuadratureRuleHolder& operator=(const QuadratureRuleHolder& other) {
        if (this != &other) {
             // Clone before releasing the old pointer
             p_rule_ = other.p_rule_ ? other.p_rule_->clone() : nullptr;
        }
        return *this;
    }

    /**
     * @brief Move constructor. Transfers ownership of the held rule.
     * 
     * @param other The other QuadratureRuleHolder to move from.
     */
    QuadratureRuleHolder(QuadratureRuleHolder&& other) noexcept = default;

    /**
     * @brief Move assignment operator. Transfers ownership of the held rule.
     * 
     * @param other The other QuadratureRuleHolder to move-assign from.
     * @return Reference to this object.
     */    
    QuadratureRuleHolder& operator=(QuadratureRuleHolder&& other) noexcept = default;
   
    /**
     * @brief Default constructor. Leaves the internal rule uninitialized.
     */
    QuadratureRuleHolder() = default;


    /**
     * @brief Integrates the given function over the specified interval using the held quadrature rule.
     * 
     * @param integrand The function to integrate. Should be callable as f(x) -> R.
     * @param lower_bound Lower limit of integration.
     * @param upper_bound Upper limit of integration.
     * @return The computed integral value.
     * @throws std::runtime_error If no quadrature rule has been initialized.
     */
    R integrate(
        const std::function<R(R)>& integrand,
        R lower_bound,
        R upper_bound) const
    {
        if (!p_rule_) {
            throw std::runtime_error("QuadratureRuleHolder is not initialized with a rule.");
        }
        return p_rule_->integrate(integrand, lower_bound, upper_bound);
    }

    /**
     * @brief Checks whether a quadrature rule is currently initialized.
     * 
     * @return True if a rule is initialized, false otherwise.
     */
    bool is_initialized() const {
        return p_rule_ != nullptr;
    }
};

} // namespace quadrature
#endif // QUADRATURE_RULE_HOLDER_HPP
