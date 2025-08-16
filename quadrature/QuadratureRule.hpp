
/**
 * @file QuadratureRule.hpp
 * @brief Provides generic quadrature (numerical integration) adapters for Boost and GSL libraries.
 *
 * This header defines two main quadrature classes:
 * - quadrature::BoostTanhSinhQuadrature: Adapter for Boost.Math's tanh_sinh quadrature, supporting finite and infinite bounds.
 * - quadrature::GSLQuadrature: Adapter for GSL's adaptive quadrature routines, supporting finite and infinite bounds with workspace management.
 *
 * Features:
 * - Type-generic (templated on floating-point type R).
 * - Handles both finite and infinite integration limits.
 * - Error control via absolute and relative tolerances.
 * - Exception-safe resource management for GSL workspaces.
 * - C++11/14 compatible, uses std::function for integrands.
 *
 *
 * Usage Example:
 * @code
 * #include "QuadratureRule.hpp"
 * 
 * quadrature::BoostTanhSinhQuadrature<double> boost_quad;
 * double result = boost_quad.integrate([](double x) { return std::exp(-x*x); }, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
 * 
 * quadrature::GSLQuadrature<double> gsl_quad;
 * double gsl_result = gsl_quad.integrate([](double x) { return std::sin(x); }, 0.0, M_PI);
 * @endcode
 *
 */
#ifndef QUADRATURE_RULE_HPP
#define QUADRATURE_RULE_HPP

#include <stdexcept>
#include <limits>
#include <cmath> 
#include <string>
#include <vector> 
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/special_functions/fpclassify.hpp> 
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h> 
#include "traits/OPOE_traits.hpp" // Assuming this contains the DataType definition

namespace quadrature {
template<typename R = traits::DataType::PolynomialField>
class BoostTanhSinhQuadrature {
    R target_relative_error_;
    // tanh_sinh internally handles max levels, etc.

public:
    /**
     * @brief Constructor for Boost tanh_sinh adapter.
     * @param relative_error Target relative error for the integration.
     */
    explicit BoostTanhSinhQuadrature(R relative_error = std::sqrt(std::numeric_limits<R>::epsilon()))
        : target_relative_error_(relative_error) {}

    /**
     * @brief Integrates using Boost.Math's tanh_sinh quadrature.
     * @param integrand The function to integrate.
     * @param lower_bound Lower integration limit (-inf allowed).
     * @param upper_bound Upper integration limit (+inf allowed).
     * @return The approximate value of the definite integral.
     */
    R integrate(
        const std::function<R(R)>& integrand,
        R lower_bound,
        R upper_bound) const
    {
        if (!(boost::math::isfinite)(lower_bound) && !(boost::math::isfinite)(upper_bound)) {
             if (lower_bound > 0 || upper_bound < 0) { // e.g., integrate from inf to inf
                 return static_cast<R>(0.0);
             }
             // Default: -inf to +inf handled directly
        } else if (lower_bound >= upper_bound) {
             return static_cast<R>(0.0);
        }

        boost::math::quadrature::tanh_sinh<R> integrator(15); // Max level refinement (adjust if needed)

        R result = 0;
        R error_estimate = 0;
        R L1_norm = 0; // Estimate of L1 norm for error scaling

        try {
             // Boost's integrate handles finite and infinite bounds automatically
             result = integrator.integrate(integrand, lower_bound, upper_bound, target_relative_error_, &error_estimate, &L1_norm);


        } catch (const std::exception& e) {
             // Handle Boost exceptions (e.g., max levels reached, convergence issues)
             throw std::runtime_error(std::string("Boost quadrature failed: ") + e.what());
        }

        return result;
    }
};


// --- C-style wrapper for GSL ---
template<typename R = traits::DataType::PolynomialField>
struct GSLIntegrationWrapper {
    // Must be static or a non-capturing lambda convertible to function pointer
    static R gsl_func_adapter(R x, void* params) {
        auto* func_ptr = static_cast<std::function<R(R)>*>(params);
        try {
            // GSL works with double, cast result if R is different (e.g., float)
            return static_cast<R>((*func_ptr)(static_cast<R>(x)));
        } catch (...) {
             // GSL doesn't handle C++ exceptions well across the C boundary.
             gsl_error("Caught C++ exception in GSL function adapter", __FILE__, __LINE__, GSL_FAILURE);
             return std::numeric_limits<R>::quiet_NaN();
        }
    }
};


template<typename R = traits::DataType::PolynomialField>
class GSLQuadrature {
private:
    size_t workspace_size_;
    R target_absolute_error_;
    R target_relative_error_;

    // RAII wrapper for gsl_integration_workspace
    // Ensures workspace is freed even if exceptions occur
    using GSLWorkspacePtr = std::unique_ptr<gsl_integration_workspace, decltype(&gsl_integration_workspace_free)>;

     // Function to create a workspace (needed if allocation is per-call)
    GSLWorkspacePtr create_workspace() const {
        gsl_integration_workspace* ws = gsl_integration_workspace_alloc(workspace_size_);
        if (!ws) {
            throw std::runtime_error("Failed to allocate GSL workspace");
        }
        return GSLWorkspacePtr(ws, gsl_integration_workspace_free);
    }


public:
    /**
     * @brief Constructor for GSL adaptive quadrature adapter.
     * @param absolute_error Target absolute error.
     * @param relative_error Target relative error.
     * @param workspace_limit Max number of subintervals for the workspace.
     */
    explicit GSLQuadrature(
        R absolute_error = 1e-6,
        R relative_error = 1e-6,
        size_t workspace_limit = 1000)
        :
        workspace_size_(workspace_limit),
        target_absolute_error_(absolute_error),
        target_relative_error_(relative_error)
    {
        
    }

    /**
     * @brief Integrates using GSL's adaptive quadrature. Selects routine based on bounds.
     * @param integrand The function to integrate.
     * @param lower_bound Lower integration limit.
     * @param upper_bound Upper integration limit.
     * @return The approximate value of the definite integral.
     */
    R integrate(
        const std::function<R(R)>& integrand,
        R lower_bound,
        R upper_bound) const
    {
        // GSL functions predominantly work with double
        R lb = static_cast<R>(lower_bound);
        R ub = static_cast<R>(upper_bound);

        if (lb >= ub) return static_cast<R>(0.0);

        // Need a non-const copy of the integrand to get a non-const pointer for GSL params
        auto non_const_integrand = integrand;
        gsl_function F;
        F.function = &GSLIntegrationWrapper<R>::gsl_func_adapter;
        F.params = &non_const_integrand;

        R result = 0.0;
        R error_estimate = 0.0;

        int status = GSL_SUCCESS;

        // Allocate workspace per call for thread safety
        GSLWorkspacePtr workspace = create_workspace();


        // Choose GSL routine based on bounds
        bool upper_inf = (ub > std::numeric_limits<R>::max() / 2);
        bool lower_inf = (lb < -std::numeric_limits<R>::max() / 2);


        if (lower_inf && upper_inf) {
            // QAGI: Integrate over (-inf, +inf)
             status = gsl_integration_qagi(&F,
                                           target_absolute_error_,
                                           target_relative_error_,
                                           workspace_size_,
                                           workspace.get(),
                                           &result, &error_estimate);
        } else if (lower_inf) {
             // QAGIL: Integrate over (-inf, b)
            status = gsl_integration_qagil(&F, ub,
                                           target_absolute_error_,
                                           target_relative_error_,
                                           workspace_size_,
                                           workspace.get(),
                                           &result, &error_estimate);
        } else if (upper_inf) {
             // QAGIU: Integrate over (a, +inf)
             status = gsl_integration_qagiu(&F, lb,
                                           target_absolute_error_,
                                           target_relative_error_,
                                           workspace_size_,
                                           workspace.get(),
                                           &result, &error_estimate);
        } else {
             // QAGS: Integrate over (a, b) - general adaptive strategy

             status = gsl_integration_qags(&F, lb, ub,
                                          target_absolute_error_,
                                          target_relative_error_,
                                          workspace_size_,
                                          workspace.get(),
                                          &result, &error_estimate);
        }

        // --- GSL Error Handling ---
        if (status != GSL_SUCCESS) {
             throw std::runtime_error(std::string("GSL integration failed: ") + gsl_strerror(status));
        }

        return static_cast<R>(result); // Cast back to original R
    }
};

} // namespace quadrature
#endif // QUADRATURE_RULE_HPP
