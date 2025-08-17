/**
 * @file Projector.hpp
 * @brief Defines the Projector class template for weighted integration of products of functions using quadrature.
 *
 * The Projector class template enables the computation of integrals of the form:
 *   Integral(lower, upper, weight_fn(x) * f1(x) * f2(x) * ... * fn(x) dx)
 * where the user supplies a quadrature rule, a weight function, and a variadic list of input functions.
 * 
 * Dependencies:
 * - QuadratureRuleHolder.hpp: For the quadrature rule implementation.
 *
 */

#ifndef PROJECTOR_HPP
#define PROJECTOR_HPP

#include <functional>                 // For std::function
#include <tuple>                      // For std::tuple to store Args...
#include <vector>                     // Potentially useful, though tuple is primary storage
#include <utility>                    // For std::forward, std::move, std::index_sequence
#include <stdexcept>                  // For exceptions
#include <type_traits>                // For std::decay_t
#include "QuadratureRuleHolder.hpp" 


namespace quadrature {
// Forward declaration if needed, or include necessary headers for R type traits
/**
 * @brief Computes the weighted integral of the product of multiple functions.
 *
 * Accepts a variable number of input functions via its constructor.
 * Calculates Integral( lower, upper, weight_fn(x) * f1(x) * f2(x) * ... * fn(x) dx )
 * using a provided quadrature rule.
 *
 * @tparam R The numeric type for calculations (e.g., double, float).
 * @tparam FuncTypes The types of the input functions (f1, f2, ... fn).
 *         All must be callable with signature R(R).
 */
template <typename R, typename... FuncTypes>
class Projector {
private:
    // Store the input functions using a tuple. std::decay_t ensures copies/values.
    std::tuple<std::decay_t<FuncTypes>...> input_functions_;

    // Store the weight function
    std::function<R(R)> weight_function_;

    // Store the integration engine (holds its own copy)
    quadrature::QuadratureRuleHolder<R> integrator_;

    // --- Helper for evaluating the product using fold expression (C++17) ---
    R evaluate_product(R x) const {
        // Use std::apply to pass the functions in the tuple as arguments to the lambda
        return std::apply(
            [x](const auto&... funcs) -> R {
                // Compile-time check: Ensure all provided types are callable as R(R)
                static_assert((std::is_invocable_r_v<R, decltype(funcs), R> && ...),
                              "All input functions must be callable with signature R(R)");

                // Fold expression to multiply the results of calling each function with x
                if constexpr (sizeof...(funcs) > 0) {
                    // (func1(x) * func2(x) * ... * funcN(x))
                    return (funcs(x) * ...);
                } else {
                    // If FuncTypes... is empty, the product is 1.0
                    return R(1.0);
                }
            },
            input_functions_ // Pass the tuple holding the functions
        );
    }

public:
    /**
     * @brief Constructor for the Projector class accepting variadic functions.
     * @param integrator An initialized QuadratureRuleHolder (passed by const ref, will be copied).
     * @param weight_fn The weighting function std::function<R(R)>.
     * @param funcs Variadic arguments representing the functions whose product (with weight_fn) will be integrated. Must be callable as R(R).
     */
    Projector(
        const quadrature::QuadratureRuleHolder<R>& integrator, // Pass integrator by const reference
        std::function<R(R)> weight_fn,
        FuncTypes&&... funcs // Use forwarding references for input functions
    ) : input_functions_(funcs...), // Perfectly forward functions into the tuple
        weight_function_(std::move(weight_fn)),           // Move the weight function
        integrator_(integrator)                           // Initialize member by COPYING from the const reference
    {
        if (!integrator_.is_initialized()) {
             throw std::invalid_argument("Projector requires an initialized QuadratureRuleHolder.");
        }
        if (!weight_function_) {
            throw std::invalid_argument("Projector requires a valid weight function.");
        }
    }

    /**
     * @brief Computes the integral of weight_fn(x) * product(input_functions(x)) dx.
     * @param lower_bound The lower limit of integration.
     * @param upper_bound The upper limit of integration.
     * @return The computed integral value.
     * @throws std::runtime_error if the integrator fails or functions are invalid at runtime.
     */
    R compute_integral(R lower_bound, R upper_bound) const {
        // Define the full integrand: weight_fn(x) * product(input_funcs(x))
        auto full_integrand = [this](R x) -> R {
            R product = evaluate_product(x);
             if (!weight_function_) [[unlikely]] { // Should have been caught in ctor
                 throw std::runtime_error("Weight function became invalid after construction.");
            }
            return weight_function_(x) * product;
        };

        // Use the integrator from the QuadratureRuleHolder
        return integrator_.integrate(full_integrand, lower_bound, upper_bound);
    }

    const std::function<R(R)>& get_weight_function() const { return weight_function_; }
    const std::tuple<std::decay_t<FuncTypes>...>& get_input_functions() const { return input_functions_; }
    size_t get_num_input_functions() const { return sizeof...(FuncTypes); }
    const quadrature::QuadratureRuleHolder<R>& get_integrator() const { return integrator_; } // Returns const ref to the internal copy
};


// --- Deduction Guide (C++17) ---
template<typename R, typename... FuncTypes>
Projector(const quadrature::QuadratureRuleHolder<R>&, std::function<R(R)>, FuncTypes...)
    -> Projector<R, FuncTypes...>;

} // namespace quadrature
#endif // PROJECTOR_HPP
