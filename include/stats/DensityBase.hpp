
/**
 * @file DensityBase.hpp
 * @brief Provides a generic base class for probability density functions using Boost.Math distributions.
 *
 * This header defines a flexible template class `BoostBaseDensity` that wraps Boost.Math distributions,
 * providing a unified interface for PDF, CDF, quantile, and parameter access, as well as domain support.
 * It uses SFINAE to enable property accessors only for distributions that support them.
 * 
 * Dependencies:
 * - Boost.Math for distribution implementations.
 * - traits/OPOE_traits.hpp for type definitions and concepts.
 *
 * Main Components:
 * - SFINAE detection helpers: Traits to detect if a Boost distribution supports mean, stddev, shape, scale, alpha, beta.
 * - DensityInterval: Represents the support (domain) of a distribution, with bounds checking.
 * - BoostBaseDensity: Template class that wraps a Boost distribution, providing PDF, CDF, quantile, and parameter access.
 *   - Stores the constructed Boost distribution and its support interval.
 *   - Provides exception-safe access to PDF, CDF, and quantile.
 *   - SFINAE-enabled getters for distribution parameters (mean, stddev, etc.).
 *   - Stores the original constructor arguments for introspection or serialization.
 * - Factory functions: Helper functions to create densities for normal, gamma, and beta distributions.
 *
 * Usage Example:
 * @code
 * auto norm = stats::make_normal_density(0.0, 1.0);
 * double p = norm.pdf(0.5);
 * double mu = norm.getMu();
 * @endcode
 */

#ifndef DENSITY_BASE_HPP
#define DENSITY_BASE_HPP
#include <stdexcept>
#include <type_traits> 
#include <limits>      
#include <utility>     
#include <tuple>      
#include <vector>      
#include <numeric>     
#include <iostream>    
#include <typeinfo>   
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/complement.hpp> 
#include <boost/math/policies/policy.hpp>         
#include "../traits/OPOE_traits.hpp"

// --- SFINAE detection helpers (Using generic Boost functions) ---
namespace stats {

/**
 * @brief Trait to detect if a Boost distribution supports `boost::math::mean`.
 * 
 * @tparam T Distribution type
 * @tparam R Result type (typically double)
 */
template<typename T, typename R, typename = void>
struct has_boost_mean : std::false_type {};
template<typename T, typename R>
struct has_boost_mean<T, R, std::void_t<decltype(boost::math::mean(std::declval<const T&>()))>> : std::true_type {};
template<typename T, typename R>
inline constexpr bool has_boost_mean_v = has_boost_mean<T, R>::value;

/**
 * @brief Trait to detect if a Boost distribution supports `boost::math::standard_deviation`.
 * 
 * @tparam T Distribution type
 * @tparam R Result type (typically double)
 */
template<typename T, typename R, typename = void>
struct has_boost_stddev : std::false_type {};
template<typename T, typename R>
struct has_boost_stddev<T, R, std::void_t<decltype(boost::math::standard_deviation(std::declval<const T&>()))>> : std::true_type {};
template<typename T, typename R>
inline constexpr bool has_boost_stddev_v = has_boost_stddev<T, R>::value;

/**
 * @brief Trait to detect if a Boost distribution possesses shape parameter.
 * 
 * @tparam T Distribution type
 * @tparam R Result type (typically double)
 */
template<typename T, typename R, typename = void>
struct has_boost_shape : std::false_type {};
template<typename T, typename R>
struct has_boost_shape<T, R, std::void_t<decltype(std::declval<const T&>().shape())>> : std::true_type {};
template<typename T, typename R>
inline constexpr bool has_boost_shape_v = has_boost_shape<T, R>::value;

/**
 * @brief Trait to detect if a Boost distribution possesses scale parameter.
 * 
 * @tparam T Distribution type
 * @tparam R Result type (typically double)
 */
template<typename T, typename R, typename = void>
struct has_boost_scale : std::false_type {};
template<typename T, typename R>
struct has_boost_scale<T, R, std::void_t<decltype(std::declval<const T&>().scale())>> : std::true_type {};
template<typename T, typename R>
inline constexpr bool has_boost_scale_v = has_boost_scale<T, R>::value;


/**
 * @brief Trait to detect if a Boost distribution possesses alpha parameter.
 * 
 * @tparam T Distribution type
 * @tparam R Result type (typically double)
 */
// Note: Boost doesn't have generic alpha/beta accessors.
// These traits check for the *member* functions specific to beta_distribution.
// If other distributions provided these members, they would also work.
template<typename T, typename R, typename = void>
struct has_boost_alpha : std::false_type {};
template<typename T, typename R>
struct has_boost_alpha<T, R, std::void_t<decltype(std::declval<const T&>().alpha())>> : std::true_type {};
template<typename T, typename R>
inline constexpr bool has_boost_alpha_v = has_boost_alpha<T, R>::value;

/**
 * @brief Trait to detect if a Boost distribution possesses beta parameter.
 * 
 * @tparam T Distribution type
 * @tparam R Result type (typically double)
 */
template<typename T, typename R, typename = void>
struct has_boost_beta_param : std::false_type {};
template<typename T, typename R>
struct has_boost_beta_param<T, R, std::void_t<decltype(std::declval<const T&>().beta())>> : std::true_type {};
template<typename T, typename R>
inline constexpr bool has_boost_beta_param_v = has_boost_beta_param<T, R>::value;



/**
 * @brief Represents a valid interval over which a density is defined.
 * 
 * @tparam R Numeric type (e.g., float or double).
 */
template<typename R>
struct DensityInterval {
    R lower = -std::numeric_limits<R>::infinity();
    R upper = std::numeric_limits<R>::infinity();

    /**
     * @brief Constructs a DensityInterval with explicit bounds.
     * @param l Lower bound
     * @param u Upper bound
     * @throws std::invalid_argument if l > u and neither bound is NaN.
     */
    DensityInterval(R l, R u) : lower(l), upper(u) {
        if (l > u && !(std::isnan(l) || std::isnan(u))) { // Allow NaN bounds? Maybe not.
             throw std::invalid_argument("Lower bound cannot be greater than upper bound in DensityInterval.");
        }
    }
    /**
     * @brief Default constructor creating interval (-inf, +inf).
     */
    DensityInterval() = default; // Default interval is (-inf, +inf)
};

/**
 * @brief Generic wrapper for Boost continuous probability distributions.
 * 
 * Provides a uniform interface for PDF, CDF, quantile, and parameter access.
 * Uses SFINAE to enable distribution-specific properties (e.g., mean, stddev).
 * 
 * @tparam BoostDist The specific Boost distribution type (e.g., boost::math::normal_distribution).
 * @tparam R Numeric type (defaults to double).
 * @tparam ConstructorArgs Constructor argument types used to instantiate BoostDist.
 */
template<typename BoostDist, typename R, typename... ConstructorArgs>
class BoostBaseDensity {
private:
    /**
     * @brief Determines the support of the distribution using boost::math::support.
     * @param dist The distribution object.
     * @return The inferred support interval or (-inf, inf) if inference fails.
     */    
    static DensityInterval<R> get_safe_support(const BoostDist& dist) {
        try {
            // Use boost::math::support to get the theoretical support interval
            auto support_pair = boost::math::support(dist);
            R lower = support_pair.first;
            R upper = support_pair.second;
            return DensityInterval<R>(lower, upper);

        } catch (const std::exception& e) {
            // If boost::math::support fails (e.g., Cauchy distribution), default to (-inf, inf)
            std::cerr << "Warning: Could not determine finite support for distribution "
                      << typeid(dist).name() << ". Using (-inf, inf). Reason: " << e.what() << std::endl;
            return DensityInterval<R>(); // Default constructor gives (-inf, inf)
        } catch (...) {
             std::cerr << "Warning: Unknown error determining support for distribution "
                      << typeid(dist).name() << ". Using (-inf, inf)." << std::endl;
            return DensityInterval<R>();
        }
    }

    // --- Helper trait to distinguish constructors from copy/move ---
    // Checks if ArgTypes... is exactly one type which decays to BoostBaseDensity itself.
    template <typename... ArgTypes>
    struct is_this_class_signature : std::false_type {};

    template <typename SingleArg> // Specialization for a single argument
    struct is_this_class_signature<SingleArg> : std::is_same<
                                                    std::decay_t<SingleArg>,
                                                    BoostBaseDensity<BoostDist, R, ConstructorArgs...>
                                                > {};

protected:
    BoostDist dist_;
    DensityInterval<R> domain_;
    std::tuple<ConstructorArgs...> constructor_params_; // Store original constructor args

public:
   /**
     * @brief Constructs the Boost distribution using forwarded arguments.
     * 
     * Enabled only if BoostDist is constructible from Args, and Args is not copy/move.
     * 
     * @tparam Args Argument types
     * @param args Arguments to construct the BoostDist
     */
    template <typename... Args,
              std::enable_if_t<
                  std::is_constructible_v<BoostDist, Args...> &&
                  !is_this_class_signature<Args...>::value
              , int> = 0>
    explicit BoostBaseDensity(Args&&... args) // Use 'explicit' to prevent unintended conversions
        : dist_(std::forward<Args>(args)...)             // Construct the distribution
        , domain_(get_safe_support(dist_))               // Determine its domain
        , constructor_params_(std::forward<Args>(args)...) // Store the arguments used
    {}

    /// @name Rule of Five
    /// @{
    BoostBaseDensity(const BoostBaseDensity& other) = default;
    BoostBaseDensity& operator=(const BoostBaseDensity& other) = default;
    BoostBaseDensity(BoostBaseDensity&& other) noexcept = default; // Ensure noexcept if members are
    BoostBaseDensity& operator=(BoostBaseDensity&& other) noexcept = default;
    ~BoostBaseDensity() = default;
     /// @}

    /**
     * @brief Probability density function.
     * @param x Point at which to evaluate the PDF.
     * @return Value of the PDF or 0 if x is outside domain.
     */
    R pdf(R x) const {
        if (!isInDomain(x)) {
             // Return 0 for points outside the domain, common convention
             return R(0.0);
        }
        try {
            return boost::math::pdf(dist_, x);
        } catch (const std::exception& e) {
            // Handle potential boost errors (e.g., pdf at boundary for some distributions)
             std::cerr << "Warning: boost::math::pdf failed for " << typeid(dist_).name()
                       << " at x=" << x << ". Returning 0. Error: " << e.what() << std::endl;
             return R(0.0); // Or NaN: std::numeric_limits<R>::quiet_NaN();
        }
    }
    /**
     * @brief Cumulative distribution function.
     * @param x Point at which to evaluate the CDF.
     * @return Value in [0,1], or NaN if Boost call fails.
     */
    R cdf(R x) const {
        if (x <= domain_.lower) return R(0.0);
        if (x >= domain_.upper) return R(1.0);
        try {
            return boost::math::cdf(dist_, x);
        } catch (const std::exception& e) {
            std::cerr << "Warning: boost::math::cdf failed for " << typeid(dist_).name()
                      << " at x=" << x << ". Returning NaN. Error: " << e.what() << std::endl;
            return std::numeric_limits<R>::quiet_NaN(); 
        }
    }
    
    /**
     * @brief Computes the quantile (inverse CDF).
     * @param p Probability in [0, 1].
     * @return Quantile value or throws if p is invalid.
     */
    R quantile(R p) const {
        if (p < 0 || p > 1) {
            throw std::domain_error("Quantile probability p must be in [0, 1].");
        }
        // Handle exact boundaries to avoid potential issues with Boost functions
        if (p == 0) return domain_.lower;
        if (p == 1) return domain_.upper;
        try {
            R result = boost::math::quantile(dist_, p);
            // Clamped result to domain to handle potential numerical inaccuracies near boundaries
            return std::max(domain_.lower, std::min(domain_.upper, result));
        } catch (const std::exception& e) {

            throw std::runtime_error("Boost::math::quantile failed for " + std::string(typeid(dist_).name()) +
                                     " at p=" + std::to_string(p) + ". Reason: " + e.what());
        }
    }

   /**
     * @brief Checks if a value lies within the domain.
     * @param x Value to check.
     * @return True if in domain, false otherwise.
     */    
    bool isInDomain(R x) const noexcept { // Mark noexcept since comparisons/infinity checks don't throw
       if (std::isnan(x)) return false; // NaN is not in any valid domain
       bool lower_ok = (x >= domain_.lower);
       bool upper_ok = (x <= domain_.upper);
       // Explicit check for infinite bounds needed because standard comparisons with infinity work as expected
       return lower_ok && upper_ok;
    }

    /// @name Accessors
    /// @{
    const DensityInterval<R>& getDomain() const noexcept { return domain_; }
    const BoostDist& getDistribution() const noexcept { return dist_; }
    const std::tuple<ConstructorArgs...>& getConstructorParameters() const noexcept {
        return constructor_params_;
    }
    /// @}

    /// @name Distribution Properties (SFINAE-enabled)
    /// @{
    /** @brief Mean (if supported). */     
    template <typename D = BoostDist, typename = std::enable_if_t<has_boost_mean_v<D, R>>>
    R getMu() const { try { return boost::math::mean(dist_); } catch(...) { return std::numeric_limits<R>::quiet_NaN(); } }

    /** @brief Standard deviation (if supported). */
    template <typename D = BoostDist, typename = std::enable_if_t<has_boost_stddev_v<D, R>>>
    R getSigma() const { try { return boost::math::standard_deviation(dist_); } catch(...) { return std::numeric_limits<R>::quiet_NaN(); } }

    /** @brief Shape parameter (if supported). */
    template <typename D = BoostDist, typename = std::enable_if_t<has_boost_shape_v<D, R>>>
    R getShape() const { try { return dist_.shape(); } catch(...) { return std::numeric_limits<R>::quiet_NaN(); } }

    /** @brief Scale parameter (if supported). */
    template <typename D = BoostDist, typename = std::enable_if_t<has_boost_scale_v<D, R>>>
    R getScale() const { try { return dist_.scale(); } catch(...) { return std::numeric_limits<R>::quiet_NaN(); } }

    /** @brief Alpha parameter (if supported). */
    template <typename D = BoostDist, typename = std::enable_if_t<has_boost_alpha_v<D, R>>>
    R getAlpha() const { try { return dist_.alpha(); } catch(...) { return std::numeric_limits<R>::quiet_NaN(); } }

    /** @brief Beta parameter (if supported). */
    template <typename D = BoostDist, typename = std::enable_if_t<has_boost_beta_param_v<D, R>>>
    R getBeta() const { try { return dist_.beta(); } catch(...) { return std::numeric_limits<R>::quiet_NaN(); } }
    /// @}
};


/**
 * @brief Factory for a BoostBaseDensity using normal_distribution.
 * @tparam R Numeric type
 * @param args Parameters to construct boost::math::normal_distribution
 * @return Wrapped normal distribution
 */
template<typename R = traits::DataType::PolynomialField, typename... Args>
auto make_normal_density(Args&&... args) {
    using Dist = boost::math::normal_distribution<R>;
    // Correctly passes R and the decayed types of constructor args
    return BoostBaseDensity<Dist, R, std::decay_t<Args>...>(std::forward<Args>(args)...);
}

/**
 * @brief Factory for a BoostBaseDensity using gamma_distribution.
 * @tparam R Numeric type
 * @param args Parameters to construct boost::math::gamma_distribution
 * @return Wrapped gamma distribution
 */
template<typename R = traits::DataType::PolynomialField, typename... Args>
auto make_gamma_density(Args&&... args) {
    using Dist = boost::math::gamma_distribution<R>;
    return BoostBaseDensity<Dist, R, std::decay_t<Args>...>(std::forward<Args>(args)...);
}

/**
 * @brief Factory for a BoostBaseDensity using beta_distribution.
 * @tparam R Numeric type
 * @param args Parameters to construct boost::math::beta_distribution
 * @return Wrapped beta distribution
 */
template<typename R = traits::DataType::PolynomialField, typename... Args>
auto make_beta_density(Args&&... args) {
    using Dist = boost::math::beta_distribution<R>;
    return BoostBaseDensity<Dist, R, std::decay_t<Args>...>(std::forward<Args>(args)...);
}

} // namespace stats


#endif // DENSITY_BASE_HPP