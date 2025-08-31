/**
 * @file CFOptionPricer.hpp
 * @brief Defines the CFOptionPricer class for closed-form (Black-Scholes) pricing of European options.
 *
 * This header provides the implementation of the CFOptionPricer class template, which specializes in
 * pricing European call and put options using the Black-Scholes closed-form solution. The class supports
 * calculation of the option price as well as the main Greeks (delta, gamma, vega, theta, rho).
 *
 * @details
 * - The pricer requires a payoff object (call or put), a risk-free rate, time to maturity, and a stochastic
 *   differential equation (SDE) model, which must be a Geometric Brownian Motion (GBM) model.
 * - The class checks at construction that the SDE model is compatible (GBM).
 * - The price and Greeks are computed using the standard Black-Scholes formulas, with internal caching
 *   of intermediate variables (d1, d2, volatility, S0) for efficiency.
 * - Throws std::logic_error if used with unsupported payoff types or SDE models.
 *
 */
#ifndef CF_OPTION_PRICER_HPP
#define CF_OPTION_PRICER_HPP

#include <cmath>
#include "BaseOptionPricer.hpp"
#include "../stats/DensityBase.hpp"
namespace options {

using Array = traits::DataType::StoringArray;

/**
 * @brief CFOptionPricer class for pricing options using the Black-Scholes model.
 * This class implements the closed-form solution for European options under the Black-Scholes model.
 * It supports both call and put options and provides methods for computing option price and Greeks.
 *
 * @tparam R The floating-point type used for calculations (default: traits::DataType::PolynomialField).
 */
template <typename R = traits::DataType::PolynomialField>
class CFOptionPricer : public BaseOptionPricer<R> {
    using Base = BaseOptionPricer<R>;
    
public:
    /**
     * @brief Constructor for CFOptionPricer.
     * @param ttm Time to maturity in years.
     * @param rate Risk-free interest rate.
     * @param payoff Unique pointer to the option payoff object (Call or Put).
     * @param sde_model Shared pointer to the SDE model (should be GeometricBrownianMotionSDE).
     */
    CFOptionPricer(R ttm, R rate,
             std::shared_ptr<IPayoff<R>> payoff,
             std::shared_ptr<SDE::ISDEModel<R>> sde_model
             )
        : Base(ttm, rate, std::move(payoff), std::move(sde_model))
          
    {
        if (!std::dynamic_pointer_cast<SDE::GeometricBrownianMotionSDE<R>>(Base::sde_model_)) {
            throw std::logic_error("CFOptionPricer is only valid for the Black-Scholes (GBM) model.");
        }
        
        recompute();
        
    }


    /**
     * @brief Prices the option using the Black-Scholes formula.
     * This method computes the option price based on the Black-Scholes closed-form solution.
     * It handles both call and put options based on the type of payoff provided.
     * @return The computed option price.
     */
    R price() const override {
        this->ensure_recomputed();
        const R r = Base::rate_;
        const R T = Base::ttm_;
        const R S = S0_;

        if (dynamic_cast<const EuropeanCallPayoff<R>*>(Base::payoff_.get())) {
            return S * normal_.cdf(d1_) - this->payoff_->getStrike() * std::exp(-r * T) * normal_.cdf(d2_);
        } else if (dynamic_cast<const EuropeanPutPayoff<R>*>(Base::payoff_.get())) {
            return this->payoff_->getStrike() * std::exp(-r * T) * normal_.cdf(-d2_) - S * normal_.cdf(-d1_);
        } else {
            throw std::logic_error("Unsupported payoff type for CFOptionPricer.");
        }
    }

    /**
     * @brief Computes the delta of the option.
     * This method calculates the sensitivity of the option price to changes in the underlying asset price.
     * @return The computed delta value.
     */
    R delta() const {
        this->ensure_recomputed();

        if (dynamic_cast<const EuropeanCallPayoff<R>*>(Base::payoff_.get())) {
            return normal_.cdf(d1_);
        } else if (dynamic_cast<const EuropeanPutPayoff<R>*>(Base::payoff_.get())) {
            return normal_.cdf(d1_) - R(1.0);
        } else {
            throw std::logic_error("Unsupported payoff type for delta.");
        }
    }

    /**
     * @brief Computes the gamma of the option.
     * This method calculates the sensitivity of the delta to changes in the underlying asset price.
     * @return The computed gamma value.
     */
    R gamma() const {
        this->ensure_recomputed();

        return normal_.pdf(d1_) / (S0_ * volatility_ * std::sqrt(Base::ttm_));
    }

    /**
     * @brief Computes the vega of the option.
     * This method calculates the sensitivity of the option price to changes in volatility.
     * @return The computed vega value.
     */
    R vega() const {
        this->ensure_recomputed();
        return S0_ * normal_.pdf(d1_) * std::sqrt(Base::ttm_);
    }

    /**
     * @brief Computes the theta of the option.
     * This method calculates the sensitivity of the option price to the passage of time.
     * @return The computed theta value.
     */
    R theta() const {
        this->ensure_recomputed();

        const R K = Base::strike_;
        const R r = Base::rate_;
        const R T = Base::ttm_;
        const R S = S0_;

        R term1 = -S * normal_.pdf(d1_) * volatility_ / (2.0 * std::sqrt(T));

        if (dynamic_cast<const EuropeanCallPayoff<R>*>(Base::payoff_.get())) {
            R term2 = -r * K * std::exp(-r * T) * normal_.cdf(d2_);
            return  (term1 + term2);
        } else if (dynamic_cast<const EuropeanPutPayoff<R>*>(Base::payoff_.get())) {
            R term2 = r * K * std::exp(-r * T) * normal_.cdf(-d2_);
            return  (term1 + term2);
        } else {
            throw std::logic_error("Unsupported payoff type for theta.");
        }
    }

    /**
     * @brief Computes the rho of the option.
     * This method calculates the sensitivity of the option price to changes in the risk-free interest rate.
     * @return The computed rho value.
     */
    R rho() const {
        this->ensure_recomputed();

        const R K = Base::strike_;
        const R r = Base::rate_;
        const R T = Base::ttm_;

        if (dynamic_cast<const EuropeanCallPayoff<R>*>(Base::payoff_.get())) {
            return K * T * std::exp(-r * T) * normal_.cdf(d2_);
        } else if (dynamic_cast<const EuropeanPutPayoff<R>*>(Base::payoff_.get())) {
            return -K * T * std::exp(-r * T) * normal_.cdf(-d2_);
        } else {
            throw std::logic_error("Unsupported payoff type for rho.");
        }
    }

private:
    mutable R volatility_;
    mutable R d1_, d2_;
    mutable R S0_;
    static inline auto normal_ = stats::make_normal_density<R>(0.0, 1.0);

    /**
     * @brief Recompute intermediate variables required for option pricing.
     *
     * This method updates all necessary internal variables for the
     * Black-Scholes closed-form pricing formula, including:
     * - Initial asset price `S0_` (from the SDE model)
     * - Volatility `volatility_` (from the Geometric Brownian Motion SDE)
     * - d1 and d2 terms used in the cumulative normal distribution functions
     *
     * Workflow:
     * 1. Retrieve the initial asset price from the SDE (`S0 = exp(x0)`).
     * 2. Retrieve the volatility from the SDE model (assumes `GeometricBrownianMotionSDE`).
     * 3. Compute d1 and d2 according to the Black-Scholes formulas:
     *    - d1 = [ln(S0 / K) + (r + 0.5 * sigma^2) * T] / (sigma * sqrt(T))
     *    - d2 = d1 - sigma * sqrt(T)
     *
     * @note This method must be called before computing the option price or Greeks.
     *       It overwrites mutable members `S0_`, `volatility_`, `d1_`, and `d2_`.
     * @throws std::bad_cast If the underlying SDE model is not a GeometricBrownianMotionSDE.
     */
    void recompute() const override {
        const R S0 = std::exp(this->sde_model_->get_x0());
        volatility_ = std::dynamic_pointer_cast<SDE::GeometricBrownianMotionSDE<R>>(Base::sde_model_)->get_v0();
        S0_ = S0;

        d1_ = (std::log(S0 / this->payoff_->getStrike()) + (Base::rate_ + 0.5 * volatility_ * volatility_) * Base::ttm_) /
              (volatility_ * std::sqrt(Base::ttm_));
        d2_ = d1_ - volatility_ * std::sqrt(Base::ttm_);
    }

};

} // namespace options

#endif // CF_OPTION_PRICER_HPP