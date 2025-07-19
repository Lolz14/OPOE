#ifndef CF_OPTION_PRICER_HPP
#define CF_OPTION_PRICER_HPP

#include <cmath>
#include "BaseOptionPricer.hpp"
#include "../stats/DensityBase.hpp"
namespace options {

using Array = traits::DataType::StoringArray;

template <typename R = traits::DataType::PolynomialField>
class CFPricer : public BaseOptionPricer<R> {
    using Base = BaseOptionPricer<R>;
    
public:
    CFPricer(R ttm, R strike, R rate,
             std::unique_ptr<IPayoff<R>> payoff,
             std::shared_ptr<SDE::ISDEModel<R>> sde_model
             )
        : Base(ttm, strike, rate, std::move(payoff), std::move(sde_model))
          
    {
        if (!std::dynamic_pointer_cast<SDE::GeometricBrownianMotionSDE<R>>(Base::sde_model_)) {
            throw std::logic_error("CFPricer is only valid for the Black-Scholes (GBM) model.");
        }

        const R S0 = std::exp(this->sde_model_->m_x0(0));
        volatility_ = std::dynamic_pointer_cast<SDE::GeometricBrownianMotionSDE<R>>(Base::sde_model_)->get_parameters().sigma;
        S0_ = S0;

        d1_ = (std::log(S0 / Base::strike_) + (Base::rate_ + 0.5 * volatility_ * volatility_) * Base::ttm_) /
              (volatility_ * std::sqrt(Base::ttm_));
        d2_ = d1_ - volatility_ * std::sqrt(Base::ttm_);
    }

    R price() const override {
        const R K = Base::strike_;
        const R r = Base::rate_;
        const R T = Base::ttm_;
        const R S = S0_;

        if (dynamic_cast<const EuropeanCallPayoff<R>*>(Base::payoff_.get())) {
            return S * normal_.cdf(d1_) - K * std::exp(-r * T) * normal_.cdf(d2_);
        } else if (dynamic_cast<const EuropeanPutPayoff<R>*>(Base::payoff_.get())) {
            return K * std::exp(-r * T) * normal_.cdf(-d2_) - S * normal_.cdf(-d1_);
        } else {
            throw std::logic_error("Unsupported payoff type for CFPricer.");
        }
    }

    // --- Greeks ---
    R delta() const {
        if (dynamic_cast<const EuropeanCallPayoff<R>*>(Base::payoff_.get())) {
            return normal_.cdf(d1_);
        } else if (dynamic_cast<const EuropeanPutPayoff<R>*>(Base::payoff_.get())) {
            return normal_.cdf(d1_) - R(1.0);
        } else {
            throw std::logic_error("Unsupported payoff type for delta.");
        }
    }

    R gamma() const {
        return normal_.pdf(d1_) / (S0_ * volatility_ * std::sqrt(Base::ttm_));
    }

    R vega() const {
        return S0_ * normal_.pdf(d1_) * std::sqrt(Base::ttm_);
    }

    R theta() const {
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

    R rho() const {
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
    R volatility_;
    R d1_, d2_;
    R S0_;
    static inline auto normal_ = stats::make_normal_density<R>(0.0, 1.0);
};

} // namespace options

#endif // CF_OPTION_PRICER_HPP