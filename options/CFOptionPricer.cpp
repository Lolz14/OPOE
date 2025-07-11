#include <cmath>
#include "../stats/DensityBase.hpp"
#include "CFOptionPricer.hpp"
#include "Payoff.hpp"

template <typename R = traits::DataType::PolynomialField>
R black_scholes(R ttm, R strike, R rate, R initial_price, R volatility,  const options::IPayoff<R>& payoff) {
    
    auto normal = stats::make_normal_density(0.0, 1.0);

    R d1 = (std::log(initial_price / strike) + (rate + 0.5 * volatility * volatility) * ttm) / (volatility * std::sqrt(ttm));
    R d2 = d1 - volatility * std::sqrt(ttm);

    if (dynamic_cast<const options::EuropeanCallPayoff<R>*>(&payoff)) {
        return initial_price * normal.cdf(d1) - strike * std::exp(-rate * ttm) * normal.cdf(d2);
    } 
    else if (dynamic_cast<const options::EuropeanPutPayoff<R>*>(&payoff)) {
        return strike * std::exp(-rate * ttm) * normal.cdf(-d2) - initial_price * normal.cdf(-d1);
    }

    throw std::logic_error("Unsupported payoff type for Black-Scholes formula.");

}
