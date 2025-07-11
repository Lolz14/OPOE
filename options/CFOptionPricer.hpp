#ifndef CF_OPTION_PRICER_HPP
#define CF_OPTION_PRICER_HPP

#include <cmath>
#include "BaseOptionPricer.hpp"
#include "CFOptionPricer.cpp"

namespace options
{

    template <typename R = traits::DataType::PolynomialField>
    class CFPricer : public BaseOptionPricer<R> {
        using Base = BaseOptionPricer<R>;
    
        public:
            CFPricer(R ttm, R strike, R rate, R initial_price, std::unique_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model)
            : Base(ttm, strike, rate, initial_price, std::move(payoff), std::move(sde_model)) {}

            R price() const override {
                if (auto gbm_model = std::dynamic_pointer_cast<SDE::GeometricBrownianMotionSDE<R>>(Base::sde_model_)) {
                    auto params = gbm_model->get_parameters();
                    return black_scholes(this->ttm_, this->strike_, this->rate_, this->x0_, params.sigma, *this->payoff_);
                }

                else {
                     throw std::logic_error("CFPricer is available only for Black and Scholes model.");
                }


                
            }


        };




    
    

} // namespace options


#endif // CF_OPTION_PRICER_HPP