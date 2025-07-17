#ifndef BASE_OPTION_PRICER_HPP
#define BASE_OPTION_PRICER_HPP

#include <cmath>
#include <memory>
#include "../traits/OPOE_traits.hpp"
#include "Payoff.hpp"
#include "../sde/FinModels.hpp"

namespace options {
    using Array = traits::DataType::StoringArray;

    template <typename R = traits::DataType::PolynomialField>
    class BaseOptionPricer {

        public:

            BaseOptionPricer(R ttm, R strike, R rate, Array initial_price, std::unique_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model)
                : ttm_(ttm), strike_(strike), rate_(rate), x0_(initial_price), payoff_(std::move(payoff)), sde_model_(sde_model) {}

            virtual ~BaseOptionPricer() = default;

            // The common pricing interface for all methods

            virtual R price() const = 0;

        protected:

        R ttm_;
        R strike_;
        R rate_;
        Array x0_;
        std::unique_ptr<IPayoff<R>> payoff_;
        std::shared_ptr<SDE::ISDEModel<R>> sde_model_;

    };


} // namespace options

#endif // BASE_OPTION_PRICER_HPP