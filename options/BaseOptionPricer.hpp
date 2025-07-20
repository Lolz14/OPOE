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

            BaseOptionPricer(R ttm, R strike, R rate, std::unique_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model)
                : ttm_(ttm), strike_(strike), rate_(rate), payoff_(std::move(payoff)), sde_model_(sde_model) {}

            virtual ~BaseOptionPricer() = default;

            // Copy constructor
            BaseOptionPricer(const BaseOptionPricer& other)
                : payoff_(other.payoff_ ? other.payoff_->clone() : nullptr) {}

            // Copy assignment
            BaseOptionPricer& operator=(const BaseOptionPricer& other) {
                if (this != &other) {
                    payoff_ = other.payoff_ ? other.payoff_->clone() : nullptr;
                }
                return *this;
            }

            // Move constructor/assignment
            BaseOptionPricer(BaseOptionPricer&&) noexcept = default;
            BaseOptionPricer& operator=(BaseOptionPricer&&) noexcept = default;

            // The common pricing interface for all methods

            virtual R price() const = 0;

            inline void set_ttm(R ttm) noexcept { ttm_ = ttm; }
            inline R get_ttm() const noexcept { return ttm_; }

            inline void set_strike(R strike) noexcept { strike_ = strike; }
            inline R get_strike() const noexcept { return strike_; }

            inline void set_rate(R rate) noexcept { rate_ = rate; }
            inline R get_rate() const noexcept { return rate_; }

        protected:

        R ttm_;
        R strike_;
        R rate_;
        std::unique_ptr<IPayoff<R>> payoff_;
        std::shared_ptr<SDE::ISDEModel<R>> sde_model_;

    };


} // namespace options

#endif // BASE_OPTION_PRICER_HPP