
/**
 * @file BaseOptionPricer.hpp
 * @brief Defines the BaseOptionPricer class, an abstract base class for option pricing models.
 *
 * This file contains the declaration of the BaseOptionPricer template class, which provides
 * a common interface and storage for option pricing parameters and payoff functions.
 * It is intended to be subclassed by specific option pricing implementations (e.g., FFTPricer, MCPricer).
 *
 * Dependencies:
 * - traits/OPOE_traits.hpp: Contains type definitions and traits for polynomial operations.
 * - Payoff.hpp: Defines the IPayoff interface for option payoffs.
 * - FinModels.hpp: Interface for stochastic differential equation models.
 */
#ifndef BASE_OPTION_PRICER_HPP
#define BASE_OPTION_PRICER_HPP

#include <cmath>
#include <memory>
#include "../traits/OPOE_traits.hpp"
#include "Payoff.hpp"
#include "../sde/FinModels.hpp"

namespace options {
    using Array = traits::DataType::StoringArray;

    /**
     * @brief Base class for option pricing.
     * This class provides a common interface for option pricing methods.
     * It holds the basic parameters of the option and the payoff function.
     * @tparam R The floating-point type used for calculations (default: traits::DataType::PolynomialField).
     * 
     * This class is designed to be inherited by specific option pricers like FFTPricer, MCPricer, etc.
     * It provides a common interface for setting and getting option parameters like time to maturity, strike price, and risk-free rate.
     * It also holds a unique pointer to the payoff function, which can be evaluated by derived classes.
     */
    template <typename R = traits::DataType::PolynomialField>
    class BaseOptionPricer {

        public:

            /**
             * @brief Constructs a BaseOptionPricer instance.
             * @param ttm Time to maturity.
             * @param strike Strike price of the option.
             * @param rate Risk-free interest rate.
             * @param payoff Payoff function for the option.
             * @param sde_model Shared pointer to the SDE model used for pricing.
             */
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