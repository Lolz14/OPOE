/**
 * @file BaseOptionPricer.hpp
 * @brief Defines the BaseOptionPricer class template for option pricing.
 *
 * This file contains the declaration of the BaseOptionPricer class, which provides a common interface
 * and base functionality for various option pricing methods (e.g., FFT, Monte Carlo).
 * The class manages option parameters such as time to maturity, risk-free rate, and the payoff function,
 * and observes changes in the underlying SDE model to ensure pricing consistency.
 * @class options::BaseOptionPricer
 * @brief Abstract base class for option pricing methods.
 *
 * This class template provides a unified interface for option pricers, encapsulating
 * the essential parameters and logic required for pricing financial derivatives.
 * It supports observer registration for SDE model changes, ensuring that derived
 * pricers can react to model updates and recompute prices as needed.
 *
 * @tparam R Floating-point type used for calculations (default: traits::DataType::PolynomialField).
 *
 * ### Key Features
 * - Stores time to maturity, risk-free rate, payoff function, and SDE model.
 * - Registers as an observer to the SDE model to track changes.
 * - Provides interface for setting/getting option parameters.
 * - Enforces implementation of price() and recompute() in derived classes.
 * - Supports copy/move semantics with correct observer management.
 *
 * ### Usage
 * Inherit from this class to implement specific pricing algorithms (e.g., FFTPricer, MCPricer).
 *
 * @see options::IPayoff
 * @see SDE::ISDEModel
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
     * It also holds a shared pointer to the payoff function, which can be evaluated by derived classes.
     */
    template <typename R = traits::DataType::PolynomialField>
    class BaseOptionPricer {

        public:

            /**
             * @brief Constructs a BaseOptionPricer instance and registers the model observer.
             * @param ttm Time to maturity.
             * @param rate Risk-free interest rate.
             * @param payoff Payoff function for the option.
             * @param sde_model Shared pointer to the SDE model used for pricing.
             */
            BaseOptionPricer(R ttm, R rate, std::shared_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model)
                : ttm_(ttm), rate_(rate), payoff_(payoff), sde_model_(sde_model) {
                    if (sde_model_) {
                // Register observer: mark_dirty() will be called whenever the model changes
                observer_id_ = sde_model_->add_observer([this]() { this->mark_dirty(); });
                }
                }

            virtual ~BaseOptionPricer(){
                if (sde_model_ && observer_id_) {
                    sde_model_->remove_observer(observer_id_);
                }
            };

            // Copy constructor
            BaseOptionPricer(const BaseOptionPricer& other)
                : payoff_(other.payoff_ ? other.payoff_->clone() : nullptr) {
                    if (sde_model_) {
                        observer_id_ = sde_model_->add_observer([this]() { this->mark_dirty(); });
                    }
                }

            // Copy assignment
             BaseOptionPricer& operator=(const BaseOptionPricer& other) {
            if (this != &other) {
                ttm_ = other.ttm_;
                rate_ = other.rate_;
                payoff_ = other.payoff_ ? other.payoff_->clone() : nullptr;
                sde_model_ = other.sde_model_;
                if (sde_model_) {
                    observer_id_ = sde_model_->add_observer([this]() { this->mark_dirty(); });
                }
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

            inline void set_rate(R rate) noexcept { rate_ = rate;}
            inline R get_rate() const noexcept { return rate_;  }

            void mark_dirty() { dirty_ = true; }
            void mark_clean() { dirty_ = false;}
        protected:

        R ttm_;
        R rate_;
        std::shared_ptr<IPayoff<R>> payoff_;
        std::shared_ptr<SDE::ISDEModel<R>> sde_model_;
        typename SDE::ObserverId observer_id_;
        mutable bool dirty_ = false;

        void ensure_recomputed() const {
        if (dirty_) {
            recompute();   // derived class must implement this
            dirty_ = false;
            }
        }

        // Derived pricers must implement recompute()
        virtual void recompute() const = 0;
        };


} // namespace options

#endif // BASE_OPTION_PRICER_HPP