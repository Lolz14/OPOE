/**
 * @file MCOptionPricer.hpp
 * @brief Defines the MCOptionPricer class for Monte Carlo option pricing.
 *
 * This header provides the implementation of a Monte Carlo pricer for financial options.
 * The MCOptionPricer class simulates multiple paths of the underlying asset using a provided
 * stochastic differential equation (SDE) model and a solver function. It evaluates the
 * option payoff at maturity for each simulated path and computes the expected discounted
 * payoff as the option price.
 *
 * Key features:
 * - Templated on the floating-point type for flexibility.
 * - Accepts custom payoff and SDE models via polymorphic interfaces.
 * - Allows injection of a custom SDE solver function.
 * - Supports configuration of the number of Monte Carlo paths and time steps.
 *
 * Dependencies:
 * - Eigen for matrix and vector operations.
 * - Standard library components for memory management and function objects.
 * - SDE.hpp for the SDE model interface.
 * - BaseOptionPricer.hpp for the base class of option pricing.
 *
 */

#ifndef MC_OPTION_PRICER_HPP
#define MC_OPTION_PRICER_HPP

#include "BaseOptionPricer.hpp"
#include "../sde/SDE.hpp"
#include <memory>
#include <Eigen/Dense>
#include <iostream>
#include <functional>

namespace options {

/**
 * @brief Monte Carlo Option Pricer class for pricing options using Monte Carlo simulation.
 *
 * This class implements the Monte Carlo method for option pricing, simulating paths of the underlying asset
 * and calculating the expected payoff at maturity.
 *
 * @tparam R The floating-point type used for calculations (default: traits::DataType::PolynomialField).
 */
    
template<typename R>
class MCOptionPricer : public BaseOptionPricer<R> {
public:
    using StoringVector = traits::DataType::StoringVector;
    using StoringMatrix = traits::DataType::StoringMatrix;

    
    using SolverFunc = std::function<StoringMatrix(
    R, R, int, int, const std::optional<StoringMatrix>& dW_opt)>;
    
    /**
     * @brief Constructs a MCOptionPricer instance.
     *
     * @param ttm Time to maturity.
     * @param strike Strike price of the option.
     * @param rate Risk-free interest rate.
     * @param payoff Payoff function for the option.
     * @param sde_model Shared pointer to the SDE model used for path generation.
     * @param solver_func Function to solve the SDE and generate paths.
     * @param num_paths Number of Monte Carlo paths to simulate (default: 10).
     * @param num_steps Number of time steps in each path (default: 3).
     */
    MCOptionPricer(R ttm, R strike, R rate,
             std::unique_ptr<IPayoff<R>> payoff,
             std::shared_ptr<SDE::ISDEModel<R>> sde_model,
             SolverFunc solver_func,
             unsigned int num_paths = 10,
             unsigned int num_steps = 3)
        : BaseOptionPricer<R>(ttm, strike, rate, std::move(payoff), std::move(sde_model)),
          solver_func_(std::move(solver_func)),
          num_paths_(num_paths),
          num_steps_(num_steps)
    {}


    /**
     * @brief Prices the option using Monte Carlo simulation.
     * This method generates paths of the underlying asset using the provided SDE model,
     * evaluates the payoff at maturity, and returns the expected discounted payoff.
     * @return The computed option price.
     * 
     */
    R price() const override {

        auto all_paths = solver_func_(0.0, this->ttm_, num_steps_, num_paths_, std::nullopt);

        int state_dim = this->sde_model_->get_state_dim();
        StoringVector terminal_column = all_paths.col(num_steps_);

        StoringVector terminal_logS(num_paths_);

        if (state_dim == 1) {
            terminal_logS = terminal_column;
        } else {
            terminal_logS = Eigen::Map<const StoringVector, 0, Eigen::InnerStride<>>(
                terminal_column.data() + 1,
                num_paths_,
                Eigen::InnerStride<>(state_dim)
            );
        }



        auto payoffs = this->payoff_->evaluate_from_log(terminal_logS);


        return std::exp(-this->rate_ * this->ttm_) * payoffs.mean();
    }

private:
    SolverFunc solver_func_;
    unsigned int num_paths_;
    unsigned int num_steps_;
};

} // namespace options

#endif // MC_OPTION_PRICER_HPP
