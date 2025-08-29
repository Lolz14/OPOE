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
    using SolverType = traits::SolverType;

    
    /**
     * @brief Constructs a MCOptionPricer instance.
     *This method generates paths of the underlying asset using the provided SDE model.

     * @param ttm Time to maturity.
     * @param rate Risk-free interest rate.
     * @param payoff Payoff function for the option.
     * @param sde_model Shared pointer to the SDE model used for path generation.
     * @param solver_type Enum indicating the solver type.
     * @param num_paths Number of Monte Carlo paths to simulate (default: 10).
     * @param num_steps Number of time steps in each path (default: 3).
     */
    MCOptionPricer(R ttm, R rate,
             std::shared_ptr<IPayoff<R>> payoff,
             std::shared_ptr<SDE::ISDEModel<R>> sde_model,
             SolverType solver_type,
             unsigned int num_paths = 10,
             unsigned int num_steps = 3)
        : BaseOptionPricer<R>(ttm, rate, std::move(payoff), std::move(sde_model)),
          solver_type_(solver_type),
          num_paths_(num_paths),
          num_steps_(num_steps)
    {
        StoringMatrix all_paths;

        switch (solver_type_) {
            case SolverType::EulerMaruyama:
                all_paths = SDE::EulerMaruyamaSolver<SDE::ISDEModel<R>, R>(*this->sde_model_)
                                .solve(0.0, this->ttm_, num_steps_, num_paths_, std::nullopt);
                break;

            case SolverType::Milstein:
                all_paths = SDE::MilsteinSolver<SDE::ISDEModel<R>, R>(*this->sde_model_)
                                .solve(0.0, this->ttm_, num_steps_, num_paths_, std::nullopt);
                break;

            case SolverType::IJK:
                all_paths = SDE::InterpolatedKahlJackelSolver<SDE::ISDEModel<R>, R>(*this->sde_model_)
                                .solve(0.0, this->ttm_, num_steps_, num_paths_, std::nullopt);
                break;

            default:
                throw std::runtime_error("Unknown solver type");
        }

        int state_dim = this->sde_model_->get_state_dim();
        StoringVector terminal_column = all_paths.col(num_steps_);
        terminal_logS_.resize(num_paths_);

        if (state_dim == 1) {
            terminal_logS_ = terminal_column;
        } else {
            terminal_logS_ = Eigen::Map<const StoringVector, 0, Eigen::InnerStride<>>(
                terminal_column.data() + 1,
                num_paths_,
                Eigen::InnerStride<>(state_dim)
            );
     
        }

    }


    /**
     * @brief Prices the option using Monte Carlo simulation.
     * Evaluates the payoff at maturity, and returns the expected discounted payoff.
     * @return The computed option price.
     * 
     */
    R price() const override {





        auto payoffs = this->payoff_->evaluate_from_log(this->terminal_logS_);


        return std::exp(-this->rate_ * this->ttm_) * payoffs.mean();
    }

private:
    SolverType solver_type_;
    StoringVector terminal_logS_;
    unsigned int num_paths_;
    unsigned int num_steps_;
};

} // namespace options

#endif // MC_OPTION_PRICER_HPP
