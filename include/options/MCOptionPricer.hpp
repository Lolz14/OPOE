/**
 * @file MCOptionPricer.hpp
 * @brief Defines the MCOptionPricer class for Monte Carlo option pricing.
 *
 * This header provides the implementation of a Monte Carlo pricer for financial options,
 * supporting multiple SDE solvers (Euler-Maruyama, Milstein, Interpolated Kahl-Jäckel).
 * The pricer simulates paths of the underlying asset and computes the expected discounted payoff.
 *
 * @class options::MCOptionPricer
 * @brief Monte Carlo Option Pricer for simulating and pricing options.
 * 
 * This class implements option pricing via Monte Carlo simulation. It generates multiple
 * paths of the underlying asset using a specified SDE model and solver, evaluates the
 * option payoff at maturity, and returns the expected discounted payoff.
 * 
 * @tparam R Floating-point type for calculations (e.g., double).
 * 
 * ### Template Parameters
 * - `R`: Floating-point type for numerical computations.
 *
 * ### Public Methods
 * - `MCOptionPricer(...)`: Constructor. Initializes the pricer with option parameters, SDE model, solver type, and simulation settings.
 * - `R price() const override`: Computes and returns the Monte Carlo price of the option.
 *
 * ### Protected Members
 * - `mutable StoringVector terminal_logS_`: Stores the terminal log-prices for all simulated paths.
 *
 * ### Private Members
 * - `SolverType solver_type_`: Enum indicating which SDE solver to use.
 * - `unsigned int num_paths_`: Number of Monte Carlo paths to simulate.
 * - `unsigned int num_steps_`: Number of time steps per path.
 *
 * ### Private Methods
 * - `void recompute() const override`: Regenerates terminal log-price paths using the selected SDE solver.
 *     - Selects the solver based on `solver_type_`.
 *     - Simulates `num_paths_` paths with `num_steps_` time steps.
 *     - Extracts the terminal log-price for each path, handling both 1D and multi-dimensional SDEs.
 *     - Throws `std::runtime_error` if an unknown solver type is provided.
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
        recompute();

    }


    /**
     * @brief Prices the option using Monte Carlo simulation.
     * Evaluates the payoff at maturity, and returns the expected discounted payoff.
     * @return The computed option price.
     * 
     */
    R price() const override {
        //Ensure that paths do not have to be regenerated.
        this->ensure_recomputed();


        auto payoffs = this->payoff_->evaluate_from_log(this->terminal_logS_);


        return std::exp(-this->rate_ * this->ttm_) * payoffs.mean();
    }
protected:
    mutable StoringVector terminal_logS_;

private:
    SolverType solver_type_;
    unsigned int num_paths_;
    unsigned int num_steps_;


    /**
     * @brief Recompute terminal log-price paths using the selected SDE solver.
     *
     * This method evolves the underlying stochastic differential equation (SDE) model
     * from time 0 to maturity (`ttm_`) using the configured solver and stores the 
     * terminal log-price vector (`terminal_logS_`).
     *
     * Workflow:
     *  1. Depending on `solver_type_`, one of the supported solvers is selected:
     *      - Euler-Maruyama
     *      - Milstein
     *      - Interpolated Kahl-Jäckel (IJK)
     *  2. The solver generates paths (`num_paths_`) with `num_steps_` timesteps.
     *  3. The terminal states (at maturity) are extracted:
     *      - If the model has state dimension = 1, the last column is the log-price.
     *      - If state dimension > 1, the log-price is assumed to be at index 1 of each path,
     *        and extracted using an Eigen::Map with stride.
     *
     * @throws std::runtime_error If an unknown solver type is requested.
     *
     */
    void recompute() const override {
        // Solve paths
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
};

} // namespace options

#endif // MC_OPTION_PRICER_HPP
