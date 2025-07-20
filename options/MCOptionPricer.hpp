#ifndef MC_OPTION_PRICER_HPP
#define MC_OPTION_PRICER_HPP

#include "BaseOptionPricer.hpp"
#include "../sde/SDE.hpp"         // Your SDE solvers and concepts
#include <memory>
#include <Eigen/Dense>

namespace options {

template<
    typename R,
    typename Solver,   // e.g. SDE::EulerMaruyamaSolver<SdeType>
    typename SdeType   // the concrete SDE model class
>
class MCPricer : public BaseOptionPricer<R> {
public:
    using StoringVector = traits::DataType::StoringVector;
    using StoringMatrix = traits::DataType::StoringMatrix;

    MCPricer(R ttm, R strike, R rate,
                   std::unique_ptr<IPayoff<R>> payoff,
                   const SdeType& sde_model,   // store concrete, not pointer!
                   unsigned int num_paths = 100,
                   unsigned int num_steps = 100)
    : BaseOptionPricer<R>(ttm, strike, rate, std::move(payoff), nullptr), // pass nullptr if you don't use the base's pointer
      sde_model_(sde_model),
      num_paths_(num_paths), num_steps_(num_steps)
    {}

    R price() const override {
        // 1. Prepare initial state (dim = SdeType::STATE_DIM)
        StoringVector x0 = sde_model_.m_x0;

        // 2. Construct solver using concrete SDE model
        Solver solver(sde_model_);

        // 3. Simulate paths
        auto all_paths = solver.solve(x0, 0.0, this->ttm_, num_steps_, num_paths_);

        std::cout << "Simulated " << num_paths_ << " paths with " << num_steps_ << " steps each." << std::endl;
        std::cout << "Paths shape: " << all_paths << std::endl;
        

        // Extract last column (terminal values for all paths and all state dims)
        StoringVector terminal_column = all_paths.col(num_steps_);

        // Prepare result vector
        StoringVector terminal_logS(num_paths_);

        // Vectorized path-wise slice: handle both cases
        if constexpr (SdeType::STATE_DIM == 1) {
            // Direct copy: all rows are from the only state dimension
            terminal_logS = terminal_column;
        } else {
            // Stride: take every STATE_DIM-th entry starting at index 1
            terminal_logS = Eigen::Map<const StoringVector, 0, Eigen::InnerStride<>>( 
                terminal_column.data() + 1, // start at second state dim
                num_paths_,                  // number of entries
                Eigen::InnerStride<>(SdeType::STATE_DIM) // stride through paths
            );
        }

        std::cout << "Terminal log-prices shape: " << terminal_logS.size() << std::endl;
        std::cout << "First 5 terminal log-prices: " << terminal_logS << std::endl;


        // Vectorized payoff evaluation (must implement batch-payoff API)
        auto payoffs = this->payoff_->evaluate_from_log(terminal_logS);

        std::cout << "Payoffs shape: " << payoffs.size() << std::endl;
        std::cout << "First 5 payoffs: " << payoffs.head(5).transpose() << std::endl;

        return std::exp(-this->rate_ * this->ttm_) * payoffs.mean();
    }

private:
    SdeType sde_model_;                 // concrete SDE, not pointer!
    unsigned int num_paths_;
    unsigned int num_steps_;
};

} // namespace options

#endif // MC_OPTION_PRICER_HPP