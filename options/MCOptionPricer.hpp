#ifndef MC_OPTION_PRICER_HPP
#define MC_OPTION_PRICER_HPP

#include "BaseOptionPricer.hpp"
#include "../sde/SDE.hpp"
#include <memory>
#include <Eigen/Dense>
#include <iostream>
#include <functional>

namespace options {

template<typename R>
class MCPricer : public BaseOptionPricer<R> {
public:
    using StoringVector = traits::DataType::StoringVector;
    using StoringMatrix = traits::DataType::StoringMatrix;

    
    using SolverFunc = std::function<StoringMatrix(
    R, R, int, int)>;

    MCPricer(R ttm, R strike, R rate,
             std::unique_ptr<IPayoff<R>> payoff,
             std::shared_ptr<SDE::ISDEModel<R>> sde_model,
             SolverFunc solver_func,
             unsigned int num_paths = 10,
             unsigned int num_steps = 3)
        : BaseOptionPricer<R>(ttm, strike, rate, std::move(payoff), sde_model),
          solver_func_(std::move(solver_func)),
          num_paths_(num_paths),
          num_steps_(num_steps)
    {}

    R price() const override {

        auto all_paths = solver_func_(0.0, this->ttm_, num_steps_, num_paths_);

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
