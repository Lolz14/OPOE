/**
 * @file OPEOptionPricer.hpp
 * @brief Defines the OPEOptionPricer class for pricing options using Orthogonal Polynomial Expansion (OPE).
 *
 * This header provides the implementation of the OPEOptionPricer class template, which inherits from BaseOptionPricer.
 * The class uses orthogonal polynomial expansions and mixture densities to efficiently price options under stochastic volatility models.
 * It supports flexible polynomial bases, customizable quadrature integration, and leverages quantization grids for path discretization.
 *
 * Dependencies:
 * - BaseOptionPricer.hpp: Base class for option pricing.
 * - SDE.hpp: Interface for stochastic differential equation models.
 * - Utils.hpp: Utility functions for enumerating polynomial bases and other operations.
 * - MixtureDensity.hpp: For handling mixture densities in polynomial expansions.
 * - QuadratureRuleHolder.hpp: For configurable quadrature integration methods.
 * 
 * @section Features
 * - Constructs an orthonormal polynomial basis for the expansion.
 * - Computes the generator matrix and propagates it via matrix exponentiation.
 * - Integrates the expected payoff using a configurable quadrature rule.
 * - Supports custom SDE solvers and stochastic volatility models.
 * - Outputs intermediate results and integrand values for debugging and analysis.
 *
 * @section Usage
 * Instantiate OPEOptionPricer with the desired template parameters, provide the required SDE model, payoff, and solver function.
 * Call the price() method to compute the discounted expected payoff.
 *
 * @section Example
 * @code
 * auto pricer = options::OPEOptionPricer<>(ttm, strike, rate, payoff, sde_model, solver_func);
 * double option_price = pricer.price();
 * @endcode
 *
 * @see BaseOptionPricer
 * @see stats::MixtureDensity
 * @see quadrature::QuadratureRuleHolder
 */
#ifndef OPTION_PRICER_HPP
#define OPTION_PRICER_HPP

#include "../utils/FileReader.hpp"
#include "../utils/Utils.hpp"
#include "../quadrature/Projector.hpp"
#include "../quadrature/QuadratureRuleHolder.hpp"
#include "../stats/MixtureDensity.hpp"
#include "../traits/OPOE_traits.hpp"
#include "BaseOptionPricer.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <variant>
#include <stdexcept>
#include <string>
#include <utility>
#include <cmath>
#include <execution>

namespace options {

/**
 * @brief OPEOptionPricer class for pricing options using Orthogonal Polynomial Expansion (OPE).
 *
 * This class implements the OPE method for option pricing, leveraging orthogonal polynomial expansions and mixture densities.
 * It supports flexible polynomial bases and integrates the expected payoff using configurable quadrature rules.
 *
 * @tparam R The floating-point type used for calculations (default: traits::DataType::PolynomialField).
 * @tparam PolynomialBaseDegree The degree of the polynomial basis (must be >= 2).
 */

template <typename R = traits::DataType::PolynomialField,
unsigned int PolynomialBaseDegree = 3>
requires (PolynomialBaseDegree >= 2 && std::is_integral_v<decltype(PolynomialBaseDegree)>) 
// Ensure the polynomial degree is at least 2, since approximation of the wiener motion with only 1 point is trivial (0, with 100% probability);
// Moreover, PolynomialBaseDegree must be an integral type
class OPEOptionPricer : public BaseOptionPricer<R> {
    static_assert(PolynomialBaseDegree >= 2, 
                  "PolynomialBaseDegree must be >= 2");

    using StoringVector = traits::DataType::StoringVector;
    using StoringMatrix = traits::DataType::StoringMatrix;
    using Array = traits::DataType::StoringArray;
    using SolverFunc = std::function<StoringMatrix(
        R, R, int, int, const std::optional<StoringMatrix>& dW_opt)>;

    using Base = BaseOptionPricer<R>;

    using DensityType = stats::BoostBaseDensity<
        boost::math::normal_distribution<R>, R, R, R>;

public:
/**
 * @brief Constructs an OPEOptionPricer instance.
 *
 * @param ttm Time to maturity.
 * @param strike Strike price of the option.
 * @param rate Risk-free interest rate.
 * @param payoff Payoff function for the option.
 * @param sde_model Shared pointer to the stochastic differential equation model.
 * @param solver_func Function to solve the SDE, returning a matrix.
 * @param integrator_param Quadrature method for integration (default: TanhSinh).
 * @param num_paths Number of paths for simulation (default: 10).
 *
 * @throws std::invalid_argument if PolynomialBaseDegree < 2.
 * @throws std::runtime_error if the polynomial basis cannot be constructed.
 *
 * This constructor initializes the OPEOptionPricer with the provided parameters,
 * constructs the orthonormal polynomial basis, and prepares the density object for pricing.
 */
    OPEOptionPricer(R ttm, R strike, R rate,
                    std::unique_ptr<IPayoff<R>> payoff,
                    std::shared_ptr<SDE::GenericSVModelSDE<R>> sde_model,
                    SolverFunc solver_func,
                    traits::QuadratureMethod integrator_param = traits::QuadratureMethod::TanhSinh,
                    unsigned int num_paths = 10)
        : Base(ttm, strike, rate, std::move(payoff), sde_model)
        , num_paths_(num_paths)
        , solver_func_(solver_func)
        , density_object_(make_density_object(ttm, num_paths, PolynomialBaseDegree,
                                              solver_func_, sde_model))
        , integrator_(integrator_param)
                                            
    {
        density_object_.constructOrthonormalBasis();
        density_object_.constructQProjectionMatrix();    
    }

    /**
     * @brief Computes the option price using the OPE method.
     *
     * This method calculates the expected value of the option payoff at maturity,
     * integrating over the log-spot price using the orthonormal polynomial basis.
     *
     * @return The computed option price.
     */
    R price() const override {
        // 1. Get the generator matrix H and the polynomial basis
        const auto& H = density_object_.getHMatrix();     // (N+1)x(N+1)
        const auto E  = Utils::enumerate_basis(PolynomialBaseDegree);
        const unsigned int M   = static_cast<int>(E.size());

        const auto mean = density_object_.mean();
        const auto stddev = std::sqrt(density_object_.variance());
        

        auto* gen_sde_model = dynamic_cast<SDE::GenericSVModelSDE<R>*>(this->sde_model_.get());
        const auto X0 = gen_sde_model->get_x0();  // Initial value
        const auto V0 = gen_sde_model->get_v0();  // Initial volatility
        
        StoringMatrix G = gen_sde_model->generator_G(E, H);         // MxM

        StoringVector h_vals = Utils::build_h_vals<R>(H, E, PolynomialBaseDegree, X0, V0);

        // Matrix exponential
        StoringMatrix expGT = (this->get_ttm() * G).exp();

        // 2. Build sparse selector matrix S (M x (N+1)), columns = e_{pi(0,n)}
        StoringMatrix S = StoringMatrix::Zero(M, PolynomialBaseDegree+1);
        for (unsigned int i = 0; i < M; ++i) {
            if (E[i].first == 0 && E[i].second <= PolynomialBaseDegree) {
                S(i, E[i].second) = 1.0;
            }
        }

        // 3. Compute all l_n in one shot: l = h_vals^T * expGT * S
        StoringVector l_values = (h_vals.transpose() * expGT * S).transpose();

        // Optional: debug output
        std::cout << "Hmatrix:\n" << H << "\n";
        std::cout << "h_vals: " << h_vals.transpose() << "\n";
        std::cout << "l_values: " << l_values.transpose() << "\n";

        
        // 4. Get integration domain 
        //stats::DensityInterval<R> support = this->density_object_.getSupport();

        auto h_functions = density_object_.getHFunctionsMixture();

        // Integrand(x) = Payoff_log(x) * [ Σ_{n=0 to N} ℓ_n * H_n(x) ] * w(x)
        // where x is the log_spot.

        auto full_integrand_function =

            [this, &h_functions, &l_values](R log_spot_price) -> R {

            // Calculate Σ_{n=0 to N} ℓ_n * H_n(log_spot_price)
            R auxiliary_density_value = static_cast<R>(0.0);

            for (unsigned int n = 0; n <= PolynomialBaseDegree; ++n) {

                // Ensure the H_n function is valid before calling
                if (!h_functions[n]) {
                        throw std::runtime_error("Invalid (null) H_n function at index " + std::to_string(n) + ".");
                }

                auxiliary_density_value += l_values[n] * h_functions[n](log_spot_price);
            
            }


            // Full term: Payoff(log_x) * (Σ ℓ_n H_n(log_x)) * w(log_x)
            return  auxiliary_density_value * this->payoff_->evaluate_from_log(log_spot_price)*this->density_object_.pdf(log_spot_price);

        };


        // 5. Perform numerical integration using the member integrator_
        R expected_value_at_maturity = this->integrator_.integrate(
            full_integrand_function,
            mean - 8*stddev, // Assuming this is the lower bound
            mean + 8*stddev // Assuming this is the upper bound
        );

    
        // 6. Discount the expected value back to today
        return std::exp(-this->rate_ * this->ttm_) * expected_value_at_maturity;

    }

private:
    unsigned int num_paths_;
    SolverFunc solver_func_;
    stats::MixtureDensity<PolynomialBaseDegree, DensityType, R> density_object_;
    quadrature::QuadratureRuleHolder<R> integrator_;


    /**
     * @brief Constructs the density object for the OPEOptionPricer.
     *
     * This method reads the quantization grid, solves for paths using the provided solver function,
     * and builds the mixture density object for polynomial expansion.
     * It computes the mean and variance of the paths and constructs the densities.
     *
     * @param ttm Time to maturity.
     * @param num_paths Number of paths for simulation.
     * @param poly_deg Degree of the polynomial basis.
     * @param solver_func Function to solve the SDE and return paths.
     * @param sde_model Shared pointer to the stochastic differential equation model.
     * @return A stats::MixtureDensity object containing the computed densities.
     * @throws std::runtime_error if the quantization grid cannot be read or if the solver function fails.
     *
     * This method is crucial for setting up the polynomial basis and densities used in the OPE method.
     */
    static stats::MixtureDensity<PolynomialBaseDegree, DensityType, R>
    make_density_object(R ttm,
                        unsigned int num_paths,
                        unsigned int poly_deg,
                        const SolverFunc& solver_func,
                        const std::shared_ptr<SDE::GenericSVModelSDE<R>>& sde_model)
    {
        // Read quantization grid
        QuantizationGrid<R> grid =
            readQuantizationGrid<R>(num_paths, poly_deg, "include/quantized_grids");

        // Double the coordinates for dw
        StoringMatrix dw(grid.coordinates.rows() * 2, grid.coordinates.cols()); 
        dw << grid.coordinates, grid.coordinates;

        // Solve for paths
        auto paths = solver_func(0.0, ttm, poly_deg, num_paths, std::optional<StoringMatrix>(dw));

        // Map vol_view
        Eigen::Map<const StoringMatrix, 0, Eigen::OuterStride<>> vol_view(
            paths.data(),
            num_paths,
            paths.cols(),
            Eigen::OuterStride<>(2 * paths.outerStride())
        );

        // Compute mean and variance
        StoringVector mean = sde_model->M_T(ttm, ttm / poly_deg, vol_view, grid.coordinates);
        StoringVector variance = sde_model->C_T(ttm, ttm / poly_deg, vol_view);

        std::cout << "Mean: " << mean.transpose() << std::endl;
        std::cout << "Variance: " << variance.transpose() << std::endl;
        std::cout << "Weights: " << grid.weights.transpose() << std::endl;

        // Build densities
        std::vector<DensityType> densities(mean.size());
        std::transform(std::execution::unseq,
                       mean.data(), mean.data() + mean.size(),
                       variance.data(),
                       densities.begin(),
                       [](R m, R v) {
                           return stats::make_normal_density<R>(m, std::sqrt(v));
                       });

        // Convert Eigen weights to std::vector
        std::vector<R> weights(grid.weights.data(),
                               grid.weights.data() + grid.weights.size());

        // Build and return MixtureDensity
        return stats::MixtureDensity<PolynomialBaseDegree, DensityType, R>(
            std::move(weights),
            std::move(densities)
        );
    }
};

} // namespace options

#endif // OPTION_PRICER_HPP
