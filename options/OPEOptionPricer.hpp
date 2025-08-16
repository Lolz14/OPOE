#ifndef OPTION_PRICER_HPP
#define OPTION_PRICER_HPP

#include "../utils/FileReader.hpp"
#include "../utils/Utils.hpp"
#include "../quadrature/Projector.hpp"
#include "../quadrature/QuadratureRuleHolder.hpp"
#include "../stats/MixtureDensity.hpp"
#include "../traits/OPOE_traits.hpp"
#include "Payoff.hpp"
#include "BaseOptionPricer.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <variant>
#include <stdexcept>
#include <string>
#include <utility>
#include <cmath>
#include <execution>

namespace options {

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
    OPEOptionPricer(R ttm, R strike, R rate,
                    std::unique_ptr<IPayoff<R>> payoff,
                    std::shared_ptr<SDE::GenericSVModelSDE<R>> sde_model,
                    SolverFunc solver_func,
                    unsigned int num_paths = 4)
        : Base(ttm, strike, rate, std::move(payoff), sde_model)
        , num_paths_(num_paths)
        , solver_func_(solver_func)
        , density_object_(make_density_object(ttm, num_paths, PolynomialBaseDegree,
                                              solver_func_, sde_model))
    {
        density_object_.constructOrthonormalBasis();
        density_object_.constructQProjectionMatrix();    
    }

    R price() const override {
        auto H = density_object_.getHMatrix();

        auto E = Utils::enumerate_basis(PolynomialBaseDegree);
        const int M = static_cast<int>(E.size());
        auto variance = density_object_.variance();

        auto gen_sde_model = dynamic_cast<SDE::GenericSVModelSDE<R>*>(this->sde_model_.get());

        auto G_matrix = gen_sde_model->generator_G(E, H);

        // Build vector of monomials [1, X0, X0^2, ..., X0^N]
        StoringVector monoms(PolynomialBaseDegree+1);
        monoms(0) = 1.0;
        for (unsigned int k = 1; k <= PolynomialBaseDegree; ++k) {
            monoms(k) = monoms(k-1) * gen_sde_model->get_x0();
        }

        // Evaluate Hermites at X0 using your H matrix
        StoringVector Hvec = H.transpose() * monoms; // size N+1
        std::cout << "Hmatrix: " << H << std::endl;
        std::cout << "Monomials: " << monoms.transpose() << std::endl;
        std::cout << "Hvec: " << Hvec.transpose() << std::endl;

        // Build h_vals = v^m * H_n(X0)
        StoringVector h_vals(M);
        for (int i = 0; i < M; ++i) {
            auto [m, n] = E[i];
            h_vals(i) = std::pow(gen_sde_model->get_v0(), m) * Hvec(n);
        }

        std::cout << "h_vals: " << h_vals.transpose() << std::endl;
  

        // Precompute dense exp(T * G)
        StoringMatrix expGT = (this->get_ttm() * G_matrix).exp();

        // Precompute index mapping for (0,n)
        std::vector<int> idx0n(PolynomialBaseDegree + 1, -1);
        for (int i = 0; i < M; ++i) {
            if (E[i].first == 0) idx0n[E[i].second] = i;
        }

        // Compute l_n = h_vals^T * expGT * e_{pi(0,n)}
        std::vector<double> l_values;
        l_values.reserve(PolynomialBaseDegree + 1);

        for (unsigned int n = 0; n <= PolynomialBaseDegree; ++n) {
            const int j = idx0n[n];
            if (j < 0) {
                l_values.push_back(0.0);
                continue;
            }
            // expGT.col(j) is exactly expGT * e_j
            double ln_val = h_vals.dot(expGT.col(j));
            l_values.push_back(ln_val);
        }

        std::cout << "l_values: ";
        for (const auto& val : l_values) {
            std::cout << val << " ";
        }
        std::cout << std::endl;



    }

private:
    unsigned int num_paths_;
    SolverFunc solver_func_;
    stats::MixtureDensity<PolynomialBaseDegree, DensityType, R> density_object_;

    // Static helper: builds the MixtureDensity before entering constructor body
    static stats::MixtureDensity<PolynomialBaseDegree, DensityType, R>
    make_density_object(R ttm,
                        unsigned int num_paths,
                        unsigned int poly_deg,
                        const SolverFunc& solver_func,
                        const std::shared_ptr<SDE::GenericSVModelSDE<R>>& sde_model)
    {
        // Read quantization grid
        QuantizationGrid<R> grid =
            readQuantizationGrid<R>(num_paths, poly_deg, "quantized_grids");

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
}




namespace options {

template<
    traits::OptionType OptionTypeValue,
    unsigned int PolynomialBaseDegree,
    typename DensityKernelType,
    typename R = double
>
class AckererOptionPricer {
public:
    using PayoffVariant = std::variant<EuropeanCallPayoff<R>, EuropeanPutPayoff<R>>;

protected:
    R ttm_;
    R rate_;
    PayoffVariant payoff_;
    stats::MixtureDensity<PolynomialBaseDegree, DensityKernelType, R> density_object_;
    quadrature::QuadratureRuleHolder<R> integrator_;

public:
    explicit AckererOptionPricer(
        R strike,
        R ttm_val,
        R rate_val,
        stats::MixtureDensity<PolynomialBaseDegree, DensityKernelType, R> density_obj_param,
        traits::QuadratureMethod integrator_param = traits::QuadratureMethod::QAGI,
        R tolerance_param = 1e-8
    ) : ttm_(ttm_val),
        rate_(rate_val),
        payoff_(
            (OptionTypeValue == traits::OptionType::Call)
            ? PayoffVariant(std::in_place_type<EuropeanCallPayoff<R>>, strike)
            : PayoffVariant(std::in_place_type<EuropeanPutPayoff<R>>, strike)
        ),
        density_object_(std::move(density_obj_param)),
        integrator_(integrator_param)
    {

    
        if (strike < static_cast<R>(0.0)) {

                throw std::invalid_argument("Strike price cannot be negative. Received: " + std::to_string(strike));

        }

        if (ttm_ < static_cast<R>(0.0)) {

            throw std::invalid_argument("Time to maturity cannot be negative. Received: " + std::to_string(ttm_));

        }

        // Consider if rate_val should also be checked (e.g., non-negative)



        static_assert(OptionTypeValue == traits::OptionType::Call || OptionTypeValue == traits::OptionType::Put,

                        "Unsupported option type provided as template parameter.");
        density_object_.constructOrthonormalBasis();
        density_object_.constructQProjectionMatrix();

    }

    virtual ~AckererOptionPricer() = default;

    virtual [[nodiscard]] R price() const {
            auto h_functions = density_object_.getHFunctionsMixture();

            std::cout << "H matrix#" << density_object_.getHMatrix() << std::endl;

           

        
   

            

        

       

            // Integrand(x) = Payoff_log(x) * [ Σ_{n=0 to N} ℓ_n * H_n(x) ] * w(x)

            // where x is typically log_spot.

            auto full_integrand_function =

                [this, &h_functions](R log_spot_price) -> R {

       

                // Calculate Σ_{n=0 to N} ℓ_n * H_n(log_spot_price)

                R auxiliary_density_value = static_cast<R>(0.0);

                for (unsigned int n = 0; n <= PolynomialBaseDegree; ++n) {

                    // Ensure the H_n function is valid before calling

                    if (!h_functions[n]) {

                         throw std::runtime_error("Invalid (null) H_n function at index " + std::to_string(n) + ".");

                    }

                    auxiliary_density_value += h_functions[n](log_spot_price)*this->density_object_.pdf(log_spot_price);

                }

       

                // Get Payoff(log_spot_price)

                R payoff_value = std::visit(

                    [log_spot_price](const auto& concrete_payoff_object) {

                        // Assuming Payoff objects have evaluate_from_log(log_spot_price)

                        return concrete_payoff_object.evaluate_from_log(log_spot_price);

                    },

                    this->payoff_

                );

       

                // Get auxiliary density w(log_spot_price)

       

                // Full term: Payoff(log_x) * (Σ ℓ_n H_n(log_x)) * w(log_x)

                return  payoff_value*auxiliary_density_value;

            };

            std::cout << "Full integrand function created successfully." << std::endl;

            std::cout << "Integrand function: " << full_integrand_function(0.5) << std::endl; // Test with log_spot_price = 0.0

       

            // 4. Get integration domain (presumably for log_spot_price)

            stats::DensityInterval<R> support = this->density_object_.getSupport();

       

            // 5. Perform numerical integration using the member integrator_

            R expected_value_at_maturity = this->integrator_.integrate(

                full_integrand_function,

                support.lower, // Assuming this is the lower bound

                support.upper // Assuming this is the upper bound

            );

       

            // 6. Discount the expected value back to today

            return std::exp(-this->rate_ * this->ttm_) * expected_value_at_maturity;

        }
};

template<
    traits::OptionType OptionTypeValue,
    unsigned int PolynomialBaseDegree,
    typename R = double
>
class GammaOptionPricer : public AckererOptionPricer<OptionTypeValue, PolynomialBaseDegree, stats::GammaDensity, R> {
public:
    using Base = AckererOptionPricer<OptionTypeValue, PolynomialBaseDegree, stats::GammaDensity, R>;
    using StoringMatrix = traits::DataType::StoringMatrix;

    explicit GammaOptionPricer(
        R strike,
        R ttm_val,
        R rate_val,
        stats::MixtureDensity<PolynomialBaseDegree, stats::GammaDensity, R> density_obj_param,
        traits::QuadratureMethod integrator_param = traits::QuadratureMethod::QAGI,
        R tolerance_param = 1e-8
    ) : Base(strike, ttm_val, rate_val, std::move(density_obj_param), integrator_param, tolerance_param) {}

    R price() const override {
        const auto& components = this->density_object_.getComponents();
        size_t num_components = components.size();
        StoringMatrix f_n(num_components, PolynomialBaseDegree + 1);
        f_n.setZero();

        // Extract necessary parameters
        R strike_price = std::visit([](const auto& payoff) { return payoff.getStrike(); },this->payoff_);
        R strike_log = std::log(strike_price);
        const auto& weights = this->density_object_.getWeights();
        const auto& Qmatrices = this->density_object_.getQProjectionMatrix();

        for (size_t i = 0; i < num_components; ++i) {
            R alpha = components[i].getShape();
            R beta = 1 / components[i].getScale();

            R eta = components[i].getDomain().upper>0 ? components[i].getDomain().lower : components[i].getDomain().upper; // Assume eta computation as per context (depends on asset)
            
            for (unsigned int n = 0; n <= PolynomialBaseDegree; ++n) {
                f_n(i, n) = compute_f_n(i, n, eta, strike_log, alpha, beta);
            }
        }

        std::cout << "Computed f_n matrix: " << f_n << std::endl;

        R sum = 0.0;
        for (size_t i = 0; i < num_components; ++i) {
            sum += weights[i] * Qmatrices[i].col(PolynomialBaseDegree).dot(f_n.row(i));
        }
        


        return sum; // Placeholder for logic.
    }
    R compute_I_n(unsigned int n, R nu, R alpha, R mu) const {
        if (n == 0) {
            // Base case: I^(α-1)_0
            return compute_I_alpha_0(mu, nu, alpha);
        } else if (n == 1) {
            // Base case: I^(α-1)_1
            return alpha * compute_I_alpha_0(mu, nu, alpha) + compute_I_alpha_0(mu, nu, alpha + 1);
        } else {
            // Recursive definition for n ≥ 2
            R term1 = (2 + (alpha - 2) / n) * compute_I_n(n - 1, nu, alpha, mu);
            R term2 = (1 + (alpha - 2) / n) * compute_I_n(n - 2, nu, alpha, mu);
            R term3 = (1.0 / n) * (compute_I_n(n - 1, nu, alpha + 1, mu) - compute_I_n(n - 2, nu, alpha + 1, mu));
            return term1 - term2 - term3;
        }
    }

    // Computes payoff coefficients f_n using the recursive results from I^(α-1)_n
    R compute_f_n(unsigned int k, unsigned int n, R eta, R strike_log, R alpha, R beta) const {
        R mu = std::max(static_cast<R>(0), beta * (strike_log - eta));
        R coeff1 = std::sqrt(std::tgamma(n + 1) / (std::tgamma(alpha + n))) * 1 / std::tgamma(alpha);
        R term1 = std::exp(eta) * compute_I_n(n, beta, alpha, mu);
        R term2 = std::exp(strike_log) * compute_I_n(n, beta, alpha, 0.0);
        //std::cout << "Computed f_n for k=" << k << ", n=" << n << ": iN=" << compute_I_n(n, 1/beta, alpha, mu) << std::endl;
        return coeff1 * (term1 + term2);
    }

private:
    // Approximation of the upper incomplete gamma function (Γ(α, x))
    static R gamma_upper(R alpha, R x) {
        if (x == 0) return std::tgamma(alpha); // Γ(α, 0) = Γ(α)
        return std::tgamma(alpha) - boost::math::gamma_p<R, R>(alpha, x); // Difference Γ(α) - lower Γ(α, x)
    }

    // Compute I^(α)_0(μ; ν) as used in the recursive calculation
    R compute_I_alpha_0(R mu, R nu, R alpha) const {

        std::cout << "Computing I_alpha_0 with mu: " << mu << ", nu: " << nu << ", alpha: " << alpha << std::endl;
        std::cout << "Gamma upper: " << gamma_upper(alpha, mu * (1 - nu)) << std::endl;
        std::cout << "Computed I_alpha_0: " <<  std::pow(1 - nu, -alpha)  << std::endl;
        return std::pow(1 - nu, -alpha) * std::tgamma(alpha) * gamma_upper(alpha, mu * (1 - nu));
    }

};

} // namespace OPOE

#endif // OPTION_PRICER_HPP
