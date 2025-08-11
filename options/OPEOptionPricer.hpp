#ifndef OPTION_PRICER_HPP
#define OPTION_PRICER_HPP

#include "../utils/FileReader.hpp"
#include "../quadrature/Projector.hpp"
#include "../quadrature/QuadratureRuleHolder.hpp"
#include "../stats/MixtureDensity.hpp"
#include "../traits/OPOE_traits.hpp"
#include "Payoff.hpp"
#include "BaseOptionPricer.hpp"
#include <variant>
#include <stdexcept>
#include <string>
#include <utility>
#include <cmath>

namespace options {

template <typename R = traits::DataType::PolynomialField,
unsigned int PolynomialBaseDegree>
class OPEPricer : public BaseOptionPricer<R> {

    using StoringVector = traits::DataType::StoringVector;
    using StoringMatrix = traits::DataType::StoringMatrix;
    using Array = traits::DataType::StoringArray;

    using Base = BaseOptionPricer<R>;
    
    public:
        OPEPricer(R ttm, R strike, R rate,
                std::unique_ptr<IPayoff<R>> payoff,
                std::shared_ptr<SDE::GenericSVModelSDE<R>> sde_model,
                unsigned int num_paths = 100
                )
            : Base(ttm, strike, rate, std::move(payoff), std::move(sde_model)),
                num_paths_(num_paths) {}

        void buildMixtureDensity() {
            // Build the mixture density Weighted-MC simulation described in the paper
            // "Option Pricing with the Orthonormal Polynomial Expansion Method" by Ackerer et al.
                QuantizationGrid<R> grid = readQuantizationGrid<R>(num_paths_, PolynomialBaseDegree, "quantized_grids");

                StoringMatrix dw(grid.coordinates.rows()*2, grid.coordinates.cols()); 
                dw << grid.coordinates, grid.coordinates;


            


           
        }
        
            
    
    private:
        unsigned int num_paths_;

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

           

        
   

            if (h_functions.size() != PolynomialBaseDegree + 1) {

                throw std::runtime_error("Number of H_n functions (" + std::to_string(h_functions.size()) +

                                         ") does not match PolynomialBaseDegree+1 (" + std::to_string(PolynomialBaseDegree + 1) + ").");

            }

         

       

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
