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
using StoringVector = traits::DataType::StoringVector;
using StoringMatrix = traits::DataType::StoringMatrix;
using Array = traits::DataType::StoringArray; 
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
requires (PolynomialBaseDegree >= 2) 
// Ensure the polynomial degree is at least 2, since approximation of the wiener motion with only 1 point is trivial (0, with 100% probability);
// Moreover, PolynomialBaseDegree must be an integral type
class OPEOptionPricer : public BaseOptionPricer<R> {
    static_assert(PolynomialBaseDegree >= 2, 
                  "PolynomialBaseDegree must be >= 2");

    using SolverType = traits::SolverType;


    using Base = BaseOptionPricer<R>;

    using DensityType = stats::BoostBaseDensity<
        boost::math::normal_distribution<R>, R, R, R>;

public:
/**
 * @brief Constructs an OPEOptionPricer instance.
 *
 * @param ttm Time to maturity.
 * @param rate Risk-free interest rate.
 * @param payoff Payoff function for the option.
 * @param sde_model Shared pointer to the stochastic differential equation model.
 * @param K Number of mixture componennts
 * @param solving_param Enum representing either the direct method, with the payoff projections computed by recursion, or the integration method (default: Integration).
 * @param solver_type Enum that represents the solver type (default: EulerMaruyama).
 * @param integrator_param Quadrature method for integration (default: TanhSinh).
 * @param num_paths Number of paths for simulation (default: 10).
 *
 * @throws std::invalid_argument if PolynomialBaseDegree < 2.
 * @throws std::runtime_error if the polynomial basis cannot be constructed.
 *
 * This constructor initializes the OPEOptionPricer with the provided parameters,
 * constructs the orthonormal polynomial basis, and prepares density object for pricing.
 */
    OPEOptionPricer(R ttm, R rate,
                    std::shared_ptr<IPayoff<R>> payoff,
                    std::shared_ptr<SDE::ISDEModel<R>> sde_model,
                    unsigned int K,
                    traits::OPEMethod solving_param = traits::OPEMethod::Integration,
                    SolverType solver_type = SolverType::EulerMaruyama,
                    traits::QuadratureMethod integrator_param = traits::QuadratureMethod::TanhSinh,
                    unsigned int num_paths = 10)
        : Base(ttm, rate, std::move(payoff), sde_model)
        , K_(K)
        , num_paths_(num_paths)
        , solver_type_(solver_type)
        , density_object_(make_density_object(ttm, num_paths, K,
                                              solver_type_, sde_model))
        , integrator_(integrator_param)
        , solving_param_(solving_param)
                                            
    {
        density_object_.constructOrthonormalBasis();
        density_object_.constructQProjectionMatrix();    
    }

    /**
     * @brief Computes the option price using the OPE (Orthogonal Polynomial Expansion) method.
     *
     * The method computes the discounted expected payoff of an option at maturity:
     *
     *   Price = e^{-rT} * E_Q[ Payoff(S_T) ]
     *
     * Depending on the OPE method chosen, this is done either:
     *   - via direct polynomial projection (fast, closed form), or
     *   - via numerical integration against the orthogonal polynomial expansion.
     *
     * @return The computed option price (call or put).
     */
    R price() const override {
        // Storage for final option price
        R option_price = static_cast<R>(0);

        // Precompute polynomial moments (ℓ_n coefficients)
        const auto l_values = poly_moments();

        // ==============================
        // 1. Direct projection method
        // ==============================
        if (solving_param_ == traits::OPEMethod::Direct) {
            const auto f_N = direct_projection();
  
            return adjustPutCall(f_N.dot(l_values));
        }

        // ==============================
        // 2. Numerical integration method
        // ==============================
        const auto& h_functions = density_object_.getHFunctionsMixture();
        const R mean   = density_object_.mean();
        const R stddev = std::sqrt(density_object_.variance());

        // --- Define equivalent payoff once (avoid constructing inside integrand loop)
        std::shared_ptr<IPayoff<R>> eq_payoff;
        if (this->payoff_->type() == traits::OptionType::Put) {
            eq_payoff = std::make_shared<EuropeanCallPayoff<R>>(this->payoff_->getStrike());
        } else {
            eq_payoff = this->payoff_->clone();
        }

        // --- Define integrand function:
        // Integrand(x) = Payoff_log(x) * [ Σ ℓ_n * H_n(x) ] * pdf(x)
        auto integrand = [&, this](R log_spot) -> R {
            R density_expansion_val = static_cast<R>(0);

            // Compute Σ ℓ_n * H_n(log_spot)
            for (unsigned int n = 0; n <= PolynomialBaseDegree; ++n) {
                if (!h_functions[n]) {
                    throw std::runtime_error("Invalid H_n function at index " + std::to_string(n));
                }
                density_expansion_val += l_values[n] * h_functions[n](log_spot);
            }

            // Multiply by payoff and true density
            return density_expansion_val *
                eq_payoff->evaluate_from_log(log_spot) *
                this->density_object_.pdf(log_spot);
        };

        // --- Numerical integration over a wide interval (±12 stddevs)
        const R expected_value = this->integrator_.integrate(
            integrand,
            mean - 12 * stddev,
            mean + 12 * stddev
        );

        // --- Discount result
        option_price = std::exp(-this->rate_ * this->ttm_) * expected_value;

        return adjustPutCall(option_price);
    }

    
private:
    unsigned int K_;
    unsigned int num_paths_;
    SolverType solver_type_;
    stats::MixtureDensity<PolynomialBaseDegree, DensityType, R> density_object_;
    quadrature::QuadratureRuleHolder<R> integrator_;
    traits::OPEMethod solving_param_;

    /**
     * @brief Constructs the mixture density object used in OPEOptionPricer.
     *
     * Steps:
     *   1. Reads the quantization grid (nodes + weights).
     *   2. Generates driving noise increments (dw).
     *   3. Simulates paths using the selected SDE solver.
     *   4. Extracts volatility paths and computes conditional means and variances.
     *   5. Builds a Gaussian mixture density from means/variances with corresponding weights.
     *
     * @param ttm         Time to maturity.
     * @param num_paths   Number of simulated paths.
     * @param n_comp      Number of quantization components (≤ 10).
     * @param solver_type Which numerical SDE solver to use.
     * @param sde_model   Shared pointer to the SDE model.
     *
     * @return A stats::MixtureDensity object containing the Gaussian mixture.
     *
     * @throws std::runtime_error If n_comp > 10, solver type unknown, 
     *         the grid cannot be read, or if the model is not a stochastic volatility model.
     */
    static stats::MixtureDensity<PolynomialBaseDegree, DensityType, R>
    make_density_object(R ttm,
                        unsigned int num_paths,
                        unsigned int n_comp,
                        SolverType solver_type,
                        const std::shared_ptr<SDE::ISDEModel<R>>& sde_model)
    {
        if (n_comp > 10) {
            throw std::runtime_error("Mixture components must be ≤ 10.");
        }

        // ==============================
        // 1. Read quantization grid
        // ==============================
        QuantizationGrid<R> grid =
            readQuantizationGrid<R>(num_paths, n_comp, "include/quantized_grids");

        // ==============================
        // 2. Build doubled coordinates for dw
        //    (needed for symmetric noise increments)
        // ==============================
        StoringMatrix dw(grid.coordinates.rows() * 2, grid.coordinates.cols());
        dw << grid.coordinates, grid.coordinates;

        // ==============================
        // 3. Solve paths with chosen SDE solver
        // ==============================
        auto solve_paths = [&](auto&& solver) {
            return solver.solve(0.0, ttm, n_comp, num_paths, std::optional<StoringMatrix>(dw));
        };

        StoringMatrix paths;
        switch (solver_type) {
            case SolverType::EulerMaruyama:
                paths = solve_paths(SDE::EulerMaruyamaSolver<SDE::ISDEModel<R>, R>(*sde_model));
                break;
            case SolverType::Milstein:
                paths = solve_paths(SDE::MilsteinSolver<SDE::ISDEModel<R>, R>(*sde_model));
                break;
            case SolverType::IJK:
                paths = solve_paths(SDE::InterpolatedKahlJackelSolver<SDE::ISDEModel<R>, R>(*sde_model));
                break;
            default:
                throw std::runtime_error("Unknown solver type.");
        }

        // ==============================
        // 4. Extract volatility view from simulated paths
        //    Using Eigen::Map for zero-copy slicing
        // ==============================
        Eigen::Map<const StoringMatrix, 0, Eigen::OuterStride<>> vol_view(
            paths.data(),
            num_paths,
            paths.cols(),
            Eigen::OuterStride<>(2 * paths.outerStride())
        );

        // ==============================
        // 5. Compute conditional mean and variance from SV model
        // ==============================
        auto* sv_model = dynamic_cast<SDE::GenericSVModelSDE<R>*>(sde_model.get());
        if (!sv_model) {
            throw std::runtime_error("Model is not a stochastic volatility model!");
        }

        const R dt = ttm / n_comp;
        StoringVector mean     = sv_model->M_T(ttm, dt, vol_view, grid.coordinates);
        StoringVector variance = sv_model->C_T(ttm, dt, vol_view);

        // ==============================
        // 6. Build Gaussian mixture densities
        // ==============================
        std::vector<DensityType> densities(mean.size());
        std::transform(std::execution::par_unseq,
                    mean.data(), mean.data() + mean.size(),
                    variance.data(),
                    densities.begin(),
                    [](R m, R v) {
                        return stats::make_normal_density<R>(m, std::sqrt(v));
                    });

        // ==============================
        // 7. Copy grid weights into std::vector
        // ==============================
        std::vector<R> weights(grid.weights.data(),
                            grid.weights.data() + grid.weights.size());

        // ==============================
        // 8. Assemble and return MixtureDensity
        // ==============================
        return stats::MixtureDensity<PolynomialBaseDegree, DensityType, R>(
            std::move(weights),
            std::move(densities)
        );
    }


    /**
     * @brief Adjusts a computed call price to the correct put price if needed.
     *
     * Uses put-call parity:
     *   Put = Call - S0 + K * e^{-rT}
     */
    inline R adjustPutCall(R call_price) const {
        if (this->payoff_->type() == traits::OptionType::Put) {
            return call_price
                - std::exp(this->sde_model_->get_x0())
                + std::exp(-this->rate_ * this->ttm_) * this->payoff_->getStrike();
        }
        return call_price;
    }

    /**
     * @brief Computes the OPE polynomial moments (ℓ_n).
     *
     * Steps:
     *   1. Build generator matrix G of the polynomial basis (MxM).
     *   2. Construct vector h_vals (initial condition).
     *   3. Exponentiate G over time-to-maturity: exp(GT).
     *   4. Build selector matrix S that picks out basis terms (M x (N+1)).
     *   5. Compute ℓ = h_valsᵀ * exp(GT) * S.
     *
     * @return StoringVector of size (N+1), containing ℓ_n.
     */
    StoringVector poly_moments() const {
        // Retrieve H-matrix (orthogonal polynomials) and basis enumeration
        const auto& H = density_object_.getHMatrix();     // (N+1)x(N+1)
        const auto   E = Utils::enumerate_basis(PolynomialBaseDegree);
        const unsigned int M = static_cast<unsigned int>(E.size());

        // Ensure SDE model is the right type
        auto* gen_sde_model = dynamic_cast<SDE::GenericSVModelSDE<R>*>(this->sde_model_.get());
        if (!gen_sde_model) {
            throw std::runtime_error("poly_moments requires GenericSVModelSDE.");
        }

        // Initial state (x0, v0)
        const R X0 = gen_sde_model->get_x0();
        const R V0 = gen_sde_model->get_v0();

        // 1. Generator matrix G (MxM)
        StoringMatrix G = gen_sde_model->generator_G(E, H);

        // 2. Initial h-values
        StoringVector h_vals = Utils::build_h_vals<R>(H, E, PolynomialBaseDegree, X0, V0);

        // 3. Matrix exponential exp(GT)
        const R T = this->get_ttm();
        StoringMatrix expGT = (T * G).exp();

        // 4. Selector matrix S (M x (N+1))
        StoringMatrix S = StoringMatrix::Zero(M, PolynomialBaseDegree + 1);
        for (unsigned int i = 0; i < M; ++i) {
            if (E[i].first == 0 && E[i].second <= static_cast<int>(PolynomialBaseDegree)) {
                S(i, E[i].second) = 1.0;
            }

        }

        // 5. Compute ℓ = h_valsᵀ * expGT * S
        return (h_vals.transpose() * expGT * S).transpose();
    }
    /**
     * @brief Computes the projection coefficients f via direct OPE projection.
     *
     * Steps:
     *   1. Wrap weights vector (avoid copy).
     *   2. Collect Gaussian mixture parameters (μ_k, σ_k).
     *   3. Build per-component coefficients f^(k).
     *   4. Apply projection matrices Q^(k).
     *   5. Weighted sum across components: f = Σ_k c_k * (Q^(k) f^(k)).
     *
     * @return StoringVector of size (N+1), containing projection coefficients f.
     */
    StoringVector direct_projection() const {
        constexpr int N = PolynomialBaseDegree;

        // 1. Wrap weights vector without copying
        const auto& w_std = density_object_.getWeights();
        const Eigen::Map<const StoringVector> w(
            w_std.data(),
            static_cast<Eigen::Index>(w_std.size())
        );

        // 2. Collect mixture components (Gaussian densities)
        const auto& components = density_object_.getComponents();
        const Eigen::Index K = static_cast<Eigen::Index>(components.size());

        if (w.size() != K) {
            throw std::runtime_error("Mismatch: weights.size != num_components.");
        }

        // Hermite polynomial basis (degree N)
        auto hermite = polynomials::HermitePolynomial<N, R>(0, 1);

        // 3. Build F matrix: columns are f^(k), size (N+1)xK
        StoringMatrix F(N + 1, K);
        const R strike_log = std::log(this->payoff_->getStrike());
        for (Eigen::Index k = 0; k < K; ++k) {
            const R mu_k    = components[k].getMu();
            const R sigma_k = components[k].getSigma();

            F.col(k).noalias() = component_coeffs<N, R>(
                this->rate_,    // r
                this->ttm_,     // T
                mu_k,
                sigma_k,
                strike_log,
                hermite
            );
        }

        // 4. Retrieve projection matrices Q^(k)
        const auto& Q_list = density_object_.getQProjectionMatrix();
        if (static_cast<Eigen::Index>(Q_list.size()) != K) {
            throw std::runtime_error("Mismatch: Q_list.size != num_components.");
        }

        for (Eigen::Index k = 0; k < K; ++k) {
            if (Q_list[k].rows() != N + 1 || Q_list[k].cols() != N + 1) {
                throw std::runtime_error("Invalid shape for Q^(k).");
            }
        }

        // 5. Compute f = Σ_k c_k * (Q^(k) f^(k))
        StoringVector f = StoringVector::Zero(N + 1);
        for (Eigen::Index k = 0; k < K; ++k) {
            StoringVector temp(N + 1);
            temp.noalias() = Q_list[k].transpose() * F.col(k);   // Projection
            f.noalias()   += w[k] * temp;            // Weighted accumulation
        }

        return f;
    }

}; 

    /**
     * @brief Compute the sequence I_n(μ, ν) used in Hermite expansions.
     *
     * Recurrence relation (for n ≥ 1):
     *   I_0 = exp(½ ν²) * Φ(ν - μ)
     *   I_n = H_{n-1}(μ) * exp(ν μ) φ(μ) + ν * I_{n-1}
     *
     * where:
     *   - H_{n} are Hermite polynomials
     *   - Φ is the standard normal CDF
     *   - φ is the standard normal PDF
     *
     * @tparam Deg Maximum polynomial degree (N).
     * @tparam T   Scalar type.
     *
     * @param mu     Argument μ.
     * @param nu     Argument ν.
     * @param I      Output vector of size Deg+1 containing {I_0, …, I_Deg}.
     * @param hermite Precomputed Hermite polynomial basis.
     */
    template<unsigned int Deg, typename T = traits::DataType::PolynomialField>
    inline void I_series(const T mu,
                        const T nu,
                        std::vector<T>& I,
                        const polynomials::HermitePolynomial<Deg, T>& hermite) 
    {
        const auto normal = stats::make_normal_density(0, 1);
        I.resize(Deg + 1);

        // Base case
        I[0] = std::exp(static_cast<T>(0.5) * nu * nu) * normal.cdf(nu - mu);
        if constexpr (Deg == 0) return;

        // Precompute exponential factor for reuse
        const T c = std::exp(nu * mu) * normal.pdf(mu);

        // Recurrence
        for (unsigned int n = 1; n <= Deg; ++n) {
            const T Hnm1 = hermite.evaluate(mu, n - 1);
            I[n] = Hnm1 * c + nu * I[n - 1];
        }
    }
    /**
     * @brief Compute Fourier coefficients f₀…f_N for one Gaussian mixture component.
     *
     * From equations (23)–(24):
     *   f₀ = e^{-rT+μ} I₀ - e^{-rT+K} Φ((μ-K)/σ)
     *   fₙ = e^{-rT+μ} (σ / √(n!)) I_{n-1},   for n ≥ 1
     *
     * where:
     *   - I_j(μ, ν) are computed via I_series()
     *   - μ, σ are component mean and stddev
     *   - K = log(strike)
     *
     * @tparam Deg Maximum polynomial degree (N).
     * @tparam T   Scalar type.
     *
     * @param r        Risk-free rate.
     * @param ttm      Time to maturity.
     * @param mu_w     Mean of component.
     * @param sigma_w  Stddev of component (must be > 0).
     * @param strike   Log-strike (log(K)).
     * @param hermite  Hermite polynomial basis.
     *
     * @return StoringVector of size Deg+1 with coefficients f.
     *
     * @throws std::invalid_argument if sigma_w ≤ 0.
     */
    template<unsigned int Deg, typename T = traits::DataType::PolynomialField>
    inline StoringVector component_coeffs(const T r,
                                        const T ttm,
                                        const T mu_w,
                                        const T sigma_w,
                                        const T strike,
                                        const polynomials::HermitePolynomial<Deg, T>& hermite)
    {
        if (!(sigma_w > 0.0)) {
            throw std::invalid_argument("sigma_w must be positive");
        }

        StoringVector f(Deg + 1);

        const auto normal = stats::make_normal_density(0, 1);

        // Parameters
        const T a   = (strike - mu_w) / sigma_w;
        const T nu  = sigma_w;

        // 1. Compute I_j sequence
        std::vector<T> I;
        I_series<Deg, T>(a, nu, I, hermite);

        // 2. Discount factors
        const T exp_m = std::exp(-r * ttm + mu_w);
        const T exp_k = std::exp(-r * ttm + strike);

        // 3. f₀ term
        f[0] = exp_m * I[0] - exp_k * normal.cdf((mu_w - strike) / sigma_w);
        if constexpr (Deg == 0) return f;

        // Helper: √(n!)
        auto sqrt_fact = [](int n) -> T { return std::sqrt(std::tgamma(n + 1)); };

        // 4. Higher order terms fₙ
        for (unsigned int n = 1; n <= Deg; ++n) {
            f[n] = exp_m * (sigma_w / sqrt_fact(n)) * I[n - 1];
        }

        return f;
    }


} // namespace options
#endif // OPTION_PRICER_HPP
