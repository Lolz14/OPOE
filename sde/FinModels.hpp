
/*
Consider Polynomial Diffusion: dZ(t) = b(Z(t)) dt + sigma(Z(t)) dW_t
where b is a vector and sigma is a matrix (Pol_n(E))
Generator is given by:
b^T*∇ + 1/2 Tr(σ^T σ ∇^2)
where ∇ is the gradient operator and Tr is the trace operator.

If the polynomial property holds, then the generator can be expressed as Q(z)Gp, with G being the matrix reperesentation, and Q is a basis of E.alignas

Moments are then given by:
Q(Z(t))^T*(exp(G(T - t)))*p = E[p(Z(T)) | F(t)]
where F(t) is the filtration up to time t.

ln = Q(Z0)^T*(exp(G(T - 0)))*Hn

dY(t) = k* (θ - Y(t))*dt + σ (Y(t)) * dW1_t
dX(t) = (r - delta)*dt  - 1/2*d[X(t)] + Sigma_1(Y(t)) * dW1_t + Sigma_2(Y(t)) * dW2_t

where:

where W1 and W2 are Brownian motions, and σ(Y(t)) is the volatility function dependent on the state variable Y(t).
where k is the speed of mean reversion, θ is the long-term mean, and σ is the volatility.





The Greeks of the option are computed by differentiating the option price with respect to one or multiple
variables. For the sensitivity analysis we fix the auxiliary density w(x), hence the basis Hn(x) and the
coefficients fn, and let only `(x) through g(x) depend on the perturbed parameters. The sensitivity of πf
with respect to the variable y is hence given by
∂y πf = ∑ (n≥0)(ln ∂y fn + fn ∂y ln) (22)
with the partial derivative ∂y = ∂/∂y. The sensitivity of ln with respect to y is given by
∂y ln = (0, ∂yQ(Z0))*exp(GT)*Hn + (1, Q(Z0))*∂yexp(GT)*Hn.
The derivative of the exponentcial operator eG T with respect to y is given by
∂ exp(GT)/∂y = [exp(xGT) ∂GT/∂y exp((1−x)GT)dx] as proved in (Wilcox 1967).
In particular for the Delta, which is the derivative of option price with respect to y = exp(X0), we have
that ∂y eG T = 0. In practice, we can efficiently approximate the option greeks by truncation the series (22)
at some finite order N . This may also prove to be valuable for the model calibration or estimation since the
objective function derivatives with respect to the model parameters can be computed which enables the use
of gradient based search algorithms.






Implementation Roadmap:
// 1. BaseStochVolModel: Base class for stochastic volatility models. (implementation of spotvariance, leverage, volvol, sigma1, sigma2,
sigma). Distribution of XT (logprice) follows a normal distribution with MT = Xo + (r - delta)T + 1/2[spotvariance(Yt)dt] +[sigma1(Yt) * dW1_t] 
CT = [sigma2(Yt)^2*dt]

Density is then defined automatically. If we want approximate the log price density with a mixture density, then it's enough to consider a
density w which is a weighted sum of densities, with weights c 
// 2. Check if prop 3.2 is applicable (MT is bounded and CT is bounded -> likelihood ratio lies in Lebesgue^2 weighted space, with weight w)

*/

#ifndef SDE_FINMODELS_HPP
#define SDE_FINMODELS_HPP

#include "SDE.hpp"
#include <stdexcept>    // For std::invalid_argument, std::runtime_error
#include <iostream>     // For std::cerr
#include <complex>

namespace SDE{

template <typename T = traits::DataType::PolynomialField>
constexpr std::complex<T> ImaginaryUnit{T(0), T(1)};

using SDEVector = SDE::SDEVector; 
using SDEComplexVector = SDE::SDEComplexVector; 
using SDEMatrix = SDE::SDEMatrix; 

template<typename T = traits::DataType::PolynomialField>
class ISDEModel {

public:

    virtual ~ISDEModel() = default; // Important for proper cleanup
    // Pure virtual functions defining the SDE
    virtual void drift(T t, const SDEVector& x, SDEVector& mu_out) const = 0;
    virtual void diffusion(T t, const SDEVector& x, SDEMatrix& sigma_out) const = 0;

    // Essential for some numerical schemes (e.g., Milstein)
    // These could be optional or provide default (e.g., zero) implementations if not always needed
    // Or, better, have specialized interfaces for models that support these

    virtual void drift_derivative_x(T t, const SDEVector& x, SDEVector& deriv_out) const {
        
        // Default implementation: numerical differentiation or throw not_implemented
        throw std::logic_error("Drift derivative not implemented for this model.");

    }


    virtual void diffusion_derivative_x(T t, const SDEVector& x, SDEMatrix& deriv_out) const {

        throw std::logic_error("Diffusion derivative not implemented for this model.");

    }

    virtual void diffusion_second_derivative_x(T t, const SDEVector& x, SDEMatrix& deriv_out) const {

        throw std::logic_error("Diffusion derivative not implemented for this model.");

    }

    virtual T generator_fn(T t, const SDEVector& x, const T df, const T ddf) const {

        // Default implementation: throw not_implemented
        throw std::logic_error("Generator function not implemented for this model.");

    }
    virtual void characteristic_fn(T /*t*/, const SDEVector& x0, const SDEComplexVector& x, SDEComplexVector& charact_out) const {

        // Default implementation: throw not_implemented
        throw std::logic_error("Characteristic function not implemented for this model.");

    }


    virtual unsigned int get_wiener_dimension() const = 0;


};

template<typename T = traits::DataType::PolynomialField>
class GeometricBrownianMotionSDE : public ISDEModel<T> {

public:

    static constexpr unsigned int WIENER_DIM = 1;
    static constexpr unsigned int STATE_DIM = 1; // Single state variable (e.g., asset price S)


    struct Parameters {
        T mu; // X is log-price ln(S), mu is (r + 0.5*sigma^2)
        T sigma; // Volatility
    };

private:

    Parameters params_;

public:

    GeometricBrownianMotionSDE(T mu, T sigma) : params_(Parameters{mu, sigma}) {

        if (params_.sigma < 0.0) {
            throw std::invalid_argument("Volatility cannot be negative.");
        }

    }

    unsigned int get_wiener_dimension() const override { return WIENER_DIM; }
    const Parameters& get_parameters() const { return params_; }
    void set_parameters(const Parameters& params) {
        params_ = params;
        if (params_.sigma < 0.0) { 
            std::invalid_argument("Volatility cannot be negative.");
        }

    }



    inline void drift(T /*t*/, const SDEVector& x, SDEVector& mu_out) const override {

        // Drift of X_t is mu
        mu_out = (params_.mu - params_.sigma * params_.sigma * static_cast<T>(0.5)) * SDEVector::Ones(STATE_DIM);

    }

    inline void diffusion(T /*t*/, const SDEVector& x, SDEMatrix& sigma_out) const override {

        // Diffusion of X_t is sigma
        sigma_out = params_.sigma * SDEMatrix::Identity(STATE_DIM, WIENER_DIM);

    }

    // Derivatives for dS_t = mu*S_t*dt + sigma*S_t*dW_t
    // Drift term a(t,S) = mu*S. Derivative w.r.t S is mu.
    // Diffusion term b(t,S) = sigma*S. Derivative w.r.t S is sigma.
    // Since we are considering log-price, then all the derivatives are 0.

    inline void drift_derivative_x(T /*t*/, const SDEVector& /*x*/, SDEVector& deriv_out) const override {

        deriv_out.setZero();

    }


    inline void diffusion_derivative_x(T /*t*/, const SDEVector& x, SDEMatrix& deriv_out) const override {
        
        deriv_out.setZero();
        
    }

    inline void diffusion_second_derivative_x(T /*t*/, const SDEVector& x, SDEMatrix& deriv_out) const override {
        // For GBM, the second derivative of diffusion is zero
        deriv_out.setZero();

    }

    inline T generator_fn(T /*t*/, const SDEVector& x, const T df, const T ddf) const override {
        // Generator for GBM: a(t,S) + 0.5 * b(t,S)^2
        // Here, a(t,S) = mu, b(t,S) = sigma
        // So, generator = mu + 0.5 * (sigma)^2
        return (params_.mu - params_.sigma * params_.sigma * static_cast<T>(0.5)) * df + static_cast<T>(0.5) * params_.sigma * params_.sigma * ddf;
    }

    inline void characteristic_fn(T t, const SDEVector& x0, const SDEComplexVector& x, SDEComplexVector& charact_out) const override{
        // Verificare se post x. array in parentesi vada mu. DEVO VEDERE OVUNQUE
        charact_out = t * ( (-params_.sigma * params_.sigma * x.array().square()) * static_cast<T>(0.5) + ImaginaryUnit<> * x.array() * ( - params_.sigma * params_.sigma * static_cast<T>(0.5)));

        charact_out = charact_out.array().exp();
    }

};

// As defined in Fast strong approximation Monte-Carlo schemes for stochastic volatility models
// The model is the following:
// dS(t) = mu*S(t)*dt + V(t)^p * S(t) * dW(t)
// dV(t) = kappa * (theta - V(t)) * dt + sigma * V(t)^q * dZ(t)
// We employ Cholesky decomposition to handle the correlation between the two Wiener processes:
// dW = rho * dZ_uncorr + sqrt(1 - rho^2) * dW_uncorr
// dZ = dZ_uncorr
// So that, we get the following system of SDEs:
// dS(t) = mu*S(t)*dt + V(t)^p * S(t) * (rho*dZ_uncorr + sqrt(1 - rho^2)*dW_uncorr)
// dV(t) = kappa * (theta - V(t)) * dt + sigma * V(t)^q * dZ_uncorr
// where dZ_uncorr and dW_uncorr are two independent Wiener motions.


template<typename T = traits::DataType::PolynomialField>
class GenericSVModelSDE : public ISDEModel<T> {

public:

    static constexpr unsigned int WIENER_DIM = 2; // Two correlated Wiener processes
    static constexpr unsigned int STATE_DIM = 2; // Two state variables (log-price and variance)

    struct Parameters {
        T asset_drift_const; // e.g., r if x(0) is log-price
        T sv_kappa;          // mean-reversion speed for x(1)
        T sv_theta;          // long-term mean for x(1)
        T sv_sigma;          // volatility of x(1)
        T correlation;       // rho
        T asset_vol_exponent; // p in your code: asset vol ~ x(1)^p
        T sv_vol_exponent;    // q in your code: sv vol ~ x(1)^q
    };

protected:

    Parameters params_;

public:

    GenericSVModelSDE(const Parameters& params) : params_(params) {

        // Parameter validation

        if (params_.sv_theta <= 0.0) {
             throw std::invalid_argument("Long-term variance theta must be positive.");
        }

        if (params_.sv_kappa < 0.0) { // kappa typically positive
            // Consider if allowing kappa=0 (CIR becomes degenerate) is desired.
            // Usually kappa > 0 for mean reversion.
            std::cerr << "Warning: Mean-reversion kappa is negative or zero.\n";

        }

        if (params_.correlation < -1.0 || params_.correlation > 1.0) {
            throw std::invalid_argument("Correlation rho must be between -1 and 1.");
        }

        if (params_.sv_sigma <= 0.0 && params_.sv_vol_exponent > 0) { // only if sv_sigma matters
             throw std::invalid_argument("Volatility of stochastic factor (sv_sigma) must be positive if it has an impact.");
        }

         // Feller-like condition might depend on 'q' if x(1) is variance.

        if (params_.sv_vol_exponent == 0.5 && params_.sv_kappa > 0 && params_.sv_theta > 0) { // Heston-like variance (x(1)=v)
            if (2.0 * params_.sv_kappa * params_.sv_theta < params_.sv_sigma * params_.sv_sigma) {
                std::cerr << "Warning: Feller condition (2*kappa*theta >= sigma_v^2) may not be satisfied; x(0) might become negative if it represents variance.\n";
            }
        }
    }

    unsigned int get_wiener_dimension() const override { return WIENER_DIM; }
    const Parameters& get_parameters() const { return params_; }

    inline void drift(T /*t*/, const SDEVector& x, SDEVector& mu_out) const override 
    {
        T asset_vol_term_squared;

        if (params_.asset_vol_exponent == static_cast<T>(0.5)) {

            // (x(1)^0.5)^2 = x(1)
            asset_vol_term_squared = x(0);

        } else if (params_.asset_vol_exponent == static_cast<T>(1.0)) {

            // (x(1)^1.0)^2 = x(1)^2
            asset_vol_term_squared = x(0) * x(0);

        } else {

            // General case: std::pow(x(1), 2.0 * params_.asset_vol_exponent)
            asset_vol_term_squared = std::pow(x(0), static_cast<T>(2.0) * params_.asset_vol_exponent);

        }

        // Drift for x(0): the stochastic volatility factor (e.g. CIR process or OU)
        mu_out(0) = params_.sv_kappa * (params_.sv_theta - x(0));

        // Drift for x(1): log-price
        mu_out(1) = params_.asset_drift_const - static_cast<T>(0.5) * asset_vol_term_squared;

    }

    inline void diffusion(T /*t*/, const SDEVector& x, SDEMatrix& sigma_out) const override {

        const T sv_factor = x(0); // The stochastic volatility factor

        // Calculate factor_p = sv_factor^p
        T factor_p;
        if (params_.asset_vol_exponent == static_cast<T>(0.5)) {
            factor_p = std::sqrt(sv_factor);
        } 
        else if (params_.asset_vol_exponent == static_cast<T>(1.0)) {
            factor_p = sv_factor;
        } 
        else {
            factor_p = std::pow(sv_factor, params_.asset_vol_exponent);
        }

        // Calculate factor_q = sv_factor^q
        T factor_q;

        if (params_.sv_vol_exponent == static_cast<T>(0.0)) {
            factor_q = static_cast<T>(1.0); // For q=0, x^0 = 1
        } 
        else if (params_.sv_vol_exponent == static_cast<T>(0.5)) {
            factor_q = std::sqrt(sv_factor);
        }
        else if (params_.sv_vol_exponent == static_cast<T>(1.0)) {
            factor_q = sv_factor;
        } 
        else {
            factor_q = std::pow(sv_factor, params_.sv_vol_exponent);
        } 


        // Row 0: volatility diffusion components
        sigma_out(0, 1) = 0.0;                 // Component for dW_uncorr
        sigma_out(0, 0) = params_.sv_sigma * factor_q; // Component for dZ

        // Row 1: Log Price diffusion components
        sigma_out(1, 1) = std::sqrt(1 - params_.correlation*params_.correlation) * factor_p ; // Corresponds to dW_uncorr
        sigma_out(1, 0) = params_.correlation * factor_p;      // Corresponds to dZ_uncorr


    }
    inline void drift_derivative_x(T /*t*/, const SDEVector& x, SDEVector& deriv_out) const override {
        // Derivative of drift w.r.t. x(0) and x(1)
        deriv_out(0) = params_.sv_kappa; // Derivative of drift w.r.t. x(0)
        deriv_out(1) = 0.0;
        }
    
    // Derivative of diffusion w.r.t. x(1) (state log-price)
    inline void diffusion_derivative_x(T /*t*/, const SDEVector& x, SDEMatrix& deriv_out) const override {
        //deriv_out(0, 0) = params_.sv_sigma * params_.sv_vol_exponent * std::pow(x(0), params_.sv_vol_exponent - 1.0); // Derivative of diffusion w.r.t. x(0) for dZ
        //deriv_out(0, 1) = 0.0; // Derivative of diffusion w.r.t. x(0) for dW_uncorr
        //deriv_out(1, 0) = params_.correlation * params_.asset_vol_exponent * std::pow(x(0), params_.asset_vol_exponent - 1.0); // Derivative of diffusion w.r.t. x(0) for dZ_unc
        //deriv_out(1, 1) = 0.0;\
        deriv_out.setZero();

    }

    inline void diffusion_second_derivative_x(T /*t*/, const SDEVector& x, SDEMatrix& deriv_out) const override {
        // For GenericSVModel, the second derivative of diffusion is zero
        deriv_out.setZero();
        //deriv_out(0, 0) = params_.sv_sigma * params_.sv_vol_exponent * (params_.sv_vol_exponent - 1.0) * std::pow(x(0), params_.sv_vol_exponent - 2.0); // Second derivative of diffusion w.r.t. x(0) for dZ
        //deriv_out(1, 0) = params_.correlation * params_.asset_vol_exponent * (params_.asset_vol_exponent - 1.0) * std::pow(x(0), params_.asset_vol_exponent - 2.0); // Second derivative of diffusion w.r.t. x(0) for dZ_unc
    }

    inline T generator_fn(T /*t*/, const SDEVector& x, const T df, const T ddf) const override {
            
        T effective_variance_term;

        if (params_.asset_vol_exponent == static_cast<T>(0.5)) {
            effective_variance_term = x(0); // x(1)^(2*0.5) = x(1)
        } 
        else if (params_.asset_vol_exponent == static_cast<T>(1.0)) {
            effective_variance_term = x(0) * x(0); // x(1)^(2*1) = x(1)^2
        } 
        else {
            effective_variance_term = std::pow(x(0), static_cast<T>(2.0) * params_.asset_vol_exponent);
        }

        return (params_.asset_drift_const - static_cast<T>(0.5) * effective_variance_term) * df + static_cast<T>(0.5) * effective_variance_term * ddf;
    }

    inline T get_p() const noexcept {
        return params_.asset_vol_exponent; // p in the model
    }

    inline T get_q() const noexcept {
        return params_.sv_vol_exponent; // q in the model
    }

    inline T get_correlation() const noexcept {
        return params_.correlation; // Correlation between the two Wiener processes
    }

    inline T get_kappa() const noexcept {
        return params_.sv_kappa; // Mean reversion speed for the variance process
    }

    inline T get_theta() const noexcept {
        return params_.sv_theta; // Long-term mean for the variance process
    }

    inline T get_sigma_v() const noexcept {
        return params_.sv_sigma; // Volatility of the variance process
    }

};

template<typename T = traits::DataType::PolynomialField>
class HestonModelSDE : public GenericSVModelSDE<T> {
public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;

    HestonModelSDE(T asset_drift_const,
                   T sv_kappa,
                   T sv_theta,
                   T sv_sigma,
                   T correlation)
        : Base(Parameters{
            asset_drift_const,
            sv_kappa,
            sv_theta,
            sv_sigma,
            correlation,
            static_cast<T>(0.5), // asset_vol_exponent p
            static_cast<T>(0.5)  // sv_vol_exponent q
        }) {}

    inline void characteristic_fn(T t, const SDEVector& x0, const SDEComplexVector& x, SDEComplexVector& charact_out) const override{

            // Step 1: gamma
        SDEComplexVector d = ((Base::params_.sv_kappa - ImaginaryUnit<> * Base::params_.correlation * Base::params_.sv_sigma * x.array()).square()
                            + Base::params_.sv_sigma * Base::params_.sv_sigma * (x.array() * (x.array() + ImaginaryUnit<>))).matrix();
        SDEComplexVector gamma = d.array().sqrt().matrix();
        // Step 2: A (exp1 + exp2)
        SDEComplexVector exp1 = ((Base::params_.sv_kappa * Base::params_.sv_theta * (Base::params_.sv_kappa - ImaginaryUnit<> * 
            Base::params_.correlation * Base::params_.sv_sigma * x.array()))
                                / (Base::params_.sv_sigma * Base::params_.sv_sigma) * t).matrix();
        SDEComplexVector exp2 = (ImaginaryUnit<> * x.array() * (Base::params_.asset_drift_const * t + x0(1))).matrix();

        SDEComplexVector A = (exp1 + exp2).array().exp().matrix();

        // Step 3: B
        SDEComplexVector gamma_t_half = (gamma.array() * (T(0.5) * t)).matrix();
        SDEComplexVector ch = (gamma_t_half.array().cosh()
                            + ((Base::params_.sv_kappa - ImaginaryUnit<> * Base::params_.correlation * Base::params_.sv_sigma * x.array()) / gamma.array()) * gamma_t_half.array().sinh()).matrix();
        SDEComplexVector B = ch.array().pow(T(2.0 * Base::params_.sv_kappa * Base::params_.sv_theta / (Base::params_.sv_sigma * Base::params_.sv_sigma))).matrix();

        // Step 4: C
        SDEComplexVector gamma_tanh = (gamma.array() * (T(0.5) * t)).tanh().matrix();
        SDEComplexVector denom = (gamma.array().cwiseQuotient(gamma_tanh.array()) + Base::params_.sv_kappa - ImaginaryUnit<> * Base::params_.correlation * Base::params_.sv_sigma * x.array()).matrix(); 

        SDEComplexVector C = (-(x.array() * x.array() + ImaginaryUnit<> * x.array()) * x0(0) / denom.array()).exp().matrix();

        // Final result
        charact_out = A.cwiseQuotient(B).cwiseProduct(C);

    }

};

template<typename T = traits::DataType::PolynomialField>
class SteinSteinModelSDE : public GenericSVModelSDE<T> {
public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;

    SteinSteinModelSDE(T asset_drift_const,
                   T sv_kappa,
                   T sv_theta,
                   T sv_sigma,
                   T correlation)
        : Base(Parameters{
            asset_drift_const,
            sv_kappa,
            sv_theta,
            sv_sigma,
            correlation,
            static_cast<T>(1.0), // asset_vol_exponent p
            static_cast<T>(0.0)  // sv_vol_exponent q
        }) {}
};

template<typename T = traits::DataType::PolynomialField>
class HullWhiteSVModelSDE : public GenericSVModelSDE<T> {
public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;

    HullWhiteSVModelSDE(T asset_drift_const,
                   T sv_kappa,
                   T sv_theta,
                   T sv_sigma,
                   T correlation)
        : Base(Parameters{
            asset_drift_const,
            sv_kappa,
            sv_theta,
            sv_sigma,
            correlation,
            static_cast<T>(1.0), // asset_vol_exponent p
            static_cast<T>(1.0)  // sv_vol_exponent q
        }) {}
};


template<typename T = traits::DataType::PolynomialField>
class JacobiModelSDE : public GenericSVModelSDE<T> {

public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;


    static constexpr unsigned int WIENER_DIM = 2; // Two correlated Wiener processes
    static constexpr unsigned int STATE_DIM = 2; // Two state variables (log-price and variance)

private:

    T y_min_;
    T y_max_;
    T q_denominator_sq_; // (sqrt(y_max) - sqrt(y_min))^2

    // Helper for Q(y)
    inline T Q_func(T y) const {

        if (q_denominator_sq_ <= 0) return 0.0; // Avoid division by zero if y_max approx y_min

        // Ensure y is within bounds for Q(y) to be non-negative, or handle appropriately
        // The SDE formulation usually assumes Y_t stays within [y_min, y_max]

        T y_clamped = std::max(y_min_, std::min(y, y_max_));
        return (y_clamped - y_min_) * (y_max_ - y_clamped) / q_denominator_sq_;

    }

    inline T Q_func_der1(T y) const {

        if (q_denominator_sq_ <= 0) return 0.0; 

        

        T y_clamped = std::max(y_min_, std::min(y, y_max_));
        return  (y_max_ - static_cast<T>(2.0)*y_clamped + y_min_) / q_denominator_sq_;

    }

    inline T Q_func_der2(T y) const {

        if (q_denominator_sq_ <= 0) return 0.0; 

        

        return  -2/q_denominator_sq_;

    }

public:

    JacobiModelSDE(T asset_drift_const, 
               T sv_kappa,
               T sv_theta,
               T sv_sigma,
               T correlation,
               T y_min,
               T y_max)
    : Base(Parameters{asset_drift_const, sv_kappa, sv_theta, sv_sigma, correlation, static_cast<T>(1.0), static_cast<T>(1.0)}),
      y_min_(y_min),y_max_(y_max)
        {
        if (Base::params_.correlation < -1.0 || Base::params_.correlation > 1.0) {
            throw std::invalid_argument("JacobiModelSDE: Rho must be between -1 and 1.");
        }

        if (y_min_ < 0.0 || y_max_ <= y_min_) {
            throw std::invalid_argument("JacobiModelSDE: Invalid y_min/y_max. y_min >= 0 and y_max > y_min required.");
        }

        if (Base::params_.sv_theta < y_min_ || Base::params_.sv_theta > y_max_) {
             std::cerr << "Warning: JacobiModelSDE: Theta is outside variance bounds [y_min, y_max].\n";
        }


        if (Base::params_.sv_sigma <= 0) {
            throw std::invalid_argument("JacobiModelSDE: Sigma (vol of vol factor) must be positive.");
        }

        if (Base::params_.sv_kappa < 0) {
             std::cerr << "Warning: JacobiModelSDE: Kappa (mean reversion speed) is negative.\n";
        }

        T sqrt_ymax = std::sqrt(y_max_);
        T sqrt_ymin = std::sqrt(y_min_);

        if (std::abs(sqrt_ymax - sqrt_ymin) < 1e-9) { // Handles y_max very close to y_min
             q_denominator_sq_ = 1.0; // Avoid division by zero, though Q(y) will be near zero

             if (y_max_ > y_min_)
                std::cerr << "Warning: JacobiModelSDE: y_max is very close to y_min, Q(y) might be ill-conditioned.\n";

        } else {

            q_denominator_sq_ = (sqrt_ymax - sqrt_ymin) * (sqrt_ymax - sqrt_ymin);

        }

    }

    // State x: x(0) = X_t (log-price), x(1) = Y_t (variance process)

    inline void drift(T /*t*/, const SDEVector& x_state, SDEVector& mu_out) const override {

        const T Y_t = x_state(0);


        // Drift for X_t (log-price)
        mu_out(1) = Base::params_.asset_drift_const  - Y_t * static_cast<T>(0.5);


        // Drift for Y_t
        mu_out(0) = Base::params_.sv_kappa * (Base::params_.sv_theta - Y_t);

    }

    inline void diffusion(T /*t*/, const SDEVector& x_state, SDEMatrix& sigma_out) const override {

        const T Y_t = x_state(0); // Current variance process value


        T q_y = Q_func(Y_t);
        T sqrt_q_y = (q_y > 0.0) ? std::sqrt(q_y) : 0.0;


        T Y_minus_rho_sq_Q = Y_t - Base::params_.correlation * Base::params_.correlation * q_y;
        T sqrt_Y_minus_rho_sq_Q = (Y_minus_rho_sq_Q > 0.0) ? std::sqrt(Y_minus_rho_sq_Q) : 0.0;
        
        // dY_t = ... + sigma*sqrt(Q(Y_t))dW_1t

        // dX_t = ... + rho*sqrt(Q(Y_t))dW_1t + sqrt(Y_t - rho^2*Q(Y_t))dW_2t


        
        // Row 0: variance (Y_t) diffusion coefficients for dW1, dW2

        sigma_out(0, 0) = Base::params_.sv_sigma * sqrt_q_y;
        sigma_out(0, 1) = 0.0;

        // Row 1: log-price (X_t) diffusion coefficients for dW1, dW2

        sigma_out(1, 0) = Base::params_.correlation * sqrt_q_y;
        sigma_out(1, 1) = sqrt_Y_minus_rho_sq_Q;


    }


    inline T generator_fn(T /*t*/, const SDEVector& x, const T df, const T ddf) const override {
            
    return (Base::params_.asset_drift_const  - x(0) * static_cast<T>(0.5) ) * df + static_cast<T>(0.5) * Base::params_.sv_sigma * Base::params_.sv_sigma * Q_func(x(0)) *ddf;
    }

};




}

#endif // SDE_FINMODELS_HPP