/**
 * @file FinModels.hpp
 * @brief Stochastic Differential Equation (SDE) models for financial applications.
 *
 * This header defines a set of classes and interfaces for modeling various stochastic processes
 * commonly used in mathematical finance, such as Geometric Brownian Motion, Heston, Stein-Stein,
 * Hull-White, and Jacobi models. These models are implemented as SDEs and provide interfaces for
 * drift, diffusion, and their derivatives, as well as characteristic functions and generator matrices.
 *
 * Main Components:
 * - ISDEModel: Abstract base class for SDE models, defining the interface for drift, diffusion,
 *   their derivatives, characteristic functions, and initial state management.
 * - GeometricBrownianMotionSDE: Implements the classic GBM model for asset prices.
 * - GenericSVModelSDE: Generic stochastic volatility model supporting flexible exponents for volatility.
 * - HestonModelSDE: Specialization of GenericSVModelSDE for the Heston model (square-root volatility).
 * - SteinSteinModelSDE: Specialization for the Stein-Stein model (OU volatility).
 * - HullWhiteModelSDE: Specialization for the Hull-White model (linear volatility).
 * - JacobiModelSDE: Specialization for the Jacobi model, with bounded variance process.
 *
 * Features:
 * - Support for both real and complex-valued state vectors and matrices.
 * - Calculation of drift, diffusion, and their derivatives for use in numerical schemes.
 * - Characteristic function computation for Fourier-based pricing methods.
 * - Construction of generator matrices for polynomial expansion methods.
 * - Parameter validation and error handling for model consistency.
 *
 * Template Parameter:
 * - T: Numeric type (default: traits::DataType::PolynomialField), allowing for flexibility in precision.
 *
 * Usage:
 * - Instantiate a model class with appropriate parameters and initial state.
 * - Use the drift and diffusion methods for simulation or numerical solution of SDEs.
 * - Use the characteristic function for pricing or calibration tasks.
 * - Use generator_G for polynomial expansion or spectral methods.
 *
 * Dependencies:
 * - SDE.hpp: Base definitions for SDE vectors and matrices.
 *
 */
#ifndef FINMODELS_HPP
#define FINMODELS_HPP

#include "SDE.hpp"
#include <stdexcept>    // For std::invalid_argument, std::runtime_error
#include <iostream>     // For std::cerr
#include <complex>
#include <memory>
namespace SDE{

template <typename T = traits::DataType::PolynomialField>
constexpr std::complex<T> ImaginaryUnit{T(0), T(1)};

using SDEVector = SDE::SDEVector; 
using SDEComplexVector = SDE::SDEComplexVector; 
using SDEMatrix = SDE::SDEMatrix; 


static constexpr long key(int m, int n) noexcept {
        return (static_cast<long>(m) << 32) | static_cast<unsigned int>(n);
    }

/**
 * @brief Base interface for all SDE models.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 */
template<typename T = traits::DataType::PolynomialField>
class ISDEModel {



public: 
        /**
         * @brief Returns the dimension of the state vector.
         * @return The number of state variables in the SDE model.
         *
         * This method provides the size of the state vector, which is essential for defining the model's structure.
         */
        virtual unsigned int state_dim() const = 0;

        /**
         * @brief Returns the dimension of the Wiener process.
         * @return The number of independent Wiener processes in the SDE model.
         *
         * This method provides the size of the Wiener process vector, which is essential for stochastic simulations.
         */
        virtual unsigned int wiener_dim() const = 0;
        /**
         * @brief Virtual destructor for proper cleanup in derived classes.
         */
        virtual ~ISDEModel() = default; 
        
        /**
         * @brief Clones the SDE model.
         * @return A shared pointer to a new instance of the SDE model.
         *
         * This method allows for polymorphic copying of the model, ensuring that derived classes can be copied correctly.
         */
        virtual std::shared_ptr<ISDEModel<T>> clone() const = 0;
        
        /**
         * @brief Computes the drift term of the SDE.
         * @param t Current time.
         * @param x Current state vector.
         * @param mu_out Output vector for the drift term.
         */
        virtual void drift(T t, const SDEVector& x, SDEVector& mu_out) const = 0;
        

        /**
         * @brief Computes the diffusion term of the SDE.
         * @param t Current time.
         * @param x Current state vector.
         * @param sigma_out Output matrix for the diffusion term.
         *
         * The diffusion term is typically a matrix that scales the Wiener increments.
         */
        virtual void diffusion(T t, const SDEVector& x, SDEMatrix& sigma_out) const = 0;

        // Essential for some numerical schemes (e.g., Milstein)
        // These could be optional or provide default (e.g., zero) implementations if not always needed
        // Or, better, have specialized interfaces for models that support these

        /**
         * @brief Computes the derivative of the drift term with respect to the state vector.
         * @param t Current time.
         * @param x Current state vector.
         * @param deriv_out Output vector for the drift derivative.
         *
         * This method is used in higher-order numerical schemes that require knowledge of how the drift changes with respect to the state.
         */
        virtual void drift_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, [[maybe_unused]] SDEVector& deriv_out) const {
            // Default implementation: numerical differentiation or throw not_implemented
            throw std::logic_error("Drift derivative not implemented for this model.");
        }
        /**
         * @brief Computes the derivative of the diffusion term with respect to the state vector.
         * @param t Current time.
         * @param x Current state vector.
         * @param deriv_out Output matrix for the diffusion derivative.
         *
         * This method is used in higher-order numerical schemes that require knowledge of how the diffusion changes with respect to the state.
         */
        virtual void diffusion_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, [[maybe_unused]] SDEMatrix& deriv_out) const {

            throw std::logic_error("Diffusion derivative not implemented for this model.");

        }

        /**
         * @brief Computes the second derivative of the diffusion term with respect to the state vector.
         * @param t Current time.
         * @param x Current state vector.
         * @param deriv_out Output matrix for the second diffusion derivative.
         *
         * This method is used in advanced numerical schemes that require knowledge of how the diffusion changes with respect to the state.
         */
        virtual void diffusion_second_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, [[maybe_unused]] SDEMatrix& deriv_out) const {

            throw std::logic_error("Diffusion derivative not implemented for this model.");

        }

        /**
         * @brief Computes the characteristic function of the SDE.
         * @param t Time parameter for the characteristic function.
         * @param x Input vector for the characteristic function.
         * @param charact_out Output vector for the characteristic function.
         *
         * The characteristic function is used in Fourier-based method.
         */
        virtual void characteristic_fn([[maybe_unused]] T t,  [[maybe_unused]] const SDEComplexVector& x, [[maybe_unused]] SDEComplexVector& charact_out) const {

            // Default implementation: throw not_implemented
            throw std::logic_error("Characteristic function not implemented for this model.");

        }

        /**
         * @brief Gets the initial state vector of the SDE.
         * @return The initial state vector.
         *
         * This method provides access to the initial state, which can be used in characteristic functions or other calculations.
         */
        virtual T get_x0() const {

            // Default implementation: throw not_implemented
            throw std::logic_error("Bump volatility not implemented for this model.");

        }

        /**
         * @brief Sets the initial state vector of the SDE.
         * @param x0 The new initial state vector.
         *
         * This method allows setting the initial state, which can be useful for simulations or recalibrations.
         */
        virtual void set_x0([[maybe_unused]] const T& x0) {

            // Default implementation: throw not_implemented
            throw std::logic_error("Set x0 not implemented for this model.");

        }

        /**
         * @brief Gets the initial volatility (v0) of the SDE.
         * @return The initial volatility.
         *
         * This method provides access to the initial volatility, which can be used in characteristic functions or other calculations.
         */
        virtual T get_v0() const {

            // Default implementation: throw not_implemented
            throw std::logic_error("Bump volatility not implemented for this model.");

        }


        /**
         * @brief Sets the initial volatility (v0) of the SDE.
         * @param v0 The new initial volatility.
         *
         * This method allows setting the initial volatility, which can be useful for simulations or recalibrations.
         */
        virtual void set_v0([[maybe_unused]] const T& v0) {

            // Default implementation: throw not_implemented
            throw std::logic_error("Set v0 not implemented for this model.");

        }

        /**
         * @brief Gets the Wiener dimension of the SDE.
         * @return The Wiener dimension.
         *
         * This method provides the number of independent Wiener processes driving the SDE.
         */
        virtual unsigned int get_wiener_dimension() const = 0;

        /**
         * @brief Gets the state dimension of the SDE.
         * @return The state dimension.
         *
         * This method provides the number of state variables in the SDE model.
         */
        virtual unsigned int get_state_dim() const = 0;

        SDEVector m_x0; // Initial state vector, can be used in characteristic functions or other calculations

};

/**
 * @brief Geometric Brownian Motion SDE model.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 *
 * This model represents the classic Geometric Brownian Motion, commonly used for modeling stock prices.
 * It includes methods for drift, diffusion, and characteristic function calculations.
 */
template<typename T = traits::DataType::PolynomialField>
class GeometricBrownianMotionSDE : public ISDEModel<T> {

public:

    static constexpr unsigned int WIENER_DIM = 1;
    static constexpr unsigned int STATE_DIM = 1; 

    unsigned int state_dim() const override { return STATE_DIM; }
    unsigned int wiener_dim() const override { return WIENER_DIM;} 

    /** * @brief Parameters for the Geometric Brownian Motion SDE.
     * Contains the drift (mu) and volatility (sigma) parameters.
     */
    struct Parameters {
        T mu; // X is log-price ln(S), mu is (r - 0.5*sigma^2)
        T sigma; // Volatility
    };

private:

    Parameters params_;

public:
    
    /**
     * @brief Geometric Brownian Motion SDE constructor.
     * @param mu Drift parameter (expected return rate).
     * @param sigma Volatility parameter (standard deviation of returns).
     * @param x0 Initial state (log-price).
     * @throws std::invalid_argument if sigma is negative.
     */
    GeometricBrownianMotionSDE(T mu, T sigma, T x0) : params_(Parameters{mu, sigma}) {
        this->m_x0 = SDEVector::Constant(STATE_DIM, x0);

        if (this->get_v0() < 0.0) {
            throw std::invalid_argument("Volatility cannot be negative.");
        }

    }

    std::shared_ptr<ISDEModel<T>> clone() const override {
        return std::make_shared<GeometricBrownianMotionSDE<T>>(*this);
    }

    unsigned int get_wiener_dimension() const override { return WIENER_DIM; }

    unsigned int get_state_dim() const override { return STATE_DIM;}

    inline void drift([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEVector& mu_out) const override {

        // Drift of X_t is mu
        mu_out = (this->get_mu() - this->get_v0() * this->get_v0() * static_cast<T>(0.5)) * SDEVector::Ones(STATE_DIM);

    }

    inline void diffusion([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEMatrix& sigma_out) const override {

        // Diffusion of X_t is sigma
        sigma_out = this->get_v0() * SDEMatrix::Identity(STATE_DIM, WIENER_DIM);

    }

    inline void drift_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEVector& deriv_out) const override {
        deriv_out.setZero();
    }
 
    inline void diffusion_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEMatrix& deriv_out) const override {    
        deriv_out.setZero();  
    }

    inline void diffusion_second_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEMatrix& deriv_out) const override {
        // For GBM, the second derivative of diffusion is zero
        deriv_out.setZero();

    }

    inline void characteristic_fn(T t, const SDEComplexVector& x, SDEComplexVector& charact_out) const override{
        // Verificare se post x. array in parentesi vada mu. DEVO VEDERE OVUNQUE
        charact_out = t * ( (-this->get_v0() * this->get_v0() * x.array().square()) * static_cast<T>(0.5) + ImaginaryUnit<> * x.array() * ( - this->get_v0() * this->get_v0() * static_cast<T>(0.5)));

        charact_out = charact_out.array().exp();
    }

    inline T get_x0() const override {
        // Return the initial state vector
        return this->m_x0(0);
    }

    inline void set_x0(const T& x0) override {
        // Set the initial state vector
        this->m_x0 = SDEVector::Constant(STATE_DIM, x0);
    }

    inline T get_v0() const noexcept override{
        return this->params_.sigma; // Return the initial variance (x(1))
    }

    inline void set_v0(const T& v0) noexcept override{
        this->params_.sigma = v0; // Set the initial variance (x(1))
    }

    inline T get_mu() const noexcept {
        return this->params_.mu; // Return the drift parameter (mu)
    }

};


/**
 * @brief Generic Stochastic Volatility SDE model.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 * This model supports flexible exponents for volatility and allows for correlation between the asset and volatility processes.
 *
 * As defined in Fast strong approximation Monte-Carlo schemes for stochastic volatility models
 * The model is the following:
 * dS(t) = mu*S(t)*dt + V(t)^p * S(t) * dW(t)
 * dV(t) = kappa * (theta - V(t)) * dt + sigma * V(t)^q * dZ(t)
 * We employ Cholesky decomposition to handle the correlation between the two Wiener processes:
 * dW = rho * dZ_uncorr + sqrt(1 - rho^2) * dW_uncorr
 * dZ = dZ_uncorr
 * So that, we get the following system of SDEs:
 * dS(t) = mu*S(t)*dt + V(t)^p * S(t) * (rho*dZ_uncorr + sqrt(1 - rho^2)*dW_uncorr)
 * dV(t) = kappa * (theta - V(t)) * dt + sigma * V(t)^q * dZ_uncorr
 * where dZ_uncorr and dW_uncorr are two independent Wiener motions.
 * Applying the Ito's lemma, we can derive the drift and diffusion terms for the SDEs.
 * dX(t) (mu - V(t)^2p/2) * dt + V(t)^p * (rho*dZ_uncorr + sqrt(1 - rho^2)*dW_uncorr
 * dY(t) = kappa * (theta - Y(t)) * dt + sigma * Y(t)^q * dZ_uncorr
*/
template<typename T = traits::DataType::PolynomialField>
class GenericSVModelSDE : public ISDEModel<T> {
        using Base = ISDEModel<T>;

public:

    static constexpr unsigned int WIENER_DIM = 2; // Two correlated Wiener processes
    static constexpr unsigned int STATE_DIM = 2; // Two state variables (log-price and variance)

    unsigned int state_dim() const override { return STATE_DIM; }
    unsigned int wiener_dim() const override { return WIENER_DIM;} 
    /**
     * @brief Parameters for the Generic Stochastic Volatility SDE.
     * Contains the drift (asset_drift_const), mean-reversion speed (sv_kappa), long-term mean (sv_theta),
     * volatility of the stochastic factor (sv_sigma), correlation (rho), and exponents for asset and volatility processes.
     * asset_vol_exponent and sv_vol_exponent are used to define the power of the stochastic volatility factor in the diffusion terms.
     * asset_vol_exponent is the exponent for the asset volatility term (p in the model).
     * sv_vol_exponent is the exponent for the stochastic volatility term (q in the model).
     */
    struct Parameters {
        T asset_drift_const; // e.g., r if x(1) is log-price
        T sv_kappa;          // mean-reversion speed for x(0)
        T sv_theta;          // long-term mean for x(0)
        T sv_sigma;          // volatility of x(0)
        T correlation;       // rho
        T asset_vol_exponent; // asset vol ~ x(0)^p
        T sv_vol_exponent;    //  sv vol ~ x(0)^q
    };

protected:

    Parameters params_;

public:

    GenericSVModelSDE(const Parameters& params, const SDEVector x0) : params_(params) {

        this->m_x0 = x0;

        // Parameter validation

        if (this->get_theta() <= 0.0) {
             throw std::invalid_argument("Long-term variance theta must be positive.");
        }

        if (this->get_kappa() < 0.0) { 
            // Usually kappa > 0 for mean reversion.
            std::cerr << "Warning: Mean-reversion kappa is negative or zero.\n";

        }

        if (this->get_correlation() < -1.0 || this->get_correlation() > 1.0) {
            throw std::invalid_argument("Correlation rho must be between -1 and 1.");
        }

        if (this->get_sigma_v() <= 0.0 && this->get_q() > 0) { 
             throw std::invalid_argument("Volatility of stochastic factor (sv_sigma) must be positive if it has an impact.");
        }

         // Feller condition check

        if (this->get_q() == 0.5 && this->get_kappa() > 0 && this->get_theta() > 0) { 
            if (2.0 * this->get_kappa() * this->get_theta() < this->get_sigma_v() * this->get_sigma_v()) {
                std::cerr << "Warning: Feller condition (2*kappa*theta >= sigma_v^2) may not be satisfied; x(0) might become negative if it represents variance.\n";
            }
        }
    }

    unsigned int get_wiener_dimension() const override { return WIENER_DIM; }
    
    unsigned int get_state_dim() const override { return STATE_DIM;}

    inline void drift([[maybe_unused]] T t, const SDEVector& x, SDEVector& mu_out) const override 
    {
        T asset_vol_term_squared;

        if (this->get_p() == static_cast<T>(0.5)) {

            asset_vol_term_squared = x(0);

        } else if (this->get_p() == static_cast<T>(1.0)) {

            asset_vol_term_squared = x(0) * x(0);

        } else {

            // General case: std::pow(x(1), 2.0 * this->get_p())
            asset_vol_term_squared = std::pow(x(0), static_cast<T>(2.0) * this->get_p());

        }

        // Drift for x(0): the stochastic volatility factor (e.g. CIR process or OU)
        mu_out(0) = this->get_kappa() * (this->get_theta() - x(0));

        // Drift for x(1): log-price
        mu_out(1) = this->get_drift() - static_cast<T>(0.5) * asset_vol_term_squared;

    }

    inline void diffusion([[maybe_unused]] T t, const SDEVector& x, SDEMatrix& sigma_out) const override {

        const T sv_factor = x(0); // The stochastic volatility factor

        // Calculate factor_p = sv_factor^p
        T factor_p;
        if (this->get_p() == static_cast<T>(0.5)) {
            factor_p = std::sqrt(sv_factor);
        } 
        else if (this->get_p() == static_cast<T>(1.0)) {
            factor_p = sv_factor;
        } 
        else {
            factor_p = std::pow(sv_factor, this->get_p());
        }

        // Calculate factor_q = sv_factor^q
        T factor_q;

        if (this->get_q() == static_cast<T>(0.0)) {
            factor_q = static_cast<T>(1.0); // For q=0, x^0 = 1
        } 
        else if (this->get_q() == static_cast<T>(0.5)) {
            factor_q = std::sqrt(sv_factor);
        }
        else if (this->get_q() == static_cast<T>(1.0)) {
            factor_q = sv_factor;
        } 
        else {
            factor_q = std::pow(sv_factor, this->get_q());
        } 


        // Row 0: volatility diffusion components
        sigma_out(0, 1) = 0.0;                 // Component for dW_uncorr
        sigma_out(0, 0) = this->get_sigma_v() * factor_q; // Component for dZ

        // Row 1: Log Price diffusion components
        sigma_out(1, 1) = std::sqrt(1 - this->get_correlation()*this->get_correlation()) * factor_p ; // Corresponds to dW_uncorr
        sigma_out(1, 0) = this->get_correlation() * factor_p;      // Corresponds to dZ_uncorr


    }
    
    inline void drift_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEVector& deriv_out) const override {
        // Derivative of drift w.r.t. x(0) and x(1)
        deriv_out(0) = this->get_kappa(); // Derivative of drift w.r.t. x(0)
        deriv_out(1) = 0.0;
        }
    
    inline void diffusion_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEMatrix& deriv_out) const override {



        deriv_out.setZero();

    }

    inline void diffusion_second_derivative_x([[maybe_unused]] T t, [[maybe_unused]] const SDEVector& x, SDEMatrix& deriv_out) const override {
        // For GenericSVModel, the second derivative of diffusion is zero
        deriv_out.setZero();
      
    }

    /**
     * @brief Struct to hold polynomial coefficients for the stochastic volatility model.
     * This struct contains the coefficients for the polynomial representation of the drift and diffusion terms.
     * It is used to build the generator matrix for polynomial expansion methods.
     *
     * The coefficients are stored as vectors, where each vector corresponds to a polynomial term:
     * - bx: Coefficients for b_x(v)
     * - axx: Coefficients for a_xx(v)
     * - bv: Coefficients for b_v(v)
     * - axv: Coefficients for a_xv(v)
     * - avv: Coefficients for a_vv(v)
     *
     * The coefficients are indexed by the polynomial degree, allowing for efficient polynomial evaluation.
     * This struct is essential for constructing the generator matrix and performing polynomial expansions in the SDE model.
     */
    struct SVPolyCoeffs {
        SDEVector bx;   // b_x(v)
        SDEVector axx;  // a_xx(v)
        SDEVector bv;   // b_v(v)
        SDEVector axv;  // a_xv(v)
        SDEVector avv;  // a_vv(v)
    };


    /**
     * @brief Builds the polynomial coefficients for the stochastic volatility model.
     * @param p Parameters of the stochastic volatility model.
     * @param N Number of polynomial coefficients to compute.
     * @return SVPolyCoeffs containing the polynomial coefficients for the drift and diffusion terms.
     */
    virtual SVPolyCoeffs build_sv_polynomials(
        int N
    ) {
        SVPolyCoeffs out;

        // b_x(v) = μ - 0.5 v^(2p)
        out.bx = SDEVector::Zero(N);
        out.bx[0] = this->get_drift();
        {
            int exp = static_cast<int>(2 * this->get_p());
            if (exp < N)
                out.bx[exp] = out.bx[exp] - T(0.5);
        }

        // a_xx(v) = v^(2p)
        out.axx = SDEVector::Zero(N);
        {
            int exp = static_cast<int>(2 * this->get_p());
            if (exp < N)
                out.axx[exp] = T(1.0);
        }

        // b_v(v) = κ θ - κ v
        out.bv = SDEVector::Zero(N);
        out.bv[0] = this->get_kappa() * this->get_theta();
        if (1 < N)
            out.bv[1] = -this->get_kappa();

        // a_xv(v) = ρ σ v^(p+q)
        out.axv = SDEVector::Zero(N);
        {
            int exp = static_cast<int>(this->get_p() + this->get_q());
            if (exp < N)
                out.axv[exp] = this->get_correlation() * this->get_sigma_v();
        }

        // a_vv(v) = σ² v^(2q)
        out.avv = SDEVector::Zero(N);
        {
            int exp = static_cast<int>(2 * this->get_q());
            if (exp < N)
                out.avv[exp] = this->get_sigma_v() * this->get_sigma_v();
        }

        return out;
    }



    /**
     * @brief Constructs the generator matrix G for the stochastic volatility model.
     * @param E Vector of pairs representing the indices of the basis functions.
     * @param N Number of basis functions.
     * @param sigma Volatility parameter.
     * @return The generator matrix G as an SDEMatrix.
     *
     * This method constructs the generator matrix G based on the polynomial coefficients derived from the stochastic volatility model.
     * The matrix is built using the coefficients for the drift and diffusion terms, and it is projected to a triangular form based on the basis functions defined in E.
     * The resulting matrix G is used in polynomial expansion methods for solving the SDE. This version of the method uses an approximation when it comes to mixture models, since
     * the sigma parameter provided is approximated through moment matching and is not the actual volatility of the process.
     */
    virtual SDEMatrix generator_G(std::vector<std::pair<int,int>> E, int N, T sigma) const {

        const int M = static_cast<int>(E.size());

        // Fast index lookup
        Utils::IndexMap idx;
        idx.reserve(M);
        for (int i = 0; i < M; ++i) {
            const auto& [m, n] = E[i];
            idx.emplace(key(m, n), i);
        }

        SDEMatrix G = SDEMatrix::Zero(M, M);

        const auto q = this->get_q();
        const auto p = this->get_p();
        const auto kappa = this->get_kappa();
        const auto theta = this->get_theta();
        const auto rho = this->get_correlation();
        const auto r = this->get_drift();
        const auto sigma_v = this->get_sigma_v();

        auto add_entry = [&](int row_m, int row_n, int col, T value) {
            if (row_m < 0 || row_n < 0) return;
            if (row_m + row_n > N) return; // out of basis range
            auto it = idx.find(key(row_m, row_n));
            if (it != idx.end()) {
                G(it->second, col) += value; // Since p and q are dynamic, the same cell may be called multiple times, so we must add to the existing value
            }
        };

        for (int col = 0; col < M; ++col) {
            const auto [m, n] = E[col];

            // 1) h_{m-2+2q, n}
            if (m >= 2 - 2 * q) {

                add_entry(m-2 + 2 * q, n, col, sigma_v * sigma_v * 0.5 * m * (m - 1));
            }
            // 2) h_{m-1+p+q, n-1}
            if (m >= 1 - p - q && n >= 1) {
            
                add_entry(m - 1 + p + q, n-1, col, rho * sigma_v * m * std::sqrt((T)n) / sigma);
            }
            // 3) h_{m-1, n}
            if (m >= 1) {
                
                add_entry(m - 1, n, col, m * kappa * theta);
            }
            // 4) h_{m, n-1}
            if (n >= 1) {
                
                add_entry(m, n-1, col, std::sqrt((T)n) / sigma * r);
            }
            // 5) h_{m+2p, n-2}
            if (n >= 2) {
                
                add_entry(m + 2 * p, n-2, col, std ::sqrt((T)n * (n - 1)) / (2.0 * sigma * sigma));
            }
            // 6) h_{m, n}
            add_entry(m, n, col, -m * kappa);

            // 7) h_{m+2p, n-1}
            if (n >= 1) {
                
                add_entry(m + 2 * p, n-1, col, -0.5 * std::sqrt((T)n) / sigma);
            }
        }

        return G;

    }; 

    /**
     * @brief Constructs the generator matrix G for the stochastic volatility model.
     * @param E Vector of pairs representing the indices of the basis functions.
     * @param H The SDEMatrix representing the current state of the model.
     * @return The generator matrix G as an SDEMatrix.
     * 
     * This method constructs the generator matrix G based on the polynomial coefficients derived from the stochastic volatility model.
     * The matrix is built using the coefficients for the drift and diffusion terms, and it is projected to a triangular form based on the basis functions defined in E.
     * The resulting matrix G is used in polynomial expansion methods for solving the SDE.
     * This version of the method uses Kronecker products to build the generator matrix G based on the provided basis H, which can be also a mixture.
     */
    SDEMatrix generator_G(std::vector<std::pair<int,int>> E, const SDEMatrix& H){

        auto const N = static_cast<int>(H.rows());

        auto coeffs = build_sv_polynomials(N);

        auto G = Utils::build_G_full(H, coeffs.bx, coeffs.axx, coeffs.bv, coeffs.axv, coeffs.avv, N);

        auto G_projected = Utils::project_to_triangular(G, E, N);

        return G_projected;

    }


    inline T get_x0() const noexcept override  {
        // Return the initial state vector
        return this->m_x0(1);
    }

    inline void set_x0(const T& x0) noexcept override  {
        // Set the initial state vector
        this->m_x0(1) = x0;
    }

    inline T get_v0() const noexcept override{
        return this->m_x0(0); // Return the initial variance (x(1))
    }

    inline void set_v0(const T& v0) noexcept override{
        this->m_x0(0) = v0; // Set the initial variance (x(1))
    }

    /**
     * @brief Getters for model parameters.
     * These methods provide access to the model parameters used in the SDE.
     * They are useful for retrieving specific parameters without exposing the entire Parameters struct.
     */
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

    inline T get_drift() const noexcept {
        return params_.asset_drift_const; // Constant drift term of log-price
    }


    /**
     * @brief Computes the Mean of the SDE at time T by employing an IJK scheme combined with weighted Monte Carlo integration.
     * @param ttm Time to maturity.
     * @param dt Time step size.
     * @param y_t Current state of the SDE (matrix form).
     * @param w_t Wiener increments (matrix form).
     * @return The mean of the SDE at time T as an SDEVector.
     *  
     * This method computes the mean of the SDE at time T using a combination of trapezoidal rule for integration and stochastic integral for the Wiener increments.
     * It handles the correlation between the asset and volatility processes and applies the appropriate transformations based on the model parameters.
     * The method assumes that y_t and w_t are matrices with compatible dimensions.
     * 
     */
    inline SDEVector M_T(T ttm, T dt, const SDEMatrix& y_t, const SDEMatrix& w_t) {
        const auto n = y_t.cols();
        const auto k = y_t.rows();

        // Quick sanity check
        assert(w_t.cols() == n - 1 && w_t.rows() == k && "y_t and w_t must have the same number of rows and w_t must have one less column than y_t");

        // Precompute .matrix() view once
        const auto& y_mat = y_t.matrix();
        const auto& w_mat = w_t.matrix();

        // Precompute powers
        auto y_t_asset = y_mat.array().pow(this->get_p()).matrix();
        auto y_t_vol   = y_mat.array().pow(this->get_q()).matrix();

        // 1. Trapezoidal rule
        auto trap_block1 = y_t_asset.array().square().block(0, 0, k, n - 1);
        auto trap_block2 = y_t_asset.array().square().block(0, 1, k, n - 1);
        auto trap = -static_cast<T>(0.5) * dt * (trap_block1 + trap_block2).rowwise().sum();

        // 2. Vanilla stochastic integral
        auto stoch = this->get_correlation() * y_t_asset.block(0, 1, k, n - 1).cwiseProduct(w_t).rowwise().sum();

        // 3. IJK term
        auto w_sq_minus_dt = w_mat.array().square() - dt;
        auto ijk = this->get_correlation() * static_cast<T>(0.5) * this->get_sigma_v()
                * y_t_vol.block(0, 1, k, n - 1).cwiseProduct(w_sq_minus_dt.matrix()).rowwise().sum();


        // Final result
        auto result = get_x0()
                    + this->get_drift() * ttm
                    + trap.array() + stoch.array() + ijk.array();

        return result;
    };

    /**
     * @brief Computes the C_T term for the SDE at time T.
     * @param ttm Time to maturity.
     * @param dt Time step size.
     * @param y_t Current state of the SDE (matrix form).
     * @return The C_T term as an SDEVector.
     * This method computes the C_T term for the SDE at time T using the trapezoidal rule.
     * It handles the correlation between the asset and volatility processes and applies the appropriate transformations based on the model parameters.
     * The method assumes that y_t is a matrix with compatible dimensions.
     */
    inline SDEVector C_T([[maybe_unused]] T ttm, T dt, const SDEMatrix& y_t) {
        const auto n = y_t.cols();
        const auto k = y_t.rows();


        // Precompute .matrix() view once
        const auto& y_mat = y_t.matrix();

        // Precompute powers
        auto y_t_asset = y_mat.array().pow(2 * this->get_p()).matrix();

        // 1. Trapezoidal rule
        auto trap_block1 = y_t_asset.block(0, 0, k, n - 1);
        auto trap_block2 = y_t_asset.block(0, 1, k, n - 1);
        auto trap = static_cast<T>(0.5) * dt * (trap_block1 + trap_block2).rowwise().sum();

        // Final result
        auto result = (1 - this->get_correlation()*this->get_correlation()) * trap.array();

        return result;
    };

};

/**
 * @brief Heston Model SDE implementation.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 */
template<typename T = traits::DataType::PolynomialField>
class HestonModelSDE : public GenericSVModelSDE<T> {
public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;

    HestonModelSDE(T asset_drift_const,
                   T sv_kappa,
                   T sv_theta,
                   T sv_sigma,
                   T correlation,
                   SDEVector x0)
        : Base(Parameters{
            asset_drift_const,
            sv_kappa,
            sv_theta,
            sv_sigma,
            correlation,
            static_cast<T>(0.5), // asset_vol_exponent p
            static_cast<T>(0.5)  // sv_vol_exponent q
        }, x0) {}

    std::shared_ptr<ISDEModel<T>> clone() const override {
        return std::make_shared<HestonModelSDE<T>>(*this);
    }


    inline void characteristic_fn(T t, const SDEComplexVector& x, SDEComplexVector& charact_out) const override{

            // Step 1: gamma
        SDEComplexVector d = ((this->get_kappa() - ImaginaryUnit<> * this->get_correlation() * this->get_sigma_v() * x.array()).square()
                            + this->get_sigma_v() * this->get_sigma_v() * (x.array() * (x.array() + ImaginaryUnit<>))).matrix();
        SDEComplexVector gamma = d.array().sqrt().matrix();

                // Step 2: A (exp1 + exp2)

        SDEComplexVector exp1 = gamma.array() + this->get_kappa() - ImaginaryUnit<> * this->get_correlation() * this->get_sigma_v() * x.array();
        SDEComplexVector exp2 = gamma.array() - this->get_kappa() + ImaginaryUnit<> * this->get_correlation() * this->get_sigma_v() * x.array();
        SDEComplexVector A = (this->get_kappa() * this->get_theta() / (this->get_sigma_v() * this->get_sigma_v())) * ((this->get_kappa() - gamma.array() -
                            ImaginaryUnit<> * this->get_correlation() * this->get_sigma_v() * x.array()) * t - 2 *((exp1.array() + exp2.array() * (-gamma.array() * t).exp())/
                            (exp1.array() + exp2.array())).log()).matrix();



        SDEComplexVector B_func_out;

        {
        const auto& x_arr = x.array();
        const auto& gamma_arr = gamma.array();
        const T one = T(1.0);
        const T two = T(2.0);

        const auto& kappa = this->get_kappa();
        const auto& rho = this->get_correlation();
        const auto& sigma = this->get_sigma_v();

        auto i_x = ImaginaryUnit<> * x_arr;
        auto x_sq = x_arr.square();
        auto exp_gamma_t = (gamma_arr * t).exp();
        auto exp_gamma_t_minus1 = exp_gamma_t - one;

        // Numerator and denominator
        auto numerator = (x_sq + i_x) * exp_gamma_t_minus1;
        auto den_term1 = gamma_arr + kappa - ImaginaryUnit<> * rho * sigma * x_arr;
        auto denominator = den_term1 * exp_gamma_t_minus1 + two * gamma_arr;

        B_func_out = numerator / denominator;

        // Fallback correction mask: detect NaN or Inf using Eigen’s select expression
        Eigen::Array<bool, Eigen::Dynamic, 1> mask = 
            (!((B_func_out.array().real().isFinite()) && (B_func_out.array().imag().isFinite())));

        if (mask.any()) {
            // Precompute fallback only for required entries
            auto fallback_num = x_sq + i_x;
            auto fallback_den = gamma_arr + kappa - ImaginaryUnit<> * rho * sigma * x_arr;

            // Apply fallback using Eigen::select
            B_func_out = mask.select(fallback_num / fallback_den, B_func_out);
            }
        }


        charact_out = (A.array() - B_func_out.array() * Base::m_x0(0)).exp().matrix();

    }

};


/**
 * @brief Stein-Stein Model SDE implementation.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 * This model is a specific case of the GenericSVModelSDE with p=1 and q=0.
 */
template<typename T = traits::DataType::PolynomialField>
class SteinSteinModelSDE : public GenericSVModelSDE<T> {
public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;

    SteinSteinModelSDE(T asset_drift_const,
                   T sv_kappa,
                   T sv_theta,
                   T sv_sigma,
                   T correlation,
                    SDEVector x0)
        : Base(Parameters{
            asset_drift_const,
            sv_kappa,
            sv_theta,
            sv_sigma,
            correlation,
            static_cast<T>(1.0), // asset_vol_exponent p
            static_cast<T>(0.0)  // sv_vol_exponent q
        }, x0) {}

    std::shared_ptr<ISDEModel<T>> clone() const override {
        return std::make_shared<SteinSteinModelSDE<T>>(*this);
    }


};

/**
 * @brief Hull-White Model SDE implementation.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 * This model is a specific case of the GenericSVModelSDE with p=1 and q=1.
 * It is commonly used in interest rate modeling.
 */
template<typename T = traits::DataType::PolynomialField>
class HullWhiteModelSDE : public GenericSVModelSDE<T> {
public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;

    HullWhiteModelSDE(T asset_drift_const,
                   T sv_kappa,
                   T sv_theta,
                   T sv_sigma,
                   T correlation,
                   SDEVector x0)
        : Base(Parameters{
            asset_drift_const,
            sv_kappa,
            sv_theta,
            sv_sigma,
            correlation,
            static_cast<T>(1.0), // asset_vol_exponent p
            static_cast<T>(1.0)  // sv_vol_exponent q
        }, x0) {}
    
    std::shared_ptr<ISDEModel<T>> clone() const override {
        return std::make_shared<HullWhiteModelSDE<T>>(*this);
    }

};

/**
 * @brief Jacobi Model SDE implementation.
 * @tparam T Numeric type for computations (default: traits::DataType::PolynomialField).
 * This model is a specific case of the GenericSVModelSDE with p=1 and q=0.
 * It is used for modeling processes with bounded variance, such as in the Jacobi process.
 */
template<typename T = traits::DataType::PolynomialField>
class JacobiModelSDE : public GenericSVModelSDE<T> {

public:
    using Base = GenericSVModelSDE<T>;
    using Parameters = typename Base::Parameters;
    using Base::generator_G;                 // bring the base class generator_G into scope


    static constexpr unsigned int WIENER_DIM = 2; // Two correlated Wiener processes
    static constexpr unsigned int STATE_DIM = 2; // Two state variables (log-price and variance)

private:

    T y_min_;
    T y_max_;
    T q_denominator_sq_; // (sqrt(y_max) - sqrt(y_min))^2

    /**
     * @brief Computes the Q function for the Jacobi model.
     * @param y The input value for which to compute Q(y).
     * @return The computed value of Q(y).
     * 
     * This function computes the Q function for the Jacobi model, which is used to transform the variance process.
     * It ensures that the input value y is within the bounds defined by y_min and y_max, and handles cases where the denominator might be zero.
     * The Q function is defined as:
     * Q(y) = (y - y_min) * (y_max - y) / (sqrt(y_max) - sqrt(y_min))^2
     * 
     * If the denominator is zero (which can happen if y_max is very close to y_min), it returns 0.0 to avoid division by zero.
     * The function also clamps the input value y to ensure it stays within the bounds [y_min, y_max].
     */
    inline T Q_func(const T y) const {

        if (q_denominator_sq_ <= 0) return 0.0; // Avoid division by zero if y_max approx y_min

        // Ensure y is within bounds for Q(y) to be non-negative, or handle appropriately
        // The SDE formulation usually assumes Y_t stays within [y_min, y_max]

        T y_clamped = std::max(y_min_, std::min(y, y_max_));
        return (y_clamped - y_min_) * (y_max_ - y_clamped) / q_denominator_sq_;

    }

    /**
     * @brief Computes the Q function for a matrix of values.
     * @param y The input matrix for which to compute Q(y).
     * @return An Eigen::Array containing the computed values of Q(y) for each element in the input matrix.
     * 
     * This function computes the Q function for each element in the input matrix y, which is expected to be an Eigen::MatrixBase type.
     * It ensures that the input values are clamped within the bounds defined by y_min and y_max, and handles cases where the denominator might be zero.
     * The Q function is defined as:
     * Q(y) = (y - y_min) * (y_max - y) / (sqrt(y_max) - sqrt(y_min))^2
     * 
     * If the denominator is zero (which can happen if y_max is very close to y_min), it returns an array of zeros with the same shape as the input matrix.
     * The function uses Eigen's array operations for efficient computation and broadcasting.
     * 
     * @tparam Derived The type of the Eigen matrix (must be derived from Eigen::MatrixBase).
     * @return An Eigen::Array containing the computed values of Q(y) for each element in the input matrix.
     * 
     * This function is useful for vectorized operations where the Q function needs to be applied to multiple values simultaneously.
     * It leverages Eigen's capabilities for efficient computation and broadcasting, making it suitable for large datasets.
     * 
     * @note The function assumes that the input matrix y is compatible with the defined bounds [y_min, y_max].
     * If the input values exceed these bounds, they will be clamped to ensure valid computations.
     */
    template<typename Derived>
    Eigen::Array<typename Derived::Scalar,
                Derived::RowsAtCompileTime,
                Derived::ColsAtCompileTime>
    Q_func(const Eigen::MatrixBase<Derived>& y) const
    {
        using R = typename Derived::Scalar;
        using ReturnType = Eigen::Array<R, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;

        if (q_denominator_sq_ <= R(0)) {
            // materialize the Zero() expression to the concrete ReturnType
            return ReturnType::Zero(y.rows(), y.cols()).eval();
        }

        // work in array-space for broadcasting with scalars
        ReturnType y_arr = y.array().template cast<R>().eval();
        ReturnType y_clamped = y_arr.min(R(y_max_)).max(R(y_min_)).eval();

        return ((y_clamped - R(y_min_)) * (R(y_max_) - y_clamped) / R(q_denominator_sq_)).eval();
    }

    /**
     * @brief Computes the first and second derivatives of the Q function with respect to y.
     * @param y The input value for which to compute the derivatives.
     * @return The computed first derivative of Q(y) with respect to y.
     * 
     * This function computes the first derivative of the Q function for the Jacobi model, which is used to transform the variance process.
     * It ensures that the input value y is within the bounds defined by y_min and y_max, and handles cases where the denominator might be zero.
     * The first derivative is defined as:
     * Q'(y) = (y_max - 2y + y_min) / (sqrt(y_max) - sqrt(y_min))^2
     * 
     * If the denominator is zero (which can happen if y_max is very close to y_min), it returns 0.0 to avoid division by zero.
     * The function also clamps the input value y to ensure it stays within the bounds [y_min, y_max]. 
     */
    
    inline T Q_func_der1(const T y) const {

        if (q_denominator_sq_ <= 0) return 0.0; 

        

        T y_clamped = std::max(y_min_, std::min(y, y_max_));
        return  (y_max_ - static_cast<T>(2.0)*y_clamped + y_min_) / q_denominator_sq_;

    }


    /**
     * @brief Computes the second derivative of the Q function with respect to y.
     * @param y The input value for which to compute the second derivative.
     * @return The computed second derivative of Q(y) with respect to y.
     * 
     * This function computes the second derivative of the Q function for the Jacobi model, which is used to transform the variance process.
     * It ensures that the input value y is within the bounds defined by y_min and y_max, and handles cases where the denominator might be zero.
     * The second derivative is defined as:
     * Q''(y) = -2 / (sqrt(y_max) - sqrt(y_min))^2
     * 
     * If the denominator is zero (which can happen if y_max is very close to y_min), it returns 0.0 to avoid division by zero.
     * The function also clamps the input value y to ensure it stays within the bounds [y_min, y_max].
     */
    inline T Q_func_der2(const T  y) const {

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
            T y_max,
            SDEVector x0)
    : Base(Parameters{asset_drift_const, sv_kappa, sv_theta, sv_sigma, correlation, static_cast<T>(1.0), static_cast<T>(1.0)}, x0),
      y_min_(y_min),y_max_(y_max)
        {
        if (this->get_correlation() < -1.0 || this->get_correlation() > 1.0) {
            throw std::invalid_argument("JacobiModelSDE: Rho must be between -1 and 1.");
        }

        if (y_min_ < 0.0 || y_max_ <= y_min_) {
            throw std::invalid_argument("JacobiModelSDE: Invalid y_min/y_max. y_min >= 0 and y_max > y_min required.");
        }

        if (this->get_theta() < y_min_ || this->get_theta() > y_max_) {
             std::cerr << "Warning: JacobiModelSDE: Theta is outside variance bounds [y_min, y_max].\n";
        }


        if (this->get_sigma_v() <= 0) {
            throw std::invalid_argument("JacobiModelSDE: Sigma (vol of vol factor) must be positive.");
        }

        if (this->get_kappa() < 0) {
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


    inline void drift([[maybe_unused]] T t, const SDEVector& x_state, SDEVector& mu_out) const override {

        const T Y_t = x_state(0);


        // Drift for X_t (log-price)
        mu_out(1) = this->get_drift()  - Y_t * static_cast<T>(0.5);


        // Drift for Y_t
        mu_out(0) = this->get_kappa() * (this->get_theta() - Y_t);

    }

    inline void diffusion([[maybe_unused]] T t, const SDEVector& x_state, SDEMatrix& sigma_out) const override {

        const T Y_t = x_state(0); // Current variance process value


        T q_y = Q_func(Y_t);
        T sqrt_q_y = (q_y > 0.0) ? std::sqrt(q_y) : 0.0;


        T Y_minus_rho_sq_Q = Y_t - this->get_correlation() * this->get_correlation() * q_y;
        T sqrt_Y_minus_rho_sq_Q = (Y_minus_rho_sq_Q > 0.0) ? std::sqrt(Y_minus_rho_sq_Q) : 0.0;
        
        // dY_t = ... + sigma*sqrt(Q(Y_t))dW_1t

        // dX_t = ... + rho*sqrt(Q(Y_t))dW_1t + sqrt(Y_t - rho^2*Q(Y_t))dW_2t


        
        // Row 0: variance (Y_t) diffusion coefficients for dW1, dW2

        sigma_out(0, 0) = this->get_sigma_v() * sqrt_q_y;
        sigma_out(0, 1) = 0.0;

        // Row 1: log-price (X_t) diffusion coefficients for dW1, dW2

        sigma_out(1, 0) = this->get_correlation() * sqrt_q_y;
        sigma_out(1, 1) = sqrt_Y_minus_rho_sq_Q;


    }

    typename Base::SVPolyCoeffs build_sv_polynomials(
        int N) override
        {
        typename Base::SVPolyCoeffs out;

        const T C = q_denominator_sq_;
        const T vmin = y_min_;
        const T vmax = y_max_;

        // b_x(v) = μ - 0.5 v
        out.bx = SDEVector::Zero(N);
        out.bx[0] = this->get_drift();
        {
            if (1 < N)
                out.bx[1] = out.bx[1]  - T(0.5);
        }

        // a_xx(v) = v
        out.axx = SDEVector::Zero(N);
        {
            if (1 < N)
                out.axx[1] = T(1.0);
        }

        // b_v(v) = κ θ - κ v
        out.bv = SDEVector::Zero(N);
        out.bv[0] = this->get_kappa() * this->get_theta();
        if (1 < N)
            out.bv[1] = -this->get_kappa();

        // a_xv(v) = ρ σ Q(v)
        out.axv = SDEVector::Constant(N, this->get_correlation() * this->get_sigma_v());
        {
            out.axv[0] *= -vmin * vmax / C; // Adjust for the constant term
            if (1 < N)
                out.axv[1] *=  (vmin + vmax) / C; 
            if (2 < N)
                out.axv[2] /= -C; 
 
        }
        // a_vv(v) = σ² Q(v)
        out.avv = SDEVector::Constant(N, this->get_sigma_v() * this->get_sigma_v());
        {
            out.avv[0] *= -vmin * vmax / C; // Adjust for the constant term
            if (1 < N)
                out.avv[1] *= (vmin + vmax) / C; 
            if (2 < N)
                out.avv[2] /= -C; 
 
        }

        return out;
    }

    SDEMatrix generator_G(std::vector<std::pair<int,int>> E, int N, T sigma) const override{
        const int M = static_cast<int>(E.size());

        // Fast index lookup
        Utils::IndexMap idx;
        idx.reserve(M);
        for (int i = 0; i < M; ++i) {
            const auto& [m, n] = E[i];
            idx.emplace(key(m, n), i);
        }

        SDEMatrix G = SDEMatrix::Zero(M, M);
        

        const T C = q_denominator_sq_;
        const T vmin = y_min_;
        const T vmax = y_max_;
        const T kappa = this->get_kappa();
        const T theta = this->get_theta();
        const T r = this->get_drift();
        const T sigma_v = this->get_sigma_v();
        const T rho = this->get_correlation();

        auto add_entry = [&](int row_m, int row_n, int col, T value) {
            if (row_m < 0 || row_n < 0) return;
            if (row_m + row_n > N) return; // out of basis range
            auto it = idx.find(key(row_m, row_n));
            if (it != idx.end()) {
                G(it->second, col) = value;
            }
        };

        for (int col = 0; col < M; ++col) {
            const auto [m, n] = E[col];


            // 1) h_{m-2, n}
            if (m >= 2) {

                add_entry(m-2, n, col, -sigma_v * sigma_v * m * (m - 1) * vmax * vmin / (2.0 * C));
            }
            // 2) h_{m-1, n-1}
            if (m >= 1 && n >= 1) {

                add_entry(m-1, n-1, col, -rho * sigma_v * m * std::sqrt((T)n) / sigma * vmax * vmin / C);
            }
            // 3) h_{m-1, n}
            if (m >= 1) {

                add_entry(m-1, n, col, m * kappa * theta + sigma_v * sigma_v * m * (m - 1) * (vmax + vmin) / (2.0 * C));
            }
            // 4) h_{m, n-1}
            if (n >= 1) {

                add_entry(m, n-1, col, std::sqrt((T)n) / sigma * (r   + m * rho * sigma_v * (vmax + vmin) / C));
            }
            // 5) h_{m+1, n-2}
            if (n >= 2) {

                add_entry(m+1, n-2, col, std::sqrt((T)n * (n - 1)) / (2.0 * sigma * sigma));
            }
            // 6) h_{m, n}
            add_entry(m, n, col, -m * kappa - sigma_v * sigma_v * m * (m - 1) / (2.0 * C));
            // 7) h_{m+1, n-1}
            if (n >= 1) {
        
                add_entry(m+1, n-1, col, -std::sqrt((T)n) / (2.0 * sigma) - rho * sigma_v * m * std::sqrt((T)n) / (sigma * C));
            }
        }

        return G;
    }

    std::shared_ptr<ISDEModel<T>> clone() const override {
        return std::make_shared<JacobiModelSDE<T>>(*this);
    }

    inline SDEVector M_T(T ttm, T dt, const SDEMatrix& y_t, const SDEMatrix& w_t) {
            const auto n = y_t.cols();
            const auto k = y_t.rows();

            // Quick sanity check
            assert(w_t.cols() == n - 1 && w_t.rows() == k && "y_t and w_t must have same shape");
            // Precompute .matrix() view once
            const auto& y_mat = y_t.matrix();
            const auto& w_mat = w_t.matrix();

            // Precompute powers
            auto y_t_vol   = Q_func(y_mat).array().sqrt().matrix().eval(); // Use eval to get rid of lazy evaluation
 
            // 1. Trapezoidal rule
            auto trap_block1 = y_mat.array().block(0, 0, k, n - 1);
            auto trap_block2 = y_mat.array().block(0, 1, k, n - 1);
            auto trap = -static_cast<T>(0.25) * dt * (trap_block1 + trap_block2).rowwise().sum();

            // 2. Vanilla stochastic integral
            auto stoch = this->get_correlation() * y_t_vol.block(0, 0, k, n - 1).cwiseProduct(w_t).rowwise().sum();

            // 3. IJK term
            auto w_sq_minus_dt = w_mat.array().square() - dt;
            auto ijk = this->get_correlation() * static_cast<T>(0.5) * this->get_sigma_v()
                    * y_t_vol.block(0, 0, k, n - 1).cwiseProduct(w_sq_minus_dt.matrix()).rowwise().sum();

            // Final result
            auto result = Base::get_x0()
                        + this->get_drift() * ttm
                        + trap.array() + stoch.array() + ijk.array();

            return result;
        };

    inline SDEVector C_T([[maybe_unused]] T ttm, T dt, const SDEMatrix& y_t) {
        const auto n = y_t.cols();
        const auto k = y_t.rows();

        // Precompute .matrix() view once
        const auto& y_mat = y_t.matrix();


        // Precompute powers
        auto y_t_asset = (y_mat - this->get_correlation() * this->get_correlation() * Q_func(y_mat).matrix()).eval(); // Use Q_func for variance process

        std::cout << "y_t_asset:\n" << Q_func(y_mat) << std::endl;

        // 1. Trapezoidal rule
        auto trap_block1 = y_t_asset.block(0, 0, k, n - 1);
        auto trap_block2 = y_t_asset.block(0, 1, k, n - 1);
        std::cout << "n = " << n << ", k = " << k << "\n";
        std::cout << "trap_block1:\n" << trap_block1 << "\n";
        std::cout << "trap_block2:\n" << trap_block2 << "\n";



        auto trap = static_cast<T>(0.5) * dt * (trap_block1 + trap_block2).rowwise().sum();

        // Final result
        auto result = trap.array();

        return result;
    };

    /**
     * @brief Getters for the y_min parameter.
     * Minimum bound for the variance process in the Jacobi model.
     * It is useful for retrieving the lower bound without exposing the entire model parameters.
     * 
     * @return The minimum bounds for the variance process.
     */
    inline T get_y_min() const noexcept {
        return y_min_;
    };
    
    /**
     * @brief Getters for the y_max parameter.
     * This method provides access to the maximum bound for the variance process in the Jacobi model.
     * It is useful for retrieving the upper bound without exposing the entire model parameters.
     * 
     * @return The maximum bound for the variance process.
     */
    inline T get_y_max() const noexcept {
        return y_max_;
    };



};




}

#endif // FINMODELS_HPP