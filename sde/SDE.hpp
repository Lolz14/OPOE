
#ifndef SDE_HPP
#define SDE_HPP

#include <cmath>        // For std::sqrt
#include <concepts>     // For C++20 concepts
#include <iostream>     // For warnings (e.g., in placeholder solvers)
#include <random>       // For random number generation
#include <stdexcept>    // For std::invalid_argument, std::runtime_error
#include <vector>       
#include <thread>       // For std::this_thread::get_id
#include "../traits/OPOE_traits.hpp"
#include "../utils/Utils.hpp"

// Define aliases for Eigen types.
// These could be adapted if your "StoringVector" and "StoringMatrix" traits map to Eigen types.



namespace SDE {
using SDEVector = traits::DataType::StoringVector; // Eigen::VectorXd or similar
using SDEComplexVector = traits::DataType::ComplexStoringVector; // Eigen::VectorXcd or similar
using SDEMatrix = traits::DataType::StoringMatrix; // Eigen::MatrixXd or similar
using SDEArray = traits::DataType::StoringArray;   // SDEArray or similar
/**

 * @brief Returns a reference to a thread-local Mersenne Twister engine for random number generation.
 * @return Reference to thread-local std::mt19937 engine, seeded with std::random_device.

 */
inline std::mt19937& get_default_rng_engine() {
    // Get time since epoch in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    // Combine with thread ID hash
    size_t thread_hash = std::hash<std::thread::id>{}(std::this_thread::get_id());

    // Mixed seed
    unsigned int seed = static_cast<unsigned int>(nanos ^ thread_hash);

    static thread_local std::mt19937 engine(seed);
    return engine;
}

/**

 * @brief Returns a reference to a thread-local standard normal distribution.
 * @return Reference to thread-local std::normal_distribution<double> with mean 0 and std dev 1.

 */

inline std::normal_distribution<double>& get_standard_normal_dist() {

    static thread_local std::normal_distribution<double> standard_normal(0.0, 1.0);
    return standard_normal;

}

/**

 * @brief Returns a reference to a thread-local uniform distribution on [0, 1).
 * @return Reference to thread-local std::uniform_real_distribution<double>.

 */

inline std::uniform_real_distribution<double>& get_uniform_distribution() {

    static thread_local std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    return uniform_distribution;

}

// --- SDE Concepts for Compile-Time Constraints ---

/** 
 * @brief Concept for general SDE models, ensuring dimensional constants and drift/diffusion methods.
 */
template<typename T>
concept SDEModel = requires(const T& sde, double t, const SDEVector& x, SDEVector& mu_out, SDEMatrix& sigma_out) {
    
    { T::WIENER_DIM } -> std::convertible_to<unsigned int>;
    { T::STATE_DIM } -> std::convertible_to<unsigned int>;
    requires T::WIENER_DIM > 0;
    requires T::STATE_DIM > 0;
    sde.drift(t, x, mu_out);       // Drift function must write to mu_out
    sde.diffusion(t, x, sigma_out); // Diffusion function must write to sigma_out

};



/**
 * @brief Concept for Stochastic Volatility (SV) SDE models, extending SDEModel with SV-specific parameters.
 */
template<typename T>
concept SVSDEModel = SDEModel<T> && requires(const T& sde) {
    
    requires T::WIENER_DIM >= 2;// At least two independent Wiener processes
    requires T::STATE_DIM >= 2;// At least two independent Wiener processes
    { sde.get_correlation() } -> std::convertible_to<double>; // Correlation parameter (rho)
    { sde.get_kappa() } -> std::convertible_to<double>;       // Mean-reversion speed
    { sde.get_theta() } -> std::convertible_to<double>;       // Long-term variance/mean
    { sde.get_sigma_v() } -> std::convertible_to<double>;     // Volatility of volatility
    { sde.get_mu() } -> std::convertible_to<double>;          // Drift or risk-neutral rate
    { sde.get_p() } -> std::convertible_to<double>;           // Power parameter for volatility (if applicable)
    { sde.get_q() } -> std::convertible_to<double>;           // Additional power parameter (if applicable)
};

/**
 * @brief Concept for 1D SDEs supporting Milstein scheme, requiring derivative information.
 */

template<typename T>
concept Milstein1DSDEModel = SDEModel<T> && requires(const T& sde, double t, double x_scalar) {

    requires T::WIENER_DIM == 1;
    requires T::STATE_DIM >= 1;

};

/** 
 * @brief Concept for 2D Stochastic Volatility SDEs supporting Milstein scheme.
 */

template<typename T>
concept Milstein2DSVSDEModel = SDEModel<T> && requires(const T& sde, double t, const SDEVector& x) {

    requires T::WIENER_DIM == 2;// Two Wiener processes
    requires T::STATE_DIM >= 2;

};



// --- Base SDE Solver Class using CRTP ---

/**

 * @brief Base class for SDE solvers using CRTP for static polymorphism.

 * @tparam DerivedSolver The derived solver class (CRTP pattern).

 * @tparam SdeType The SDE model type satisfying SDEModel concept.

 */


template <typename DerivedSolver, SDEModel SdeType>
class SDESolverBase {
public:
    SDESolverBase(const SdeType& sde, std::mt19937& rng_engine = SDE::get_default_rng_engine())
        : sde_ref_(sde), rng_engine_ref_(rng_engine) {}

    virtual ~SDESolverBase() = default;

    /**
     * @brief Generates Wiener process increments for simulation.
     * @param dt Time step size.
     * @param dW_out Matrix to store increments (WIENER_DIM x num_steps).
     * @param num_steps Number of time steps.
     */

    void generate_wiener_increments(double dt, SDEMatrix& dW_out, int num_steps, int num_paths) const {
        if (dt < 0.0) throw std::invalid_argument("dt must be non-negative");

        if (dt == 0.0) {
            dW_out.setZero(num_paths * SdeType::WIENER_DIM, num_steps);
            return;
        }

        const int num_wiener = SdeType::WIENER_DIM;
        const double sqrt_dt = std::sqrt(dt);

        #pragma omp parallel for
        for (int w = 0; w < num_wiener; ++w) {
            // Unique seed per Wiener dimension (you can make this more robust if needed)
            SDEMatrix block = Utils::sampler<double>(
                get_default_rng_engine(),
                SDE::get_standard_normal_dist(),
                num_paths, num_steps
            ) * sqrt_dt;

            // Write the block into the correct location of dW_out
            dW_out.block(w * num_paths, 0, num_paths, num_steps) = block;
        }
}
    /**
     * @brief Solves the SDE path and invokes an observer at each step.
     * @param initial_x Initial state vector (wiener_dim x 1). We do not permit to use different initial states for each path.
     *                  This is a common assumption in SDE solvers for simplicity.
     * @param t_start Starting time.
     * @param t_end Ending time.
     * @param num_steps Number of time steps.
     * @param num_paths Number of paths to simulate.
     * @param observer Callback function with signature void(double time, const SDEVector& state).
     */

    void solve(const SDEVector& initial_x, double t_start, double t_end, int num_steps, int num_paths,
           const std::function<void(unsigned int, const SDEVector&)>& observer) const {

        if (initial_x.size() != SdeType::STATE_DIM) 
            throw std::invalid_argument("initial_x must have STATE_DIM");

        if (num_steps <= 0 || t_end <= t_start)
            throw std::invalid_argument("Invalid time range or num_steps");

        double dt = (t_end - t_start) / num_steps;

        // Flattened state vector: num_paths * STATE_DIM
        SDEVector current_x(num_paths * SdeType::STATE_DIM);

        // initial_x is a column vector: [0.25, 100]
        SDEMatrix initial_states = initial_x.transpose().replicate(num_paths, 1); 
        current_x = Eigen::Map<SDEVector>(initial_states.data(), initial_states.size());

                
        SDEVector next_x(num_paths * SdeType::STATE_DIM);

        // dW is matrix: (num_paths * WIENER_DIM, num_steps)
        SDEMatrix dW(num_paths * SdeType::WIENER_DIM, num_steps);

        
        
        generate_wiener_increments(dt, dW, num_steps, num_paths);

        

        double current_t = t_start;
        observer(0, current_x);

        for (int i = 0; i < num_steps; ++i) {
            
            this->derived().step(current_t, current_x, dt, dW.col(i), num_paths, next_x);
            current_x = next_x;
            current_t += dt;
            if (i == num_steps - 1) current_t = t_end;
            observer(i + 1, current_x);
        }
    }


    /**
     * @brief Solves the SDE and returns the full path.
     * @return Vector of state vectors at each time step (size num_steps + 1).
     */

    SDEMatrix solve(const SDEVector& initial_x, double t_start, double t_end, int num_steps, int num_paths) const {
        SDEMatrix path_flat(num_paths * SdeType::STATE_DIM, num_steps + 1);

        

        solve(initial_x, t_start, t_end, num_steps, num_paths,
            [&path_flat](unsigned int idx, const SDEVector& x) {
                path_flat.col(idx) = x;

            });

        return path_flat;
    }

protected:

    const SdeType& sde_ref_;
    std::mt19937& rng_engine_ref_;
    const DerivedSolver& derived() const { return *static_cast<const DerivedSolver*>(this); }
    DerivedSolver& derived() { return *static_cast<DerivedSolver*>(this); }

};



// --- Concrete Solver Implementations ---

// --- Euler-Maruyama Solver Implementation ---

/**

 * @brief Euler-Maruyama solver for general SDEs, a first-order method.

 * @tparam SdeType SDE model type satisfying SDEModel concept.

 */

template <SDEModel SdeType>
class EulerMaruyamaSolver : public SDESolverBase<EulerMaruyamaSolver<SdeType>, SdeType> {
public:
    using Base = SDESolverBase<EulerMaruyamaSolver<SdeType>, SdeType>;
    using Base::Base;

void step(
    double t, const SDEVector& current_x, double dt, const SDEVector& dW_t,
    int num_paths, SDEVector& next_x
    ) const {
        const int state_dim = SdeType::STATE_DIM;
        const int wiener_dim = SdeType::WIENER_DIM;

        #pragma omp parallel
        {
        // Thread-local storage for temporary buffers to avoid heap allocations
        SDEVector mu(state_dim);
        SDEMatrix sigma(state_dim, wiener_dim);

        #pragma omp for 
        for (int p = 0; p < num_paths; ++p) {
            const int offset = p * state_dim;

            // Map current and next state slices
            Eigen::Map<const SDEVector> x_p(current_x.data() + offset, state_dim);
            Eigen::Map<SDEVector> next_x_p(next_x.data() + offset, state_dim);

            // Compute drift and diffusion
            this->sde_ref_.drift(t, x_p, mu);
            this->sde_ref_.diffusion(t, x_p, sigma);

            // Map dW for this path, spaced by num_paths per Wiener dimension
            Eigen::Map<const SDEVector, 0, Eigen::InnerStride<>> dW_p(dW_t.data() + p, wiener_dim, Eigen::InnerStride<>(num_paths));

            // Euler-Maruyama update
            next_x_p.noalias() = x_p + mu * dt + sigma * dW_p;
        }
    }
    }
};


// --- Milstein Solver Implementation ---

/**
 * @brief Milstein solver for SDEs, a higher-order method using derivative information.
 * @tparam SdeType SDE model type satisfying SDEModel concept.
 */

template <SDEModel SdeType>
class MilsteinSolver : public SDESolverBase<MilsteinSolver<SdeType>, SdeType> {

public:

    using Base = SDESolverBase<MilsteinSolver<SdeType>, SdeType>;
    using Base::Base;

    void step(double t, const SDEVector& current_x, double dt, const SDEVector& dW_t, int num_paths, SDEVector& next_x) const {

        const int state_dim = SdeType::STATE_DIM;
        const int wiener_dim = SdeType::WIENER_DIM;

        #pragma omp parallel
        {
        // Thread-local working memory
        SDEVector mu(state_dim);
        SDEMatrix sigma(state_dim, wiener_dim);

        // Optional buffers for Milstein models
        SDEVector drift_prime_at_x(state_dim);
        SDEMatrix sigma_prime_at_x(state_dim, wiener_dim);
        SDEMatrix sigma_second_at_x(state_dim, wiener_dim);

        #pragma omp for
        for (int p = 0; p < num_paths; ++p) {
            const int offset = p * state_dim;

            Eigen::Map<const SDEVector> x_p(current_x.data() + offset, state_dim);
            Eigen::Map<SDEVector> next_x_p(next_x.data() + offset, state_dim);

            this->sde_ref_.drift(t, x_p, mu);
            this->sde_ref_.diffusion(t, x_p, sigma);

    

            Eigen::Map<const SDEVector, 0, Eigen::InnerStride<>> dW_p(
                dW_t.data() + p, // start at row=p
                wiener_dim,
                Eigen::InnerStride<>(num_paths)
            );

            next_x_p.noalias() = x_p + mu * dt + sigma * dW_p;

            // Milstein correction term:
            if constexpr (Milstein1DSDEModel<SdeType>) {
            // This block is only compiled if SdeType meets the Milstein1DSDEModel concept.

            this->sde_ref_.drift_derivative_x(t, x_p, drift_prime_at_x);
            this->sde_ref_.diffusion_derivative_x(t, x_p, sigma_prime_at_x);
            this->sde_ref_.diffusion_second_derivative_x(t, x_p, sigma_second_at_x);

            const SDEMatrix correction = 0.5 * sigma
                .cwiseProduct(sigma_prime_at_x
                .cwiseProduct((dW_p.array().square() - dt).matrix()));            
            const SDEMatrix correction_two = 
            (1.0 / 6.0) *
            (
                sigma
                    .cwiseProduct(sigma_prime_at_x)
                    .cwiseProduct(sigma_prime_at_x)
                + 
                sigma_second_at_x
                    .cwiseProduct(sigma)
                    .cwiseProduct(sigma)
            ).cwiseProduct(dW_p.array().cube().matrix());
            const SDEMatrix term1_3 = sigma.array().colwise() * drift_prime_at_x.array(); // σ · μ'
            const SDEMatrix term2_3 = sigma_prime_at_x.array().colwise() * mu.array();    // σ' · μ
            const SDEMatrix term3_3 = sigma
                .cwiseProduct(sigma_prime_at_x)
                .cwiseProduct(sigma_prime_at_x);                                              // σ · σ'^2
            const SDEMatrix term4_3 = 0.5 * sigma_second_at_x
                .cwiseProduct(sigma)
                .cwiseProduct(sigma);                                                     // 0.5 · σ'' · σ²
            const SDEMatrix correction_three = 0.5 * (term1_3 + term2_3 - term3_3 - term4_3)
                .cwiseProduct(dW_p) * dt;

        

            next_x_p.noalias() += (correction + correction_two + correction_three);
        }

        if constexpr (Milstein2DSVSDEModel<SdeType>) {
            // This block is only compiled if SdeType meets the Milstein1DSDEModel concept.

            // TODO : CHECK IF DERIVATIVES ARE TO BE USED AND CHECK THE COEFFS OF DW
        

            const double pEXP = this->sde_ref_.get_p();
            const double q = this->sde_ref_.get_q();
            const double rho = this->sde_ref_.get_correlation();
            const double rho_p = std::sqrt(1.0 - rho * rho);
            const double sigma_v = this->sde_ref_.get_sigma_v();
            auto Q = Utils::sampler<double>(get_default_rng_engine(), get_uniform_distribution(), 1);

            const double X = dt / M_PI * std::log(Q(0)/(1.0 - Q(0))); // X is the log-normal variable
            const double Y = std::sqrt((dW_p(1)*dW_p(1) + dW_p(0)*dW_p(0))*dt/3)* Utils::sampler<double>(get_default_rng_engine(), get_standard_normal_dist(), 1)(0); // Assuming the second component is the volatility variable
            
            const double double_integral = 0.5 * dW_p(1) * dW_p(0) - 0.5 * (X + Y); // Integral term for the correction
            const double correction_vol = sigma_v * pEXP * std::pow(x_p(0), pEXP + q - 1) *
            (rho_p * double_integral + rho * 0.5 * (dW_p(0) * dW_p(0) - dt));

            
            next_x_p(0) += correction_vol;
        } 
           

        }
    }
};
};

// --- Interpolated Kahl-Jackel Solver Implementation ---

/**
 * @brief Interpolated Kahl-Jackel solver for stochastic volatility SDEs.
 * @tparam SdeType SDE model type satisfying SDEModel concept.
 */
template <SDEModel SdeType>
class InterpolatedKahlJackelSolver : public SDESolverBase<InterpolatedKahlJackelSolver<SdeType>, SdeType> {
public:
    using Base = SDESolverBase<InterpolatedKahlJackelSolver<SdeType>, SdeType>;
    using Base::Base;

    void step(double t_current, const SDEVector& current_x,
              double dt, const SDEVector& dW_t, int num_paths,
              SDEVector& next_x) const {
        // static bool ikj_warning_shown = false;
        // if (!ikj_warning_shown) {
        //    std::cerr << "Warning: InterpolatedKahlJackelSolver is a placeholder and currently falls back to EulerMaruyamaSolver's step logic." << std::endl;
        //    ikj_warning_shown = true;
        // }
        
        // Fallback to Euler-Maruyama as this is a placeholder.
        // For a real IKJ, this step would be very different and model-specific.
        const int state_dim = SdeType::STATE_DIM;
        const int wiener_dim = SdeType::WIENER_DIM;
        EulerMaruyamaSolver<SdeType> euler_solver(this->sde_ref_, this->rng_engine_ref_);
        euler_solver.step(t_current, current_x, dt, dW_t, num_paths, next_x);

        if constexpr (Milstein2DSVSDEModel<SdeType>) {
            const double pEXP = this->sde_ref_.get_p();
            const double q = this->sde_ref_.get_q();
            const double rho = this->sde_ref_.get_correlation();
            const double rho_p = std::sqrt(1.0 - rho * rho);
            const double sigma_v = this->sde_ref_.get_sigma_v();


            // Map to Eigen views
            Eigen::Map<const SDEArray, 0, Eigen::InnerStride<>> v_curr(current_x.data(), num_paths,  Eigen::InnerStride<>(SdeType::STATE_DIM));        
            Eigen::Map<const SDEArray, 0, Eigen::InnerStride<>> v_next(next_x.data(), num_paths,  Eigen::InnerStride<>(SdeType::STATE_DIM));        
            Eigen::Map<const SDEArray> dW0(dW_t.data(), num_paths);
            Eigen::Map<const SDEArray> dW1(dW_t.data() + num_paths, num_paths);

            // Compute all power terms once
            SDEArray v_pow_2p = v_curr.pow(2 * pEXP);
            SDEArray v_next_pow_2p = v_next.pow(2 * pEXP);
            SDEArray v_pow_p = v_curr.pow(pEXP);
            SDEArray v_next_pow_p = v_next.pow(pEXP);
            SDEArray v_pow_pm1 = v_curr.pow(pEXP - 1.0);


            // Full correction formula
            SDEArray correction_ijk = 
                -0.25 * (v_pow_2p + v_next_pow_2p) * dt
                + rho * v_pow_p * dW0
                + 0.5 * rho_p * (v_pow_p + v_next_pow_p) * dW1
                + 0.5 * rho * rho_p * v_pow_pm1 * sigma_v * (dW0.square() - dt);


            
            Eigen::Map<SDEArray, 0, Eigen::InnerStride<>> x_next(next_x.data() + 1, num_paths,  Eigen::InnerStride<>(SdeType::STATE_DIM));   
            x_next += correction_ijk;

    }
    }
};



} // namespace SDE

#endif // SDE_HPP