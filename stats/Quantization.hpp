/**
 * @file Quantization.hpp
 * @brief Implements quantization algorithms for vector quantization, including Newton's method (1D) and Competitive Learning Vector Quantization (CLVQ) for arbitrary dimensions.
 *
 * This header defines the `stats::Quantizer` class template and its specializations for different quantization procedures.
 * It provides a flexible framework for optimal quantizer design, supporting both batch (Newton's method for 1D) and online (CLVQ) learning.
 *
 * Main Components:
 * - `QuantizerBase`: Base class providing storage and access for quantizer centroids (hints) and the final quantization grid.
 * - `Quantizer<T, N_points, 1, QuantizationProcedure::Newton>`: Specialization for 1D quantization using Newton's method, optimized for the standard normal distribution.
 *   - Initializes centroids on a uniform grid.
 *   - Iteratively updates centroids using Newton's method, computing gradients and Hessians based on the quantization distortion for the normal distribution.
 * - `Quantizer<T, N_points, Dim_val, QuantizationProcedure::CLVQ>`: Specialization for arbitrary dimension using Competitive Learning Vector Quantization (CLVQ).
 *   - Initializes centroids by sampling from the standard normal distribution.
 *   - Performs online updates of centroids using a learning rate schedule and a winner-takes-all update rule.
 *   - Maintains and updates centroid weights (frequency of selection).
 *   - Tracks and reports distortion estimates during training.
 * Dependencies:
 * - Eigen for matrix/vector operations.
 * - OpenMP for parallelization.
 * - Utilities for sampling, argmin, and distortion calculation.
 *
 * Usage:
 * - Instantiate the appropriate `Quantizer` specialization with desired parameters.
 * - Call `run()` to perform quantization.
 * - Access the resulting quantization grid via `getQuantization()`.
 *
 * References:
 * - Optimal quantizers for standard Gaussian distribution (Printems, Paige)
 * - Optimal quadratic quantization for a Gauss-Markov source (Paige/Printems)
 *
 */
#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP

#include <iostream>
#include <vector>
#include <array>
#include <cmath>        // For std::pow, std::abs
#include <algorithm>
#include <random>
#include <limits>       // For std::numeric_limits
#include <iomanip>      // For std::fixed, std::setprecision
#include <tuple>        // For std::tuple from Utils::argmin
#include <omp.h>        // OpenMP for parallelization
#include "../traits/OPOE_traits.hpp"
#include "../stats/DensityBase.hpp" // For OPOE::make_normal_density
#include "../utils/Utils.hpp"       // For Utils::argmin, Utils::sample_standard_normal_eigen, Utils::calculate_overall_distortion


// --- Namespace for Quantization ---
namespace stats {

/// @brief Type alias for quantization procedures defined in traits.
using QuantizationProcedure = traits::QuantizationProcedure;using StoringVector = traits::DataType::StoringVector;

/// @brief Type alias for vector storage type used in quantization.
using StoringVector = traits::DataType::StoringVector;

/// @brief Type alias for vector storage type used in quantization.
using QuantizerGrid = traits::DataType::StoringMatrix; // Eigen::Matrix<T, Dim, N> or Dynamic

/**
 * @brief Forward declaration of Quantizer class.
 * 
 * @tparam T Scalar type used (e.g., float, double).
 * @tparam N_points Number of quantization points (centroids).
 * @tparam Dim_val Dimension of the input space.
 * @tparam Procedure Quantization procedure strategy (e.g., Lloyd, Random).
 */
template <std::size_t N_points, std::size_t Dim_val, QuantizationProcedure Procedure, typename T = traits::DataType::PolynomialField>
class Quantizer;

/**
 * @brief Base class template for quantizer implementations.
 * 
 * This class provides a generic interface and common storage for quantization algorithms.
 * It defines and manages the quantizer grid (centroids) and hints (initial positions).
 * 
 * @tparam T Scalar type (e.g., float or double).
 * @tparam N_points Number of quantization points (must be > 0).
 * @tparam Dim_val Dimensionality of the input space (must be > 0).
 */
template <std::size_t N_points, std::size_t Dim_val, typename T = traits::DataType::PolynomialField>
class QuantizerBase {
public:
    static_assert(N_points > 0, "Number of quantization points (N_points) must be greater than 0.");
    static_assert(Dim_val > 0, "Dimension (Dim_val) must be greater than 0.");

    /**
     * @brief Default constructor for the base quantizer.
     * 
     * Initializes the `hints` and `quantization` matrices to zero-filled matrices
     * with dimensions Dim_val × N_points.
     */
    explicit QuantizerBase()
        : hints(Dim_val, N_points),      // Dim x N matrix
          quantization(Dim_val, N_points) {} // Dim x N matrix

    /**
     * @brief Virtual destructor.
     */
    virtual ~QuantizerBase() = default;

    /**
     * @brief Retrieves the current quantization grid (centroids).
     * 
     * @return const QuantizerGrid& Reference to the quantization matrix (Dim × N).
     */
    [[nodiscard]] const QuantizerGrid& getQuantization() const noexcept { return quantization; }

    /**
     * @brief Gets the number of quantization points.
     * 
     * @return std::size_t Number of quantizer centroids (N_points).
     */
    [[nodiscard]] std::size_t getNumPoints() const noexcept { return N_points; }

    /**
     * @brief Gets the dimensionality of the quantization space.
     * 
     * @return std::size_t Dimension of each quantization point (Dim_val).
     */
    [[nodiscard]] std::size_t getDim() const noexcept { return Dim_val; }

    /**
     * @brief Retrieves the initialization hints for the quantizer.
     * 
     * @return const QuantizerGrid& Reference to the hint matrix (Dim × N).
     */
    [[nodiscard]] const QuantizerGrid& getHints() const noexcept { return hints; }

protected:
    QuantizerGrid hints;         ///< Hint matrix (initial centroids) of size Dim × N_points.
    QuantizerGrid quantization;  ///< Final optimized quantizer grid of size Dim × N_points.
};

/**
 * @brief Specialization of Quantizer for 1D using Newton's method.
 * 
 * This class implements a 1D quantizer using Newton's method to minimize the distortion function
 * for a Gaussian distribution. It inherits from QuantizerBase and provides an optimization routine
 * based on analytical gradients and Hessians.
 * 
 * @tparam T Scalar type (e.g., float or double).
 * @tparam N_points Number of quantization points.
 */
template <std::size_t N_points, typename T>
class Quantizer< N_points, 1, QuantizationProcedure::Newton, T> : public QuantizerBase<N_points, 1, T> {
private:
    static constexpr std::size_t Dim = 1; // Explicitly state Dim for this specialization
public:
    /**
     * @brief Parameters for controlling Newton's method.
     */
    struct Params {
        T tolerance{1e-6};            ///< Convergence threshold for gradient norm.
        std::size_t max_iter{100};    ///< Maximum allowed iterations for optimization.

        /**
         * @brief Constructor for custom parameters.
         * 
         * @param tol Desired tolerance for convergence.
         * @param max_it Maximum number of iterations.
         */
        Params(T tol = 1e-5, std::size_t max_it = 1000)
            : tolerance(tol), max_iter(max_it) {}
    };

    /**
     * @brief Constructor for the 1D Newton quantizer.
     * 
     * Initializes the hint values and stores user-specified parameters.
     * 
     * @param params Parameters for the Newton optimization routine.
     */
    explicit Quantizer(const Params& params = Params{})
        : QuantizerBase<N_points, Dim, T>(), params_(params) {
        initializeHints();
    }

    /**
     * @brief Initializes hints using a uniform grid over [-4, 4].
     * 
     * The hints provide starting centroids for the optimization, evenly spaced in 1D.
     */
    void initializeHints() {
        constexpr T a_range = -4.0; // Range for uniform initialization [-4, 4]
        constexpr T b_range = 4.0;

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N_points; ++i) {
            // Uniform grid logic for Dim = 1
            // i-th point (0-indexed) value: a + (b-a) * (2*(i+1) - 1) / (2*N)
            // or more simply: a + (b-a) * (i + 0.5) / N
            this->hints(0, i) = a_range + (b_range - a_range) * (static_cast<T>(i) + static_cast<T>(0.5)) / static_cast<T>(N_points);
        }

        std::cout << "[Debug] Hints initialized for Newton (Dim=1, N=" << N_points << "):\n";
    }

    /**
     * @brief Runs Newton's method to optimize quantizer centroids.
     * 
     * Uses analytical gradients and diagonal Hessian approximation to iteratively update centroids.
     * Updates the `quantization` matrix in the base class with the final result.
     */
    void run() {
        StoringVector x = this->hints.row(0).transpose(); ///< Current quantizer positions.
        StoringVector grad(N_points);                     ///< Gradient vector.
        StoringVector hessian_diag(N_points);             ///< Diagonal of Hessian matrix.

        std::cout << "[Info] Newton: Starting optimization for " << N_points << " points." << std::endl;

        for (std::size_t iter = 0; iter < params_.max_iter; ++iter) {
            computeGradientAndHessian(x, grad, hessian_diag);

            T grad_norm = grad.template lpNorm<Eigen::Infinity>(); // Max absolute gradient component
            // std::cout << "[Debug] Newton Iter " << iter << ": Grad Norm = " << grad_norm << std::endl;

            if (grad_norm < params_.tolerance) {
                std::cout << "[Info] Newton: Converged after " << iter << " iterations." << std::endl;
                break;
            }

            // Newton step: delta = -H^{-1} * grad. For diagonal H, H_ii^{-1} = 1/H_ii
            // Ensure no division by zero in hessian_diag before this step
            StoringVector delta = -grad.array() / hessian_diag.array();
            x += delta;

            if (iter == params_.max_iter - 1) {
                std::cout << "[Warning] Newton: Reached max iterations (" << params_.max_iter << ")." << std::endl;
            }
        }
        this->quantization.row(0) = x.transpose(); // Store result
        std::cout << "[Debug] Newton: Final Quantization Grid:\n" << this->quantization.transpose() << std::endl;
    }

private:
    Params params_; ///< Newton method parameters.

    /**
     * @brief Computes the gradient and diagonal Hessian of the distortion function.
     * 
     * Uses Gaussian PDF and CDF to evaluate analytical expressions for each quantizer point.
     * 
     * @param q_points Current positions of quantizer centroids.
     * @param grad Output: gradient vector (N_points).
     * @param hess_diag Output: diagonal of Hessian matrix (N_points).
     */

    void computeGradientAndHessian(const StoringVector& q_points, // q_points is N_points x 1
                                   StoringVector& grad,
                                   StoringVector& hess_diag) {
        // grad and hess_diag should be N_points x 1
        grad.setZero();
        hess_diag.setConstant(static_cast<T>(1e-9)); // Small positive for stability

        auto normal_density = stats::make_normal_density(static_cast<T>(0.0), static_cast<T>(1.0));

        StoringVector a_boundaries(N_points), b_boundaries(N_points);

        // Voronoi boundaries: a_i = (q_{i-1} + q_i)/2, b_i = (q_i + q_{i+1})/2
        a_boundaries(0) = -std::numeric_limits<T>::infinity();
        if (N_points > 1) {
            for (std::size_t i = 1; i < N_points; ++i) {
                a_boundaries(i) = static_cast<T>(0.5) * (q_points(i - 1) + q_points(i));
            }
        }

        b_boundaries(N_points - 1) = std::numeric_limits<T>::infinity();
        if (N_points > 1) {
            for (std::size_t i = 0; i < N_points - 1; ++i) {
                b_boundaries(i) = static_cast<T>(0.5) * (q_points(i) + q_points(i + 1));
            }
        }
        
        // This is the formulation from "Optimal quantizers for standard Gaussian distribution" by Printems (2001) 
        // grad_i = 2 * ( q_i * (cdf(b_i) - cdf(a_i)) - (pdf(b_i) - pdf(a_i)) )  -- this formula seems reversed from a minimization perspective
        // Or from "Optimal quadratic quantization for a gauss Markov source" (Paige/Printems) where dD/dx_i is given
        // dD/dx_i = 2 * integral_{x_i in V_i} (x_i - y) phi(y) dy
        // For 1D, integral (x_i - y)phi(y)dy = x_i * (F(b_i)-F(a_i)) - (phi(a_i)-phi(b_i)) (using -phi'(y)=y*phi(y) so integral y*phi(y) = -phi(y))
        // The gradient of D = sum_i integral_{y in V_i} (y - q_i)^2 phi(y) dy w.r.t q_i is -2 * integral_{y in V_i} (y - q_i) phi(y) dy
        // = -2 * [ integral y*phi(y)dy - q_i * integral phi(y)dy ]
        // = -2 * [ (-phi(b_i) - (-phi(a_i))) - q_i * (F(b_i) - F(a_i)) ]
        // = -2 * [ phi(a_i) - phi(b_i) - q_i * (F(b_i) - F(a_i)) ]
        // =  2 * [ q_i * (F(b_i) - F(a_i)) - (phi(a_i) - phi(b_i)) ]
        // The Hessian (diagonal elements for 1D Lloyd-Max type problem) is d^2D/dq_i^2 = 2 * (F(b_i) - F(a_i))

        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < N_points; ++i) {
            T Fa = normal_density.cdf(a_boundaries(i));
            T Fb = normal_density.cdf(b_boundaries(i));
            T pdf_a = normal_density.pdf(a_boundaries(i));
            T pdf_b = normal_density.pdf(b_boundaries(i));

            T Z_i = Fb - Fa; // Probability mass in Voronoi cell i

            if (Z_i > std::numeric_limits<T>::epsilon()) { // Avoid division by zero or instability
                // Gradient component for q_i
                grad(i) = static_cast<T>(2.0) * (q_points(i) * Z_i - (pdf_a - pdf_b));
                // Hessian diagonal component for q_i
                hess_diag(i) = static_cast<T>(2.0) * Z_i;
                if (hess_diag(i) < static_cast<T>(1e-9)) { // Floor hessian component
                     hess_diag(i) = static_cast<T>(1e-9);
                }
            } else {
                grad(i) = static_cast<T>(0.0);
                hess_diag(i) = static_cast<T>(1.0); // Or a small stable value if Z_i is effectively zero
            }
        }
    }
};


/**
 * @brief Specialization of Quantizer using the Competitive Learning Vector Quantization (CLVQ) algorithm.
 * 
 * This implementation performs unsupervised quantization using a stochastic online learning procedure
 * where centroids (hints) are updated based on sampled data points and a dynamic learning rate schedule.
 * 
 * @tparam T           Scalar type (e.g., float, double).
 * @tparam N_points    Number of centroids (quantization points).
 * @tparam Dim_val     Dimensionality of the input space.
 */
template <std::size_t N_points, std::size_t Dim_val, typename T>
class Quantizer<N_points, Dim_val, QuantizationProcedure::CLVQ, T> : public QuantizerBase<N_points, Dim_val, T> {
public:
    /**
     * @brief Parameters for CLVQ quantizer training.
     */
    struct Params {
        T initial_gamma_0_base{0.1};               ///< Base learning rate scale factor.
        std::size_t epochs{10000};                 ///< Number of training iterations (samples processed).
        T convergence_tolerance{1e-5};             ///< Convergence threshold for distortion improvement.
        int distortion_check_interval{1000};       ///< Frequency (in iterations) to evaluate distortion.
        int num_samples_for_distortion_eval{5000}; ///< Number of samples used for evaluating distortion.

        /**
         * @brief Constructor for CLVQ parameters.
         * 
         * @param gamma_base   Base learning rate.
         * @param ep           Total epochs (iterations).
         * @param tol          Convergence tolerance.
         * @param dist_interval Distortion check frequency.
         * @param dist_samples Number of samples for distortion computation.
         */
        Params(T gamma_base = 0.1, std::size_t ep = 10000, T tol = 1e-5, int dist_interval = 1000, int dist_samples = 5000)
            : initial_gamma_0_base(gamma_base), epochs(ep), convergence_tolerance(tol),
              distortion_check_interval(dist_interval), num_samples_for_distortion_eval(dist_samples) {}
    };

    /**
     * @brief Constructor for the CLVQ Quantizer.
     * 
     * @param params Optional parameter struct for training configuration.
     */
    explicit Quantizer(const Params& params = Params{})
        : QuantizerBase<N_points, Dim_val, T>(), params_(params),
          weights_(StoringVector::Constant(N_points, static_cast<T>(1.0) / static_cast<T>(N_points))) {
        initializeHints();
    }

    /**
     * @brief Initializes centroids (hints) with samples from a standard normal distribution.
     */
    void initializeHints() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(static_cast<T>(0.0), static_cast<T>(1.0));

        // this->hints is Dim_val x N_points
        this->hints = Utils::sampler<T>(gen, dist, Dim_val, N_points);
        std::cout << "[Debug] Hints initialized for CLVQ (Dim=" << Dim_val << ", N=" << N_points << "):\n";
    }

    /**
     * @brief Runs the CLVQ training loop.
     * 
     * Samples data points from a standard normal distribution, updates centroids using a
     * soft competitive learning rule, and monitors convergence via distortion.
     */
    void run() {
        std::random_device rd_train, rd_dist;
        std::mt19937 training_generator(rd_train());
        std::mt19937 distortion_eval_generator(rd_dist());
        std::normal_distribution<T> standard_normal_dist(static_cast<T>(0.0), static_cast<T>(1.0));

        // --- Initial Overall Distortion ---
        T last_overall_distortion = Utils::calculate_overall_distortion<T, Dim_val>(
            this->hints, params_.num_samples_for_distortion_eval, distortion_eval_generator, standard_normal_dist
        );
        std::cout << "[Info] CLVQ Initial Overall Distortion: " << last_overall_distortion << std::endl;

        // --- Learning Rate (gamma) Parameters ---
        const T initial_gamma_0 = params_.initial_gamma_0_base; // Simpler: use base directly

        const T K_centroids = static_cast<T>(N_points);

        const T const_a_lr = 4.0 * std::pow(K_centroids, 1.0 / static_cast<T>(Dim_val));
        const T const_b_lr = M_PI * M_PI * std::pow(K_centroids, -2.0 / static_cast<T>(Dim_val));

        // --- Online Distortion Estimate (EMA) ---
        T online_distortion_estimate_ema = last_overall_distortion; // Initialize with the first true calculation

        std::cout << "[Info] CLVQ: Starting training for " << params_.epochs << " samples." << std::endl;
        std::cout << "[Info] CLVQ: Centroid matrix dimensions: " << this->hints.rows() << "x" << this->hints.cols() << std::endl;

        for (std::size_t iter_idx = 0; iter_idx < params_.epochs; ++iter_idx) {
            // 1. Sample a data point (eta)
            StoringVector eta = Utils::sampler<T>(
                training_generator, standard_normal_dist, Dim_val // Dim_val for dimension, 1 sample
            );

            // 2. Find the winning centroid for eta
            auto argmin_result = Utils::argmin<T>(this->hints, eta); // Returns std::tuple<int, T, RowVector>
            int winner_index = std::get<0>(argmin_result);
            T l2_dist_eta_to_winner = std::get<1>(argmin_result);

            if (winner_index == -1) {
                std::cerr << "[Warning] CLVQ Iter " << iter_idx << ": No winning centroid found. Skipping update." << std::endl;
                continue;
            }

            // 3. Calculate learning rate (gamma)
            T time_step_t = static_cast<T>(iter_idx + 1);
            T gamma = initial_gamma_0 * const_a_lr / (time_step_t * const_b_lr * initial_gamma_0 + const_a_lr);



            // 4. Update centroid
            this->hints.col(winner_index) += gamma * (eta - this->hints.col(winner_index));

            // 5. Update weights (conscience mechanism / frequency counting)
            if (this->weights_.size() == N_points) { // Safety check
                 this->weights_ *= (static_cast<T>(1.0) - gamma); // Decay all
                 this->weights_(winner_index) += gamma;         // Increment winner
            }


            // 6. Update online distortion estimate (EMA)
            T squared_l2_dist_eta_to_winner = l2_dist_eta_to_winner * l2_dist_eta_to_winner;
            online_distortion_estimate_ema = (static_cast<T>(1.0) - gamma) * online_distortion_estimate_ema + gamma * squared_l2_dist_eta_to_winner;

            
        }

        // --- Final Evaluation & Printouts ---
        T final_overall_distortion = Utils::calculate_overall_distortion<T, Dim_val>(
            this->hints, params_.num_samples_for_distortion_eval * 2, // Use more samples for final
            distortion_eval_generator, standard_normal_dist
        );
        this->quantization = this->hints; // Store the final learned hints as the quantization result

        std::cout << "[Info] CLVQ training finished." << std::endl;
        std::cout << "[Info] CLVQ Final Overall Distortion: " << std::fixed << std::setprecision(6) << final_overall_distortion << std::endl;
        if (this->weights_.size() > 0) {
             std::cout << "[Info] CLVQ Sum of final weights: " << this->weights_ << std::endl;
             // std::cout << "[Info] CLVQ Final weights (transpose):\n" << this->weights_.transpose() << std::endl;
        }
    }

private:
    Params params_;
    StoringVector weights_; // StoringVector should be N_points x 1
};

} // namespace stats
#endif  // QUANTIZATION_HPP