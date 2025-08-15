
/**
 * @file MixtureDensity.hpp
 * @brief Defines the MixtureDensity class template for representing and manipulating mixtures of probability density functions (PDFs) with orthogonal polynomial bases.
 *
 * @details
 * This header provides a flexible framework for constructing mixtures of various probability distributions (e.g., Normal, Gamma, Beta) and associating each component with its corresponding orthogonal polynomial basis (e.g., Hermite, Laguerre, Jacobi).
 * The MixtureDensity class supports:
 *   - Validation and storage of mixture weights and component densities.
 *   - Automatic selection and construction of the appropriate polynomial basis for each component via DensityToPolyTraits.
 *   - Computation of recursion coefficients for orthonormal polynomial basis construction using the mixture's weighted inner product.
 *   - Construction of the orthonormal polynomial basis for the mixture, including storage of polynomial coefficients and function representations.
 *   - Calculation of the mixture's PDF and CDF at arbitrary points.
 *   - Access to component parameters, polynomial bases, Jacobi matrices, and projection matrices.
 *   - Prevention of copy semantics for safety, with move semantics enabled.
 *
 * @note
 * - Requires C++17 or later for features such as std::variant, std::apply, and fold expressions.
 * - Relies on Eigen for matrix and vector operations, and OpenMP for parallelization.
 * - Assumes that DensityType classes provide methods: pdf(x), cdf(x), getMu(), getSigma(), getAlpha(), getBeta(), getShape(), getScale(), getDomain(), getDistribution(), and getConstructorParameters().
 * - The polynomial basis construction is tailored to the type of each component via DensityToPolyTraits specializations.
 *
 * @section MixtureDensity_Usage Usage Example
 * @code
 * using Mixture = stats::MixtureDensity<3, stats::NormalDensity>;
 * std::vector<double> weights = {0.5, 0.5};
 * std::vector<stats::NormalDensity> components = {stats::make_normal_density(0.0, 1.0), stats::make_normal_density(2.0, 1.0)};
 * Mixture mixture(weights, components);
 * mixture.constructOrthonormalBasis();
 * double value = mixture.pdf(1.0);
 * @endcode
 *
 *  */
#ifndef MIXTURE_DENSITY_HPP
#define MIXTURE_DENSITY_HPP



#include <vector>
#include <variant>
#include <numeric>      // For std::accumulate
#include <stdexcept>
#include <limits>
#include <iostream>
#include <typeinfo>     // For typeid
#include <tuple>        // For std::apply
#include <utility>      // For std::move
#include <functional>  // For std::function
#include "DensityBase.hpp" 
#include "../polynomials/OrthogonalPolynomials.hpp"

namespace stats {
/// @brief Alias for a normal distribution wrapped in BoostBaseDensity.
/// @note Uses example parameters (mean = 0.0, stddev = 1.0).
using NormalDensity = decltype(stats::make_normal_density(0.0, 1.0)); 

/// @brief Alias for a gamma distribution wrapped in BoostBaseDensity.
/// @note Uses example parameters (shape = 1.0, scale = 1.0).
using GammaDensity  = decltype(stats::make_gamma_density(1.0, 1.0));  

/// @brief Alias for a beta distribution wrapped in BoostBaseDensity.
/// @note Uses example parameters (alpha = 1.0, beta = 1.0).
using BetaDensity   = decltype(stats::make_beta_density(1.0, 1.0));   

/**
 * @brief Alias for a compile-time polynomial type.
 * 
 * @tparam N Degree of the polynomial.
 * @tparam R Real number type (default: double).
 */
template <unsigned int N, typename R = traits::DataType::PolynomialField>
using Polynomial = polynomials::Polynomial<N, R>;

/// @brief A variant type that can hold any supported Boost density type.
using DensityVariant = std::variant<
        NormalDensity,
        GammaDensity,
        BetaDensity>;  
/**
 * @brief Traits class mapping a Density type to its associated orthogonal polynomial type.
 * 
 * This base template is used to generate a meaningful static_assert failure for unsupported types.
 * 
 * @tparam Density The density distribution type.
 * @tparam N Degree of the polynomial.
 * @tparam R Real number type.
 */
template<typename Density, unsigned int N, typename R>
struct DensityToPolyTraits {
    // Error for unsupported types
    static_assert(!std::is_same_v<Density, Density>, "Unsupported DensityType for MixtureDensity");
};

/**
 * @brief Specialization mapping NormalDensity to HermitePolynomial.
 * 
 * @tparam N Degree of the polynomial.
 * @tparam R Real number type.
 */
template<unsigned int N, typename R>
struct DensityToPolyTraits<NormalDensity, N, R> {
    using PolyType = polynomials::HermitePolynomial<N, R>; 

    /**
     * @brief Creates a Hermite polynomial associated with the given normal density.
     * 
     * @param density_component The normal distribution component.
     * @return HermitePolynomial initialized with mean and stddev.
     */
    static PolyType create(const NormalDensity& density_component) {
    
        return PolyType(density_component.getMu(), density_component.getSigma());
    }
};

/**
 * @brief Specialization mapping BetaDensity to JacobiPolynomial.
 * 
 * @tparam N Degree of the polynomial.
 * @tparam R Real number type.
 */
template<unsigned int N, typename R>
struct DensityToPolyTraits<BetaDensity, N, R> {
    using PolyType = polynomials::JacobiPolynomial<N, R>;
    /**
     * @brief Specialization mapping BetaDensity to JacobiPolynomial.
     * 
     * @tparam N Degree of the polynomial.
     * @tparam R Real number type.
     */

    static PolyType create(const BetaDensity& density_component) {
        
        return PolyType(density_component.getAlpha(), density_component.getBeta());
    }
};

/**
 * @brief Specialization mapping GammaDensity to LaguerrePolynomial.
 * 
 * @tparam N Degree of the polynomial.
 * @tparam R Real number type.
 */

template<unsigned int N, typename R>
struct DensityToPolyTraits<GammaDensity, N, R> {
    using PolyType = polynomials::LaguerrePolynomial<N, R>; 
    
    /**
     * @brief Creates a Laguerre polynomial associated with the given gamma density.
     * 
     * @param density_component The gamma distribution component.
     * @return LaguerrePolynomial initialized with shape - 1 and scale.
     */
    static PolyType create(const GammaDensity& density_component) {

        return PolyType(density_component.getShape() - 1, 
                        density_component.getScale()); // Adjust if your constructor differs
    }
};


/**
 * @brief MixtureDensity class template for polynomial-based mixture density estimation.
 * 
 * @tparam PolynomialBaseDegree Degree of the polynomial basis.
 * @tparam DensityType Type of the density components (e.g., NormalDensity, GammaDensity).
 * @tparam R Floating-point type used for calculations (default is double).
 * 
 * This class represents a mixture model composed of weighted density components.
 * It supports polynomial basis construction for density approximation and
 * provides methods for computing PDF, CDF, and orthonormal polynomial bases.
 */
template<unsigned int PolynomialBaseDegree, typename DensityType, typename R = traits::DataType::PolynomialField>
class MixtureDensity{

public:
    using PolyType = Polynomial<PolynomialBaseDegree, R>;  // Polynomial class/type
    using StoringArray = traits::DataType::StoringArray;  // Storage for polynomials
    using StoringVector = traits::DataType::StoringVector;  // Storage for polynomials
    using JacobiMatrixType = traits::DataType::SparseStoringMatrix;  // Storage for matrix J
    using StoringMatrix = traits::DataType::StoringMatrix;  // Storage for matrix J

protected: // Protected so derived classes can potentially access if needed
    std::vector<R> weights_;                   // Weights for each component in the mixture
    std::vector<DensityType> components_;      // Density components of the mixture
    std::vector<JacobiMatrixType> Js;          // Jacobi matrices for polynomial evaluation
    StoringArray a;                            // Coefficients for polynomial evaluation
    StoringArray b;                            // Coefficients for polynomial evaluation
    std::vector<PolyType> polynomials;         // Precomputed polynomial solutions
    StoringMatrix H;                           // Matrix for polynomial coefficients (H matrix)
    std::vector<StoringMatrix> Hs;             // Vector of H matrices for each component
    std::vector<std::vector<std::function<R(R)>>> HPol_components; // Vector of function pointers for each mixture component
    std::vector<std::function<R(R)>> HPol_mixture;                  // Vector of function pointers for the overall mixture
    std::vector<StoringMatrix> Qs;                                 // Projection matrices for each component

public:

    /**
     * @brief Constructor initializing the mixture density with given weights and components.
     * 
     * @param weights Vector of component weights. Must be non-empty and sum to 1.
     * @param components Vector of density components, must match size of weights.
     * 
     * @throws std::invalid_argument if weights are empty, do not sum to 1, contain negatives,
     *         or if size mismatch with components.
     */    
    MixtureDensity(std::vector<R> weights, std::vector<DensityType> components)
        : weights_(std::move(weights)), components_(std::move(components)), a(PolynomialBaseDegree + 1), b(PolynomialBaseDegree), polynomials(PolynomialBaseDegree + 2), H(PolynomialBaseDegree + 1, PolynomialBaseDegree + 1)
 
    {
        if (weights_.empty()) {
             throw std::invalid_argument("Mixture Density cannot be empty (no weights).");
        }
        if (weights_.size() != components_.size()) {
            throw std::invalid_argument("Number of weights must match the number of components.");
        }

        // Validate weights sum to approximately 1
        R sum_weights = std::accumulate(weights_.begin(), weights_.end(), R(0.0));
        constexpr R epsilon_multiplier = 100;
        if (std::abs(sum_weights - R(1.0)) > std::numeric_limits<R>::epsilon() * epsilon_multiplier) {
             throw std::invalid_argument("Component weights must sum to 1. Current sum: " + std::to_string(sum_weights));
        }

        // Ensure no negative weights
        for(const auto& w : weights_) {
            if (w < R(0.0)) {
                 throw std::invalid_argument("Component weights cannot be negative.");
            }
        }

        Js.reserve(components.size());
        for (const auto& component : components_) {
    
            // 1. Create the polynomial traits for the current component
            // Note: DensityToPolyTraits is a template that maps the DensityType to the appropriate polynomial type
            using PolyTraits = DensityToPolyTraits<DensityType, PolynomialBaseDegree, R>;

            auto curr_comp_poly = PolyTraits::create(component); 
        
            // 2. Get the Jacobi matrix (assuming it's available)
            Js.push_back(curr_comp_poly.getJacobiMatrix());
            Hs.push_back(curr_comp_poly.getHMatrix());
            std::vector<std::function<R(R)>> curr_vec;
            curr_vec.reserve(PolynomialBaseDegree + 1);

            for (unsigned int idx = 0; idx <= PolynomialBaseDegree; ++idx) {
                // This line requires the full definition of std::function<R> (or std::function<double>)
                auto temp_poly = Polynomial<PolynomialBaseDegree, R>(curr_comp_poly.getHMatrix().col(idx));

                curr_vec.push_back(temp_poly.as_function());
            }
            HPol_components.push_back(curr_vec);

        }

    }

    
    /**
     * @brief Computes the combined support interval of the mixture density.
     * 
     * @return stats::DensityInterval<R> representing the overall support covering all components.
     * @throws std::runtime_error if no components exist.
     */
    stats::DensityInterval<R> getSupport() const {
        // Assuming all components have the same support
        if (components_.empty()) {
            throw std::runtime_error("No components available to determine support.");
        }
    
        stats::DensityInterval<R> overall_support = components_[0].getDomain(); // Assumes components_[0] exists due to the check above

        // Iterate through the rest of the components to find the min lower and max upper bounds
        for (size_t i = 1; i < components_.size(); ++i) {
            stats::DensityInterval<R> component_support = components_[i].getDomain();
            
            if (component_support.lower < overall_support.lower) {
                overall_support.lower = component_support.lower;
            }
            if (component_support.upper > overall_support.upper) {
                overall_support.upper = component_support.upper;
            }
        }
    
        // The DensityInterval constructor already checks if lower > upper.
        // If overall_support.lower ended up > overall_support.upper (e.g., if components_
        // had disjoint and strangely ordered supports, which shouldn't happen for typical densities),
        // the DensityInterval constructor will throw.
        return overall_support;
    }
    
    /**
     * @brief Computes the recursion coefficients (a, b) for the orthonormal polynomial basis.
     * 
     * This method calculates the three-term recurrence coefficients
     * for the polynomial basis constructed from the mixture components.
     * 
     * @throws std::runtime_error on numerical instability or invalid inputs.
     */

    void computeRecursionCoeffs() {
        unsigned int N_J = PolynomialBaseDegree + 1; // Number of basis elements (0 to N) -> Size of vectors/coeffs
        unsigned int N_K = weights_.size();          // Number of mixture components
        StoringArray csi = StoringArray::Zero(N_J);  // Intermediate norm^2 values (psi in paper notation) 

        if (N_K == 0) {
             throw std::runtime_error("Cannot construct basis with zero components.");
        }
         if (N_J == 0) {
             // Handle degree 0 case: basis is just H_0(x) = 1.
             a.setZero(); // Size 1
             b.setZero(); // Size 1 or 0 depending on convention
             csi.setOnes(); // csi[0] = <1,1>_w = 1
             return;
         }


        // --- Initialization ---
        std::vector<StoringVector> z_prev(N_K); // Stores z_{i-1} for component j
        std::vector<StoringVector> z_curr(N_K); // Stores z_i for component j

        #pragma omp parallel for
        for (size_t j = 0; j < N_K; ++j) {
            z_prev[j] = StoringVector::Zero(N_J);
            z_curr[j] = StoringVector::Zero(N_J);
            if (N_J > 0) {
               z_curr[j](0) = R(1.0); // H_0 = 1, so initial vector is e_1
            }
        }

        constexpr R tolerance = std::numeric_limits<R>::epsilon() * 100;

        // Temporary storage for the next iteration's vectors (for parallel update)
        std::vector<StoringVector> z_next(N_K);
        #pragma omp parallel for
        for(size_t j=0; j< N_K; ++j) {
             z_next[j].resize(N_J); // Pre-allocate
        }

        // --- Main Recurrence Loop (over degree i) ---
        for (size_t i = 0; i < N_J; ++i) {
            R phi_i_local = R(0.0); // Accumulator for <z_i, J z_i>_w for this iteration
            R csi_i_local = R(0.0); // Accumulator for <z_i, z_i>_w for this iteration

            // --- Parallel Calculation of phi_i and csi_i ---
            #pragma omp parallel for reduction(+:phi_i_local, csi_i_local) schedule(static)
            for (size_t j = 0; j < N_K; ++j) {
                // Ensure Js[j] and z_curr[j] are valid
                 if (z_curr[j].size() != N_J || Js[j].rows() != N_J || Js[j].cols() != N_J) {
                     #pragma omp critical
                     {
                         throw std::runtime_error("Dimension mismatch inside parallel loop at i="
                            + std::to_string(i) + ", j=" + std::to_string(j));
                     }
                 }

                // Weighted inner product for phi: w_j * z_curr[j]^T * Js[j] * z_curr[j]
                R contribution_phi = (z_curr[j].transpose() * Js[j] * z_curr[j]).value();
                phi_i_local += weights_[j] * contribution_phi;

                // Weighted squared norm for csi: w_j * ||z_curr[j]||^2
                R contribution_csi = z_curr[j].squaredNorm();
                csi_i_local += weights_[j] * contribution_csi;
            } // End parallel reduction loop

            // Store result in the variable `csi`
            csi[i] = csi_i_local;


            // --- Compute Recurrence Coefficients a[i] and b[i] ---
            if (std::abs(csi[i]) < tolerance) {
                 // Handling of potential division by zero or instability
                 throw std::runtime_error("Numerical instability: csi[" + std::to_string(i) + "] is near zero.");
            }

            a[i] = phi_i_local / csi[i];

            // Calculate b coefficient needed for the *next* step's z update
            // (This loop computes up to a_N, b_{N-1}, csi_N based on i < N_J)
            // We actually need b[i] based on csi[i+1] for updating z_{i+1} later.

            // The update formula is: z_{k+1} = (J_k - a_k I)z_k - b_{k-1}^2 z_{k-1} 


            R b_im1_sq = R(0.0);
            if (i > 0) {
                 // We need b[i-1], which depends on csi[i] / csi[i-1]
                 if (std::abs(csi[i - 1]) < tolerance) {
                    throw std::runtime_error("Numerical instability: csi[" + std::to_string(i-1) + "] is near zero when calculating b[" + std::to_string(i-1) + "].");
                 }
                 R ratio = csi[i] / csi[i - 1];
                 if (ratio < -tolerance) { // Check for significant negativity
                    throw std::runtime_error("Numerical instability: Negative ratio csi["+std::to_string(i)+"]/csi["+std::to_string(i-1)+"] for b[" + std::to_string(i-1) + "] calculation.");
                 }
                 ratio = std::max(R(0.0), ratio); // Clamp near-zero negative due to precision
                 b[i - 1] = std::sqrt(ratio); // Store b[i-1]
                 b_im1_sq = ratio;            // Use the ratio (b[i-1]^2) for the update step
            } else {
                 // b[-1] is not defined / needed. b[0] might be computed if i=0 loop calculates csi[1].
                 // Assuming b[0] = sqrt(csi[1]/csi[0]) would be calculated in the *next* (i=1) iteration.
                 // The update formula needs b_{i-1}^2, so for i=0, this term is zero.
            }

            // --- Parallel Update of z vectors ---
            // Calculate z_{next} = (J - a[i] * I) * z_curr - b[i-1]^2 * z_prev
            // If i == N_J - 1 (last iteration), this step might be skippable unless z vectors are needed later.
             if (i < PolynomialBaseDegree) // Only need to update z if we haven't reached the max degree
             {
                 #pragma omp parallel for schedule(static)
                 for(size_t j = 0; j < N_K; ++j) {
                     // Compute (Js[j] - a[i] * I) * z_curr[j]
                     z_next[j].noalias() = Js[j] * z_curr[j]; // Sparse matrix-vector product is efficient
                     z_next[j].noalias() -= a[i] * z_curr[j]; // Scale and subtract

                     // Subtract term involving z_prev if i > 0
                     if (i > 0) {
                         z_next[j].noalias() -= b_im1_sq * z_prev[j]; // Use precomputed square b[i-1]^2
                     }
                 } // End parallel update loop

                 // --- Update z_prev and z_curr for the next iteration (Serial) ---
                 // Using swap is often robust for parallel temporary storage patterns
                 for(size_t j = 0; j < N_K; ++j) {
                     z_prev[j] = std::move(z_curr[j]); // Old z_curr becomes z_prev
                     z_curr[j] = std::move(z_next[j]); // Result from z_next becomes new z_curr
                     // z_next[j] is now in a moved-from state, ready for reallocation/reuse
                     // For Eigen, resize might be needed if move semantics empty it:
                     z_next[j].resize(N_J); // Ensure it's ready for the next iteration's write
                 }
            } // end if (i < PolynomialBaseDegree)

        } // End i loop (main recurrence)

    } // End computeRecursionCoeffs
    
    /**
     * @brief Constructs the orthonormal polynomial basis for the mixture density.
     * 
     * Calls computeRecursionCoeffs() internally and then builds the polynomial
     * basis up to the specified PolynomialBaseDegree.
     */
    void constructOrthonormalBasis() {
        computeRecursionCoeffs();
        
        #pragma omp parallel for simd
        for (unsigned int i = 0; i < PolynomialBaseDegree + 2; ++i) {
            polynomials[i] = PolyType(StoringArray::Zero(PolynomialBaseDegree + 1));
        }
        
        // P₀(x) is implicitly zero.
        // P₁(x) = 1 (coefficient of x⁰ is 1)
        if (PolynomialBaseDegree + 1 > 0) {
            polynomials[1].get_coeff()(0) = 1.0;
        }
        H.col(0) = polynomials[1].get_coeff();
        HPol_mixture.push_back(polynomials[1].as_function()); // Store the first polynomial as a function
        if constexpr (PolynomialBaseDegree == 0) {
            
                return;
        }
    
    
        // --- Sequential Recurrence Calculation ---
    
        StoringArray P_kp1_coeffs(PolynomialBaseDegree + 1);
  
        for (unsigned int k = 1; k <= PolynomialBaseDegree; ++k) {

    
            // Get references to coefficients of P_k and P_{k-1}
            const auto& P_k_coeffs = polynomials[k].get_coeff();
            const auto& P_km1_coeffs = polynomials[k - 1].get_coeff();
            const auto& beta_kp = b[k - 1]; // beta_k+1
            const auto& alpha_k = a[k - 1]; // alpha_k-1
            const auto& beta_k = (k == 1) ?  0 : b[k - 2];

    
            // 1. Compute x * P_k (results in higher degree, store temporarily)
            P_kp1_coeffs.setZero(); // Clear previous iteration's result
            // Shift P_k's coefficients: P_k[i] -> (x*P_k)[i+1]
            // Copies P_k[0..N-1] to P_kp1_coeffs[1..N]
            P_kp1_coeffs.segment(1, PolynomialBaseDegree) = P_k_coeffs.head(PolynomialBaseDegree);
    
            // 2. Subtract alpha_km1 * P_k
            P_kp1_coeffs -= alpha_k * P_k_coeffs;
    
            // 3. Subtract beta_km1 * P_{k-1}
            // P_kp1_coeffs -= beta_km1 * P_km1_coeffs;

            P_kp1_coeffs -= beta_k* P_km1_coeffs;
            P_kp1_coeffs /= beta_kp; // Normalize by sqrt(beta_k)
            // --- End of Eigen calculation ---
    
            // Store the final coefficients for P_{k+1}
            H.col(k) = P_kp1_coeffs;
            polynomials[k + 1].set_coeff(P_kp1_coeffs);
            HPol_mixture.push_back(polynomials[k + 1].as_function()); 
        }


        } // End constructOrthonormalBasis
    

    /**
     * @brief Constructs the Q-projection matrices for the mixture.
     * 
     * Placeholder function to compute Q matrices based on H matrices of components.
     * Implementation depends on specific application needs.
     */
    void constructQProjectionMatrix() {
        // This function implements the logic to construct the Q-projection matrix
        // based on the orthonormal basis and other relevant parameters.

        for (const auto & hk : Hs) {
            Qs.push_back(hk.triangularView<Eigen::Upper>().solve(getHMatrix())); 
        }
    }

    /**
     * @brief Virtual destructor.
     */
    virtual ~MixtureDensity() = default;

    /**
     * @brief Evaluates the probability density function (PDF) of the mixture at a given point.
     * 
     * @param x The point at which to evaluate the PDF.
     * @return R The PDF value at x.
     */  
    R pdf(R x) const {
        R total_pdf = R(0.0);
        for (size_t i = 0; i < components_.size(); ++i) {
             // Directly call pdf on the known component type
            total_pdf += weights_[i] * components_[i].pdf(x);
        }
        return total_pdf;
    }

    /**
     * @brief Evaluates the cumulative distribution function (CDF) of the mixture at a given point.
     * 
     * @param x The point at which to evaluate the CDF.
     * @return R The CDF value at x, clamped between 0 and 1.
     */
    R cdf(R x) const {
        R total_cdf = R(0.0);
        for (size_t i = 0; i < components_.size(); ++i) {
            // Directly call cdf on the known component type
            total_cdf += weights_[i] * components_[i].cdf(x);
        }
        // Clamp CDF result
        return std::max(R(0.0), std::min(R(1.0), total_cdf));
    }

    /**
     * @brief Computes the mean of the mixture density.
     * 
     * @return R The weighted mean of the mixture components.
     */
    R mean() const {
        R total_mean = R(0.0);
        for (size_t i = 0; i < components_.size(); ++i) {
            total_mean += weights_[i] * components_[i].getMu();
        }
        return total_mean;
    }

    /** * @brief Computes the variance of the mixture density.
     * 
     * @return R The weighted variance of the mixture components.
     */
    R variance() const {
        R total_variance = R(0.0);
        for (size_t i = 0; i < components_.size(); ++i) {
            R mu = components_[i].getMu();
            R sigma = components_[i].getSigma();
            total_variance += weights_[i] * (sigma * sigma + mu * mu);
        }
        return total_variance - mean() * mean();
    }
    

    /**
     * @brief Prints the parameters of each component in the mixture to standard output.
     * 
     * Assumes DensityType provides getDistribution() and getConstructorParameters().
     */



    void printComponentParameters() const {
        std::cout << "--- Mixture Components ---\n";
        if (components_.empty()) {
            std::cout << "  (No components)\n";
            std::cout << "------------------------------------\n";
            return;
        }

        // Print overall type once (assuming all components have same underlying Boost type)
         const auto& first_comp_dist = components_[0].getDistribution();
         std::cout << "  Mixture Type: " << typeid(first_comp_dist).name() << "\n"; // Might be mangled

        for (size_t i = 0; i < components_.size(); ++i) {
            std::cout << "  Component " << i << " (Weight: " << weights_[i] << "): ";

            // Get the tuple containing the original constructor parameters
            const auto& params_tuple = components_[i].getConstructorParameters();

            // Use std::apply and a fold expression (C++17) to print tuple elements
            std::cout << "Params=(";
            std::apply([](const auto&... args){
                bool first = true;
                auto print_arg = [&](const auto& arg){
                    if (!first) std::cout << ", ";
                    std::cout << arg;
                    first = false;
                };
                (print_arg(args), ...);
            }, params_tuple);
            std::cout << ")" << std::endl;
        }
         std::cout << "------------------------------------\n";
    }

    /**
     * @brief Retrieves the orthonormal polynomial of a specified degree.
     * 
     * @param degree Degree of the polynomial requested (0-based).
     * @return const PolyType& Reference to the polynomial of the requested degree.
     * @throws std::out_of_range if degree exceeds PolynomialBaseDegree.
     */
    const PolyType& getPolynomial(unsigned int degree) const {
        if (degree > PolynomialBaseDegree) {
            throw std::out_of_range("Requested degree exceeds max polynomial degree.");
        }
        return polynomials[degree + 1];
    }

    /**
     * @brief Retrieves the matrix H storing polynomial coefficients.
     * 
     * @return StoringMatrix Matrix H.
     */
    StoringMatrix getHMatrix() const noexcept {
        return H;
    }

    /**
     * @brief Retrieves the vector of Q projection matrices.
     * 
     * @return auto Vector of Q matrices.
     */
    const auto getQProjectionMatrix() const noexcept {
        return Qs;
    }

    /**
     * @brief Retrieves the vector of H matrices for each component.
     * 
     * @return auto Vector of H matrices.
     */
    const auto getHMatrixs() const noexcept {
        return Hs;
    }

    /**
     * @brief Retrieves the vector of Jacobi matrices for each component.
     * 
     * @return auto Vector of Jacobi matrices.
     */
    const auto getJacobiMatrix() const noexcept {
        return Js;
    }

    /**
     * @brief Retrieves the vector of orthonormal polynomial functions for each component.
     * 
     * @return auto Vector of function pointers for each component's polynomial.
     */

    const auto getHFunctionsComponents() const noexcept {
        return HPol_components;
    }

    /**
     * @brief Retrieves the vector of orthonormal polynomial functions for the mixture.
     * 
     * @return auto Vector of function pointers for the mixture's polynomial.
     */
    const auto getHFunctionsMixture() const noexcept {
        return HPol_mixture;
    }

    /// @name Accessors
    /// @{
    /**
     * @brief Retrieves the mixture's components, weights, Jacobi matrices, and coefficients.
     * 
     * @return const std::vector<DensityType>& Reference to the vector of density components.
     */
    const std::vector<DensityType>& getComponents() const noexcept { return components_; }
    const std::vector<R>& getWeights() const noexcept { return weights_; }
    const std::vector<JacobiMatrixType>& getMatrices() const noexcept { return Js; }
    const StoringArray& getA() const noexcept { return a; }
    const StoringArray& getB() const noexcept { return b; }
    /// @}

    /**
     * @brief Returns the number of components in the mixture.
     * 
     * @return size_t Number of components.
     */
    size_t size() const noexcept { return components_.size(); }

    /**
     * @brief Checks if the mixture is empty (no components).
     * 
     * @return true if the mixture has no components, false otherwise.
     */
    bool empty() const noexcept { return components_.empty(); }
    
    /// Prevent copy semantics for safety
    // Copy constructor and assignment operator are deleted to prevent copying
    // This is to ensure that the MixtureDensity object is not copied, which could lead to issues
    // with shared ownership of components and weights.
    // Move semantics are allowed for efficiency
    // This is to ensure that the MixtureDensity object can be moved, which is efficient and safe.
    MixtureDensity(const MixtureDensity&) = delete;
    MixtureDensity& operator=(const MixtureDensity&) = delete;
    MixtureDensity(MixtureDensity&&) = default;
    MixtureDensity& operator=(MixtureDensity&&) = default;


}; // End class MixtureDensity


} // End namespace stats

#endif // MIXTURE_DENSITY_HPP


