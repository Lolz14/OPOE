
/**
 * @file OrthogonalPolynomials.hpp
 * @brief Defines a generic CRTP-based framework for orthogonal polynomials and their specializations.
 *
 * This header provides a base class template, `OrthogonalPolynomialBase`, for constructing and working with
 * families of orthogonal polynomials using recurrence relations, weight functions, and domain validation.
 * It also provides concrete implementations for several classical orthogonal polynomial families:
 * Legendre, Hermite, Chebyshev, Laguerre, Jacobi, and Gegenbauer polynomials.
 *
 * ## Main Components
 *
 * - **OrthogonalPolynomialBase**: 
 *   - CRTP base class for orthogonal polynomials of degree N, with support for custom weight functions,
 *     recurrence coefficients, domain intervals, and parameter validation.
 *   - Handles polynomial generation, evaluation, and construction of the Jacobi matrix.
 *   - Provides access to recurrence coefficients, weight function, and precomputed polynomials.
 *
 * - **Specializations**:
 *   - `LegendrePolynomial`: Legendre polynomials on [-1, 1] with weight 1.
 *   - `HermitePolynomial`: Hermite polynomials with configurable mean and variance.
 *   - `ChebychevPolynomial`: Chebyshev polynomials of the first kind on [-1, 1].
 *   - `LaguerrePolynomial`: Generalized Laguerre polynomials with parameters alpha and theta.
 *   - `JacobiPolynomial`: Jacobi polynomials with parameters alpha and beta.
 *   - `GegenbauerPolynomial`: Gegenbauer polynomials with parameter mu.
 *
 *
 *
 * ## Usage Example
 * @code
 * polynomials::LegendrePolynomial<5> legendre;
 * double value = legendre.evaluate(0.5, 3); // Evaluate P_3(0.5)
 * auto J = legendre.getJacobiMatrix();      // Get Jacobi matrix
 * @endcode
 *
 * @note Requires Eigen for matrix/vector types and OpenMP for parallelization.
 * @note All polynomials are normalized according to their respective weight functions.
 *
 */
#ifndef HH_ORTHOGONAL_POLYNOMIALS_HH
#define HH_ORTHOGONAL_POLYNOMIALS_HH

#include "Polynomials.hpp"
#include "OrthogonalValidator.cpp"
#include <functional>
#include <memory>


namespace polynomials {


/**
 * @brief Base class for orthogonal polynomials using CRTP.
 * 
 * @tparam Derived Derived class (e.g., LegendrePolynomial)
 * @tparam N Degree of polynomial
 * @tparam R Floating point type (e.g., float, double)
 * @tparam Params Variadic template for parameter validation (e.g., α, β)
 */
template<typename Derived, unsigned int N, typename R, typename... Params>
requires std::floating_point<R> 
class OrthogonalPolynomialBase {
public:
    using PolyType = Polynomial<N, R>;  // Polynomial class/type
    using ValidatorType = PolynomialDomainValidator<R, Params...>; // Variadic validator type
    using StoringArray = traits::DataType::StoringArray;  // Storage for polynomials
    using JacobiMatrixType = traits::DataType::SparseStoringMatrix;  // Storage for matrix J
    using HMatrixType = traits::DataType::StoringMatrix;  // Storage for matrix J
    using Triplets = traits::DataType::Triplets;  // Triplet type for sparse matrix construction
    using Triplet = traits::DataType::Triplet;  // Triplet type for sparse matrix construction


protected:
    Function<R> weight_function;                          // Weight function for the polynomial
    RecurrenceCoefficients<R> recurrence_coeffs;          // Recurrence coefficient system
    DomainInterval<R> evaluation_domain;                  // Valid domain for evaluation
    std::unique_ptr<ValidatorType> domain_validator;      // Parameter domain validator
    std::vector<PolyType> polynomials;                    // Precomputed polynomial solutions
    StoringArray alphas;                                  // Coefficients for polynomial evaluation
    StoringArray betas;                                   // Coefficients for polynomial evaluation
    JacobiMatrixType J;                                   // Jacobi matrix for polynomial evaluation
    HMatrixType H;

    // Access the derived class
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

public:
    /**
     * @brief Construct base class with weight function, recurrence coefficients, domain, and validator.
     * 
     * @param weight Weight function
     * @param coeffs Recurrence coefficients
     * @param domain Evaluation domain
     * @param validator Parameter validator
     */
    OrthogonalPolynomialBase(
        const Function<R>& weight,
        const RecurrenceCoefficients<R>& coeffs,
        const DomainInterval<R>& domain,
        ValidatorType&& validator // Forwarding the validator instance
    )
        : weight_function(weight)
        , recurrence_coeffs(coeffs)
        , evaluation_domain(domain)
        , domain_validator(std::make_unique<ValidatorType>(std::move(validator)))
        , polynomials(N + 2)
        , alphas(N + 1)
        , betas(N + 1)
        , J(N + 1, N + 1),
        H(N + 1, N + 1)	
    {
        // Validate parameters directly upon construction
        domain_validator->validateParameters();
        generatePolynomials();
        buildJacobiMatrix();
    }

     /**
     * @brief Validate polynomial domain parameters.
     * @throws ParameterDomainError if parameters are invalid.
     */    void validateDomains() const {
        if (!domain_validator->validateParameters()) {
            throw ParameterDomainError("Domain validation failed: Parameters are out of bounds.");
        }
    }

    /**
     * @brief Evaluate polynomial at given point x and degree.
     * 
     * @param x Input point
     * @param degree Degree of polynomial
     * @return Polynomial value at x
     * @throws EvaluationDomainError if x is outside domain.
     * @throws std::out_of_range if degree > N
     */
    constexpr R evaluate(R x, unsigned int degree) const {
        if (!evaluation_domain.contains(x)) {
            throw EvaluationDomainError("Evaluation point outside valid domain.");
        }
        if (degree > N) {
            throw std::out_of_range("Requested degree exceeds max polynomial degree.");
        }

        return derived().getPolynomial(degree).operator()(x); 
    }

    /**
     * @brief Generate orthogonal polynomials up to degree N+1 using recurrence relations.
     */
    void generatePolynomials() {
        #pragma omp parallel for simd
        for (unsigned int i = 0; i < N + 2; ++i) {
            polynomials[i] = PolyType(StoringArray::Zero(N + 1));
        }
    
        // P₀(x) is implicitly zero.
        // P₁(x) = 1 (coefficient of x⁰ is 1)
        if (N + 1 > 0) {
            polynomials[1].get_coeff()(0) = 1.0/std::sqrt(recurrence_coeffs.beta_0(0.0));
            //polynomials[1].get_coeff()(0) = 1.0;
        }
        H.col(0) = polynomials[1].get_coeff();

        if constexpr (N == 0) {
             // Pre-calculate and store alpha_0 and beta_0 if needed elsewhere
             if (N + 1 > 0) { // Need alpha_0 for J matrix
                alphas[0] = recurrence_coeffs.alpha_k(0);
                // beta_0 is often special or not used directly in J matrix
                // betas[0] = recurrence_coeffs.beta_0(0.0); // Store if needed
             }
             return;
        }
    
    
        // --- Sequential Recurrence Calculation ---
    
        // Temporary storage for the coefficients of P_{k+1}
        // Allocated once outside the loop to avoid repeated allocations.
        StoringArray P_kp1_coeffs(N + 1);
    
        // Calculate P₂, P₃, ..., P_{N+1} sequentially
        for (unsigned int k = 1; k <= N; ++k) {
            // Calculate recurrence coefficients for this step
            const R alpha_km1 = recurrence_coeffs.alpha_k(k - 1); // α_{k-1}
            const R beta_km1 = (k == 1) ? recurrence_coeffs.beta_0(0.0)  // β₀
                                       : recurrence_coeffs.beta_k(k - 1); // β_{k-1}
            const R beta_kp1 = recurrence_coeffs.beta_k(k); // β_{k}
    
            // Store coefficients for later use (e.g., Jacobi matrix, evaluation)
            alphas[k - 1] = alpha_km1;
            betas[k - 1] = beta_km1; // Store β₀, β₁, ..., β_{N-1}
    
            // Get references to coefficients of P_k and P_{k-1}
            const auto& P_k_coeffs = polynomials[k].get_coeff();
            const auto& P_km1_coeffs = polynomials[k - 1].get_coeff();
    
            // --- Calculation using Eigen's vectorized operations ---
            // P_kp1 = (x * P_k) - alpha_km1 * P_k - beta_km1 * P_km1
    
            // 1. Compute x * P_k (results in higher degree, store temporarily)
            P_kp1_coeffs.setZero(); // Clear previous iteration's result
            // Shift P_k's coefficients: P_k[i] -> (x*P_k)[i+1]
            // Copies P_k[0..N-1] to P_kp1_coeffs[1..N]
            P_kp1_coeffs.segment(1, N) = P_k_coeffs.head(N);
    
            // 2. Subtract alpha_km1 * P_k
            P_kp1_coeffs -= alpha_km1 * P_k_coeffs;
    
            // 3. Subtract beta_km1 * P_{k-1}
            // P_kp1_coeffs -= beta_km1 * P_km1_coeffs;

            P_kp1_coeffs -= std::sqrt(beta_km1) * P_km1_coeffs;
            P_kp1_coeffs /= std::sqrt(beta_kp1); // Normalize by sqrt(beta_k)
            // --- End of Eigen calculation ---
    
            // Store the final coefficients for P_{k+1}
            H.col(k) = P_kp1_coeffs;
            polynomials[k + 1].set_coeff(P_kp1_coeffs);
        }


        // Store coefficients for later use (e.g., Jacobi matrix, evaluation)
        alphas[N] = recurrence_coeffs.alpha_k(N);
        betas[N] = recurrence_coeffs.beta_k(N);
            }
        
    
    /// @name Accessors for α and β recurrence coefficients
    /// @{
    auto getAlpha(unsigned int k) const noexcept {
        return alphas[k];
    }

    auto getBeta(unsigned int k) const noexcept {
        return betas[k];
    }
    
    auto getAlphas() const noexcept {
        return alphas;
    }
    auto getBetas() const noexcept {
        return betas;
    }
    /// @}

    /**
     * @brief Retrieve the polynomial of given degree.
     * 
     * @param degree Degree of polynomial
     * @return Const reference to polynomial
     * @throws std::out_of_range if degree > N
     */
    const PolyType& getPolynomial(unsigned int degree) const {
        if (degree > N) {
            throw std::out_of_range("Requested degree exceeds max polynomial degree.");
        }
        return polynomials[degree + 1];
    }

    /**
     * @brief Evaluate the weight function at point x.
     * @param x Point at which to evaluate
     * @return Value of weight function
     * @throws EvaluationDomainError if x is outside valid domain
     */
    R getWeight(R x) const {

        if (!evaluation_domain.contains(x)) {

            throw EvaluationDomainError("Evaluation point outside valid domain");

        }

        return derived().weight_function(x);

    }

    /**
     * @brief Retrieve recurrence coefficients object.
     * @return RecurrenceCoefficients<R>
     */
    RecurrenceCoefficients<R> getRecurrenceCoefficients() const {

        return recurrence_coeffs;
    }

    /**
     * @brief Build Jacobi matrix used in spectral methods and eigenvalue problems.
     */
    void buildJacobiMatrix() {
        // Fill the Jacobi matrix with recurrence coefficients
        Triplets triplets(N + 1);        
        const int expectedNonZeros = 3 * N + 1;  // Main diagonal + upper + lower
        triplets.reserve(expectedNonZeros);
         

        #pragma omp parallel
        {
        // Each thread maintains its own local vector of triplets
        Triplets localTriplets;
        localTriplets.reserve(expectedNonZeros / omp_get_num_threads());

        #pragma omp for nowait
        for(unsigned int i = 0; i < N + 1; ++i) {
            // Main diagonal
            localTriplets.emplace_back(i, i, alphas[i]);
            
            // Lower and upper diagonals
            if(i > 0) {
                localTriplets.emplace_back(i, i-1, std::sqrt(betas[i]));    // Lower
                localTriplets.emplace_back(i-1, i, std::sqrt(betas[i]));    // Upper
            }
        }

        // Merge local triplets into global vector
        #pragma omp critical
        {
            triplets.insert(triplets.end(), 
                          localTriplets.begin(), 
                          localTriplets.end());
        }
    }

    // Set matrix from triplets
    J.resize(N + 1, N + 1);
    J.setFromTriplets(triplets.begin(), triplets.end());
    J.makeCompressed();
    };

    /**
     * @brief Get the constructed Jacobi matrix.
     * @return Sparse Jacobi matrix
     */
    JacobiMatrixType getJacobiMatrix() const noexcept {
        return J;
    }
    
    /**
     * @brief Get matrix of polynomial coefficients.
     * @return Dense matrix of polynomial coefficients
     */
    HMatrixType getHMatrix() const noexcept {
        return H;
    }
};

/**
 * @brief Legendre polynomials on [-1, 1] with uniform weight.
 */
template<unsigned int N, class R = traits::DataType::PolynomialField>
class LegendrePolynomial
    : public OrthogonalPolynomialBase<LegendrePolynomial<N, R>, N, R> {

public:
    using Base = OrthogonalPolynomialBase<LegendrePolynomial<N, R>, N, R>;

    LegendrePolynomial()
        : Base(
            [](R x) { return 1.0; }, // Weight function
            RecurrenceCoefficients<R>{
                [](R n) { return (void)n; 0.0; },                           // alpha_k
                [](R n) { return 1.0 / (4 - std::pow(n, -2)); },  // beta_k
                [](R n) { return (void)n; 2.0; }                           // beta_0
            },
            DomainInterval<R>{-1, 1}, // Valid domain for Legendre polynomials
            PolynomialDomainValidator<R>() // No additional parameter validation required
        ) {}
};

/**
 * @brief Hermite polynomials with parameters μ and σ.
 */
template<unsigned int N, class R = traits::DataType::PolynomialField>
class HermitePolynomial
    : public OrthogonalPolynomialBase<HermitePolynomial<N, R>, N, R, HermiteMu<R>, HermiteSigma<R>> {

private:
    R mu_; // Parameter specific to Hermite polynomials
    R sigma_; // Parameter specific to Hermite polynomials

public:
    using Base = OrthogonalPolynomialBase<HermitePolynomial<N, R>, N, R, HermiteMu<R>, HermiteSigma<R>>;

    HermitePolynomial(R mu = 0, R sigma = M_SQRT2)
        : Base(
            [mu, sigma](R x) { return 1/(sigma*std::sqrt(M_PI*2)) * std::exp(-std::pow(x-mu, 2)/(2*sigma*sigma)); }, // Weight function
            RecurrenceCoefficients<R>{
                [mu](R n) { (void)n; return mu; },                  // alpha_k
                [sigma](R n) { return n*sigma*sigma; },             // beta_k
                [sigma](R n) { (void)n; return sigma; }             // beta_0
            },
            DomainInterval<R>{std::numeric_limits<R>::lowest(), 
                              std::numeric_limits<R>::max()}, // Valid domain for Hermite
            PolynomialDomainValidator<R, HermiteMu<R>, HermiteSigma<R>>(HermiteMu<R>(mu), HermiteSigma<R>(sigma)) // Parameter validator
        ),
        mu_(mu),
        sigma_(sigma) 
    {
        // Core parameter validation happens automatically in Base class validator
    }
};

/**
 * @brief Chebyshev polynomials of the first kind.
 */
template<unsigned int N, class R = traits::DataType::PolynomialField>
class ChebychevPolynomial
    : public OrthogonalPolynomialBase<ChebychevPolynomial<N, R>, N, R> {

public:
    using Base = OrthogonalPolynomialBase<ChebychevPolynomial<N, R>, N, R>;

    ChebychevPolynomial()
        : Base(
            [](R x) { return std::pow(1 - x * x, -0.5); }, // Weight function
            RecurrenceCoefficients<R>{
                [](R n) { (void)n; return 0.0; },                  // alpha_k
                [](R n) { return n == 1 ? 0.5 : 0.25; },           // beta_k
                [](R n) { (void)n; return M_PI; }                  // beta_0
            },
            DomainInterval<R>{-1, 1}, // Valid domain for Chebyshev polynomials
            PolynomialDomainValidator<R>() // No additional parameter validation required
        ) {}
};

/**
 * @brief Generalized Laguerre polynomials with parameters α and θ.
 */
template<unsigned int N, class R = traits::DataType::PolynomialField>
class LaguerrePolynomial 
    : public OrthogonalPolynomialBase<LaguerrePolynomial<N, R>, N, R, LaguerreAlpha<R>, LaguerreTheta<R>> {
private:
    R alpha_;   // Parameter specific to Laguerre polynomials
    R theta_;   // Parameter specific to Laguerre polynomials

public:
    using Base = OrthogonalPolynomialBase<LaguerrePolynomial<N, R>, N, R, LaguerreAlpha<R>, LaguerreTheta<R>>;
    
    LaguerrePolynomial(R alpha, R theta = 1) 
        : Base(
            [alpha, theta](R x) { return std::exp(-x/theta)*std::pow(x, alpha)/(std::pow(theta, alpha + 1)*std::tgamma(1 + alpha)); },
            RecurrenceCoefficients<R>{
            [alpha, theta](R n){ return (2*n + alpha + 1)*theta;},              // alpha_k
            [alpha, theta](R n){ return n*(n + alpha)*theta*theta;},            // beta_k
            [alpha](R n){ (void)n;return 1;}},                                  // beta_0
            DomainInterval<R>{0.0, std::numeric_limits<R>::max()},
            PolynomialDomainValidator<R, LaguerreAlpha<R>, LaguerreTheta<R>>(LaguerreAlpha<R>(alpha), LaguerreTheta<R>(theta))
            ), alpha_(alpha), theta_(theta)
            {    }



};

/**
 * @brief Jacobi polynomials with parameters α and β.
 * 
 */
template<unsigned int N, class R = traits::DataType::PolynomialField>
class JacobiPolynomial
    : public OrthogonalPolynomialBase<JacobiPolynomial<N, R>, N, R, JacobiAlpha<R>, JacobiBeta<R>> {

private:
    R alpha_; // Alpha parameter specific to Jacobi polynomials
    R beta_;  // Beta parameter specific to Jacobi polynomials

public:
    using Base = OrthogonalPolynomialBase<JacobiPolynomial<N, R>, N, R, JacobiAlpha<R>, JacobiBeta<R>>;

    JacobiPolynomial(R alpha, R beta)
        : Base(
            [alpha, beta](R x) { 
                return std::pow(1 - x, alpha) * std::pow(1 + x, beta); 
            }, // Weight function
            RecurrenceCoefficients<R>{
                [alpha, beta](R n) { 
                    return (beta * beta - alpha * alpha) / 
                           ((2 * n + alpha + beta) * (2 * n + alpha + beta + 2)); 
                }, // alpha_k
                [alpha, beta](R n) { 
                    return 4 * n * (n + beta) * (n + alpha) * (n + alpha + beta) /
                           ((2 * n + alpha + beta) * (2 * n + alpha + beta) * 
                            (2 * n + alpha + beta - 1) * (2 * n + alpha + beta + 1)); 
                }, // beta_k
                [alpha, beta](R n) {
                    (void)n; 
                    return std::pow(2, alpha + beta + 1) * 
                           std::tgamma(alpha + 1) * std::tgamma(beta + 1) / 
                           std::tgamma(alpha + beta + 1); 
                } // beta_0
            },
            DomainInterval<R>{-1, 1}, // Valid domain
            PolynomialDomainValidator<R, JacobiAlpha<R>, JacobiBeta<R>>(
                JacobiAlpha<R>(alpha),
                JacobiBeta<R>(beta)
            ) // Parameter validator
        ),
        alpha_(alpha), 
        beta_(beta) 
    {
        // Validation happens automatically in the base class
    }
};

/**
 * @brief Gegenbauer polynomials with parameter λ (lambda).
 * 
 * Gegenbauer polynomials are a generalization of Legendre polynomials and are orthogonal on the interval [-1, 1]
 * with respect to the weight function (1 - x^2)^(λ - 0.5).
 */
template<unsigned int N, class R = traits::DataType::PolynomialField>
class GegenbauerPolynomial
    : public OrthogonalPolynomialBase<GegenbauerPolynomial<N, R>, N, R, GegenbauerLambda<R>> {

private:
    R lambda_; // Lambda parameter specific to Gegenbauer polynomials

public:
    using Base = OrthogonalPolynomialBase<GegenbauerPolynomial<N, R>, N, R, GegenbauerLambda<R>>;

    GegenbauerPolynomial(R lambda)
        : Base(
            [lambda](R x) { 
                return std::pow(1 - x * x, lambda - 0.5); 
            }, // Weight function
            RecurrenceCoefficients<R>{
                [](R n) { 
                    (void)n;
                    return 0.0; 
                }, // alpha_k
                [lambda](R n) { 
                    return n * (n + 2 * lambda - 1) / 
                           (4 * (n + lambda) * (n + lambda - 1)); 
                }, // beta_k
                [lambda](R n) { 
                    (void)n;
                    return 2 / M_2_SQRTPI * std::tgamma(lambda + 0.5) / 
                           std::tgamma(lambda + 1.0); 
                } // beta_0
            },
            DomainInterval<R>{-1, 1}, // Valid domain
            PolynomialDomainValidator<R, GegenbauerLambda<R>>(GegenbauerLambda<R>(lambda)) // Validator
        ),
        lambda_(lambda)
    {
        // Validation happens automatically in the base class
    }
};



};// namespace OPOE

#endif // HH_ORTHOGONAL_POLYNOMIALS_HH

