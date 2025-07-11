#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <memory>
#include <limits>
#include <functional>
#include <cassert>
#include <algorithm>
#include <type_traits>  
#include <unsupported/Eigen/MatrixFunctions>
#include "quadrature/QuadratureRuleHolder.hpp" // Assume this includes the QuadratureRuleHolder definition
#include "sde/FinModels.hpp"
#include "sde/SDE.hpp"
#include "options/Payoff.hpp"
#include "options/CFOptionPricer.hpp"
#include "options/FFTOptionPricer.hpp"

// Implementare FFT, MC, Ackerer in OptionPricer
// Calibrazione
// Python
// Done


int main() {
    using namespace options;
    using namespace SDE;
    GeometricBrownianMotionSDE gbm_model(0.05, 0.2);
    auto T = 1;
    auto K = 100;
    auto r = 0.05;
    auto x0 = 100;

    auto payoff = std::make_unique<options::EuropeanCallPayoff<double>>(K);
    std::shared_ptr<SDE::GeometricBrownianMotionSDE<double>> model = std::make_shared<SDE::GeometricBrownianMotionSDE<double>>(0.05, 0.20);

    auto pricer = std::make_unique<options::CFPricer<double>>(
    T, K, r, x0,
    std::move(payoff),
    model  // must be a shared_ptr, not a raw reference
    );

    auto price2 = std::make_unique<options::FFTPricer<double>>(
    T, K, r, x0,
    std::move(payoff),
    model,  // must be a shared_ptr, not a raw reference
    20,
    100
);
    price2 ->price();






    std::cout << "Monte Carlo Call Price: " << pricer->price() << std::endl;

    
}





/*

   
int main() {
    try {
        using DataType = double;

        constexpr unsigned int N = 5; // Degree of basis


        // Define drift and volatility
        auto mu = 0.05;
        auto sigma = 0.2;
        
        stats::MixtureDensity<5, stats::GammaDensity> gamma_density({0.5,0.5},{
            stats::make_gamma_density(2.0, 1.0), // Shape and scale parameters
            stats::make_gamma_density(3.0, 1.5)}  // Another component
        );
        

        SDE::GeometricBrownianMotionSDE<DataType>::Parameters gbm_params{mu, sigma};
        SDE::GeometricBrownianMotionSDE<DataType> gbm_model(gbm_params);
        // Define domain and integrator
        auto domain = gamma_density.getSupport(); // Assumed to return {lower, upper}
        quadrature::QuadratureRuleHolder<double> integrator(quadrature::QuadratureType::TanhSinh);
        gamma_density.constructOrthonormalBasis();

        // Allocate generator matrix
        Eigen::MatrixXd G(N, N);

        for (unsigned int j = 0; j < N; ++j) {
            const auto& Hj = gamma_density.getPolynomial(j);
            auto dHj = polynomials::der<1>(Hj);
            auto d2Hj = polynomials::der<2>(Hj);

          
            for (unsigned int i = 0; i < N; ++i) {
                const auto& Hi = gamma_density.getPolynomial(i).as_function();

                // Weight function: Hi(x) * L(Hj)(x)
                auto weight = [&](double x) {
                    double val = Hi(x) * gbm_model.generator_fn(0.0, SDE::SDEVector::Ones(1)*x, dHj(x), d2Hj(x)) * gamma_density.pdf(x); // weighted inner product
                    return std::isfinite(val) ? val : 0.0;
                };

                G(i, j) = integrator.integrate(weight, domain.lower, domain.upper);
            }
        }

        std::cout << "G matrix (in monomial basis):\n" << G << std::endl;

        // --- Option parameters ---
        DataType K = 100.0;        // Strike price
        DataType r = 0.05;         // Risk-free rate
        DataType T = 1.0;          // Time to maturity (1 year)



        // --- Simulation settings ---
        int num_steps = 100;
        int num_paths = 1000; // Increase for more accuracy
        DataType dt = T / num_steps;
        DataType initial_state = std::log(100.0);

        // Create initial state vector: all paths start at initial_state
        SDE::SDEVector initial_x = SDE::SDEVector::Ones(1) * initial_state;
        // Parameters
    double mu = 0.21;
    double kappa = 2;
    double theta = 0.20;
    double sigma_v = 0.35;
    double rho = -0.7;

    JacobiModelSDE heston(mu, kappa, theta, sigma_v, rho, 0.1, 0.9);
    EulerMaruyamaSolver<JacobiModelSDE<double>> solver(heston);


    // Simulation settings
    double T = 1.0;
    int num_steps = 252;
    int num_paths = 10000;

    VectorXd x0(2);
    x0 <<  0.25,std::log(100.0); // Initial [S, v]
    auto start_time = std::chrono::high_resolution_clock::now();
    auto paths = solver.solve(x0, 0.0, T, num_steps, num_paths);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Simulation completed in " << elapsed_time.count() << " seconds." << std::endl;


    std::cout << "Simulated paths (first 5 paths):\n";
    // Output results
    std::cout<< "Paths" << paths << std::endl;

    bool found_nan = false;

    for (int i = 0; i < paths.rows(); ++i) {
        for (int j = 0; j < paths.cols(); ++j) {
            if (std::isnan(paths(i, j))) {
                std::cerr << "NaN detected at row " << i << ", col " << j << std::endl;
                found_nan = true;
            }
        }
    }

    if (!found_nan) {
        std::cout << "No NaNs found in paths matrix." << std::endl;
    }

    std::cout << "Simulation complete. Results written to heston_paths.csv\n";
    return 0;


}















// Check correctnes of G, since Brownian motion is split
int main() {
    options::EuropeanCallPayoff<double> call_payoff(100.0); // Example payoff
   options::EuropeanPutPayoff put_payoff(100.0); // Example payoff


   // Now we can compute the G matrix using the mixture density and a GBM model
quadrature::QuadratureRuleHolder<double> integrator(quadrature::QuadratureType::TanhSinh); // Example integrator
   stats::GammaDensity gamma_density(3, 3); // Example instantiation
   stats::MixtureDensity<5, stats::GammaDensity> mixture_density(
       {0.5, 0.5}, 
       {gamma_density, gamma_density} // Using the same density for simplicity
   );

   // Construct the orthonormal basis for the mixture density
   mixture_density.constructOrthonormalBasis(); // Generate polynomials based on the density

    // Now we can compute the G matrix

    SDE::GeometricBrownianMotionSDE gbm_model({0.05, 0.2}); // Example GBM model    
    Eigen::MatrixXd G(5, 5); // G matrix for 5 basis polynomials

    for (unsigned int j = 0; j < 5; ++j) {

        auto Hj = mixture_density.getPolynomial(j);



        
        auto dHj = polynomials::der<1>(Hj);

        auto d2Hj = polynomials::der<2>(Hj);

        

        // Define L_Hj as mu(x) * dHj + 0.5 * sigma^2(x) * d2Hj

        auto L_Hj = [&](double x) {

            return gbm_model.generator_fn(0.0, SDE::SDEVector::Ones(1)*x, dHj(x), d2Hj(x)); // Assuming t=0 for simplicity

        };


    

        for (unsigned int i = 0; i < 5; ++i) {
            


            auto Hi = mixture_density.getPolynomial(i);

            
            auto weight = [&gamma_density, &Hi, &L_Hj](double x){ double y = gamma_density.pdf(x)*Hi.as_function()(x)*L_Hj(x); if (!std::isfinite(y)) {return 0.0; }return y;}; // Define weight function as product of densities and polynomials

            //auto weight = [&gamma_density](double x){return gamma_density.pdf(x);};

               // Compute inner product <Hi, L_Hj> w.r.t. mixture_density measure


            double inner_product = integrator.integrate(weight, gamma_density.getDomain().lower, gamma_density.getDomain().upper); // Pseudo-code


            G(i, j) = inner_product;

    }

    }

// Output or use G matrix

    std::cout << "G Matrix:\n" << G << std::endl;
    std::cout << "G exp Matrix:\n" << G.exp() << std::endl;

   };
   
   */