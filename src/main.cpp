#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iomanip>
#include "../include/sde/FinModels.hpp"
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
#include "../include/polynomials/OrthogonalPolynomials.hpp"
#include "../include/utils/Utils.hpp"
#include "../include/options/OPEOptionPricer.hpp"
#include "../include/options/FFTOptionPricer.hpp"
#include "../include/options/MCOptionPricer.hpp"

using namespace SDE;
using namespace options;

using Eigen::MatrixXd;
using StoringVector = Eigen::VectorXd;


int main() {
    const unsigned int N = 5; // total degree



    StoringVector x0(2);

    x0 << 0.20, std::log(100.0);
    using R = double;

    auto hull_model = std::make_shared<HestonModelSDE<R>>(0.04, 2.0, 0.10, 0.3, -0.7, x0);


    OPEOptionPricer<R, N> ope_pricer(1.0, 100.0, 0.05, std::make_unique<EuropeanCallPayoff<R>>(100.0), hull_model, 
    [hull_model](R t0, R ttm, int num_steps, int num_paths, const std::optional<SDEMatrix>& dW_opt) {
        return EulerMaruyamaSolver<HestonModelSDE<R>, R> (*hull_model).solve(t0, ttm, num_steps, num_paths, dW_opt);});

    std::cout << "Option Price OPE: " << ope_pricer.price() << std::endl;

    std::cout << "Option Price FFT: " << FFTOptionPricer<R>(1.0, 100.0, 0.05, std::make_unique<EuropeanCallPayoff<R>>(100.0), hull_model).price() << std::endl;

    std::cout << "Option Price MC: " << MCOptionPricer<R>(1.0, 100.0, 0.05, std::make_unique<EuropeanCallPayoff<R>>(100.0), hull_model, [hull_model](R t0, R ttm, int num_steps, int num_paths, const std::optional<SDEMatrix>& dW_opt) {
        return EulerMaruyamaSolver<HestonModelSDE<R>, R> (*hull_model).solve(t0, ttm, num_steps, num_paths, dW_opt);}, 1000, 100 ).price() << std::endl;


   

  
}
