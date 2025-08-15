#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iomanip>
#include <sde/FinModels.hpp>
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
#include "polynomials/OrthogonalPolynomials.hpp"
#include "utils/Utils.hpp"

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;
using namespace SDE;

static std::vector<std::pair<int,int>> enumerate_basis(int N) {
    std::vector<std::pair<int,int>> E;
    E.reserve((N+1)*(N+2)/2);
    for (int m = 0; m <= N; ++m) {
        for (int n = 0; n <= N - m; ++n) {
            E.emplace_back(m, n);
        }
    }
    return E;
}

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <vector>
#include <utility>

using Eigen::MatrixXd;
using StoringVector = Eigen::VectorXd;


int main() {
    int N = 2; // total degree
    auto E_tri = enumerate_basis(N-1);
    int dim_tri = E_tri.size();

    int Nv_full = N+1;
    int Nx_full = N+1; // we need Nx = N+1 to cover all n
    int dim_full = Nv_full * Nx_full;

    using R = double;
    polynomials::HermitePolynomial<2, R> hermite(0.0, 1.0); // mu=0, sigma=1
    std::cout << "Hermite polynomial of degree " << Nv_full-1 << ":\n";
    for (int i = 0; i < Nv_full; ++i) {
        std::cout << "H_" << i << "(x) = " << hermite.getPolynomial(i) << "\n"; // Evaluate at x=0
    }
    R S0 = 100.0;     // Spot price
    // R K = 100.0;      // Strike
    R r = 0.05;       // Risk-free rate
    // R T = 1.0;        // Time to maturity
    R sigma = 0.2;    // Volatility


    Eigen::VectorXd x0(2);
    x0 << 0.20, std::log(100.0);

    // GBM model (no need to configure beyond placeholder here)
    auto heston_model = std::make_shared<HullWhiteModelSDE<R>>(0.05, 2.0, 0.20, 0.3, -0.7, x0);


    

    // Example H for Nx_full=3
    MatrixXd H(3,3);
    H << 1, 0, -std::sqrt(2.0)/2,
         0, 1,  0,
         0, 0,  std::sqrt(2.0)/2;

    MatrixXd HS(2, 2);
    HS << 1, 0,
         0, 1;
    



    // Project onto triangular basis4
    MatrixXd G_tri_projected = heston_model->generator_G(E_tri, HS);

    // Compare with generator_G
    SDEMatrix G_tri_generator = heston_model->generator_G(E_tri, 2, 1);
    std::cout << "G:" << G_tri_generator << "\n";

    // Pretty-print differences
    std::cout << "Differences G_tri_projected - G_tri_generator:\n";
    std::cout << (G_tri_projected - G_tri_generator) << "\n";

    std::cout << "vG_tri:" << G_tri_projected << "\n";

  
}
