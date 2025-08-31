
/*
    bindings.cpp - Pybind11 bindings for stochastic volatility SDE models, solvers, and option pricers.

    This module exposes C++ classes and functions to Python, enabling the use of advanced stochastic differential equation (SDE) models,
    numerical solvers, and option pricing engines in Python workflows. The bindings are designed for flexibility and extensibility,
    supporting multiple models, solvers, and pricer types.

    Main Features:
    --------------
    - SDE Models:
        * Heston, Hull-White, Stein-Stein, Jacobi, and Geometric Brownian Motion models.
        * All models inherit from a common ISDEModel interface.
        * Model parameters and state dimensions are accessible from Python.

    - SDE Solvers:
        * Euler-Maruyama, Milstein, and Interpolated Kahl-Jackel solvers.
        * Generic binding for solver classes, supporting custom SDE models.
        * Solvers can simulate SDE paths with optional user-supplied Brownian increments.

    - Option Payoffs:
        * Base IPayoff interface for extensibility.
        * European call and put payoffs with strike getter/setter and evaluation methods.

    - Option Pricers:
        * Monte Carlo (MCOptionPricer), FFT-based (FFTOptionPricer), and Operator Expansion (OPEOptionPricer) pricers.
        * OPEOptionPricer is templated for different expansion orders (N = 3, 5, 7, 9, 10).
        * All pricers accept model, payoff, and numerical method parameters.

    - Enumerations:
        * QuadratureMethod: TanhSinh, QAGI.
        * SolverType: EulerMaruyama, Milstein, IJK.

    Usage:
    ------
    Import the module in Python as `opoe` and instantiate models, solvers, payoffs, and pricers as needed.
    Example:
        import opoe
        model = opoe.HestonModel(...)
        payoff = opoe.EuropeanCallPayoff(K=100)
        pricer = opoe.MCOptionPricer(ttm=1.0, rate=0.05, payoff=payoff, model=model)
        price = pricer.price()

    Notes:
    ------
    - All numerical types (Real, Matrix, Vector) are defined via the traits system for flexibility.
    - Shared pointers are used for memory management and Python interoperability.
    - The bindings are designed to be extensible for additional models, solvers, and pricer types.

*/
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <optional>

#include "../include/sde/FinModels.hpp"              
#include "../include/options/MCOptionPricer.hpp"     
#include "../include/options/FFTOptionPricer.hpp"    
#include "../include/options/OPEOptionPricer.hpp"    
#include "../include/options/CFOptionPricer.hpp"    
#include "../include/options/Payoff.hpp"            
#include "../include/traits/OPOE_traits.hpp" 

namespace py = pybind11;

// Short-hands
using Real   = traits::DataType::PolynomialField; 
using Matrix = traits::DataType::StoringMatrix; 
using Vector = traits::DataType::StoringVector;

/**
 * @brief Helper function to expose a solver class to Python via pybind11.
 *
 * This function creates a Python binding for a given solver class by registering
 * it in the specified pybind11 module. It binds the constructor and the
 * `solve` method of the solver to Python.
 *
 * @tparam SolverClass  The concrete solver class to bind.
 * @param m             The pybind11 module where the class will be registered.
 * @param name          The name of the solver class in the Python module.
 *
 * The bound Python class provides:
 * - A constructor taking a reference to an SDE model (`SDE::ISDEModel<Real>`).
 * - A `solve` method that computes the SDE solution over a specified time interval.
 */
template <typename SolverClass>
void bind_solver(py::module_ &m, const char *name) {
    py::class_<SolverClass, std::shared_ptr<SolverClass>>(m, name)
        .def(py::init<SDE::ISDEModel<Real> &>(), py::arg("model"))
        .def("solve",
             [](SolverClass &self,
                Real t0, Real ttm,
                int num_steps, int num_paths,
                const std::optional<Matrix> &dW_opt) -> Matrix {
                 // Call the solve method of the specific solver instance
                 return self.solve(t0, ttm, num_steps, num_paths, dW_opt);
             },
             py::arg("t0"), py::arg("ttm"),
             py::arg("num_steps"), py::arg("num_paths"),
             py::arg("dW_opt") = std::nullopt,
             "Solves the SDE path over the given time interval.");
}

/**
 * @brief Bind a generic stochastic volatility (SV) model to Python via pybind11.
 *
 * This template function wraps a C++ SDE model (`ModelT`) so it can be used in Python.
 * It exposes the constructor, common getters, and setters for model parameters, 
 * as well as utility methods like `state_dim` and `wiener_dim`.
 *
 * @tparam ModelT The specific SDE model type (e.g., HestonModelSDE, JacobiModelSDE).
 * @param m The Python module where the class should be added.
 * @param py_name The name of the Python class to expose.
 *
 * @details
 * The binding includes:
 * - Constructor:
 *   - Accepts initial variance `v0`, mean reversion `kappa`, long-term variance `theta`,
 *     volatility of variance `sigma`, correlation `rho`, and initial log-price `x0`.
 * - Getters (read-only access to model parameters):
 *   - `get_x0()`, `get_v0()`, `get_kappa()`, `get_theta()`, `get_sigma_v()`, `get_correlation()`
 * - Setters (allows modification of model parameters from Python):
 *   - `set_correlation(rho)`, `set_kappa(kappa)`, `set_theta(theta)`, `set_sigma_v(sigma)`, `set_drift(mu)`, 'set_v0(v0)', 'set_x0(x0)'
 *     - Each setter calls `notify_observers()` internally to update dependent calculations.
 * - Utility methods:
 *   - `state_dim()` – returns the dimension of the state vector.
 *   - `wiener_dim()` – returns the dimension of the Wiener process.
 *
 * @note
 * The setters use lambda functions to ensure proper type conversions and allow for
 * additional logic (e.g., domain checks) in C++ to be enforced when called from Python.
 * 
 */
template <typename ModelT>
void bind_model(py::module_& m, const char* py_name) {
    py::class_<ModelT, SDE::ISDEModel<Real>, std::shared_ptr<ModelT>>(m, py_name)
        // generic constructor (adjust if not always valid)
        .def(py::init<Real, Real, Real, Real, Real, const Vector&>(),
             py::arg("v0"), py::arg("kappa"), py::arg("theta"),
             py::arg("sigma"), py::arg("rho"), py::arg("x0"))
        // common getters (these must exist in all models)
        .def("get_x0", py::overload_cast<>(&ModelT::get_x0, py::const_))
        .def("get_v0", py::overload_cast<>(&ModelT::get_v0, py::const_))
        .def("get_kappa", py::overload_cast<>(&ModelT::get_kappa, py::const_))
        .def("get_theta", py::overload_cast<>(&ModelT::get_theta, py::const_))
        .def("get_sigma_v", py::overload_cast<>(&ModelT::get_sigma_v, py::const_))
        .def("get_correlation", py::overload_cast<>(&ModelT::get_correlation, py::const_))
        // Setters
        .def("set_x0", [](ModelT &self, Real x0) { self.set_x0(x0); })
        .def("set_v0", [](ModelT &self, Real v0) { self.set_v0(v0); })
        .def("set_correlation", [](ModelT &self, Real rho) { self.set_correlation(rho); })
        .def("set_kappa", [](ModelT &self, Real kappa) { self.set_kappa(kappa); })
        .def("set_theta", [](ModelT &self, Real theta) { self.set_theta(theta); })
        .def("set_sigma_v", [](ModelT &self, Real sigma) { self.set_sigma_v(sigma); })
        .def("set_drift", [](ModelT &self, Real mu) { self.set_drift(mu); })
        .def("state_dim", &ModelT::state_dim)
        .def("wiener_dim", &ModelT::wiener_dim);
}


/**
 * @brief Bind the Geometric Brownian Motion (GBM) model to Python via pybind11.
 *
 * This specialization wraps the `SDE::GeometricBrownianMotionSDE` class so it can be used
 * from Python. It exposes the constructor, getters, setters, and utility functions for
 * state and Wiener process dimensions.
 *
 * @param m The Python module where the class should be added.
 * @param py_name The name of the Python class to expose.
 *
 * @details
 * The binding includes:
 * - Constructor:
 *   - `__init__(mu, sigma, x0)` where:
 *     - `mu` is the drift of the asset.
 *     - `sigma` is the volatility of the asset.
 *     - `x0` is the initial value of the log-price.
 * - Getters (read-only access to model parameters):
 *   - `get_x0()` – returns the initial state.
 *   - `get_sigma()` – returns the volatility parameter.
 *   - `get_mu()` – returns the drift parameter.
 * - Setters (modify parameters from Python):
 *   - `set_x0(x0)` – sets the initial state.
 *   - `set_sigma(sigma)` – sets the volatility.
 *   - `set_mu(mu)` – sets the drift.
 * - Utility methods:
 *   - `state_dim()` – returns the dimension of the state vector.
 *   - `wiener_dim()` – returns the dimension of the Wiener process.
 *
 * @note
 * The GBM model is simpler than general stochastic volatility models, so no domain checks
 * or observer notifications are required for the setters.
 *
 */
template <>
void bind_model<SDE::GeometricBrownianMotionSDE<Real>>(py::module_& m, const char* py_name) {
    py::class_<SDE::GeometricBrownianMotionSDE<Real>, SDE::ISDEModel<Real>,
               std::shared_ptr<SDE::GeometricBrownianMotionSDE<Real>>>(m, py_name)
        .def(py::init<Real, Real, Real>(),
             py::arg("mu"), py::arg("sigma"), py::arg("x0"))
        .def("get_x0", py::overload_cast<>(&SDE::GeometricBrownianMotionSDE<Real>::get_x0, py::const_))
        .def("get_sigma", py::overload_cast<>(&SDE::GeometricBrownianMotionSDE<Real>::get_v0, py::const_))
        .def("get_mu", py::overload_cast<>(&SDE::GeometricBrownianMotionSDE<Real>::get_mu, py::const_))
        .def("set_x0", &SDE::GeometricBrownianMotionSDE<Real>::set_x0)
        .def("set_sigma", &SDE::GeometricBrownianMotionSDE<Real>::set_v0)
        .def("set_mu", &SDE::GeometricBrownianMotionSDE<Real>::set_mu)
        .def("state_dim", &SDE::GeometricBrownianMotionSDE<Real>::state_dim)
        .def("wiener_dim", &SDE::GeometricBrownianMotionSDE<Real>::wiener_dim);
}

/**
 * @brief Bind the Jacobi Stochastic Volatility (SV) model to Python via pybind11.
 *
 * This specialization wraps the `SDE::JacobiModelSDE` class so it can be used from Python.
 * It exposes the constructor, getters, setters, and utility functions for state and Wiener
 * process dimensions.
 *
 * @param m The Python module where the class should be added.
 * @param py_name The name of the Python class to expose.
 *
 * @details
 * The binding includes:
 * 
 * - Constructor:
 *   - `__init__(v0, kappa, theta, sigma, rho, y_min, y_max, x0)` where:
 *     - `v0` is the initial variance.
 *     - `kappa` is the mean reversion speed of the variance process.
 *     - `theta` is the long-term mean of the variance process.
 *     - `sigma` is the volatility of volatility (SV process volatility).
 *     - `rho` is the correlation between the asset and variance Wiener processes.
 *     - `y_min` is the lower bound for the variance process.
 *     - `y_max` is the upper bound for the variance process.
 *     - `x0` is the initial asset state.
 *
 * - Getters (read-only access to model parameters):
 *   - `get_x0()` – initial asset value.
 *   - `get_v0()` – initial variance.
 *   - `get_kappa()` – mean reversion speed.
 *   - `get_theta()` – long-term mean.
 *   - `get_sigma_v()` – volatility of the variance process.
 *   - `get_correlation()` – correlation parameter.
 *   - `get_y_min()` – lower variance bound.
 *   - `get_y_max()` – upper variance bound.
 *
 * - Setters (update parameters from Python with domain checks and notifications):
 *   - `set_x0(x0)` – sets initial asset state.
 *   - `set_v0(v0)` – sets initial variance.
 *   - `set_kappa(kappa)` – sets mean reversion speed.
 *   - `set_theta(theta)` – sets long-term mean.
 *   - `set_sigma_v(sigma)` – sets volatility of variance process.
 *   - `set_correlation(rho)` – sets correlation.
 *   - `set_y_min(ymin)` – sets lower variance bound (must be positive and < y_max).
 *   - `set_y_max(ymax)` – sets upper variance bound (must be > y_min and positive).
 *
 * - Utility methods:
 *   - `state_dim()` – dimension of the state vector.
 *   - `wiener_dim()` – dimension of the Wiener process.
 *
 * @note
 * Setters perform domain checks and trigger observer notifications if the value is updated.
 * Python users must provide values satisfying these constraints to avoid `std::domain_error`.
 *
 */
template <>
void bind_model<SDE::JacobiModelSDE<Real>>(py::module_& m, const char* py_name) {
    py::class_<SDE::JacobiModelSDE<Real>, SDE::ISDEModel<Real>, std::shared_ptr<SDE::JacobiModelSDE<Real>>>(m, py_name)
        // generic constructor (adjust if not always valid)
        .def(py::init<Real, Real, Real, Real, Real, Real, Real, const Vector&>(),
             py::arg("v0"), py::arg("kappa"), py::arg("theta"),
             py::arg("sigma"), py::arg("rho"), py::arg("y_min"), 
             py::arg("y_max"), py::arg("x0"))
        // common getters (these must exist in all models)
        .def("get_x0", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_x0, py::const_))
        .def("get_v0", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_v0, py::const_))
        .def("get_kappa", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_kappa, py::const_))
        .def("get_theta", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_theta, py::const_))
        .def("get_sigma_v", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_sigma_v, py::const_))
        .def("get_correlation", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_correlation, py::const_))
        .def("get_y_min", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_y_min, py::const_))
        .def("get_y_max", py::overload_cast<>(&SDE::JacobiModelSDE<Real>::get_y_max, py::const_))
        // Setters
        .def("set_x0", &SDE::JacobiModelSDE<Real>::set_x0)
        .def("set_v0", &SDE::JacobiModelSDE<Real>::set_v0)
        .def("set_kappa", &SDE::JacobiModelSDE<Real>::set_kappa, py::arg("kappa"))
        .def("set_theta", &SDE::JacobiModelSDE<Real>::set_theta, py::arg("theta"))
        .def("set_sigma_v", &SDE::JacobiModelSDE<Real>::set_sigma_v, py::arg("sigma"))
        .def("set_correlation", &SDE::JacobiModelSDE<Real>::set_correlation, py::arg("rho"))
        .def("set_y_min", &SDE::JacobiModelSDE<Real>::set_y_min, py::arg("y_min"))
        .def("set_y_max", &SDE::JacobiModelSDE<Real>::set_y_max, py::arg("y_max"))
        .def("state_dim", &SDE::JacobiModelSDE<Real>::state_dim)
        .def("wiener_dim", &SDE::JacobiModelSDE<Real>::wiener_dim);
}

/**
 * @brief Bind the OPEOptionPricer class template to Python.
 *
 * This function template exposes the `options::OPEOptionPricer<Real, N>` 
 * class to Python using pybind11. The option pricer computes option prices 
 * under a given SDE model using the Orthonormal Polynomial Expansion.
 *
 * @tparam N         The number of dimensions (state variables) handled by the pricer.
 * @param m          The pybind11 module where the class will be registered.
 * @param class_name The name of the Python class to expose.
 *
 * ### Bound constructor
 * - `__init__(ttm: Real, rate: Real, payoff: IPayoff, model: ISDEModel,
 *             solver_type: SolverType = EulerMaruyama,
 *             integrator: QuadratureMethod = TanhSinh,
 *             num_paths: int = 10)`  
 *
 *   Constructs an OPE option pricer with:
 *   - `ttm` : Time-to-maturity.  
 *   - `rate` : Risk-free interest rate.  
 *   - `payoff` : A payoff function (must implement `IPayoff<Real>`).  
 *   - `model` : An SDE model (must implement `ISDEModel<Real>`).  
 *   - `solver_type` : Numerical SDE solver (default: Euler–Maruyama).  
 *   - `integrator` : Quadrature integration method (default: Tanh–Sinh).  
 *   - `num_paths` : Number of Monte Carlo paths (default: 10).  
 *
 * ### Bound methods
 * - `price() -> Real` : Computes the option price.  
 *
 * @note The class is registered as `OPEOptionPricer<N>` in Python.
 */
template <int N>
void bind_OPEOptionPricer(py::module& m, const std::string& class_name) {
    using OPEType = options::OPEOptionPricer<Real, N>;

    py::class_<OPEType, std::shared_ptr<OPEType>>(m, class_name.c_str())
        .def(py::init(
            [](Real ttm, Real rate,
               std::shared_ptr<options::IPayoff<Real>> payoff,
               std::shared_ptr<SDE::ISDEModel<Real>> model,
                unsigned int K,
               traits::OPEMethod solving_param = traits::OPEMethod::Integration,
               traits::SolverType solver_type = traits::SolverType::EulerMaruyama,
               traits::QuadratureMethod integrator = traits::QuadratureMethod::TanhSinh,
               unsigned int num_paths = 10) 
            {
                return std::make_shared<OPEType>(
                    ttm, rate,
                    std::move(payoff), model, K, 
                    solving_param, solver_type, integrator, num_paths
                );
            }),
            py::arg("ttm"), py::arg("rate"),
            py::arg("payoff"), py::arg("model"), py::arg("K"),
            py::arg("solving_param") = traits::OPEMethod::Integration,
            py::arg("solver_type") = traits::SolverType::EulerMaruyama,
            py::arg("integrator") = traits::QuadratureMethod::TanhSinh,
            py::arg("num_paths") = 10)
        .def("price", &OPEType::price);
}


PYBIND11_MODULE(opoe, m) {
    m.doc() = "Stochastic vol SDE models, solvers, and option pricers (pybind11)";

    // ----- Enums -----
    /**
     * @brief Quadrature integration methods available for option pricing.
     *
     * This enum defines the numerical integration strategies used in 
     * quadrature-based pricing. It is exposed to Python as `QuadratureMethod`.
     *
     * ### Values
     * - `QuadratureMethod.TanhSinh` : Tanh–Sinh quadrature method 
     *   (suitable for singular integrals, highly accurate).
     * - `QuadratureMethod.QAGI` : GSL’s QAGI algorithm for improper integrals.
     */
    py::enum_<traits::QuadratureMethod>(m, "QuadratureMethod")
    .value("TanhSinh", traits::QuadratureMethod::TanhSinh)
    .value("QAGI", traits::QuadratureMethod::QAGI) 
    .export_values();

    /**
     * @brief SDE solver schemes for stochastic simulation.
     *
     * This enum defines the numerical discretization methods used to 
     * approximate solutions of stochastic differential equations (SDEs).  
     * It is exposed to Python as `SolverType`.
     *
     * ### Values
     * - `SolverType.EulerMaruyama` : Explicit Euler–Maruyama scheme 
     *   (first-order weak, simple, widely used).
     * - `SolverType.Milstein` : Milstein scheme 
     *   (higher order, captures diffusion more accurately).
     * - `SolverType.IJK` : Custom/advanced solver (Ito–Taylor variant, if defined).
     */
    py::enum_<traits::SolverType>(m, "SolverType")
    .value("EulerMaruyama", traits::SolverType::EulerMaruyama)
    .value("Milstein", traits::SolverType::Milstein) 
    .value("IJK", traits::SolverType::IJK)
    .export_values();

        /**
     * @brief Enum containing Method type.
     *
     * This enum defines the solving method for OPEPricer.
     *
     * ### Values
     * - `OPEMethod.Integration` : integrates the payoff wrt approximated mixture density.
     * - `OPEMethod.Direct` : uses payoff projection computed via recursion schemes, when available.
     */
    py::enum_<traits::OPEMethod>(m, "OPEMethod")
    .value("Direct", traits::OPEMethod::Direct)
    .value("Integration", traits::OPEMethod::Integration) 
    .export_values();

    //----- Models -----

    py::class_<SDE::ISDEModel<Real>, std::shared_ptr<SDE::ISDEModel<Real>>>(m, "ISDEModel");

    bind_model<SDE::HestonModelSDE<Real>>(m, "HestonModel");
    bind_model<SDE::HullWhiteModelSDE<Real>>(m, "HullWhiteModel");
    bind_model<SDE::SteinSteinModelSDE<Real>>(m, "SteinSteinModel");
    bind_model<SDE::GeometricBrownianMotionSDE<Real>>(m, "GeometricBrownianMotionModel");
    bind_model<SDE::JacobiModelSDE<Real>>(m, "JacobiModel");

    // ----- Solvers -----
    bind_solver<SDE::EulerMaruyamaSolver<SDE::ISDEModel<Real>, Real>>(m, "EulerMaruyamaSolver");
    bind_solver<SDE::MilsteinSolver<SDE::ISDEModel<Real>, Real>>(m, "MilsteinSolver");
    bind_solver<SDE::InterpolatedKahlJackelSolver<SDE::ISDEModel<Real>, Real>>(m, "InterpolatedKahlJackelSolver");

    // ----- Payoffs -----
    // Base payoff interface
    /**
     * @brief Interface for option payoff functions.
     *
     * This is the abstract base class for all option payoff types.
     * Exposed to Python as `IPayoff`.
     *
     * Payoffs define how the terminal asset price `S_T` is transformed
     * into a cashflow (e.g. max(S_T - K, 0) for a call).
     */
    py::class_<options::IPayoff<Real>, std::shared_ptr<options::IPayoff<Real>>>(m, "IPayoff");

    // European call
    /**
     * @brief European Call option payoff.
     *
     * Payoff function: \f$\max(S_T - K, 0)\f$
     */
    py::class_<options::EuropeanCallPayoff<Real>, options::IPayoff<Real>, std::shared_ptr<options::EuropeanCallPayoff<Real>>>(m, "EuropeanCallPayoff")
    .def(py::init<Real>(), py::arg("K"))
    .def("evaluate", py::overload_cast<Real>(&options::EuropeanCallPayoff<Real>::evaluate, py::const_), py::arg("S_T"))
    .def("evaluate_from_log", py::overload_cast<Real>(&options::EuropeanCallPayoff<Real>::evaluate_from_log, py::const_), py::arg("log_S_T"))
    .def("evaluate", py::overload_cast<options::StoringVector>(&options::EuropeanCallPayoff<Real>::evaluate, py::const_), py::arg("S_T_vec"))
    .def("evaluate_from_log", py::overload_cast<options::StoringVector>(&options::EuropeanCallPayoff<Real>::evaluate_from_log, py::const_), py::arg("log_S_T_vec"))
    .def("get_strike", &options::EuropeanCallPayoff<Real>::getStrike)
    .def("set_strike", &options::EuropeanCallPayoff<Real>::setStrike, py::arg("K"));

    // European put
    /**
     * @brief European Put option payoff.
     *
     * Payoff function: \f$\max(K - S_T, 0)\f$
     */
    py::class_<options::EuropeanPutPayoff<Real>, options::IPayoff<Real>, std::shared_ptr<options::EuropeanPutPayoff<Real>>>(m, "EuropeanPutPayoff")
    .def(py::init<Real>(), py::arg("K"))
    .def("evaluate", py::overload_cast<Real>(&options::EuropeanPutPayoff<Real>::evaluate, py::const_), py::arg("S_T"))
    .def("evaluate_from_log", py::overload_cast<Real>(&options::EuropeanPutPayoff<Real>::evaluate_from_log, py::const_), py::arg("log_S_T"))
    .def("evaluate", py::overload_cast<options::StoringVector>(&options::EuropeanPutPayoff<Real>::evaluate, py::const_), py::arg("S_T_vec"))
    .def("evaluate_from_log", py::overload_cast<options::StoringVector>(&options::EuropeanPutPayoff<Real>::evaluate_from_log, py::const_), py::arg("log_S_T_vec"))
    .def("get_strike", &options::EuropeanPutPayoff<Real>::getStrike)
    .def("set_strike", &options::EuropeanPutPayoff<Real>::setStrike, py::arg("K"));


    // ----- Pricers -----
    /**
     * @brief Monte Carlo Option Pricer.
     *
     * Prices European-style options by simulating paths of the underlying SDE
     * using the specified solver.
     *
     * @tparam Real Floating point type (e.g., double).
     *
     * ### Python Example
     * ```python
     * model = GeometricBrownianMotionSDE(mu=0.05, sigma=0.2, x0=100.0)
     * payoff = EuropeanCallPayoff(K=100.0)
     * pricer = MCOptionPricer(ttm=1.0, rate=0.01,
     *                         payoff=payoff, model=model,
     *                         solver_type=SolverType.EulerMaruyama,
     *                         num_paths=50000, num_steps=100)
     * price = pricer.price()
     */
    py::class_<options::MCOptionPricer<Real>, std::shared_ptr<options::MCOptionPricer<Real>>>(m, "MCOptionPricer")
    .def(py::init(
    [](Real ttm, Real rate,
        std::shared_ptr<options::IPayoff<Real>> payoff,
        std::shared_ptr<SDE::ISDEModel<Real>> model,
        traits::SolverType solver_type = traits::SolverType::EulerMaruyama,
        int num_paths, int num_steps) {

        return std::make_shared<options::MCOptionPricer<Real>>(
            ttm, rate,
            std::move(payoff), model,
            solver_type, num_paths, num_steps
        );
    }),
    py::arg("ttm"), py::arg("rate"),
    py::arg("payoff"), py::arg("model"), py::arg("solver_type") = traits::SolverType::EulerMaruyama,
    py::arg("num_paths") = 10000, py::arg("num_steps") = 100)
    .def("price", &options::MCOptionPricer<Real>::price);



    /**
     * @brief FFT Option Pricer.
     *
     * Prices European-style options using the **Carr–Madan FFT method**.  
     * This method is efficient for computing prices across a range of strikes.
     *
     * @tparam Real Floating point type (e.g., double).
     *
     * @param ttm  Time-to-maturity
     * @param rate Risk-free interest rate
     * @param payoff Payoff function (e.g. EuropeanCallPayoff)
     * @param model Stochastic model for underlying
     * @param Npow Power-of-two grid size for FFT (2^Npow points)
     * @param A Damping factor for characteristic function
     *
     * ### Python Example
     * ```python
     * model = GeometricBrownianMotionSDE(mu=0.05, sigma=0.2, x0=100.0)
     * payoff = EuropeanPutPayoff(K=100.0)
     * pricer = FFTOptionPricer(ttm=1.0, rate=0.01,
     *                          payoff=payoff, model=model,
     *                          Npow=12, A=10)
     * price = pricer.price()
     * ```
     */
    py::class_<options::FFTOptionPricer<Real>, std::shared_ptr<options::FFTOptionPricer<Real>>>(m, "FFTOptionPricer")
    .def(py::init([](Real ttm, Real rate,
                     std::shared_ptr<options::IPayoff<Real>> payoff,
                     std::shared_ptr<SDE::ISDEModel<Real>> sde_model,
                     unsigned int Npow, unsigned int A) {
        return std::make_shared<options::FFTOptionPricer<Real>>(
            ttm, rate, std::move(payoff), sde_model, Npow, A
        );
    }),
    py::arg("ttm"), py::arg("rate"),
    py::arg("payoff"), py::arg("model"),
    py::arg("Npow") = 10, py::arg("A") = 10)
    .def("price", &options::FFTOptionPricer<Real>::price);


 

    /**
     * @brief Exposing the various pricers depending on the number of components.
     *
     */
    bind_OPEOptionPricer<3>(m, "OPEOptionPricerN3"); 
    bind_OPEOptionPricer<5>(m, "OPEOptionPricerN5"); 
    bind_OPEOptionPricer<7>(m, "OPEOptionPricerN7"); 
    bind_OPEOptionPricer<9>(m, "OPEOptionPricerN9"); 
    bind_OPEOptionPricer<10>(m, "OPEOptionPricerN10"); 
    bind_OPEOptionPricer<15>(m, "OPEOptionPricerN15"); 
    bind_OPEOptionPricer<20>(m, "OPEOptionPricerN20"); 
    bind_OPEOptionPricer<25>(m, "OPEOptionPricerN25"); 
    bind_OPEOptionPricer<30>(m, "OPEOptionPricerN30"); 

    /**
     * @brief Closed Formula (CF) Option Pricer.
     *
     * Prices European-style options using the closed formulae. This method is only available for Geometric Brownian Motion.
     *
     * @tparam Real Floating point type (e.g., double).
     *
     * ### Python Example
     * ```python
     * model = GeometricBrownianMotionSDE(mu=0.05, sigma=0.2, x0=100.0)
     * payoff = EuropeanCallPayoff(K=100.0)
     * pricer = CFOptionPricer(ttm=1.0, rate=0.01,
     *                         payoff=payoff, model=model)
     * price = pricer.price()
     * ```
     */
    py::class_<options::CFOptionPricer<Real>, std::shared_ptr<options::CFOptionPricer<Real>>>(m, "CFOptionPricer")
    .def(py::init(
    [](Real ttm, Real rate,
        std::shared_ptr<options::IPayoff<Real>> payoff,
        std::shared_ptr<SDE::ISDEModel<Real>> model) {

        return std::make_shared<options::CFOptionPricer<Real>>(
            ttm, rate,
            std::move(payoff), model);
    }),
    py::arg("ttm"), py::arg("rate"),
    py::arg("payoff"), py::arg("model"))
    .def("price", &options::CFOptionPricer<Real>::price);




}