// bindings/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <optional>

// YOUR headers (rename to your actual paths/types)
#include "../include/sde/FinModels.hpp"              // HestonModelSDE<double>
#include "../include/options/MCOptionPricer.hpp"     // MCOptionPricer<double>
#include "../include/options/FFTOptionPricer.hpp"    // FFTOptionPricer<double>
#include "../include/options/OPEOptionPricer.hpp"    // OPEOptionPricer<double, N>
#include "../include/options/Payoff.hpp"            // IPayoff<double>, EuropeanCallPayoff<double>
#include "../include/traits/OPOE_traits.hpp" // DataType, PolynomialField, etc.
namespace py = pybind11;

// Short-hands
using Real   = traits::DataType::PolynomialField; ///< Default scalar type for polynomials
using Matrix = traits::DataType::StoringMatrix; ///< Dynamic-size matrix type.
using Vector = traits::DataType::StoringVector;

// Generic template function to bind any solver class
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
             // Add a docstring for the solve method
             "Solves the SDE path over the given time interval.");
}

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
        .def("state_dim", &ModelT::state_dim)
        .def("wiener_dim", &ModelT::wiener_dim);
}

template <>
void bind_model<SDE::GeometricBrownianMotionSDE<Real>>(py::module_& m, const char* py_name) {
    py::class_<SDE::GeometricBrownianMotionSDE<Real>, SDE::ISDEModel<Real>,
               std::shared_ptr<SDE::GeometricBrownianMotionSDE<Real>>>(m, py_name)
        .def(py::init<Real, Real, Real>(),
             py::arg("mu"), py::arg("sigma"), py::arg("x0"))
        .def("get_x0", py::overload_cast<>(&SDE::GeometricBrownianMotionSDE<Real>::get_x0, py::const_))
        .def("get_sigma", py::overload_cast<>(&SDE::GeometricBrownianMotionSDE<Real>::get_v0, py::const_))
        .def("get_mu", py::overload_cast<>(&SDE::GeometricBrownianMotionSDE<Real>::get_mu, py::const_))
        .def("state_dim", &SDE::GeometricBrownianMotionSDE<Real>::state_dim)
        .def("wiener_dim", &SDE::GeometricBrownianMotionSDE<Real>::wiener_dim);
}

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
        .def("state_dim", &SDE::JacobiModelSDE<Real>::state_dim)
        .def("wiener_dim", &SDE::JacobiModelSDE<Real>::wiener_dim);
}

template <int N>
void bind_OPEOptionPricer(py::module& m, const std::string& class_name) {
    using OPEType = options::OPEOptionPricer<Real, N>;

    py::class_<OPEType, std::shared_ptr<OPEType>>(m, class_name.c_str())
        .def(py::init(
            [](Real ttm, Real rate,
               std::unique_ptr<options::IPayoff<Real>> payoff,
               std::shared_ptr<SDE::ISDEModel<Real>> model,
               py::function solver_fn,
               traits::QuadratureMethod integrator = traits::QuadratureMethod::TanhSinh,
               unsigned int num_paths = 10) 
            {
                using SolverFunc = std::function<Matrix(Real, Real, int, int, const std::optional<Matrix>&)>;

                SolverFunc solver_lambda = [solver_fn](Real t0, Real ttm, int n_steps, int n_paths,
                                                      const std::optional<Matrix>& dW_opt) -> Matrix {
                    py::gil_scoped_acquire gil;
                    py::object arg5 = dW_opt ? py::cast(*dW_opt) : py::none();
                    py::object ret  = solver_fn(t0, ttm, n_steps, n_paths, arg5);
                    return ret.cast<Matrix>();
                };

                return std::make_shared<OPEType>(
                    ttm, rate,
                    std::move(payoff), model,
                    solver_lambda, integrator, num_paths
                );
            }),
            py::arg("ttm"), py::arg("rate"),
            py::arg("payoff"), py::arg("model"), py::arg("solver_fn"),
            py::arg("integrator") = traits::QuadratureMethod::TanhSinh,
            py::arg("num_paths") = 10)
        .def("price", &OPEType::price);
}








PYBIND11_MODULE(sdefin, m) {
    m.doc() = "Stochastic vol SDE models, solvers, and option pricers (pybind11)";

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
    py::class_<options::IPayoff<Real>, py::smart_holder>(m, "IPayoff");

    py::class_<options::EuropeanCallPayoff<Real>, options::IPayoff<Real>, py::smart_holder>(m, "EuropeanCallPayoff")      
    .def(py::init<Real>(), py::arg("K"))
    .def("evaluate",py::overload_cast<Real>(&options::EuropeanCallPayoff<Real>::evaluate, py::const_),
    py::arg("S_T"))
    .def("evaluate_from_log", py::overload_cast<Real>(&options::EuropeanCallPayoff<Real>::evaluate_from_log, py::const_),
    py::arg("log_S_T"))
    .def("get_strike", &options::EuropeanCallPayoff<Real>::getStrike)
    .def("set_strike", &options::EuropeanCallPayoff<Real>::setStrike, py::arg("K"));
      
    py::class_<options::EuropeanPutPayoff<Real>, options::IPayoff<Real>, py::smart_holder>(m, "EuropeanPutPayoff")      
    .def(py::init<Real>(), py::arg("K"))
    .def("evaluate",py::overload_cast<Real>(&options::EuropeanPutPayoff<Real>::evaluate, py::const_),
    py::arg("S_T"))
    .def("evaluate_from_log", py::overload_cast<Real>(&options::EuropeanPutPayoff<Real>::evaluate_from_log, py::const_),
    py::arg("log_S_T"))
    .def("get_strike", &options::EuropeanPutPayoff<Real>::getStrike)
    .def("set_strike", &options::EuropeanPutPayoff<Real>::setStrike, py::arg("K"));

    // ----- Pricers -----

    // Single generic MCOptionPricer binding
    py::class_<options::MCOptionPricer<Real>, std::shared_ptr<options::MCOptionPricer<Real>>>(m, "MCOptionPricer")
    .def(py::init(
    [](Real ttm, Real rate,
        std::unique_ptr<options::IPayoff<Real>> payoff,
        std::shared_ptr<SDE::ISDEModel<Real>> model,
        py::function solver_fn,
        int num_paths, int num_steps) {

        using SolverFunc = std::function<Matrix(Real, Real, int, int, const std::optional<Matrix>&)>;

        SolverFunc solver_lambda = [solver_fn](Real t0, Real ttm, int n_steps, int n_paths,
                                              const std::optional<Matrix>& dW_opt) -> Matrix {
            py::gil_scoped_acquire gil;
            py::object arg5 = dW_opt ? py::cast(*dW_opt) : py::none();
            py::object ret  = solver_fn(t0, ttm, n_steps, n_paths, arg5);
            return ret.cast<Matrix>();
        };

        return std::make_shared<options::MCOptionPricer<Real>>(
            ttm, rate,
            std::move(payoff), model,
            solver_lambda, num_paths, num_steps
        );
    }),
    py::arg("ttm"), py::arg("rate"),
    py::arg("payoff"), py::arg("model"), py::arg("solver_fn"),
    py::arg("num_paths") = 10000, py::arg("num_steps") = 100)
    .def("price", &options::MCOptionPricer<Real>::price);

    // FFT Option Pricer Binding
    py::class_<options::FFTOptionPricer<Real>, std::shared_ptr<options::FFTOptionPricer<Real>>>(m, "FFTOptionPricer")
    .def(py::init([](Real ttm, Real rate,
                     std::unique_ptr<options::IPayoff<Real>> payoff,
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


    py::enum_<traits::QuadratureMethod>(m, "QuadratureMethod")
    .value("TanhSinh", traits::QuadratureMethod::TanhSinh)
    .value("QAGI", traits::QuadratureMethod::QAGI) // add all your methods here
    .export_values();

 
    bind_OPEOptionPricer<3>(m, "OPEOptionPricerN3"); 
    bind_OPEOptionPricer<5>(m, "OPEOptionPricerN5"); 
    bind_OPEOptionPricer<7>(m, "OPEOptionPricerN7"); 
    bind_OPEOptionPricer<9>(m, "OPEOptionPricerN9"); 
    bind_OPEOptionPricer<10>(m, "OPEOptionPricerN10"); 

}

