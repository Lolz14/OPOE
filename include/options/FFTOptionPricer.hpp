/**
 * @file FFTOptionPricer.hpp
 * @brief Defines the FFTOptionPricer class for pricing options using the Fast Fourier Transform (FFT) method.
 *
 * This header provides the FFTOptionPricer class template, which implements option pricing via FFT,
 * leveraging the characteristic function of the underlying asset's stochastic process. The class supports
 * flexible polynomial bases, configurable quadrature rules, and spline interpolation for efficient and accurate
 * computation of option prices and Greeks (delta, gamma, vega, rho, theta).
 *
 * Key Features:
 * - FFT-based option pricing using characteristic functions.
 * - Support for arbitrary payoff functions and SDE models.
 * - Efficient computation of option Greeks via bump-and-difference methods.
 * - Spline interpolation for smooth price evaluation across strikes.
 * - Configurable FFT grid parameters for accuracy and performance tuning.
 *
 * Template Parameters:
 * - R: Floating-point type used for calculations (default: traits::DataType::PolynomialField).
 *
 * Dependencies:
 * - Eigen (for arrays and splines)
 * - FFTW (for FFT computations)
 * - BaseOptionPricer (base class for option pricers)
 * - SDE model and payoff interfaces
 *
 * Usage:
 * Instantiate FFTOptionPricer with the desired types and parameters, then use the price() and Greeks methods
 * to evaluate option prices and sensitivities.
 */
#ifndef FFT_OPTION_PRICER_HPP
#define FFT_OPTION_PRICER_HPP

#include <cmath>
#include <unsupported/Eigen/Splines>
#include "BaseOptionPricer.hpp"
#include "../utils/FFTW.hpp"

namespace options
{
using Array = traits::DataType::StoringArray;
using ComplexArray = traits::DataType::ComplexArray;
using ComplexStoringVector = traits::DataType::ComplexStoringVector;

template <typename T, typename R>
concept GetterFunction = requires(T obj) {
    { obj() } -> std::convertible_to<R>;
};

template <typename T, typename R>
concept SetterFunction = requires(T obj, R value) {
    { obj(value) };
};

/**
 * @brief FFTOptionPricer class for pricing options using Fast Fourier Transform (FFT).
 *
 * This class implements the FFT method for option pricing, leveraging the characteristic function of the underlying asset.
 * It supports flexible polynomial bases and integrates the expected payoff using configurable quadrature rules.
 * @tparam R The floating-point type used for calculations (default: traits::DataType::PolynomialField).
 */
template <typename R = traits::DataType::PolynomialField>
class FFTOptionPricer : public BaseOptionPricer<R> {
    using Base = BaseOptionPricer<R>;

    public:
        /**
         * @brief Constructs an FFTOptionPricer instance.
         * @param ttm Time to maturity.
         * @param rate Risk-free interest rate.
         * @param payoff Payoff function for the option.
         * @param sde_model Shared pointer to the SDE model used for pricing.
         * @param Npow Power of 2 for the FFT grid size (default: 20).
         * @param A Scaling factor for the FFT grid (default: 1200).
         */
        FFTOptionPricer(R ttm, R rate, std::shared_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model, unsigned int Npow = 10, unsigned int A = 10)
        : Base(ttm, rate, std::move(payoff), std::move(sde_model)), Npow_(Npow), A_(A) {
            initialize_fft_grid();
            recompute();
        }
        
        /**
         * @brief Price the option using FFT.
         * This method computes the option price by evaluating the characteristic function of the underlying asset
         * and applying the FFT to obtain the expected payoff. If the sde model was subject to changes, it calls the recompute function.
         * @return The computed option price.
         */
        R price() const override {
            this->ensure_recomputed();
            return interpolate_price(this->payoff_->getStrike());
        }

        /**
         * @brief Computes the delta of the option using a bump-and-difference method.
         * This method perturbs the initial state of the SDE model and calculates the change in option price.
         * @return The computed delta value.
         */
        R delta() const  {
            return bump_and_diff(
                [this]() { return this->sde_model_->get_x0(); },
                [this](R v) { this->sde_model_->set_x0(v); },
                /* bump as log-spot */ true
            );
        }
        
        /**
         * @brief Computes the gamma of the option using a bump-and-difference method.
         * This method perturbs the initial state of the SDE model twice and calculates the second derivative of the option price.
         * @return The computed gamma value.
         */
        R gamma() const  {
            return bump_and_diff(
                [this]() { return this->sde_model_->get_x0(); },
                [this](R v) { this->sde_model_->set_x0(v); },
                true, true
            );
        }

        /**
         * @brief Computes the vega of the option using a bump-and-difference method.
         * This method perturbs the volatility parameter of the SDE model and calculates the change in option price.
         * @return The computed vega value.
         */
        R vega() const  {
            return bump_and_diff(
                [this]() { return this->sde_model_->get_v0(); },
                [this](R v) { this->sde_model_->set_v0(v); }
            );
        }

        /**
         * @brief Computes the rho of the option using a bump-and-difference method.
         * This method perturbs the risk-free interest rate and calculates the change in option price.
         * @return The computed rho value.
         */
        R rho() const  {
            return bump_and_diff(
                [this]() { return this->get_rate(); },
                [this](R v)  { 
                    const_cast<FFTOptionPricer*>(this)->set_rate(v); 
                }
            );
        }

        /**
         * @brief Computes the theta of the option using a bump-and-difference method.
         * This method perturbs the time to maturity and calculates the change in option price.
         * @return The computed theta value.
         */
        R theta() const {
            return -bump_and_diff(
                [this]() { return this->get_ttm(); },
                [this](R v) {
                    const_cast<FFTOptionPricer*>(this)->set_ttm(v);
                }
            );
}


    private:
        /**
         * @brief Initializes the FFT grid parameters.
         * This method sets up the grid size, eta, and lambda based on the provided Npow and A parameters.
         * It also initializes the complex vectors v_ and k_ used in the FFT computation.
         * 
         */
        void initialize_fft_grid() {
            N_ = std::pow(2, Npow_);
            eta_ = A_ / static_cast<R>(N_);
            lambda_ = 2 * M_PI / (N_ * eta_);

            v_ = Array::LinSpaced(N_, std::numeric_limits<R>::epsilon(), A_ * (N_ - 1) / N_ + eta_).template cast<std::complex<R>>();
            k_ = (-lambda_ * N_ / 2) + lambda_ * Array::LinSpaced(N_, 0, N_ - 1);
            w_ = Array::Ones(N_);   
            w_(0) = w_(N_ - 1) = R(0.5);
        }

        /**
         * @brief Updates the characteristic function and computes the spline for option pricing.
         * This method evaluates the characteristic function of the underlying asset using the SDE model,
         * computes the necessary terms for the FFT, and builds a spline for interpolating option prices.
         * It is called before pricing the option to ensure the characteristic function is up-to-date.
         */
        void recompute() const override {
            ComplexStoringVector res(N_);
            this->sde_model_->characteristic_fn(this->ttm_, v_ - SDE::ImaginaryUnit<R>, res);

            auto exp_term = (SDE::ImaginaryUnit<R> * this->rate_ * v_ * this->ttm_).exp();
            ComplexArray numerator = res.array() - std::complex<R>(1.0, 0.0);
            ComplexArray denominator = SDE::ImaginaryUnit<R> * v_ * (v_ * SDE::ImaginaryUnit<R> + std::complex<R>(1.0, 0.0));
            ComplexArray Z_k = exp_term.array() * numerator.array() / denominator.array();



            ComplexArray lin = Array::LinSpaced(N_, 0, N_ - 1).template cast<std::complex<R>>();
            ComplexArray x = w_.template cast<std::complex<R>>() * std::complex<R>(eta_, 0) *
                            (SDE::ImaginaryUnit<R> * M_PI * lin).array().exp() * Z_k.array();

            auto fft_result = Utils::forwardFFT(x / M_PI);
            Array z_k = fft_result.real();


            R S0 = std::exp(this->sde_model_->get_x0());
            Array C = S0 * (z_k + (1.0 - (k_.array() - this->rate_ * this->ttm_).exp()).cwiseMax(0.0));
            strikes_ = S0 * k_.array().exp();


            // Build spline once per price computation
            spline_ = Eigen::SplineFitting<Eigen::Spline<R, 1>>::Interpolate(C.transpose(), 3, strikes_.transpose());
        }

        /**
         * @brief Interpolates the option price based on the computed spline and strike price.
         * This method evaluates the spline at the given strike price and adjusts the result based on the option type.
         * @param K The strike price at which to interpolate the option price.
         * @return The interpolated option price.
         */
        R interpolate_price(R K) const {
            R result = spline_(K)(0)* std::exp(-this->rate_ * this->ttm_);
            if (this->payoff_->type() == traits::OptionType::Put)
                return result - std::exp(this->sde_model_->get_x0()) + K * std::exp(-this->rate_ * this->ttm_);
            return result;
        }

        
        using Getter = std::function<R()>;
        using Setter = std::function<void(R)>;
        /**
         * @brief Bump-and-difference method for computing derivatives.
         * This method perturbs a value using a getter and setter function, computes the option price,
         * and returns the derivative based on the specified bump size.
         * @param getter Function to retrieve the current value.
         * @param setter Function to set the perturbed value.
         * @param log_spot Whether to apply the bump as a log-spot perturbation (default: false).
         * @param second Whether to compute the second derivative (default: false).
         * @return The computed derivative value.
         */
        R bump_and_diff(Getter getter, Setter setter, bool log_spot = false, bool second = false) const
        requires GetterFunction<Getter, R> && SetterFunction<Setter, R>
        {            
            this->ensure_recomputed();
            R original = getter();
            R bump = log_spot ? bump_size_ / std::exp(original) : bump_size_;

            setter(original + bump);
            R plus = this->price();

            if (second) {
                setter(original - bump);
                R minus = this->price();
                setter(original); // restore
                R base = this->price();
                return (plus - 2 * base + minus) / (bump_size_ * bump_size_);
            } else {
                setter(original - bump);
                R minus = this->price();
                setter(original); // restore
                return (plus - minus) / (2 * bump_size_);
            }
        }

        static constexpr R bump_size_ = 1e-6;

        // FFT parameters
        unsigned int Npow_, A_;
        mutable unsigned int N_;
        mutable R eta_, lambda_;
        mutable Array k_, w_, strikes_;
        mutable ComplexArray v_;
        mutable Eigen::Spline<R, 1> spline_;


};

} // namespace options


#endif // FFT_OPTION_PRICER_HPP