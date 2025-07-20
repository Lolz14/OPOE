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


    template <typename R = traits::DataType::PolynomialField>
    class FFTPricer : public BaseOptionPricer<R> {
        using Base = BaseOptionPricer<R>;
    
        public:
            FFTPricer(R ttm, R strike, R rate, std::unique_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model, unsigned int Npow = 20, unsigned int A = 1200)
            : Base(ttm, strike, rate, std::move(payoff), std::move(sde_model)), Npow_(Npow), A_(A) {
                initialize_fft_grid();
            }

           R price() const override {
                update_characteristic_fn();
                return interpolate_price(this->strike_);
            }

            using Getter = std::function<R()>;
            using Setter = std::function<void(R)>;

            R delta() const  {
                return bump_and_diff(
                    [this]() { return this->sde_model_->get_x0(); },
                    [this](R v) { this->sde_model_->set_x0(v); },
                    /* bump as log-spot */ true
                );
            }

            R gamma() const  {
                return bump_and_diff(
                    [this]() { return this->sde_model_->get_x0(); },
                    [this](R v) { this->sde_model_->set_x0(v); },
                    true, true
                );
            }

            R vega() const  {
                return bump_and_diff(
                    [this]() { return this->sde_model_->get_v0(); },
                    [this](R v) { this->sde_model_->set_v0(v); }
                );
            }

            R rho() const  {
                return bump_and_diff(
                    [this]() { return this->get_rate(); },
                    [this](R v)  { 
                        const_cast<FFTPricer*>(this)->set_rate(v); 
                    }
                );
            }

            R theta() const {
                return -bump_and_diff(
                    [this]() { return this->get_ttm(); },
                    [this](R v) {
                        const_cast<FFTPricer*>(this)->set_ttm(v);
                    }
                );
}


        private:
            void initialize_fft_grid() {
                N_ = std::pow(2, Npow_);
                eta_ = A_ / static_cast<R>(N_);
                lambda_ = 2 * M_PI / (N_ * eta_);

                v_ = Array::LinSpaced(N_, std::numeric_limits<R>::epsilon(), A_ * (N_ - 1) / N_ + eta_).template cast<std::complex<R>>();
                k_ = (-lambda_ * N_ / 2) + lambda_ * Array::LinSpaced(N_, 0, N_ - 1);
                w_ = Array::Ones(N_);   
                w_(0) = w_(N_ - 1) = R(0.5);
            }

            // --- Run characteristic function ---
            void update_characteristic_fn() const {
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

            R interpolate_price(R K) const {
                R result = spline_(K)(0);
                if (this->payoff_->type() == traits::OptionType::Put)
                    return result - this->sde_model_->get_x0() + K * std::exp(-this->rate_ * this->ttm_);
                return result;
            }

            // --- Greek computation via bumping ---
            R bump_and_diff(Getter getter, Setter setter, bool log_spot = false, bool second = false) const
            requires GetterFunction<Getter, R> && SetterFunction<Setter, R>
            {
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

            static constexpr R bump_size_ = 1e-4;

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