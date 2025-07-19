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


    template <typename R = traits::DataType::PolynomialField>
    class FFTPricer : public BaseOptionPricer<R> {
        using Base = BaseOptionPricer<R>;
    
        public:
            FFTPricer(R ttm, R strike, R rate, std::unique_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model, unsigned int Npow = 20, unsigned int A = 1200)
            : Base(ttm, strike, rate, std::move(payoff), std::move(sde_model)), Npow_(Npow), A_(A) {}

            R price() const override {
            // FFT grid size
            const unsigned int N = static_cast<unsigned int>(std::pow(2, this->Npow_));  // 2^Npow_
            const R eta =  static_cast<R>(this->A_) / static_cast<R>(N);

            // Create real linspace, then cast to complex
            Array v_real = 
                Array::LinSpaced(N, std::numeric_limits<R>::epsilon(), this->A_ * (N - 1) / N + eta);

            ComplexArray v = v_real.template cast<std::complex<R>>();

            // Lambda grid
            const R lambd = M_PI * 2 / (N * eta);
            Array k_real = 
                (-lambd * N / 2) + lambd * Array::LinSpaced(N, 0, N - 1);
            
            // Prepare container for characteristic function results
            ComplexStoringVector res(N);


            // Evaluate characteristic function at v - i (imag unit)
            this->sde_model_->characteristic_fn(
                this->ttm_, 
                v - SDE::ImaginaryUnit<R>,  // Make sure ImaginaryUnit<R>() returns std::complex<R>(0,1)
                res
            );

            auto initial_value = this->sde_model_->m_x0.size() == 1 ? std::exp(this->sde_model_->m_x0(0)) : std::exp(this->sde_model_->m_x0(1));
            
            // Compute Z_k per Carr-Madan or similar formula
            auto exp_term = (SDE::ImaginaryUnit<R> * this->rate_ * v * this->ttm_).exp();

            auto numerator = res.array() - std::complex<R>(R(1.0), R(0.0));
            auto denominator = SDE::ImaginaryUnit<R> * v * (v * SDE::ImaginaryUnit<R> + std::complex<R>(R(1.0), R(0.0)));
            auto Z_k = exp_term * numerator.cwiseQuotient(denominator);

            // Weights for trapezoidal rule
            Array w = Array::Ones(N);
            w(0) = R(0.5);
            w(N - 1) = R(0.5);

            // Compose vector for FFT input
            ComplexArray linspaced = Array::LinSpaced(N, 0, N - 1).template cast<std::complex<R>>();
            ComplexArray x = w.template cast<std::complex<R>>() * std::complex<R>(eta, 0) * (SDE::ImaginaryUnit<R> * M_PI * linspaced).array().exp() * Z_k.array();

            // Run FFT and get real part
            auto fft_result = Utils::forwardFFT(x / M_PI);
            auto z_k = fft_result.real();

            // Transform z_k to C
            Array C = initial_value * (z_k + (R(1.0) - (k_real - this->rate_ * this->ttm_).array().exp()).cwiseMax(R(0.0)));

            // Convert log-strikes to strikes
            Array K = initial_value * k_real.array().exp();

            // Prepare for spline interpolation
            Eigen::Spline<R, 1> spline = Eigen::SplineFitting<Eigen::Spline<R, 1>>::Interpolate(C.transpose(), 3, K.transpose());


            R result = spline(this->strike_)(0);


      
            switch (this->payoff_->type()) {
            case traits::OptionType::Call:
                return result;  // Direct from FFT/interpolation
            case traits::OptionType::Put:
                return result - initial_value + this->strike_ * std::exp(-this->rate_ * this->ttm_);
            default:
                throw std::runtime_error("Unsupported payoff type");
            }
          
        }
        // --- FFT Greeks: central differences as member functions ---

        R delta() const  {

            R p_plus = bumped_S0(+bump_size).price();
            R p_minus = bumped_S0(-bump_size).price();
            return (p_plus - p_minus) / (2 * bump_size);

        }

        R gamma() const  {

            R p_plus = bumped_S0(+bump_size).price();
            R p_0    = this->price();
            R p_minus= bumped_S0(-bump_size).price();
            return (p_plus - 2*p_0 + p_minus) / (bump_size * bump_size);

        }

        R vega() const  {

            R v_plus = bumped_vol(+bump_size).price();
            R v_minus= bumped_vol(-bump_size).price();
            return (v_plus - v_minus) / (2 * bump_size);

        }

        R rho() const  {

            R r_plus = bumped_r(+bump_size).price();
            R r_minus= bumped_r(-bump_size).price();
            return (r_plus - r_minus) / (2 * bump_size);

        }

        R theta() const  {

        R t_plus = bumped_T(+bump_size).price();
        R t_minus= bumped_T(-bump_size).price();

        // Negative sign ensures theta is "decay" per finance convention
        return -(t_plus - t_minus) / (2 * bump_size); 
        }
        


        

        private:

            FFTPricer bumped_S0(R bump) const { FFTPricer t(this->ttm_,
            this->strike_,
            this->rate_,
            this->payoff_->clone(),
            this->sde_model_->clone(),
            this->Npow_,
            this->A_);; t.sde_model_->bump_S0(std::log(1.0 + bump)); return t; }
            FFTPricer bumped_vol(R bump) const { FFTPricer t(this->ttm_,
            this->strike_,
            this->rate_,
            this->payoff_->clone(),
            this->sde_model_->clone(),
            this->Npow_,
            this->A_);; t.sde_model_->bump_volatility(bump); return t; }
            FFTPricer bumped_r(R bump) const { FFTPricer t(this->ttm_,
            this->strike_,
            this->rate_,
            this->payoff_->clone(),
            this->sde_model_->clone(),
            this->Npow_,
            this->A_);; t.rate_ += bump; return t; }
            FFTPricer bumped_T(R bump) const { FFTPricer t(this->ttm_,
            this->strike_,
            this->rate_,
            this->payoff_->clone(),
            this->sde_model_->clone(),
            this->Npow_,
            this->A_);; t.ttm_ += bump; return t; }

            static constexpr R bump_size = 1e-4;

            unsigned int Npow_;
            unsigned int A_;




    
    };

} // namespace options


#endif // FFT_OPTION_PRICER_HPP