#ifndef FFT_OPTION_PRICER_HPP
#define FFT_OPTION_PRICER_HPP

#include <cmath>
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
            FFTPricer(R ttm, R strike, R rate, R initial_price, std::unique_ptr<IPayoff<R>> payoff, std::shared_ptr<SDE::ISDEModel<R>> sde_model, unsigned int Npow = 20, unsigned int A = 1200)
            : Base(ttm, strike, rate, initial_price, std::move(payoff), std::move(sde_model)), Npow_(Npow), A_(A) {}

            R price() const override {
            // FFT grid size
            const unsigned int N = static_cast<unsigned int>(1u << this->Npow_);  // 2^Npow_
            const R eta = this->A_ / N;

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
                Array::Ones(1) * this->x0_, 
                v - SDE::ImaginaryUnit<R>,  // Make sure ImaginaryUnit<R>() returns std::complex<R>(0,1)
                res
            );

            // Compute Z_k per Carr-Madan or similar formula
            auto exp_term = (SDE::ImaginaryUnit<R> * this->rate_ * v * this->ttm_).exp();
            auto numerator = res.array() - std::complex<R>(1.0, 0.0);
            auto denominator = SDE::ImaginaryUnit<R> * v * (v * SDE::ImaginaryUnit<R> + std::complex<R>(1.0, 0.0));
            auto Z_k = exp_term * numerator.cwiseQuotient(denominator);

            // Weights for trapezoidal rule
            Array w = Array::Ones(N);
            w(0) = 0.5;
            w(N - 1) = 0.5;

            // Compose vector for FFT input
            ComplexArray linspaced = Array::LinSpaced(N, 0, N - 1).template cast<std::complex<R>>();
            ComplexArray x = w.template cast<std::complex<R>>() * std::complex<R>(eta, 0) * (SDE::ImaginaryUnit<R> * M_PI) * linspaced;

            // Run FFT and get real part
            auto fft_result = Utils::forwardFFT(x / M_PI);
            auto z_k = fft_result.real();

            // Print debug info (optional)
            std::cout << "FFT input vector (x): " << x.transpose() << std::endl;
            std::cout << "FFT output real part (z_k): " << z_k.transpose() << std::endl;

            // TODO: Continue your pricing logic with z_k, k_real, Z_k, etc.
            // Return a dummy value to compile:
            return static_cast<R>(0);
        }




        

        private:
            unsigned int Npow_;
            unsigned int A_;




    
    };

} // namespace options


#endif // FFT_OPTION_PRICER_HPP