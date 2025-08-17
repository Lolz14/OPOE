/**
 * @file FFTW.hpp
 * @brief Utilities for performing FFT-based polynomial multiplication using FFTW and Eigen.
 *
 * This header provides functions to compute the forward and inverse Fast Fourier Transform (FFT)
 * using the FFTW library, as well as utilities for pointwise multiplication in the frequency domain
 * and polynomial multiplication via FFT.
 *
 * Dependencies:
 * - FFTW3 library for FFT computations.
 * - Eigen library for array and matrix operations.
 * - OPOE_traits.hpp for type definitions.
 *
 * Namespace: Utils
 *
 * Types:
 * - Complex: Alias for std::complex<double>.
 * - Array: Alias for Eigen array of complex numbers as defined in traits::DataType::ComplexArray.
 *
 * Functions:
 * - forwardFFT: Computes the forward FFT of a given array, with zero-padding to the next power of two.
 * - inverseFFT: Computes the inverse FFT of a given frequency-domain array, with normalization.
 * - pointwiseMultiply: Performs element-wise multiplication of two arrays in the frequency domain.
 * - fftMultiply: Template function to multiply two polynomials using FFT-based convolution.
 */
#ifndef FFTW_HPP
#define FFTW_HPP

#include <fftw3.h>
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include "../traits/OPOE_traits.hpp"

namespace Utils {
template <typename R = traits::DataType::PolynomialField>
using Complex = std::complex<R>;

using Array = traits::DataType::ComplexArray;

/**
 * @brief Computes the forward FFT.
 * @param data Input time-domain data.
 * @return Frequency-domain data.
 */
template <typename R = traits::DataType::PolynomialField>
inline Array forwardFFT(const Array& data) {
    int n = data.size();
    int paddedSize = 1;
    while (paddedSize < n) {
        paddedSize <<= 1;
    }

    Array paddedData = Array::Zero(paddedSize);
    paddedData.head(n) = data;

    Array freqData(paddedSize);

    fftw_plan plan = fftw_plan_dft_1d(n,
        reinterpret_cast<fftw_complex*>(const_cast<Complex<R>*>(paddedData.data())),
        reinterpret_cast<fftw_complex*>(freqData.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    return freqData;
}
/**
 * @brief Computes the inverse FFT.
 * @param freqData Input frequency-domain data.
 * @return Time-domain data.
 */
template <typename R = traits::DataType::PolynomialField>
inline Array inverseFFT(const Array& freqData) {
    int n = freqData.size();
    int paddedSize = 1;
    while (paddedSize < n) {
        paddedSize <<= 1;
    }
    Array paddedData = Array::Zero(paddedSize);
    paddedData.head(n) = freqData;

    Array timeData(paddedSize);

    fftw_plan plan = fftw_plan_dft_1d(n,
        reinterpret_cast<fftw_complex*>(const_cast<Complex<R>*>(paddedData.data())),
        reinterpret_cast<fftw_complex*>(timeData.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan);
    fftw_destroy_plan(plan);



    return timeData / static_cast<double>(paddedSize);
}

/**
 * @brief Performs pointwise multiplication in the frequency domain.
 * @param A First transformed polynomial.
 * @param B Second transformed polynomial.
 * @return Pointwise multiplied result.
 */
inline Array pointwiseMultiply(const Array& A, const Array& B) {
    return A.cwiseProduct(B);
}

/**
 * @brief Performs FFT-based polynomial multiplication.
 * @tparam T Floating-point type.
 * @param lhs First polynomial coefficients.
 * @param rhs Second polynomial coefficients.
 * @return Resulting polynomial coefficients.
 */
template <typename T = traits::DataType::PolynomialField>
inline Eigen::Array<T, Eigen::Dynamic, 1> fftMultiply(const Eigen::Array<T, Eigen::Dynamic, 1>& lhs,
                                                      const Eigen::Array<T, Eigen::Dynamic, 1>& rhs) {
    int sizeA = lhs.size();
    int sizeB = rhs.size();
    int n = 1;

    while (n < sizeA + sizeB - 1) {
        n <<= 1;
    }


    // Allocate and copy input
    Array fa = Array::Zero(n);
    Array fb = Array::Zero(n);
    fa.head(sizeA) = lhs.template cast<Complex>();
    fb.head(sizeB) = rhs.template cast<Complex>();
    
    // Perform FFT
    Array freqA = forwardFFT(fa);
    Array freqB = forwardFFT(fb);

    // Multiply in frequency domain
    Array freqResult = pointwiseMultiply(freqA, freqB);

    // Inverse FFT to get result
    Array timeResult = inverseFFT(freqResult);

    // Extract real parts
    return timeResult.head(sizeA + sizeB - 1).real().template cast<T>();
}

} // namespace Utils

#endif // FFTW_HPP

