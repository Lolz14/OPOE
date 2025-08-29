/*!
 * @file Polynomials.hpp
 * @brief Defines the polynomials::Polynomial template class and related polynomial operations.
 *
 * This header provides a generic, fixed-degree polynomial class template with support for
 * various arithmetic operations, evaluation methods, and utilities. The implementation
 * leverages Eigen for efficient array operations and supports both Horner's
 * and direct evaluation methods. Additional features include conversion to std::function,
 * pretty-printing, and compile-time degree information.
 *
 * 
 * Dependencies:
 * - Eigen for array operations.
 * - C++17 or later (C++20 for comparison operators).
 * - Custom traits and utility headers.
 * 
 * @details
 * Main features:
 * - Template class `Polynomial<N, R>` for polynomials of degree N over field R.
 * - Supports construction from coefficient arrays, copy/move semantics, and assignment from
 *   polynomials of lower degree.
 * - Evaluation via operator() with selectable method (Horner/direct).
 * - Conversion to std::function for functional programming use cases.
 * - Arithmetic operations: addition, subtraction, multiplication (including FFT-based),
 *   and scalar operations.
 * - Derivative computation via the `der<M>(p)` template function.
 * - Pretty-printing via operator<<.
 * - C++20 three-way comparison operator (if available).
 *
 *
 * Usage example:
 * @code
 * using Poly = polynomials::Polynomial<3, double>;
 * Poly p({1.0, 2.0, 3.0, 4.0});
 * double val = p(2.0); // Evaluate at x=2
 * auto dp = polynomials::der<1>(p); // First derivative
 * std::cout << p << std::endl;
 * @endcode
 */
#ifndef HH_POLYNOMIALS_HH
#define HH_POLYNOMIALS_HH

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <type_traits>
#include <utility>
#include <ranges>
#include <cmath>
#include <functional>
#include "../traits/OPOE_traits.hpp"
#include "../utils/Utils.hpp"
#include "../utils/FFTW.hpp"

#if __cplusplus >= 202002L
#include <compare> // for c++20 style comparison operators
#endif

namespace polynomials {
/*!
* @class Polynomial
* @brief Template class for polynomials
* @tparam N Polynomial degree
* @tparam R Polynomial field
*/

template <unsigned int N, class R = traits::DataType::PolynomialField>
class Polynomial
{
public:
   
    Polynomial() { M_coeff.setZero(); }

    //! Constructor taking coefficients
    Polynomial(const traits::DataType::StoringArray &c) : M_coeff{c} {}

    //! Constructor with evaluation method
    Polynomial(const traits::DataType::StoringArray &c, traits::EvalMethod method) : M_coeff{c}, eval_method{method} {}

    //! I can initialize with another polynomial, but only if Degree<=
    template <unsigned int M>
    Polynomial(Polynomial<M, R> const &right) noexcept
    {
        static_assert(M <= N, "Cannot assign a polynomial of higher degree");
        M_coeff.setZero();
        M_coeff.head(M + 1) = right.get_coeff().head(M + 1);      
    }

    //! For polynomial of the same type I use the implicit copy-constructor
    Polynomial(Polynomial<N, R> const &) = default;

    //! Move constructor is the implicit one
    Polynomial(Polynomial<N, R> &&) = default;

    //! I can also assign a polynomial of smaller degree
    template <unsigned int M>
    Polynomial &operator=(Polynomial<M, R> const &right) noexcept
    {
        static_assert(M <= N, "Cannot assign a polynomial of higher degree");
        M_coeff.setZero();
        M_coeff.head(M + 1) = right.get_coeff().head(M + 1);
        return *this;
    }

    //! Copy assignment is the synthesized one
    Polynomial &operator=(Polynomial<N, R> const &) = default;

    //! Move assignment is the synthesized one
    Polynomial &operator=(Polynomial<N, R> &&) = default;

    /*!
     * Relational operators among polynomials of same type and order (C++20)
     *
     * It relies on the fact that <=> operator is defined for std::array<R,M>
     */
#if __cplusplus >= 202002L
    friend auto operator<=>(Polynomial<N, R> const &, Polynomial<N, R> const &) = default;
#endif

    //! Set coefficients
    void set_coeff(const traits::DataType::StoringArray &c)
    {
        M_coeff = c;
    }

    //! Get coefficients
    auto get_coeff() const noexcept
    {
        return M_coeff;
    }

    //! Get coefficient as reference (a nicer alternative to setter).
    auto &get_coeff() noexcept
    {
        return M_coeff;
    }

    //! Evaluate polynomial with selected method
    //! @param x The evaluation point
    auto constexpr operator()(R const &x) const noexcept
    {
        return std::visit([this, &x](auto &&method) { return evaluate(x, method); }, eval_method);
    }

    /*!
     * @brief Returns a std::function representing the polynomial evaluation.
     *
     * The returned function captures a *copy* of the polynomial's state
     * (coefficients and evaluation method) at the time this method is called.
     * It remains valid even if the original Polynomial object is modified or destroyed.
     *
     * @return std::function<R(R)> A function object that takes an argument of type R
     *         and returns the polynomial evaluated at that argument.
     */
    std::function<R(R)> as_function() const {
        // Capture a copy of the current polynomial object state by value.
        // This ensures the lambda is self-contained.
        auto captured_poly_copy = *this;

        auto evaluator_lambda =
            // Move the captured copy into the lambda's state
            [poly_copy = std::move(captured_poly_copy)](R x) -> R {
            // Call the operator() of the captured copy.
            // This automatically uses the evaluation method stored in the copy.
            return poly_copy(x);
        };
        // The lambda is implicitly converted to std::function
        return evaluator_lambda;
    }
    // --- END NEW METHOD ---

    //! Unary minus
    auto operator-() noexcept
    {
        M_coeff = -M_coeff;
        return *this;
    }
    
    //! Unary minus (returns a new polynomial with inverted sign)
    auto operator-() const noexcept
    {
        Polynomial<N, R> result = *this;
        result.get_coeff() = -result.get_coeff();
        return result;
    }
    

    //! Unary plus
    auto operator+() noexcept
    {
        return *this;
    }

    //! The polynomial degree
    /*!
     *  implemented as a constexpr function since the degree is
     *  here a constant expression. It must be static since
     *  constexpr methods should be static methods.
     */
    static constexpr unsigned int degree()
    {
        return N;
    }

private:
    //! Coefficients a_0---a_n
    traits::DataType::StoringArray M_coeff;

    //! Evaluation method
    std::variant<traits::EvalMethod> eval_method = traits::EvalMethod::Horner;

    /*!
    * @brief Evaluates the polynomial at a given point x using the specified method.
    * @param x The point at which to evaluate the polynomial.
    * @param method The evaluation method to use (Horner or Direct).
    * @return The evaluated polynomial value at x.
    * @note This method uses Horner's method for efficient evaluation if the method is set to Horner.
    *       If the method is set to Direct, it computes powers of x and sums the products with coefficients.
    */
    auto constexpr evaluate(R const &x, traits::EvalMethod method) const noexcept
    {
        if (method == traits::EvalMethod::Horner){

        return M_coeff.reverse().redux([&](R a, R b) { return a * x + b; });

        }
        else // Direct evaluation
        {
            traits::DataType::StoringArray x_powers(N + 1);
            x_powers.setOnes();  // Set all elements to 1


            for (unsigned int i = 1; i <= N; ++i) {
                x_powers[i] = x_powers[i - 1] * x; 
            }
            
        return (M_coeff*x_powers).sum();
        }
    };
    
  

};

/*!
 * Outputs the polynomial in a pretty-print way
 * @tparam N The degree
 * @tparam R The field
 * @param out The output stream
 * @param p The polynomial
 * @return The stream
 */
template <unsigned int N, typename R>
std::ostream &operator<<(std::ostream &out, Polynomial<N, R> const &p)
{
    const auto &coeffs = p.get_coeff();  // Avoid multiple function calls
    out << coeffs[0];  // Print the constant term

    for (unsigned int i = 1; i <= N; ++i)
    {
        if (coeffs[i] != 0)  // Avoid printing zero terms
        {
            out << (coeffs[i] > 0 ? " + " : " - ") << std::abs(coeffs[i]) << "x^" << i;
        }
    }

    return out;
}



/*!
* Polynomial subtraction
 * @tparam LDegree The degree of the left polynomial
 * @tparam RDegree The degree of the right polynomial
 * @tparam R The scalar field
 * @param left The left polynomial
 * @param right The right polynomial
 * @return The subtraction of the two polynomials
*/
template <unsigned int LDegree, unsigned int RDegree, typename R>
auto
operator-(Polynomial<LDegree, R> const &left,
          Polynomial<RDegree, R> const &right) noexcept
{
  constexpr unsigned int NMAX = (LDegree > RDegree) ? LDegree : RDegree;
  Polynomial<NMAX, R> res;
  
  if constexpr (LDegree > RDegree) {
    res.get_coeff() = left.get_coeff();
    res.get_coeff() -= right.get_coeff();
  } else {
    res.get_coeff() = -right.get_coeff();
    res.get_coeff() += left.get_coeff();
  }
  
  return res;
}

/*!
* Polynomial subtraction with scalar (polynomial - scalar)
 * @tparam N The degree of the polynomial
 * @tparam R The scalar field
 * @param left The polynomial
 * @param right The scalar
 * @return The result of subtracting the scalar from the polynomial
*/
template <unsigned int N, typename R>
auto operator-(const Polynomial<N, R>& left, const R& right) noexcept
{
    Polynomial<N, R> res(left);  // Create copy of left polynomial
    res.get_coeff()[0] -= right; // Subtract scalar from constant term
    return res;
}

/*!
* Polynomial subtraction with scalar (scalar - polynomial)
 * @tparam N The degree of the polynomial
 * @tparam R The scalar field
 * @param left The scalar
 * @param right The polynomial
 * @return The result of subtracting the polynomial from the scalar
*/
template <unsigned int N, typename R>
auto operator-(const R& left, const Polynomial<N, R>& right) noexcept
{
    Polynomial<N, R> res(-right);  // Create negated copy of right polynomial
    res.get_coeff()[0] += left;    // Add scalar to constant term
    return res;
}

/*!
*Polynomial multiplication
 * @tparam LDegree The degree of the left polynomial
 * @tparam RDegree The degree of the right polynomial
 * @tparam R The scalar field
 * @param left The left polynomial
 * @param right The right polynomial
 * @return The product of the two polynomials
*/
template <unsigned int LDegree, unsigned int RDegree, typename R>
auto operator*(Polynomial<LDegree, R> const &left, Polynomial<RDegree, R> const &right) noexcept
{
    constexpr unsigned int NRES = LDegree + RDegree;
    Polynomial<NRES, R> res;

    // Perform FFT-based multiplication
    res.get_coeff() = Utils::fftMultiply(left.get_coeff(), right.get_coeff());

    return res;
}


/*!
* Multiplication of a polynomial with a scalar
 * @tparam RDegree The degree of the polynomial
 * @tparam R The scalar field
 * @param scalar The scalar
 * @param right The polynomial
 * @return The polynomial multiplied by the scalar
*/
template <unsigned int RDegree, typename R>
auto operator*(R const &scalar, Polynomial<RDegree, R> const &right) noexcept
{
    Polynomial<RDegree, R> res;
    res.get_coeff() = scalar * right.get_coeff();
    return res;
}

/*!
* Multiplication of a polynomial with a scalar
 * This is the same as the previous one, but with the arguments swapped.
 * It is here to allow for a more natural syntax.
 * @tparam RDegree The degree of the polynomial
 * @tparam R The scalar field
 * @param left The polynomial
 * @param scalar The scalar
 * @return The polynomial multiplied by the scalar
*/
template <unsigned int RDegree, typename R>
auto operator*( Polynomial<RDegree, R> const &left, R const &scalar) noexcept
{
    Polynomial<RDegree, R> res;
    res.get_coeff() = scalar * left.get_coeff();
    return res;
}





/*!
 * Derivative of a Polynomial
 * Usage: der<M>(p) (M>=0)
 *
 * @tparam M The derivative order
 * @tparam RDegree The degree of the polynomial
 * @tparam R The scalar field
 * @param p The polynomial
 * @return \f$\frac{d^{M}(p)}{dx^{M}}\f$
 */
template <unsigned M, unsigned RDegree, typename R>
auto
der(Polynomial<RDegree, R> const &p)
{
  if constexpr(M == 0u)
    return p;
  else if constexpr(RDegree < M)
    return Polynomial<0u, R>{{R(0)}};
  else
{
    // Create a vector containing [1, 2, 3, ..., RDegree]
    traits::DataType::StoringArray multipliers = traits::DataType::StoringArray::LinSpaced(RDegree, 1, RDegree);

    // Perform element-wise multiplication with the polynomial coefficients (ignoring the constant term)
    traits::DataType::StoringArray C = multipliers * p.get_coeff().segment(1, RDegree);

      return der<M - 1>(Polynomial<RDegree - 1, R>{C});
    }
};

}; // namespace polynomials


#endif // HH_POLYNOMIALS_HH