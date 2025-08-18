
/**
 * @file Monomials.hpp
 * @brief Provides constexpr functions to compute monomials of a given degree.
 *
 * This header defines a set of template functions in the `polynomials` namespace
 * for evaluating monomials (i.e., powers of a variable) at compile time.
 * The functions use template metaprogramming and type traits to recursively
 * compute x^N for a given value x and non-negative integer N. 
 * 
* Dependencies:
 * - traits/OPOE_traits.hpp: For the default type of R.
 * 
 * Templates:
 * - R: The type of the variable and result (defaults to traits::DataType::PolynomialField).
 * - N: The degree of the monomial (unsigned int).
 *
 * Functions:
 * - monomial(const R, traits::IntToType<0>): Base case, returns 1 for x^0.
 * - monomial(const R x, traits::IntToType<N>): Recursive case, computes x^N as x * x^(N-1).
 * - monomial(const R x): Convenience function to compute x^N using IntToType.
 *
 * Requirements:
 * - The type R must support multiplication and construction from integer 1.
 * 

 * *
 * Example usage:
 * @code
 * constexpr auto result = polynomials::monomial<3>(2); // Computes 2^3 = 8 at compile time
 * @endcode
 */
#ifndef HH_MONOMIALS_HPP
#define HH_MONOMIALS_HPP    
#include "../traits/OPOE_traits.hpp"

namespace polynomials {

template <class R=traits::DataType::PolynomialField>
constexpr R
monomial(const R, traits::IntToType<0>)
{
  return 1;
}

template <unsigned int N, class R=traits::DataType::PolynomialField>
constexpr R
monomial(const R x, traits::IntToType<N>)
{
  return monomial(x, traits::IntToType<N - 1>()) * x;
}

template <unsigned int N, class R=traits::DataType::PolynomialField>
constexpr R 
monomial(const R x)
{
  return monomial(x, traits::IntToType<N>());
}

}
#endif // HH_MONOMIALS_HPP