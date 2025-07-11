#ifndef HH_MONOMIALS_HPP
#define HH_MONOMIALS_HPP    
#include "../traits/OPOE_traits.hpp"

namespace polynomials {

//! Overload for N=0: Returns 1 as the base case for the monomial function when the exponent is zero.
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