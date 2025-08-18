/**
 * @brief Validates and debugs polynomial domain parameters.
 *
 * This file contains the implementation of member functions for the
 * PolynomialDomainValidator class template, which is responsible for
 * validating the parameters of a polynomial domain and providing
 * debugging information.
 * 
 * Dependencies:
 * - OrthogonalValidator.hpp: Header file for the PolynomialDomainValidator class.
 *
 * @tparam R The return type or result type associated with the validator.
 * @tparam Params Variadic template parameters representing the types of the parameters to validate.
 *
 * Member Functions:
 * - validateParameters(): Checks if all parameters are valid according to their domain.
 *   Throws a std::runtime_error with a detailed error message if any parameter is invalid.
 * - debugParameters(): Outputs the name and value of each parameter to std::cout for debugging purposes.
 * - buildErrorMessage(): Constructs a detailed error message listing all invalid parameters,
 *   their values, and the expected domain.
 *
 * Usage:
 *   Instantiate PolynomialDomainValidator with the appropriate parameter types,
 *   then call validateParameters() to ensure all parameters are valid before proceeding.
 */
#include <iostream>
#include <sstream>
#include "OrthogonalValidator.hpp"

namespace polynomials {

// PolynomialDomainValidator member functions
template<typename R, typename... Params>
void PolynomialDomainValidator<R, Params...>::validateParameters() const {
    // Check if all parameters are valid using fold expression
    // This checks if each parameter's isValid() method returns true
    // If any parameter is invalid, it throws an exception with a detailed message
    // Using std::apply to unpack the tuple and check each parameter
    bool allValid = std::apply([](const auto&... params) {
        return (params.isValid() && ...);
    }, parameters_);

    if (!allValid) {
        std::string errorMessage = buildErrorMessage();
        throw std::runtime_error(errorMessage);
    }
}

template<typename R, typename... Params>
void PolynomialDomainValidator<R, Params...>::debugParameters() const {
    // Print each parameter's name and value to standard output
    // Using std::apply to unpack the tuple and print each parameter
    // This uses a fold expression to iterate over each parameter in the tuple
    std::apply([](const auto&... params) {
        ((std::cout << "Parameter " << params.name << ": " << params.value << "\n"), ...);
    }, parameters_);
}

template<typename R, typename... Params>
std::string PolynomialDomainValidator<R, Params...>::buildErrorMessage() const {
    // Build a detailed error message listing all invalid parameters
    // Using std::apply to unpack the tuple and check each parameter
    // This constructs a string that includes the name, value, and expected domain for each invalid parameter
    std::string errorDetail = "Parameter validation failed:\n";

    std::apply([&errorDetail](const auto&... params) {
        (
            [&]() {
                if (!params.isValid()) {
                    errorDetail += " - Parameter \"" + std::string(params.name) + "\" is invalid (value: " 
                                   + std::to_string(params.value) + "). Expected: " + std::string(params.getDomain()) + ".\n";
                }
            }(),
            ...
        );
    }, parameters_);

    return errorDetail;
} }