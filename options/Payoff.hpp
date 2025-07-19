
/**
 * @file Payoff.hpp
 * @brief Defines interfaces and concrete classes for option payoff evaluation in quantitative finance.
 *
 * This header provides a generic interface (`IPayoff`) for evaluating the payoff of financial derivatives,
 * such as options, given the final price or log-price of the underlying asset. It also includes concrete
 * implementations for European call and put options, supporting both price and log-price evaluation.
 *
 * Classes:
 * - IPayoff<T>: Abstract base class for option payoffs, supporting polymorphic evaluation and cloning.
 * - EuropeanCallPayoff<T>: Implements the payoff for a European call option.
 * - EuropeanPutPayoff<T>: Implements the payoff for a European put option.
 *
 *
 * Usage:
 *   - Instantiate concrete payoff classes with the desired strike price.
 *   - Use `evaluate` for payoff calculation from price, or `evaluate_from_log` for log-price.
 *   - Use `clone` for polymorphic copying.
 *
 * Example:
 *   options::EuropeanCallPayoff<double> call(100.0);
 *   double payoff = call.evaluate(105.0); // Returns 5.0
 */
#ifndef PAYOFF_HPP
#define PAYOFF_HPP
#include <algorithm> // For std::max
#include <cmath>     // For std::exp (if evaluating from log-price)
#include <memory>    // For std::unique_ptr
#include <stdexcept> // For exceptions
#include "../traits/OPOE_traits.hpp" // For traits::DataType::PolynomialField


namespace options {

template <typename T = traits::DataType::PolynomialField>
class IPayoff {
public:
    // Virtual destructor is essential for polymorphic base classes
    virtual ~IPayoff() = default;

    /**
     * @brief Evaluates the payoff for a given final underlying asset price.
     * @param underlying_price The final price (S_T) of the underlying asset.
     * @return The calculated payoff value.
     */
    virtual T evaluate(T underlying_price) const = 0;   
    
    /**
     * @brief Returns the type of the option (Call or Put).
     * @return The type of the option as an enum value.
     */
    virtual traits::OptionType type() const = 0;

    /**
     * @brief Creates a copy of the payoff object.
     * @return A std::unique_ptr to the new IPayoff object.
     */
    virtual std::unique_ptr<IPayoff<T>> clone() const = 0;

    // Optional: Add a method to evaluate based on log-price if frequently needed
    /**
     * @brief Evaluates the payoff for a given final log-price of the underlying.
     * @param log_underlying_price The final log-price (x_T = ln(S_T)) of the underlying asset.
     * @return The calculated payoff value.
     */
    virtual T evaluate_from_log(T log_underlying_price) const {
        // Default implementation converts log-price to price
        // Derived classes can override if a more direct calculation exists
        return evaluate(std::exp(log_underlying_price));
    }
};

// --- Concrete European Call Payoff Template ---

template <typename T = traits::DataType::PolynomialField>
class EuropeanCallPayoff final : public IPayoff<T> {
private:
    T strike_price_;

public:
    /**
     * @brief Constructor for European Call Payoff.
     * @param strike_price The strike price (K) of the option.
     */
    explicit EuropeanCallPayoff(T strike_price) : strike_price_(strike_price) {
        if (strike_price < static_cast<T>(0.0)) {
            throw std::invalid_argument("Strike price cannot be negative.");
        }
    }

    /**
     * @brief Returns the type of the option (Call).
     * @return The type of the option as traits::OptionType::Call.
     */
    traits::OptionType type() const override {
        return traits::OptionType::Call; // Return the type of the option
    }

    /**
     * @brief Evaluates the call payoff: max(S_T - K, 0).
     * @param underlying_price The final price (S_T) of the underlying asset.
     * @return The calculated call payoff value.
     */
    T evaluate(T underlying_price) const override {
        return std::max(underlying_price - strike_price_, static_cast<T>(0.0));
    }

     /**
     * @brief Evaluates the call payoff from log-price: max(exp(x_T) - K, 0).
     * @param log_underlying_price The final log-price (x_T = ln(S_T)) of the underlying asset.
     * @return The calculated call payoff value.
     */
    T evaluate_from_log(T log_underlying_price) const override {
         return std::max(std::exp(log_underlying_price) - strike_price_, static_cast<T>(0.0));
    }

    /**
     * @brief Creates a copy of the EuropeanCallPayoff object.
     * @return A std::unique_ptr to the new IPayoff object.
     */
    std::unique_ptr<IPayoff<T>> clone() const override {
        // Use make_unique for safe and efficient memory allocation
        return std::make_unique<EuropeanCallPayoff<T>>(*this);
    }

    /**
     * @brief Gets the strike price.
     * @return The strike price (K).
     */
    T getStrike() const noexcept {
        return strike_price_;
    }
};

// --- Concrete European Put Payoff Template ---

template <typename T = traits::DataType::PolynomialField>
class EuropeanPutPayoff final : public IPayoff<T> {
private:
    T strike_price_;

public:
    /**
     * @brief Constructor for European Put Payoff.
     * @param strike_price The strike price (K) of the option.
     */
    explicit EuropeanPutPayoff(T strike_price) : strike_price_(strike_price) {
         if (strike_price < static_cast<T>(0.0)) {
            throw std::invalid_argument("Strike price cannot be negative.");
        }
    }

    /**
     * @brief Returns the type of the option (Put).
     * @return The type of the option as traits::OptionType::Put.
     */
    traits::OptionType type() const override {
        return traits::OptionType::Put;
    }

    /**
     * @brief Evaluates the put payoff: max(K - S_T, 0).
     * @param underlying_price The final price (S_T) of the underlying asset.
     * @return The calculated put payoff value.
     */
    T evaluate(T underlying_price) const override {
        return std::max(strike_price_ - underlying_price, static_cast<T>(0.0));
    }

    /**
     * @brief Evaluates the put payoff from log-price: max(K - exp(x_T), 0).
     * @param log_underlying_price The final log-price (x_T = ln(S_T)) of the underlying asset.
     * @return The calculated put payoff value.
     */
    T evaluate_from_log(T log_underlying_price) const override {
         return std::max(std::log(strike_price_) - log_underlying_price, static_cast<T>(0.0));
    }


    /**
     * @brief Creates a copy of the EuropeanPutPayoff object.
     * @return A std::unique_ptr to the new IPayoff object.
     */
    std::unique_ptr<IPayoff<T>> clone() const override {
        return std::make_unique<EuropeanPutPayoff<T>>(*this);
    }

    /**
     * @brief Gets the strike price.
     * @return The strike price (K).
     */
    T getStrike() const noexcept{
        return strike_price_;
    }
};

} // namespace options

#endif // PAYOFF_HPP
