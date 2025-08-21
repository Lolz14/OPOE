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
#include <algorithm> 
#include <cmath>     
#include <memory>    
#include <stdexcept> 
#include "../traits/OPOE_traits.hpp" 


namespace options {

using StoringVector = traits::DataType::StoringVector;
using Array = traits::DataType::StoringArray;

/**
 * @brief Interface for option payoffs.
 * This class defines the contract for evaluating the payoff of financial derivatives,
 * such as options, based on the final price or log-price of the underlying asset.
 *
 * @tparam T The numeric type used for the strike price and evaluations (default: traits::DataType::PolynomialField).
 */
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
     * @brief Evaluates the payoff for a given vector of final underlying asset prices.
     * @param underlying_prices The final prices (S_T) of the underlying asset.
     * @return The calculated payoff value for each underlying.
     */
    virtual StoringVector evaluate(StoringVector underlying_prices) const = 0;   
    
    
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

    /**
     * @brief Evaluates the payoff for a vector of final log-prices of the underlying.
     * @param log_underlying_prices The final log-prices (x_T = ln(S_T)) of the underlying asset.
     * @return A vector of calculated payoffs for each log-price.
     */
    virtual StoringVector evaluate_from_log(StoringVector log_underlying_prices) const {
        // Default implementation converts log-prices to prices
        // Derived classes can override if a more direct calculation exists
        return evaluate(log_underlying_prices.array().exp());
    }

    virtual T getStrike() const = 0;
    /**
     * @brief Sets the strike price.
     * @param strike_price The new strike price (K).
     * @throws std::invalid_argument if the strike price is negative.
     */
    virtual void setStrike(T strike_price) = 0;
};

/**
 * @brief Specialization of IPayoff for European Call options.
 *
 * This class implements the payoff evaluation for European call options,
 * defined as max(S_T - K, 0), where S_T is the final price of the underlying asset
 * and K is the strike price.
 *
 * @tparam T The numeric type used for the strike price and evaluations (default: traits::DataType::PolynomialField).
 */
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
     * @brief Evaluates the call payoff for a vector of underlying prices.
     * @param underlying_prices The final prices (S_T) of the underlying asset.
     * @return A vector of calculated call payoffs for each underlying price.
     */
    StoringVector evaluate(StoringVector underlying_prices) const override {
        // Vectorized evaluation for multiple underlying prices
        StoringVector payoffs(underlying_prices.size());
        
        payoffs = (underlying_prices.array() - strike_price_).cwiseMax(static_cast<T>(0.0));
        return payoffs;
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
    T getStrike() const noexcept override {
        return strike_price_;
    }

    /**
     * @brief Sets the strike price.
     * @param strike_price The new strike price (K).
     * @throws std::invalid_argument if the strike price is negative.
     */
    void setStrike(T strike_price) override {
        if (strike_price < static_cast<T>(0.0)) {
            throw std::invalid_argument("Strike price cannot be negative.");
        }
        strike_price_ = strike_price;
    }
};


/**
 * @brief Specialization of IPayoff for European Put options.
 *
 * This class implements the payoff evaluation for European put options,
 * defined as max(K - S_T, 0), where S_T is the final price of the underlying asset
 * and K is the strike price.
 *
 * @tparam T The numeric type used for the strike price and evaluations (default: traits::DataType::PolynomialField).
 */
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
     * @brief Evaluates the put payoff for a vector of underlying prices.
     * @param underlying_prices The final prices (S_T) of the underlying asset.
     * @return A vector of calculated put payoffs for each underlying price.
     */
    StoringVector evaluate(StoringVector underlying_prices) const override {
        // Vectorized evaluation for multiple underlying prices
        StoringVector payoffs(underlying_prices.size());
        payoffs = (strike_price_ - underlying_prices.array()).cwiseMax(static_cast<T>(0.0));
        return payoffs;
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
    T getStrike() const noexcept override {
        return strike_price_;
    }

        /**
     * @brief Sets the strike price.
     * @param strike_price The new strike price (K).
     * @throws std::invalid_argument if the strike price is negative.
     */
    void setStrike(T strike_price) override {
        if (strike_price < static_cast<T>(0.0)) {
            throw std::invalid_argument("Strike price cannot be negative.");
        }
        strike_price_ = strike_price;
    }

};

} // namespace options

#endif // PAYOFF_HPP
