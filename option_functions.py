import numpy as np
import math
from scipy.stats import norm


def black_scholes(sigma, K, r, S0, T, is_call=True, q=0):
    """
    Black-Scholes option pricing
    :param sigma: volatility of the underlying (decimal)
    :param K: strike price
    :param r: risk-free rate (decimal)
    :param S0: initial asset price
    :param T: time until expiration (years)
    :param is_call: True for call, False for put
    :param q: dividend yield (decimal)

    :return: price of option
    """
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        price = S0 * norm.cdf(d1) * math.exp(-q * T) - K * math.exp(-r * T) * norm.cdf(d2)
        return float(price)
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1) * math.exp(-q * T)
        return float(price)


def binomial_price(sigma, K, r, S0, T, m, is_call=True, is_euro=True, q=0):
    """
    CRR option pricing
    :param sigma: volatility of the underlying (decimal)
    :param K: strike price
    :param r: risk-free rate (decimal)
    :param S0: initial asset price
    :param T: time until expiration (years)
    :param m: number of periods after initial
    :param is_call: True for call, False for put
    :param is_euro: True for European, False for American
    :param q: dividend yield (decimal)

    :return: price of option
    """
    dt = T / m
    u = np.exp(sigma * np.sqrt(dt))  # up movement size
    d = 1 / u  # down movement size
    p = (np.exp((r - q) * dt) - d) / (u - d)  # risk-neutral up probability

    price_tree = np.zeros((m + 1, m + 1))
    option_values = np.zeros((m + 1, m + 1))

    for j in range(m + 1):
        for i in range(j + 1):
            price_tree[i, j] = S0 * (u ** (j - i)) * (d ** i)

    if is_call:
        option_values[:, m] = np.maximum(np.zeros(m + 1), price_tree[:, m] - K)
    else:
        option_values[:, m] = np.maximum(np.zeros(m + 1), K - price_tree[:, m])

    for j in range(m - 1, -1, -1):
        for i in range(j + 1):
            cont = np.exp(-r * dt) * (p * option_values[i, j + 1] + (1 - p) * option_values[i + 1, j + 1])
            if is_euro or is_call:
                option_values[i, j] = cont
            else:
                option_values[i, j] = max(K - price_tree[i, j], cont)

    return option_values[0, 0]


def bs_greeks(sigma, K, r, S0, T, is_call=True, q=0):
    """
    Same as black_scholes but returns a few Greeks
    :return: price, delta, gamma, vega, theta
    """
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Gamma and vega are the same for calls and puts
    gamma = norm.pdf(d1) / (S0 * sigma * math.sqrt(T))
    vega = S0 * norm.pdf(d1) * math.sqrt(T)

    if is_call:
        delta = norm.cdf(d1)
        theta = -(S0 * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) - r * K * math.exp(-r * T) * norm.cdf(d2)
        price = S0 * norm.cdf(d1) * math.exp(-q * T) - K * math.exp(-r * T) * norm.cdf(d2)
        return float(price), float(delta), float(gamma), float(vega), float(theta)
    else:
        delta = norm.cdf(d1) - 1
        theta = -(S0 * norm.pdf(d1) * sigma / (2 * math.sqrt(T))) + r * K * math.exp(-r * T) * norm.cdf(-d2)
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1) * math.exp(-q * T)
        return float(price), float(delta), float(gamma), float(vega), float(theta)


def implied_vol(price, K, r, S0, T, is_call=True, epsilon=0.0001, check_bounds=True):
    """
    Finding implied volatility via Newton-Raphson method
    """
    if check_bounds:
        if price > K * math.exp(-r * T):
            raise ValueError("Price is greater than the discounted strike")
        elif is_call and price < max(S0 - K * math.exp(-r * T), 0):
            raise ValueError("Price below lower bound")
        elif not is_call and price < max(K * math.exp(-r * T) - S0, 0):
            raise ValueError("Price below lower bound")

    n = 1
    x0 = np.sqrt((2 * np.abs(np.log(S0 / K) + r * T)) / T)
    x_list = [x0]

    looking = True
    while looking:
        fx = black_scholes(x_list[n - 1], K, r, S0, T, is_call=is_call, q=0) - price
        d1 = (np.log(S0 / K) + (r + 0.5 * (x_list[n - 1]) ** 2) * T) / ((x_list[n - 1]) * np.sqrt(T))
        dfx = S0 * np.sqrt(T) * norm.pdf(d1)
        new_x = x_list[n - 1] - (fx / dfx)
        x_list.append(new_x)
        if np.abs(x_list[n] - x_list[n - 1]) < epsilon:
            looking = False
        n += 1

    return x_list[n - 1]
