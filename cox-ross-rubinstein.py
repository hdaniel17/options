import numpy as np


def binomial_price(sigma, K, r, S0, T, m, is_call=True, is_euro=True, q=0):
    """
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

# print(binomial_price(0.2, 100, 0.05, 100, 1, 2, is_call=False, is_euro=True, q=0))
