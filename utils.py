import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm

def price_contract(r,sigma,T,guarantee, n, initial_price=100):
    dt = T/n
    R = np.exp(r*dt)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (R-d)/(u-d)

    # create matrix to store stock prices
    stock = np.zeros((n+1,n+1))
    stock[0,0] = initial_price
    for i in range(1,n+1):
        stock[i,0] = stock[i-1,0]*u
        for j in range(1,i+1):
            stock[i,j] = stock[i-1,j-1]*d

    # create matrix to store option prices
    option = np.zeros((n+1,n+1))
    option[n,:] = np.maximum(guarantee-stock[n,:],0)

    # calculate option prices at each node
    for i in range(n-1,-1,-1):
        for j in range(i+1):
            option[i,j] = np.exp(-r*dt)*(p*option[i+1,j]+(1-p)*option[i+1,j+1])

    return option[0,0]






# obviously price contract can be vectorized using numpy to make it faster
# so I will do that

def price_contract_vectorized(r,sigma,T,guarantee, n, initial_price=100):
    dt = T/n
    R = np.exp(r*dt)
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (R-d)/(u-d)

    # init asset prices at maturity
    stock = initial_price * u ** np.arange(0,n+1,1) * d ** np.arange(n,-1,-1)

    # init option prices at maturity
    stock = np.maximum(guarantee-stock, np.zeros(n+1))

    # calculate option prices at each node
    for i in np.arange(n,0,-1):
        stock = np.exp(-r*dt) * (p * stock[1:i+1] + (1-p) * stock[0:i])

    return stock[0]


# create a method to plot the price of the contract as a function of the number of steps
def plot_price(r, sigma, T, guarantee, n, pricing=price_contract_vectorized, analytical=None):
    steps = np.arange(1, n + 1)
    price = np.zeros(len(steps))
    for i in range(len(steps)):
        price[i] = pricing(r, sigma, T, guarantee, steps[i], n)
    plt.plot(steps, price)
    if analytical is not None:
        plt.plot(steps, analytical * np.ones(len(steps)))
    plt.xlabel("Number of steps")
    plt.ylabel("Price of contract")
    plt.legend(["Binomial Tree", "Analytical"])
    plt.title("r={}, sigma={}, T={}, guarantee={}".format(r, sigma, T, guarantee))
    plt.show()


def d1(S, t, T, r, sigma):
    return ( np.log(S/100) + (r + 0.5 * sigma ** 2) * (T - t)) / (sigma * np.sqrt(T - t))


# Function to price the option contract and calculate delta at each node
def price_contract_with_delta(r, sigma, T, guarantee, n, initial_price=100):
    dt = T / n
    R = np.exp(r * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (R - d) / (u - d)

    # Create matrices to store stock prices and option deltas
    stock = np.zeros((n + 1, n + 1))
    delta = np.zeros((n + 1, n + 1))
    stock[0, 0] = initial_price

    # Populate the stock price tree
    for i in range(1, n + 1):
        stock[i, 0] = stock[i - 1, 0] * u
        for j in range(1, i + 1):
            stock[i, j] = stock[i - 1, j - 1] * d

    # Create matrix to store option prices
    option = np.zeros((n + 1, n + 1))
    option[n, :] = np.maximum(guarantee - stock[n, :], 0)

    # Calculate option prices and deltas at each node
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_up = option[i + 1, j]
            option_down = option[i + 1, j + 1]
            stock_up = stock[i + 1, j]
            stock_down = stock[i, j] * d

            # Calculate delta using the formula from above
            delta[i, j] = stats.norm.cdf(
                d1(stock[i, j], i, 10, r, 0.15))  # (option_up - option_down) / (stock_up - stock_down)

            # Calculate option price
            option[i, j] = np.exp(-r * dt) * (p * option_up + (1 - p) * option_down)

    return stock, option, delta




# characteristic function of brownian motion WT
def charfct(xi, Wt, Tt):
    return np.exp(1j * xi * Wt - 0.5 * xi * xi * Tt)

# fourier transform of tilted option payoff
def optft(a, xi, S0, K, T, sig):
    rsig2t = -0.5 * sig * sig * T
    # compute Wbar
    wbar = (np.log(K / S0) - rsig2t) / sig
    return -S0 * np.exp(rsig2t + (1j * xi + sig - a) * wbar) / (1j * xi + sig - a) + K * np.exp(
        (1j * xi - a) * wbar) / (1j * xi - a)

# define auxillary function for integration
def ftint2(xi, a):
    return np.real(charfct(-xi - 1j * a, 0, T) * optft(a, xi, S0, K, T, sig) / (2 * np.pi))



# create a method for the monte carlo pricing
def MC_price(S0, r, sigma, T, K, M, n, type="call"):
    dt = T/n # timesteps
    # geometric brownian motion
    St = np.exp(
        (r - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(M,n)).T
    )

    # include array of 1's
    St = np.vstack([np.ones(M), St])

    # multiply through by S0 and get cumprod of elements along sim path
    St = S0 * St.cumprod(axis=0)

    # payoff funciton
    def payoff(S, K):
        if type == "call":
            return np.maximum(S[-1] - K, 0)
        else:
            return np.maximum(S[-1], K)

    # discounting
    def discounting(r, t):
        return np.exp(-r*t)

    # get the payoff of each path and take mean
    payoff = payoff(St, K)
    discounting = discounting(r, T)
    price = np.mean(payoff * discounting)


    return np.round(price, 3)


def black_scholes_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

