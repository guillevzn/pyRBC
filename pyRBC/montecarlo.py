import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

class MonteCarloSimulation:
    def __init__(self, data, intervals, iterations):
        self.data = data
        self.log_returns = np.log(1 + self.data.close.pct_change())
        self.u = self.log_returns.mean()
        self.var = self.log_returns.var()
        self.drift = self.u - (0.5 * self.var)
        self.stdev = self.log_returns.std()
        self.t_intervals = intervals
        self.iterations = iterations

    def run_simulation(self):
        daily_returns = np.exp(self.drift + self.stdev * norm.ppf(np.random.rand(self.t_intervals, self.iterations)))
        price_list = np.zeros_like(daily_returns)
        S0 = self.data.close.iloc[-1]
        price_list[0] = S0

        for t in range(1, self.t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]

        price_list = pd.DataFrame(price_list)
        price_list['close'] = price_list[0]
        return price_list

    def plot_simulation(self, monte_carlo_forecast):
        plt.figure(figsize=(17, 8))
        plt.plot(monte_carlo_forecast)
        plt.show()
