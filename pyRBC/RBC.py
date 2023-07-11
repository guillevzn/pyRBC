import matplotlib.pyplot as plt
import numpy as np

class RBC:
    def __init__(self, observed, params, shocks):

        '''
        Observed variables
        '''
        self.Y = observed[0] # Steady state output
        self.G = observed[1] # Steady state government spending
        self.I = observed[2] # Steady state investment
        self.C = self.Y - self.G - self.I # Steady state consumption
        self.N = observed[3] # Steady-State fraction of hours worked
        self.K = observed[4] # Steady state capital/output

        '''
        Parameters
        '''
        self.theta = params[0] # Capital income share
        self.v = params[1] # Frisch elasticity

        self.rho_z = params[2]
        self.sigma_z = params[3]
        self.rho_f = params[4]
        self.sigma_f = params[5]
        self.xi = params[6]

        '''
        Shocks
        '''
        self.epsilon = shocks[0]
        self.omega = shocks[1]

    def steady_state(self):
        phi = ((1 - self.theta) * self.Y / self.N) / (self.C * (self.N ** (1 / self.v)))  # MRS leisure-consumption and labor demand
        delta = self.I / self.K  # Steady state capital accumulation
        R = self.theta * self.Y / self.K  # Demand for capital
        r = -1 + (R + 1 - delta)  # Real interest rate
        beta = 1 / (1 + r)  # Subjective discount rate
        A = self.Y / ((self.K ** self.theta) * (self.N ** (1 - self.theta)))  # Technological scale parameter
        w = (1 - self.theta) * self.Y / self.N  # Real wage
        return beta, phi, A, self.theta, delta, self.G, self.v
        
    def simulate(self, K0, L, r0, T):
        K = [K0]
        L = [L] * T
        R = [r0]
        Y = [self.production_function(K0, L[0])]
        I = [self.investment_function(Y[0], K[0])]
        C = [self.consumption_function(Y[0], I[0])]
        
        for t in range(1, T):
            r = R[-1]
            K_t = K[-1] + I[-1] - C[-1]
            Y_t = self.production_function(K_t, L[t])
            I_t = self.investment_function(Y_t, K_t)
            C_t = self.consumption_function(Y_t, I_t)
            euler_error = self.euler_equation(K_t, L[t], r)
            #while abs(euler_error) > 1e-8:
            #    r += euler_error / 10
            #    euler_error = self.euler_equation(K_t, L[t], r)
            K.append(K_t)
            Y.append(Y_t)
            I.append(I_t)
            C.append(C_t)
            R.append(r)
        
        return K, Y, I, C, R

    def plot_simulation(self, K, Y, I, C, R):
        T = len(K)
        fig, axs = plt.subplots(2, 2, figsize=(12,8))
        axs[0, 0].plot(range(T), K)
        axs[0, 0].set_title('Capital accumulation')
        axs[0, 1].plot(range(T), Y)
        axs[0, 1].set_title('Production')
        axs[1, 0].plot(range(T), I)
        axs[1, 0].set_title('Investment')
        axs[1, 1].plot(range(T), C)
        axs[1, 1].set_title('Consumption')
        fig.suptitle('RBC simulation')
        plt.show()
