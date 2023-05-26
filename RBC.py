import matplotlib.pyplot as plt
import numpy as np

class RBC:
    def __init__(self, alpha, beta, delta):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        
    def production_function(self, K, L):
        Y = K**self.alpha * L**(1-self.alpha)
        return Y
        
    def investment_function(self, Y, K):
        I = (1 - self.delta) * K + Y - self.production_function(K, 1)
        return I
        
    def consumption_function(self, Y, I):
        C = Y - I
        return C
        
    def euler_equation(self, K, L, r):
        Y = self.production_function(K, L)
        I = self.investment_function(Y, K)
        C = self.consumption_function(Y, I)
        euler_error = self.beta * (1 + r) * self.production_function(K, L) / (C * (1 + self.delta - r)) - 1
        return euler_error
        
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
    
    def montecarlo_simulation(self, K0_range, r0_range, L, T, num_simulations):
        K_simulations = []
        Y_simulations = []
        I_simulations = []
        C_simulations = []
        R_simulations = []
        
        for _ in range(num_simulations):
            K0 = np.random.uniform(*K0_range)
            r0 = np.random.uniform(*r0_range)
            
            K, Y, I, C, R = self.simulate(K0, L, r0, T)
            
            if K is not None:  # Verificar que la simulación no haya fallado
                K_simulations.append(K)
                Y_simulations.append(Y)
                I_simulations.append(I)
                C_simulations.append(C)
                R_simulations.append(R)
        
        return K_simulations, Y_simulations, I_simulations, C_simulations, R_simulations
    
    def plot_montecarlo_simulation(self, K_simulations, Y_simulations, I_simulations, C_simulations, R_simulations):
        num_simulations = len(K_simulations)
        T = len(K_simulations[0])
        
        # Graficar trayectorias individuales
        for i in range(num_simulations):
            plt.plot(range(T), K_simulations[i], color='blue', alpha=0.2)
            plt.plot(range(T), Y_simulations[i], color='green', alpha=0.2)
            plt.plot(range(T), I_simulations[i], color='red', alpha=0.2)
            plt.plot(range(T), C_simulations[i], color='orange', alpha=0.2)
            plt.plot(range(T), R_simulations[i], color='purple', alpha=0.2)
        
        # Calcular y graficar la línea de tendencia promedio
        K_avg = np.mean(K_simulations, axis=0)
        Y_avg = np.mean(Y_simulations, axis=0)
        I_avg = np.mean(I_simulations, axis=0)
        C_avg = np.mean(C_simulations, axis=0)
        R_avg = np.mean(R_simulations, axis=0)
        
        plt.plot(range(T), K_avg, color='blue', label='Capital')
        plt.plot(range(T), Y_avg, color='green', label='Producción')
        plt.plot(range(T), I_avg, color='red', label='Inversión')
        plt.plot(range(T), C_avg, color='orange', label='Consumo')
        plt.plot(range(T), R_avg, color='purple', label='Tasa de interés')
        
        plt.title('Montecarlo Simulation')
        plt.xlabel('Periodo')
        plt.ylabel('Valor')
        plt.legend()
        plt.show()

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
