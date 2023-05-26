#%%
import matplotlib.pyplot as plt
from RBC import RBC

rbc_model = RBC(alpha=0.33, beta=0.95, delta=0.02)

K, Y, I, C, R = rbc_model.simulate(K0=50, L=1, r0=0.02, T=100)
rbc_model.plot_simulation(K, Y, I, C, R)


# %%
import matplotlib.pyplot as plt
from RBC import RBC

# Parámetros del modelo
alpha = 0.3
beta = 0.95
delta = 0.1

# Crear una instancia del modelo RBC
rbc_model = RBC(alpha, beta, delta)

K_simulations, Y_simulations, I_simulations, C_simulations, R_simulations = rbc_model.montecarlo_simulation(K0_range=(40, 60), r0_range=(0.01, 0.03), L=1, T=100, num_simulations=100)
rbc_model.plot_montecarlo_simulation(K_simulations, Y_simulations, I_simulations, C_simulations, R_simulations)

# %%
