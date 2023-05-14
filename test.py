#%%
import matplotlib.pyplot as plt
from RBC import RBC

rbc_model = RBC(alpha=0.33, beta=0.95, delta=0.02)

K, Y, I, C, R = rbc_model.simulate(K0=50, L=1, r0=0.02, T=100)
rbc_model.plot_simulation(K, Y, I, C, R)


# %%
