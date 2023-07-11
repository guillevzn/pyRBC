#%%
import matplotlib.pyplot as plt
from pyRBC.RBC import RBC

rbc_model = RBC(model_type='0', alpha=0.33, beta=0.95, delta=0.02)

K, Y, I, C, R = rbc_model.simulate(K0=50, L=1, r0=0.02, T=100)
rbc_model.plot_simulation(K, Y, I, C, R)

# %%
# Ejemplo de uso Montecarlo
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

from pyRBC.montecarlo import MonteCarloSimulation
import easyfred
fred = easyfred.Fred(api_key='733fa628f2c2813c263501118e80e79c')
data = fred.get_table('SP500')
data = data[['date', 'value']]
data.columns = ['date', 'close']

data.close = pd.to_numeric(data.close, errors='coerce')
data = data[data.close != '.']
data.close = pd.to_numeric(data.close)

simulation = MonteCarloSimulation(data=data, intervals=250, iterations=10)
forecast = simulation.run_simulation()
simulation.plot_simulation(forecast)
# %%
