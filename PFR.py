import cantera as ct
import os
import numpy as np
import matplotlib.pyplot as plt

T_0 = 1500.0  # inlet temperature [K]
pressure = ct.one_atm  # constant pressure [Pa]
composition_0 = 'CH4:2, O2:1, AR:0.1'
length = 1.5e-7  # *approximate* PFR length [m]
u_0 = .006  # inflow velocity [m/s]
area = 1.e-4  # cross-sectional area [m**2]

# input file containing the reaction mechanism
reaction_mechanism = os.path.join('data', 'ucsdmech.yaml')

# Resolution: The PFR will be simulated by 'n_steps' time steps or by a chain
# of 'n_steps' stirred reactors.
n_steps = 2000

gas1 = ct.Solution(reaction_mechanism)
gas1.TPX = T_0, pressure, composition_0
mass_flow_rate1 = u_0 * gas1.density * area
