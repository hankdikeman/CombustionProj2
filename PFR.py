import cantera as ct
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.style.use('seaborn-bright')

# ignore warnings from bad mech file
warnings.filterwarnings("ignore")

T_0 = 1000.0  # inlet temperature [K]
pressure = ct.one_atm  # constant pressure [Pa]
phi = 1
composition_0 = {'CH4': phi, 'O2': 2 / phi, 'N2': 2 * 3.76 / phi}
length = 0.001  # *approximate* PFR length [m]
u_0 = 0.001  # inflow velocity [m/s]
area = 1.e-4  # cross-sectional area [m**2]

# input file containing the reaction mechanism
reaction_mechanism = os.path.join('data', 'ucsdmech.yaml')

# Resolution: The PFR will be simulated by 'n_steps' time steps or by a chain
# of 'n_steps' stirred reactors.
n_steps = 80000

gas1 = ct.Solution(reaction_mechanism)
gas1.TPX = T_0, pressure, composition_0
mass_flow_rate1 = u_0 * gas1.density * area

# create a new reactor
r1 = ct.IdealGasConstPressureReactor(gas1)
# create a reactor network for performing time integration
sim1 = ct.ReactorNet([r1])

# approximate a time step to achieve a similar resolution as in the next method
t_total = length / u_0
dt = t_total / n_steps
# define time, space, and other information vectors
t1 = (np.arange(n_steps) + 1) * dt
z1 = np.zeros_like(t1)
u1 = np.zeros_like(t1)
states1 = ct.SolutionArray(r1.thermo)
for n1, t_i in enumerate(t1):
    # perform time integration
    sim1.advance(t_i)
    # compute velocity and transform into space
    u1[n1] = mass_flow_rate1 / area / r1.thermo.density
    z1[n1] = z1[n1 - 1] + u1[n1] * dt
    states1.append(r1.thermo.state)

plt.figure()
plt.plot(t1, states1.T, label='Lagrangian Particle')
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.legend(loc=0)
plt.show()

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('CH4')],
         label='Lagrangian Particle')
plt.xlabel('$t$ [s]')
plt.ylabel('$X_{H_2}$ [-]')
plt.legend(loc=0)
plt.show()
