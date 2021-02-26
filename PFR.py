"""
This program evaluates the compositions of major and minor species of methane
combustion for air-methane combustion at T = 1100 K, p = 1 atm, and phi = 1.
Ignition was defined as a temperature rise greater than 150 K.
"""
import cantera as ct
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.style.use('seaborn-bright')

# ignore warnings from bad mech file
warnings.filterwarnings("ignore")

T_0 = 1100.0  # inlet temperature [K]
pressure = ct.one_atm  # constant pressure [Pa]
phi = 1
composition_0 = {'CH4': phi, 'O2': 2 / phi, 'N2': 2 * 3.76 / phi}
length = 0.001  # *approximate* PFR length [m]
u_0 = 0.001  # inflow velocity [m/s]
area = 1.e-4  # cross-sectional area [m**2]

# input file containing the reaction mechanism
reaction_mechanism = os.path.join('data', 'ucsdmech.yaml')

# Resolution: The PFR will be simulated by 'n_steps' time steps
n_steps = 200000

gas1 = ct.Solution(reaction_mechanism)
gas1.TPX = T_0, pressure, composition_0
mass_flow_rate1 = u_0 * gas1.density * area

# create a new reactor
r1 = ct.IdealGasConstPressureReactor(gas1)
# create a reactor network for performing time integration
sim1 = ct.ReactorNet([r1])

# approximate a time step to achieve a similar resolution as in the next method
t_total = 0.08
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
plt.plot(t1, states1.T, label='Temperature')
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.title('Temperature with Respect to Time')
plt.legend(loc=0)
figname = 'Figures/Sim1/TemperatureZoomOut.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.T, label='Temperature')
plt.xlabel('$t$ [s]')
plt.ylabel('$T$ [K]')
plt.title('Temperature with Respect to Time')
plt.legend(loc=0)
plt.xlim(0.055, 0.06)
figname = 'Figures/Sim1/TemperatureZoomIn.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H2O')], 'g-',
         label='H$_2$O')
plt.plot(t1, states1.X[:, gas1.species_index('CH4')], 'r-',
         label='CH$_4$')
plt.plot(t1, states1.X[:, gas1.species_index('CO2')], 'b-',
         label='CO$_2$')
plt.plot(t1, states1.X[:, gas1.species_index('O2')], 'm-',
         label='O$_2$')
plt.title('Major Species Concentrations with Respect to Time')
plt.xlabel('Residence Time $t$ [s]')
plt.ylabel('$X$ [-]')
plt.legend(loc=0)
figname = 'Figures/Sim1/MajorZoomOut.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H2O')], 'g-',
         label='H$_2$O')
plt.plot(t1, states1.X[:, gas1.species_index('CH4')], 'r-',
         label='CH$_4$')
plt.plot(t1, states1.X[:, gas1.species_index('CO2')], 'b-',
         label='CO$_2$')
plt.plot(t1, states1.X[:, gas1.species_index('O2')], 'm-',
         label='O$_2$')
plt.title('Major Species Concentrations with Respect to Time')
plt.xlabel('Residence Time $t$ [s]')
plt.ylabel('$X$ [-]')
plt.legend(loc=0)
plt.xlim(0.055, 0.06)
figname = 'Figures/Sim1/MajorZoomIn.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H')], 'r-',
         label='H')
plt.plot(t1, states1.X[:, gas1.species_index('OH')], 'b-',
         label='OH')
plt.plot(t1, states1.X[:, gas1.species_index('O')], 'm-',
         label='O')
plt.title('Minor Species Concentrations with Respect to Time')
plt.xlabel('Residence Time $t$ [s]')
plt.xlim(0.0565, 0.05775)
plt.ylabel('$X$ [-]')
plt.legend(loc=0)
figname = 'Figures/Sim1/MinorZoomIn.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H')], 'r-',
         label='H')
plt.plot(t1, states1.X[:, gas1.species_index('OH')], 'b-',
         label='OH')
plt.plot(t1, states1.X[:, gas1.species_index('O')], 'm-',
         label='O')
plt.title('Minor Species Concentrations with Respect to Time')
plt.xlabel('Residence Time $t$ [s]')
plt.xlim(0.055, 0.06)
plt.yscale('log')
plt.ylabel('$X$ [-]')
plt.legend(loc=0)
figname = 'Figures/Sim1/MinorZoomInLog.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H')], 'r-',
         label='H')
plt.plot(t1, states1.X[:, gas1.species_index('OH')], 'b-',
         label='OH')
plt.plot(t1, states1.X[:, gas1.species_index('O')], 'm-',
         label='O')
plt.title('Minor Species Concentrations with Respect to Time')
plt.xlabel('Residence Time $t$ [s]')
plt.yscale('log')
plt.ylabel('$X$ [-]')
plt.legend(loc=0)
figname = 'Figures/Sim1/MinorZoomOutLog.png'
plt.savefig(figname)
plt.show(block=False)

plt.figure()
plt.plot(t1, states1.X[:, gas1.species_index('H')], 'r-',
         label='H')
plt.plot(t1, states1.X[:, gas1.species_index('OH')], 'b-',
         label='OH')
plt.plot(t1, states1.X[:, gas1.species_index('O')], 'm-',
         label='O')
plt.title('Minor Species Concentrations with Respect to Time')
plt.xlabel('Residence Time $t$ [s]')
plt.ylabel('$X$ [-]')
plt.legend(loc=0)
figname = 'Figures/Sim1/MinorZoomOut.png'
plt.savefig(figname)
plt.show()
