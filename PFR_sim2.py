"""
This code simulates a PFR undergoing nitrogen combustion with oxidizers of
varying concentrations of radical species and diluents. Ignition delay
for these oxidizers are then plotted for phi = 0.5, T = 1300 K, P = 1 atm
"""
import cantera as ct
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

plt.style.use('seaborn-bright')

# constant values
MAX_TIME = 1
reaction_mechanism = os.path.join('data', 'ucsdmech.yaml')
# plot colors must be same length as phi!!!
plotcolors = ["red", "blue", "green", "orange"]
PHI = 0.5

# ignore warnings from bad mech file
warnings.filterwarnings("ignore")


init_temps = np.array([1300])
init_pressure = np.array([1])
init_phi = np.array(['air', 'oxygen', 'O radicals', 'OH radicals'])
init_compositions = {
    'air': {'CH4': 1, 'O2': 2 / PHI, 'N2': 2 * 3.76 / PHI},
    'oxygen': {'CH4': 1, 'O2': 2 / PHI},
    'O radicals': {'CH4': 1, 'O2': 2 / PHI, 'N2': 0.79 / 0.207 * 2 / PHI, 'O': 0.003 / 0.207 * 2 / PHI},
    'OH radicals': {'CH4': 1, 'O2': 2 / PHI, 'N2': 0.79 / 0.207 * 2 / PHI, 'OH': 0.003 / 0.207 * 2 / PHI}
}

# store ignition times
ignition_times = np.empty(np.shape(init_temps)[0] * np.shape(init_phi)[0])

for pressure in init_pressure:
    ignition_times = np.empty((np.shape(init_temps)[0], np.shape(init_phi)[0]))
    equiv_ratios = np.zeros_like(ignition_times)
    initial_temperatures = np.zeros_like(ignition_times)
    phi_count = 0
    for n_phi, phi in enumerate(init_phi):
        temp_count = 0
        for n_temp, T_0 in enumerate(init_temps):
            # describe composition of mixture
            composition_0 = init_compositions[phi]

            # Resolution: The PFR will be simulated by 'n_steps' time steps
            n_steps = int(200000)

            # generate gas solution and set initial conditions
            gas1 = ct.Solution(reaction_mechanism)
            gas1.TPX = T_0, pressure * ct.one_atm, composition_0

            # create a new reactor
            r1 = ct.IdealGasConstPressureReactor(gas1)
            # create a reactor network for performing time integration
            sim1 = ct.ReactorNet([r1])

            # approximate a time step to achieve a similar resolution as in the next method
            dt = MAX_TIME / n_steps
            # define timesteps
            timesteps = (np.arange(n_steps) + 1) * dt
            for t_i in timesteps:
                # perform time integration
                sim1.advance(t_i)
                # store current time
                sim_time = t_i
                # compute velocity and transform into space
                # print(r1.thermo.state.T[0])
                if(r1.thermo.state.T[0] > T_0 + 150):
                    break

            # assign temps and equiv ratios to condition matrix
            initial_temperatures[n_temp, n_phi] = T_0
            ignition_times[n_temp, n_phi] = sim_time
            temp_count += 1
        phi_count += 1
    # plot results at the given pressure
    print(ignition_times)
    plt.figure()
    plt.bar([i for i, _ in enumerate(init_phi)],
            ignition_times[0, :], color=plotcolors)
    plt.title('Auto-Ignition Times of Different Oxidizer Compositions')
    plt.xlabel('$Oxidizer Species$')
    plt.ylabel('Autoignition Delay $t$ [s]')
    plt.legend(loc=0)
    plt.grid(True, axis='y')
    plt.xticks([i for i, _ in enumerate(init_phi)], init_phi)
    figname = 'Figures/Sim2/P' + str(pressure) + 'atm.png'
    plt.savefig(figname)
    plt.show()
