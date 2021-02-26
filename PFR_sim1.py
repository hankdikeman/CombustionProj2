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

# ignore warnings from bad mech file
warnings.filterwarnings("ignore")

init_temps = np.array([1100, 1200, 1300, 1400, 1500])
init_pressure = np.array([1, 5, 10])
init_phi = np.array([0.3, 0.5, 1, 1.5])

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
            composition_0 = {'CH4': phi, 'O2': 2 / phi, 'N2': 2 * 3.76 / phi}

            # Resolution: The PFR will be simulated by 'n_steps' time steps
            n_steps = int(200000 * phi)

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
            equiv_ratios[n_temp, n_phi] = phi
            ignition_times[n_temp, n_phi] = sim_time
            temp_count += 1
        phi_count += 1
    # plot results at the given pressure
    plt.figure()
    for n, phi in enumerate(init_phi):
        plt.plot(initial_temperatures[:, n], ignition_times[:, n], 'o-', color=plotcolors[n],
                 label='phi=' + str(phi))
    plt.title('Auto-Ignition Times at P = ' + str(pressure) + ' atm')
    plt.xlabel('$T$ [K]')
    plt.ylabel('Autoignition Delay $t$ [s]')
    plt.yscale('log')
    plt.ylim(0.9E-4, 2.5E-1)
    plt.legend(loc=0)
    plt.grid(True, axis='y')
    figname = 'Figures/Sim1/P' + str(pressure) + 'atm.png'
    plt.savefig(figname)
    plt.show()


composition_0 = {'CH4': 0.3, 'O2': 2 / 0.3, 'N2': 2 * 3.76 / 0.3}
gas1 = ct.Solution(reaction_mechanism)
gas1.TPX = 1300, ct.one_atm, composition_0
gas1.equilibrate('HP')
print(gas1.T)
