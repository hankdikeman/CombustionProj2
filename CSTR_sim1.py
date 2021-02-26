"""
This code simulates a CSTR reactor for various residence time and evaluates
the presence of important pollutants such as NO and CO
"""
import cantera as ct
import numpy as np
import matplotlib.pyplot as plt

# create the gas mixture and composition arrays
gas = ct.Solution('gri30.xml')
residence_times = [np.power((x + 1) / 20, 2) for x in range(20)]
print("res times:", residence_times)
NO_conc = np.zeros_like(residence_times)
CO_conc = np.zeros_like(residence_times)


for n, t in enumerate(residence_times):
    print('res_time:', t)
    # pressure = 60 Torr, T = 750 K
    p = ct.one_atm
    T = 750
    phi = 0.85

    gas.TPX = T, p, {'CH4': phi, 'O2': 2 / phi, 'N2': 2 * 3.76 / phi}

    # create an upstream reservoir that will supply the reactor. The temperature,
    # pressure, and composition of the upstream reservoir are set to those of the
    # 'gas' object at the time the reservoir is created.
    upstream = ct.Reservoir(gas)

    # Now create the reactor object with the same initial state
    cstr = ct.IdealGasReactor(gas)

    # Set its volume to 1000 cm^3. In this problem, the reactor volume is fixed, so
    # the initial volume is the volume at all later times.
    cstr.volume = 1000.0 * 1.0e-6

    # Connect the upstream reservoir to the reactor with a mass flow controller
    # (constant mdot). Set the mass flow rate to 1.25 sccm.
    res_time = t
    mdot = res_time * cstr.volume * gas.density
    print("mdot:", mdot, '\n')
    sccm = 1.25
    vdot = sccm * 1.0e-6 / 60.0 * \
        ((ct.one_atm / gas.P) * (gas.T / 273.15))  # m^3/s
    # mdot = gas.density * vdot  # kg/s
    mfc = ct.MassFlowController(upstream, cstr, mdot=mdot)

    # now create a downstream reservoir to exhaust into.
    downstream = ct.Reservoir(gas)

    # connect the reactor to the downstream reservoir with a valve, and set the
    # coefficient sufficiently large to keep the reactor pressure close to the
    # downstream pressure of 60 Torr.
    mfc = ct.MassFlowController(cstr, downstream, mdot=mdot)
    # v = ct.Valve(cstr, downstream, K=1.0e-9)

    # create the network
    network = ct.ReactorNet([cstr])

    # now integrate in time
    time = 0.0
    dt = 0.1

    states = ct.SolutionArray(gas, extra=['t'])
    while time < 400.0:
        time += dt
        network.advance(time)
        states.append(cstr.thermo.state, t=time)

    NO_conc[n] = states('NO').X[-1]
    CO_conc[n] = states('CO').X[-1]

species = ['CO', 'NO']

if __name__ == '__main__':
    plt.figure(1)
    plt.plot(residence_times, NO_conc, 'ro-', label='[NO]$_{SS}$')
    plt.title('NO Production as a Function of Residence Time')
    plt.legend(loc='upper right')
    plt.xlabel('time [s]')
    plt.yscale('log')
    plt.ylabel('mol fraction')
    figname = 'Figures/Sim3/NOProd.png'
    plt.savefig(figname)
    plt.show()
    plt.figure(2)
    plt.plot(residence_times, CO_conc, 'bo-', label='[CO]$_{SS}$')
    plt.title('CO Production as a Function or Residence Time')
    plt.legend(loc='upper right')
    plt.xlabel('time [s]')
    plt.yscale('log')
    figname = 'Figures/Sim3/COProd.png'
    plt.savefig(figname)
    plt.ylabel('mol fraction')
    plt.show()
