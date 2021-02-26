import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
# create the gas mixture
gas = ct.Solution('gri30.xml')
residence_times = [np.power(x / 10, 2) for x in range(20)]
NO_conc = np.zeros_like(residence_times)
CO_conc = np.zeros_like(residence_times)


for t in residence_times:
    # pressure = 60 Torr, T = 770 K
    p = ct.one_atm
    t = 650
    phi = 0.85

    gas.TPX = t, p, {'CH4': phi, 'O2': 2 / phi, 'N2': 2 * 3.76 / phi}

    # create an upstream reservoir that will supply the reactor. The temperature,
    # pressure, and composition of the upstream reservoir are set to those of the
    # 'gas' object at the time the reservoir is created.
    upstream = ct.Reservoir(gas)

    # Now create the reactor object with the same initial state
    cstr = ct.IdealGasReactor(gas)

    # Set its volume to 10 cm^3. In this problem, the reactor volume is fixed, so
    # the initial volume is the volume at all later times.
    cstr.volume = 1000.0 * 1.0e-6

    # Connect the upstream reservoir to the reactor with a mass flow controller
    # (constant mdot). Set the mass flow rate to 1.25 sccm.
    res_time = 0.1
    mdot = res_time * cstr.volume * gas.density
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
    v = ct.Valve(cstr, downstream, K=1.0e-9)

    # create the network
    network = ct.ReactorNet([cstr])

    # now integrate in time
    t = 0.0
    dt = 0.01

    states = ct.SolutionArray(gas, extra=['t'])
    while t < 200.0:
        t += dt
        network.advance(t)
        states.append(cstr.thermo.state, t=t)


species = ['CO', 'NO']

if __name__ == '__main__':
    plt.figure(1)
    for spc in species:
        plt.plot(states.t, states(spc).Y, label=spc)
    plt.legend(loc='upper right')
    plt.xlabel('time [s]')
    plt.ylabel('mass fraction')
    plt.show()
