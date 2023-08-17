import numpy as np


def emitted_radiation(Temperature, model='linear'):
    """
    This function computes the emitted radiation based on the selected model.
    """

    number_of_seconds_in_a_year = 365.25 * 24 * 60 * 60

    if model == 'linear':
        A = -339.647 * (number_of_seconds_in_a_year ** 3)  # -339.647 W/m^2
        B = 2.218 * (number_of_seconds_in_a_year ** 3)  # 2.218 W/(m^2 K)
        emitted_radiation = A + B * Temperature
    elif model == 'black body':
        Stefan_Boltzmann_constant = 5.67e-8 * (number_of_seconds_in_a_year ** 3)  # W/(m^2 K^4)
        emitted_radiation = Stefan_Boltzmann_constant * (Temperature ** 4)
    else:
        raise ValueError("Invalid model selection. Choose 'linear' or 'black body'.")

    return emitted_radiation


def periodic_forcing(time, A, omega):
    """
    This function computes the periodic forcing applied to the system
    """
    return 1+A*np.cos(omega*time)


def F(Temperature,
      C, tau,
      T3, T1, T2,
      model='linear',
      mu=1):

    """
    This function returns the value of F(T) given a certain value for the temperature using a specified emission model for the computation of beta.
    """

    beta = -((C / (tau * emitted_radiation(T3, model))) * ((T1 * T2 * T3) / ((T1 - T3) * (T2 - T3))))
    F_value = (emitted_radiation(Temperature, model) / C) * ((mu / (1 + beta * (1 - (Temperature / T1)) * (1 - (Temperature / T2)) * (1 - (Temperature / T3)))) - 1)

    return F_value

def simulate_ito(T_start, t_start, dt, num_steps, num_simulations,
		 C, tau,
                 T3, T1, T2,
                 A, omega,
                 sigma, noise,
                 model = 'linear',
                 forcing = 'constant'):

    seed_value = 42
    np.random.seed(seed_value)

    t = np.arange(t_start, t_start+num_steps*dt + dt, dt) #len(t) = num_steps
    T = np.zeros((num_simulations, num_steps))

    T[:, 0] = T_start

    if noise == True:
        W = np.random.normal(0, np.sqrt(dt), (num_simulations, num_steps))
    elif noise == False:
        W = np.zeros((num_simulations, num_steps))
    else:
        raise ValueError("Invalid value for 'noise'. Please use True or False")

    if forcing == "constant":
        forcing_values = np.ones(num_steps)
    elif forcing == "varying":
        forcing_values = periodic_forcing(t, A, omega)
    else:
        raise ValueError("Invalid value for 'forcing'. Please use 'constant' or 'varying'")

    for i in range(num_steps-1):
        Fi = F(T[:, i], C, tau, T3, T1, T2, model, forcing_values[i])
        dT = Fi*dt + sigma*W[:, i]
        T[:, i+1] = T[:, i] + dT

    return t, T
