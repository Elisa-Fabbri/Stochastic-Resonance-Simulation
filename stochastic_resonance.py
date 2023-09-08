import numpy as np
import configparser
import sys
import aesthetics as aes

config = configparser.ConfigParser()

try:
    config.read(sys.argv[1])
except IndexError:
    with aes.red_text():
        print('Error: You must specify a configuration file as an argument!')
        sys.exit()

T1 = config['settings'].getfloat('stable_temperature_solution_1')
T2 = config['settings'].getfloat('unstable_temperature_solution')
T3 = config['settings'].getfloat('stable_temperature_solution_2')

C_j_per_m2_K = config['settings'].getfloat('surface_earth_thermal_capacity')
tau = config['settings'].getfloat('relaxation_time')
emission_model = config['settings'].get('emission_model')
A = config['settings'].getfloat('forcing_amplitude')
period = config['settings'].getfloat('forcing_period')

num_sec_in_a_year = 365.25*24*60*60
C_years = C_j_per_m2_K*np.power(num_sec_in_a_year,2)
omega = (2*np.pi)/period

num_steps_default = config['settings'].getint('num_steps')
num_simulations_default = config['settings'].getint('num_simulations')
time_step_default = config['settings'].getfloat('time_step')


def emitted_radiation(Temperature, model=emission_model,
                      conversion_factor=num_sec_in_a_year):
    """
    Calculate emitted radiation from the Earth's surface based on the selected emission model.

    This function computes the emitted radiation from the Earth's surface based on the selected emission model,
    which can be either 'linear' or 'black body'. The emitted radiation is calculated using the provided temperature and
    conversion factor.

    Parameters:
    - Temperature (float): The temperature of the Earth's surface in Kelvin.
    - model (str, optional): The emission model to use, which can be 'linear' or 'black body'. Default is 'linear'.
    - conversion_factor (float, optional): Conversion factor for time units to calculate the emitted radiation in kg/year^3.
      Default is the number of seconds in a year.

    Returns:
    - emitted_radiation (float): The calculated emitted radiation in kg/year^3.

    Raises:
    - ValueError: If an invalid emission model is selected. Valid options are 'linear' or 'black body'.
    """
    if model == 'linear':
        A = -339.647 * (conversion_factor ** 3)  # -339.647 W/m^2
        B = 2.218 * (conversion_factor ** 3)  # 2.218 W/(m^2 K)
        emitted_radiation = A + B * Temperature
    elif model == 'black body':
        Stefan_Boltzmann_constant = 5.67e-8 * (conversion_factor ** 3)  # W/(m^2 K^4)
        emitted_radiation = Stefan_Boltzmann_constant * (Temperature ** 4)
    else:
        raise ValueError('Invalid emission model selection. Choose "linear" or "black body".')

    return emitted_radiation

def periodic_forcing(time, amplitude=A, angular_frequency=omega):
    """
    Calculate the periodic forcing applied to the system.

    Parameters:
    - time (float or array-like): The time or times at which to calculate the periodic forcing.
    - amplitude (float, optional): The amplitude of the periodic forcing. If not provided, the default value is read from
      the configuration file.
    - angular_frequency (float, optional): The angular frequency of the periodic forcing. If not provided, the default value
      is read from the configuration file.

    Returns:
    - periodic_forcing (float or array-like): The calculated periodic forcing values corresponding to the input time(s).
    """
    return 1 + amplitude * np.cos(angular_frequency * time)

def F(Temperature,
      surface_heat_capacity = C_years,
      relaxation_time = tau,
      stable_solution_1 = T1, unstable_solution = T2, stable_solution_2 = T3,
      model=emission_model,
      periodic_forcing_value=1):

    """
    Calculate the rate of temperature change (F(T) = dT/dt) given a certain temperature (T) using a specified emission model
    for the computation of the constant beta.

    Parameters:
    - Temperature (float): The temperature in Kelvin at which to calculate the rate of temperature change (F(T)).
    - surface_heat_capacity (float, optional): The surface heat capacity in kg/(K*year^2). If not provided, the default
    value is read from the configuration file.
    - relaxation_time (float, optional): The relaxation time in seconds. If not provided, the default value is read from
    the configuration file.
    - stable_solution_1 (float, optional): The first stable temperature solution in Kelvin. If not provided, the default value
    is read from the configuration file.
    - unstable_solution (float, optional): The unstable temperature solution in Kelvin. If not provided, the default value
    is read from the configuration file.
    - stable_solution_2 (float, optional): The second stable temperature solution in Kelvin. If not provided, the default value
    is read from the configuration file.
    - model (str, optional): The emission model to use for beta computation. If not provided, the default value is read
    from the configuration file.
    - periodic_forcing_amplitude (float, optional): The amplitude of the periodic forcing in the equation. Default is 1.

    Returns:
    - F(T) (float): The calculated rate of temperature change (F(T)) at the specified temperature (T).
    """

    beta = -((surface_heat_capacity / (relaxation_time * emitted_radiation(stable_solution_2, model))) *
         ((stable_solution_1 * unstable_solution * stable_solution_2) /
          ((stable_solution_1 - stable_solution_2) * (unstable_solution - stable_solution_2))))

    F_value = (emitted_radiation(Temperature, model) / surface_heat_capacity) * (
        (periodic_forcing_value / (1 + beta * (1 - (Temperature / stable_solution_1)) *
        (1 - (Temperature / unstable_solution)) * (1 - (Temperature / stable_solution_2)))) - 1)

    return F_value

def emission_models_comparison(*temperatures):
    """
    Compare emitted radiation and rate of temperature change for 'linear' and 'black body' emission models.

    Given a list of temperatures, this function calculates and compares the emitted radiation and the rate of temperature
    change ('F(T)') for both the 'linear' and 'black body' emission models over a temperature range that spans 10 degrees
    below the minimum temperature in the input list and 10 degrees above the maximum temperature.

    Parameters:
    - *temperatures (float): A variable number of temperatures at which to compare the emission models.

    Returns:
    - T (array): An array of temperatures over the specified temperature range.
    - emitted_radiation_values (array): An array of emitted radiation values for both 'linear' and 'black body' models
      corresponding to the temperature range.
    - F_values (array): An array of rate of temperature change ('F(T)') values for both 'linear' and 'black body' models
      corresponding to the temperature range.
    """

    T_start = min(temperatures) - 10
    T_end = max(temperatures) + 10

    num_points = int((T_end - T_start))*10

    T = np.linspace(T_start, T_end, num = num_points)
    emitted_radiation_linear_values = [emitted_radiation(Ti, model = 'linear') for Ti in T]
    emitted_radiation_black_body_values = [emitted_radiation(Ti, model = 'black body') for Ti in T]

    emitted_radiation_values = np.array([emitted_radiation_linear_values, emitted_radiation_black_body_values])

    F_linear_values = [F(Ti, model = 'linear') for Ti in T]
    F_black_body_values = [F(Ti, model = 'black body') for Ti in T]

    F_values = np.array([F_linear_values, F_black_body_values])

    return T, emitted_radiation_values, F_values

def simulate_ito(
    T_start,
    t_start=0,
    noise_variance=0,
    dt=time_step_default,
    num_steps=num_steps_default,
    num_simulations=num_simulations_default,
    surface_heat_capacity=C_years,
    relaxation_time=tau,
    stable_solution_1=T1,
    unstable_solution=T2,
    stable_solution_2=T3,
    forcing_amplitude=A,
    forcing_angular_frequency=omega,
    noise=True,
    model=emission_model,
    forcing='varying'
):
    """
    Simulate the Ito stochastic differential equation for temperature.

    This function simulates the Ito stochastic differential equation for temperature given initial conditions and various parameters.

    Parameters:
    - T_start (float): The initial temperature in Kelvin at t_start.
    - t_start (float, optional): The starting time. Default is 0.
    - noise_variance (float, optional): The variance of the noise term. Default is 0.
    - dt (float, optional): The time step for the simulation. Default is the value of 'time_step_default', read from the
      configuration file.
    - num_steps (int, optional): The number of time steps in the simulation. Default is the value of 'num_steps_default, read
      from the configuration file'.
    - num_simulations (int, optional): The number of independent simulations to run. Default is the value of
      'num_simulations_default', read from the configuration file.
    - surface_heat_capacity (float, optional): The surface heat capacity in kg/(K*year^2). If not provided, the default value
      is read from the configuration file.
    - relaxation_time (float, optional): The relaxation time in seconds. If not provided, the default value is computed using
      the one read from the configuration file.
    - stable_solution_1 (float, optional): The fist stable temperature solution in Kelvin. If not provided, the default value
      is read from the configuration file.
    - unstable_solution (float, optional): The unstable temperature solution in Kelvin. If not provided, the default value is
      read from the configuration file.
    - stable_solution_2 (float, optional): The stable second temperature solution in Kelvin. If not provided, the default value
      is read from the configuration file.
    - forcing_amplitude (float, optional): The amplitude of the periodic forcing. Default is the value of 'A', read from the
      configuration file.
    - forcing_angular_frequency (float, optional): The angular frequency of the periodic forcing. Default is 'omega', computed
      using the value of the 'period' read from the configuration file.
    - noise (bool, optional): Whether to include noise in the simulation. Default is True.
    - model (str, optional): The emission model to use for temperature change computation. Default is the value of
      'emission_model', read from the confuguration file.
    - forcing (str, optional): The type of forcing applied to the system. Use 'constant' for constant forcing or 'varying'
      for periodic forcing. Default is 'varying'.

    Returns:
    - t (array): An array of time values.
    - T (array): An array of temperature values for each simulation.
    """
    seed_value = 42
    np.random.seed(seed_value)
    sigma = np.sqrt(noise_variance)
    t = np.arange(t_start, t_start + num_steps * dt, dt)  # len(t) = num_steps
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
        forcing_values = periodic_forcing(t, forcing_amplitude, forcing_angular_frequency)
    else:
        raise ValueError("Invalid value for 'forcing'. Please use 'constant' or 'varying'")
    for i in aes.progress(range(num_steps - 1)):
        Fi = F(
            T[:, i],
            surface_heat_capacity,
            relaxation_time,
            stable_solution_1,
            unstable_solution,
            stable_solution_2,
            model,
            forcing_values[i]
        )
        dT = Fi * dt + sigma * W[:, i]
        T[:, i + 1] = T[:, i] + dT
    with aes.green_text():
        print('finished!')
    return t, T

def calculate_evolution_towards_steady_states(temperatures):
    """
    Calculate the temperature evolution towards steady states.

    This function calculates the temperature evolution towards steady states for a given set of initial temperatures. It
    simulates the Ito stochastic differential equation with constant forcing and no noise.

    Parameters:
    - temperatures (array-like): An array of initial temperatures representing stable and unstable solutions.

    Returns:
    - time (array): An array of time values.
    - temperature (array): An array of temperature values for each simulation.

    Note:
    - The function uses the simulate_ito function with constant forcing and no noise to calculate the temperature
      evolution towards steady states. The input temperatures should include the two stable solutions and one unstable
      solution. The initial temperatures for simulation are generated by adding and subtracting epsilon (1.75K) from the
      input temperatures.
    """

    epsilon = 1.75
    T_start = np.array([temp - epsilon for temp in temperatures] + temperatures + [temp + epsilon for temp in temperatures])
    t_start = 0

    time, temperature = simulate_ito(T_start, t_start, num_steps = 100, num_simulations = len(T_start), noise = False, forcing = 'constant')

    return time, temperature

def find_peak_indices(frequencies, period = period):
    """
    Find indices of theoretically predicted peaks in a frequency spectrum.

    This function calculates the indices of the theoretically predicted peaks in a frequency spectrum based on the
    specified period.

    Parameters:
    - frequencies (array-like): An array of frequencies.
    - period (float): The period used for peak prediction. Default is the period of the periodic forcing, read
      from the configuration file.

    Returns:
    - peaks_indices (array): An array of indices corresponding to the closest frequencies to the predicted peak
      frequencies based on the given period.
    """
    peaks_indices = np.abs(frequencies - (1/period)).argmin(axis = 1)
    return peaks_indices

def calculate_peaks(frequencies, PSD_mean, peaks_indices):
    """
    Calculate the values of peaks in a power spectral density.

    This function calculates the values of peaks in a power spectral density (PSD) based on the provided frequencies,
    PSD_mean, and peak indices.

    Parameters:
    - frequencies (array-like): An array of frequencies.
    - PSD_mean (array-like): An array of mean power spectral density values.
    - peaks_indices (array-like): An array of indices representing the positions of the peaks.

    Returns:
    - peaks (array): An array of peak values corresponding to the provided peak indices.
    """

    peaks = PSD_mean[np.arange(frequencies.shape[0]), peaks_indices]
    return peaks

def calculate_peaks_base(frequencies, PSD_mean, peaks_indices, num_neighbors = 2):
    """
    Calculate the values of the base of the peaks in a power spectral density.

    This function calculates the values of the base of the peaks in a power spectral density (PSD) based on the provided
    frequencies, PSD_mean, peak indices, and the number of neighboring points to consider for the base calculation.

    Parameters:
    - frequencies (array-like): An array of frequencies.
    - PSD_mean (array-like): An array of mean power spectral density values.
    - peaks_indices (array-like): An array of indices representing the positions of the peaks.
    - num_neighbors (int, optional): The number of neighboring points on each side of a peak to consider for calculating
      the peak's base. Default is 2.

    Returns:
    - peaks_base (array): An array of base values corresponding to the provided peak indices.
    """

    peaks_base = np.empty_like(peaks_indices, dtype = float)

    for i in range(frequencies.shape[0]):
        current_indices = peaks_indices[i]
        neighbor_indices = np.arange(current_indices - num_neighbors, current_indices + num_neighbors + 1)
        valid_indices = np.clip(neighbor_indices, 0, PSD_mean.shape[1]-1)
        neighbor_values = PSD_mean[i, valid_indices]
        peaks_base[i] = np.mean(neighbor_values)

    return peaks_base

def calculate_peak_height(peaks, peaks_base):
    """
    Calculate the heights of peaks in a power spectral density.

    This function calculates the heights of peaks in a power spectral density (PSD) based on the provided peak values and
    peak base values.

    Parameters:
    - peaks (array-like): An array of peak values.
    - peaks_base (array-like): An array of base values corresponding to the peaks.

    Returns:
    - peak_height (array): An array of peak heights calculated as the difference between peak values and peak base
      values.
    """
    peak_height = peaks - peaks_base
    return peak_height

def simulate_ito_combinations_and_collect_results(initial_temperature, noise_variance):
    """
    Simulate the Ito stochastic differential equation for multiple combinations of noise and forcing types.

    This function simulates the Ito stochastic differential equation for temperature for various combinations of noise and
    forcing types. It collects the simulation results for each combination.

    Parameters:
    - initial_temperature (float): The initial temperature for the simulations.
    - noise_variance (float): The variance of the noise term for simulations.

    Returns:
    - times (list of arrays): A list of arrays containing time values for each simulation.
    - temperatures (list of arrays): A list of arrays containing temperature values for each simulation.
    """
    times = []
    temperatures = []

    for noise_status, forcing_type in [(False, 'constant'), (True, 'constant'), (False, 'varying'), (True, 'varying')]:
        time, temperature = simulate_ito(T_start= initial_temperature,
					 noise_variance=noise_variance if noise_status else 0, num_simulations=1,
					 noise=noise_status, forcing=forcing_type)
        times.append(time)
        temperatures.append(temperature)

    return times, temperatures
