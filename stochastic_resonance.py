"""
This module contains the definition of the functions used in the stochastic resonance project.

In particular, it contains the following functions:
- emitted_radiation: Calculate the emitted radiation from the Earth's surface based on the 
  chosen emission model.
- periodic_forcing: Calculate the periodic forcing applied to the system.
- calculate_rate_of_temperature_change: Calculate the rate of temperature change (dT/dt) 
  given a certain temperature (T)
  using a specified emission model for the computation of the constant beta.
- emission_models_comparison: Compare emitted radiation and rate of temperature change for 'linear' and 'black body'
  emission models.
- simulate_ito: Simulate temperature using the Itô stochastic differential equation.
- calculate_evolution_towards_steady_states: Calculate temperature evolution towards steady states.
- find_peak_indices: Find indices of theoretically predicted peaks in a frequency spectrum.
- calculate_peaks: Calculate the values of peaks in a power spectral density.
- calculate_peaks_base: Calculate the values of the base of the peaks in a power spectral density.
- calculate_peak_height: Calculate the heights of peaks in a power spectral density.
- simulate_ito_combinations_and_collect_results: Simulate temperature with various combinations of 
  noise and forcing.
"""

import numpy as np
import aesthetics as aes


#Default values for the functions:

# Stable and unstable temperatures solutions in Kelvin:
stable_temperature_solution_1_default = 280
unstable_temperature_solution_default = 285
stable_temperature_solution_2_default = 290

surface_earth_thermal_capacity_j_per_m2_K = 0.31e9 #average earth surface thermal capacity [J/(m^2 K)]
relaxation_time_default = 13 #in years
emission_model_default = 'linear'

#periodic forcing constants:
forcing_amplitude_default = 0.0005
forcing_period_default = 1e5
forcing_angular_frequency_default = (2*np.pi)/forcing_period_default #computes the angular frequency 
                                                                     #of the periodic forcing

num_sec_in_a_year = 365.25*24*60*60
#the following line of code converts the earth thermal capacity from J/(m^2 K) to kg/(year^2 K)
surface_earth_thermal_capacity_in_years = surface_earth_thermal_capacity_j_per_m2_K*(num_sec_in_a_year**2)

#parameters for the simulation (used in the simulate ito function):
time_step_default = 1 # year
num_steps_default = 1000000 # years
num_simulations_default = 10

def emitted_radiation(temperature,
                      emission_model=emission_model_default,
                      conversion_factor=num_sec_in_a_year):
    """
    Calculate the emitted radiation from the Earth's surface based on the chosen emission model.

    This function calculates the radiation emitted from the Earth's surface based on the 
    specified emission model.
    The emission model can be either 'linear' or 'black body'. 
    The emitted radiation is computed using the given temperature and conversion factor.

    Parameters:
    - temperature (float): The temperature of the Earth's surface in Kelvin. It must be a non-negative value.
    - emission_model (str, optional): The emission model to use, which can be 'linear' or 'black body'.
      Default is 'linear'.
    - conversion_factor (float, optional): The conversion factor for time units to calculate the 
      emitted radiation, measured in kg/year^3. 
      The default is set to the number of seconds in a year.

    Returns:
    - emitted_radiation (float): The calculated emitted radiation in kg/year^3.

    Raises:
    - ValueError: If an invalid emission model is selected. Valid options are 'linear' or 'black body'.
    - ValueError: If the temperature is not a non-negative value.

    Notes:
    For an explanation on the linear model and on the values of the parameters A and B,
    see `<https://www.pnas.org/doi/10.1073/pnas.1809868115>`.
    """

    if np.any(temperature < 0):
        raise ValueError('Temperature in Kelvin must be non-negative.')
    
    if emission_model == 'linear':
        A = -339.647 * (conversion_factor ** 3)  # converts from W/(m^2 K) to kg/year^3 K
        B = 2.218 * (conversion_factor ** 3)  # converts from W/(m^2 K) to kg/year^3 K
        emitted_radiation = A + B * temperature
    elif emission_model == 'black body':
        Stefan_Boltzmann_constant = 5.67e-8 * (conversion_factor ** 3)  # W/(m^2 K^4)
        emitted_radiation = Stefan_Boltzmann_constant * (temperature ** 4)
    else:
        raise ValueError('Invalid emission model selection. Choose "linear" or "black body".')

    return emitted_radiation

def periodic_forcing(time,
                     amplitude = forcing_amplitude_default,
                     angular_frequency = forcing_angular_frequency_default):
    """
    Calculate the periodic forcing applied to the system.

    This function calculates the periodic forcing applied to a system at a given time or times. 
    The periodic forcing is modeled as an oscillatory function of time.

    Parameters:
    - time (float or array-like): The time or times at which to calculate the periodic forcing.
    - amplitude (float, optional): The amplitude of the periodic forcing. Default is 0.0005.
    - angular_frequency (float, optional): The angular frequency of the periodic forcing. 
      Default is (2 * pi) / 1e5.

    Returns:
    - periodic_forcing (float or array-like): The calculated periodic forcing values corresponding to 
      the input time(s).

    Raises:
    - ValueError: If the time is a negative value.
    """
    if np.any(np.array(time) < 0):
        raise ValueError('Time must be non-negative.')
    
    periodic_forcing = np.array(amplitude) * np.cos(np.array(angular_frequency) * time)
    return 1 + periodic_forcing

def calculate_rate_of_temperature_change(temperature,
      					 surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
      					 relaxation_time = relaxation_time_default,
      					 stable_temperature_solution_1 = stable_temperature_solution_1_default,
      					 unstable_temperature_solution = unstable_temperature_solution_default,
      					 stable_temperature_solution_2 = stable_temperature_solution_2_default,
      					 emission_model = emission_model_default,
      					 periodic_forcing_value = 1):

    """
    Calculate the rate of temperature change (dT/dt) given a certain temperature (T) using a 
    specified emission model for the computation of the constant beta.

    This function calculates the rate of temperature change (dT/dt) at a specified temperature (T) 
    using a specified emission model for the computation of the constant beta.

    Parameters:
    - temperature (float): The temperature in Kelvin at which to calculate the rate of 
      temperature change (dT/dt).
    - surface_thermal_capacity (float, optional): The surface thermal capacity in kg per square year per Kelvin. 
      Default corresponds to 0.31e9 joules per square meter per Kelvin.
    - relaxation_time (float, optional): The relaxation time in years. Default is 13.
    - stable_temperature_solution_1 (float, optional): The first stable temperature solution in Kelvin. 
      Default is 280.
    - unstable_temperature_solution (float, optional): The unstable temperature solution in Kelvin. 
      Default is 285.
    - stable_temperature_solution_2 (float, optional): The second stable temperature solution in Kelvin. 
      Default is 290.
    - emission_model (str, optional): The emission model to use for beta computation. 
      Default is 'linear'.
    - periodic_forcing_value (float, optional): The amplitude of the periodic forcing in the equation. 
      Default is 1.

    Returns:
    - dT/dt (float): The calculated rate of temperature change (dT/dt) at the specified temperature (T).
    """

    beta = -((surface_thermal_capacity / (relaxation_time * emitted_radiation(
        temperature=stable_temperature_solution_2,
        emission_model=emission_model))) *
        ((stable_temperature_solution_1 * unstable_temperature_solution * stable_temperature_solution_2) /
        ((stable_temperature_solution_1 - stable_temperature_solution_2) *
         (unstable_temperature_solution - stable_temperature_solution_2))))

    dT_dt = (emitted_radiation(temperature=temperature, emission_model=emission_model) /
        surface_thermal_capacity) * (
       (periodic_forcing_value /
        (1 + beta * (1 - (temperature / stable_temperature_solution_1)) *
         (1 - (temperature / unstable_temperature_solution)) *
         (1 - (temperature / stable_temperature_solution_2)))) - 1)
    
    return dT_dt



def emission_models_comparison(surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
			       relaxation_time = relaxation_time_default,
			       stable_temperature_solution_1 = stable_temperature_solution_1_default,
			       unstable_temperature_solution = unstable_temperature_solution_default,
			       stable_temperature_solution_2 = stable_temperature_solution_2_default):
    """
    Compare emitted radiation and rate of temperature change for 'linear' and 'black body' emission models.

    This function calculates and compares the emitted radiation and the rates of temperature change (dT_dt)
    for both the 'linear' and 'black body' emission models over a temperature range that spans 10 degrees 
    below the minimum temperature (stable_temperature_solution_1) and 10 degrees above the maximum temperature
    (stable_temperature_solution_2).

    Parameters:
    - surface_thermal_capacity (float): The thermal capacity of the Earth's surface in kg/(year^2 K).
    Default is the average Earth surface thermal capacity in kg/(years^2 K), 
    which corresponds to 0.31e9 J/(m^2 K).
    - relaxation_time (int): The relaxation time in years. Default is 13 years.
    - stable_temperature_solution_1 (float): The stable temperature solution 1 in Kelvin.
    Default is 280 K.
    - unstable_temperature_solution (float): The unstable temperature solution in Kelvin.
    Default is 285 K.
    - stable_temperature_solution_2 (float): The stable temperature solution 2 in Kelvin.
    Default is 290 K.

    Returns:
    - T (numpy.ndarray): An array of temperature values over the specified range.
    - emitted_radiation_values (numpy.ndarray): An array containing emitted radiation values
    for 'linear' and 'black body' emission models.
    - dT_dt_values (numpy.ndarray): An array containing temperature change rate values
    for 'linear' and 'black body' emission models.
    """

    T_start = stable_temperature_solution_1 - 10
    T_end = stable_temperature_solution_2 + 10

    num_points = int((T_end - T_start))*10

    T = np.linspace(T_start, T_end, num = num_points)
    emitted_radiation_linear_values = [emitted_radiation(temperature = Ti, 
                                                         emission_model = 'linear') for Ti in T]
    emitted_radiation_black_body_values = [emitted_radiation(temperature = Ti, 
                                                             emission_model = 'black body') for Ti in T]

    emitted_radiation_values = np.array([emitted_radiation_linear_values, 
                                         emitted_radiation_black_body_values])

    dT_dt_linear_values = [calculate_rate_of_temperature_change(temperature = Ti,
			 				    surface_thermal_capacity = surface_thermal_capacity,
			 				    relaxation_time = relaxation_time,
			 				    stable_temperature_solution_1 = stable_temperature_solution_1,
			 				    unstable_temperature_solution = unstable_temperature_solution,
			 				    stable_temperature_solution_2 = stable_temperature_solution_2,
			 				    emission_model = 'linear'
			 				    ) for Ti in T]
    dT_dt_black_body_values = [calculate_rate_of_temperature_change(temperature = Ti,
			     					surface_thermal_capacity = surface_thermal_capacity,
			     					relaxation_time = relaxation_time,
			     					stable_temperature_solution_1 = stable_temperature_solution_1,
			     					unstable_temperature_solution = unstable_temperature_solution,
			     					stable_temperature_solution_2 = stable_temperature_solution_2,
			     					emission_model = 'black body'
			     					) for Ti in T]

    dT_dt_values = np.array([dT_dt_linear_values, dT_dt_black_body_values])

    return T, emitted_radiation_values, dT_dt_values

def simulate_ito(
    T_start = stable_temperature_solution_2_default,
    t_start=0,
    noise_variance = 0,
    dt = time_step_default,
    num_steps = num_steps_default,
    num_simulations = num_simulations_default,
    surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
    relaxation_time = relaxation_time_default,
    stable_temperature_solution_1 = stable_temperature_solution_1_default,
    unstable_temperature_solution = unstable_temperature_solution_default,
    stable_temperature_solution_2 = stable_temperature_solution_2_default,
    forcing_amplitude = forcing_amplitude_default,
    forcing_angular_frequency = forcing_angular_frequency_default,
    noise = True,
    emission_model = emission_model_default,
    forcing = 'varying',
    seed_value = 0
    ):

    """
    Simulate temperature using the Itô stochastic differential equation.

    This function simulates temperature evolution using the Itô stochastic differential equation (SDE).
    It allows for the modeling of temperature dynamics with various parameters, noise, and forcing.

    Parameters:
    - T_start (float): The initial temperature value at t_start. Default is 290.
    - t_start (float): The starting time of the simulation. Default is 0.
    - noise_variance (float): The variance of the noise. Default is 0 (no noise).
    - dt (float): The time step size for the simulation in years. Default is 1 year.
    - num_steps (int): The number of time steps in the simulation. Default is 1000000.
    - num_simulations (int): The number of simulation runs. Default is 10.
    - surface_thermal_capacity (float): The thermal capacity of the Earth's surface in kg/(year^2 K).
    Default is the average Earth surface thermal capacity in kg/(year^2 K), 
    which corresponds to 0.31e9 J/(m^2 K).
    - relaxation_time (int): The relaxation time in years. Default is 13 years.
    - stable_temperature_solution_1 (float): The stable temperature solution 1 in Kelvin.
    Default is 280 K.
    - unstable_temperature_solution (float): The unstable temperature solution in Kelvin.
    Default is 285 K.
    - stable_temperature_solution_2 (float): The stable temperature solution 2 in Kelvin.
    Default is 290 K.
    - forcing_amplitude (float): The amplitude of periodic forcing. Default is 0.0005.
    - forcing_angular_frequency (float): The angular frequency of periodic forcing.
    Default is (2 * pi) / 1e5.
    - noise (bool): A flag indicating whether to include noise in the simulation. Default is True.
    - emission_model (str): The emission model to use ('linear' or 'black body'). Default is 'linear'.
    - forcing (str): The type of forcing to apply ('constant' or 'varying'). Default is 'varying'.
    - seed_value (int): The seed value for the random number generator. Default is 0.

    Returns:
    - t (numpy.ndarray): An array of time values for the simulation.
    - T (numpy.ndarray): An array of temperature values for each simulation run and time step.
    """
    
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
        forcing_values = periodic_forcing(time = t, amplitude = forcing_amplitude, 
                                          angular_frequency = forcing_angular_frequency)
    else:
        raise ValueError("Invalid value for 'forcing'. Please use 'constant' or 'varying'")
    for i in aes.progress(range(num_steps - 1)):
        Fi = calculate_rate_of_temperature_change(
            temperature = T[:, i],
            surface_thermal_capacity = surface_thermal_capacity,
            relaxation_time = relaxation_time,
            stable_temperature_solution_1 = stable_temperature_solution_1,
            unstable_temperature_solution = unstable_temperature_solution,
            stable_temperature_solution_2 = stable_temperature_solution_2,
            emission_model = emission_model,
            periodic_forcing_value = forcing_values[i]
        )
        dT = Fi * dt + sigma * W[:, i]
        T[:, i + 1] = T[:, i] + dT
    with aes.green_text():
        print('finished!')
    return t, T

def calculate_evolution_towards_steady_states(
                          surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
					                relaxation_time = relaxation_time_default,
					                stable_temperature_solution_1 = stable_temperature_solution_1_default,
					                unstable_temperature_solution = unstable_temperature_solution_default,
					                stable_temperature_solution_2 = stable_temperature_solution_2_default,
					                forcing_amplitude = forcing_amplitude_default,
					                forcing_angular_frequency = forcing_angular_frequency_default,
					                emission_model = emission_model_default,
                          seed_value = 0
                          ):
    """
    Calculate temperature evolution towards steady states.

    This function calculates the temperature evolution towards steady states for a given set of initial
    temperatures by simulating temperature dynamics using the Itô stochastic differential equation (SDE)
    with constant forcing and no noise.

    Parameters:
    - surface_thermal_capacity (float): The thermal capacity of the Earth's surface in kg/(year^2 K).
    Default is the average Earth surface thermal capacity in kg/(year^2 K), 
    which corresponds to 0.31e9 J/(m^2 K).
    - relaxation_time (int): The relaxation time in years. Default is 13 years.
    - stable_temperature_solution_1 (float): The stable temperature solution 1 in Kelvin.
    Default is 280 K.
    - unstable_temperature_solution (float): The unstable temperature solution in Kelvin.
    Default is 285 K.
    - stable_temperature_solution_2 (float): The stable temperature solution 2 in Kelvin.
    Default is 290 K.
    - forcing_amplitude (float): The amplitude of periodic forcing. Default is 0.0005.
    - forcing_angular_frequency (float): The angular frequency of periodic forcing.
    Default is (2 * pi) / 1e5.
    - emission_model (str): The emission model to use ('linear' or 'black body'). Default is 'linear'.
    - seed_value (int): The seed value for the random number generator. Default is 0.

    Returns:
    - time (numpy.ndarray): An array of time values for the simulation.
    - temperature (numpy.ndarray): An array of temperature values for each simulation run and time step.

    Notes:
    This function initializes temperatures around the given stable and unstable solutions, simulates
    their evolution, and returns the resulting time series of temperatures. Noise is not included in this
    simulation, and a constant forcing is applied.
    """

    temperatures = [stable_temperature_solution_1, 
                    unstable_temperature_solution, 
                    stable_temperature_solution_2]

    epsilon = 1.75
    T_start = np.array([temp - epsilon for temp in temperatures] + 
                       temperatures + [temp + epsilon for temp in temperatures])

    time, temperature = simulate_ito(T_start = T_start,
				                             num_steps = 100,
				                             num_simulations = len(T_start),
				                             surface_thermal_capacity = surface_thermal_capacity,
				                             relaxation_time = relaxation_time,
				                             stable_temperature_solution_1 = stable_temperature_solution_1,
				                             unstable_temperature_solution = unstable_temperature_solution,
				                             stable_temperature_solution_2 = stable_temperature_solution_2,
				                             forcing_amplitude = forcing_amplitude,
				                             forcing_angular_frequency = forcing_angular_frequency,
				                             noise = False,
				                             emission_model = emission_model,
				                             forcing = 'constant',
                                     seed_value = seed_value)

    return time, temperature

def find_peak_indices(frequencies, period = forcing_period_default):
    """
    Find indices of theoretically predicted peaks in a frequency spectrum.

    This function calculates the indices of the theoretically predicted peaks in a frequency 
    spectrum based on the specified period.

    Parameters:
    - frequencies (array-like): An array of frequencies.
    - period (float): The period used for peak prediction which should be the period of the periodic 
    forcing applied to the system. Default is 1e5.

    Returns:
    - peaks_indices (array): An array of indices corresponding to the closest frequencies to the 
      predicted peak frequencies based on the given period.
    """
    peaks_indices = np.abs(frequencies - (1/period)).argmin(axis = 1)
    return peaks_indices

def calculate_peaks(PSD_mean, peaks_indices):
    """
    Calculate the values of peaks in a power spectral density.

    This function calculates the values of peaks in a power spectral density (PSD) based on the 
    provided PSD_mean and peak indices.

    Parameters:
    - PSD_mean (array-like): An array of mean power spectral density values.
    - peaks_indices (array-like): An array of indices representing the positions of the peaks.

    Returns:
    - peaks (array): An array of peak values corresponding to the provided peak indices.
    """

    peaks = PSD_mean[np.arange(PSD_mean.shape[0]), peaks_indices]
    return peaks

def calculate_peaks_base(PSD_mean, peaks_indices, num_neighbors=2):
    """
    Calculate the values of the base of the peaks in a power spectral density.

    This function calculates the values of the base of the peaks in a power spectral density (PSD) based on 
    the provided PSD_mean, peak indices, and the number of neighboring points to consider for 
    the base calculation.

    Parameters:
    - PSD_mean (array-like): An array of mean power spectral density values.
    - peaks_indices (array-like): An array of indices representing the positions of the peaks.
    - num_neighbors (int, optional): The number of neighboring points on each side of a peak to consider 
      for calculating the peak's base. Default is 2.

    Returns:
    - peaks_base (array): An array of base values corresponding to the provided peak indices.
    """

    peaks_base = np.empty_like(peaks_indices, dtype=float)

    for i in range(PSD_mean.shape[0]):
        current_indices = peaks_indices[i]
        neighbor_indices = np.arange(current_indices - num_neighbors, 
                                     current_indices + num_neighbors + 1)
        valid_indices = np.clip(neighbor_indices, 0, PSD_mean.shape[1] - 1)
        valid_indices = valid_indices[valid_indices != current_indices]
        valid_indices = np.unique(valid_indices)
        neighbor_values = PSD_mean[i, valid_indices]
        peaks_base[i] = np.mean(neighbor_values)
    return peaks_base


def calculate_peak_height(peaks, peaks_base):
    """
    Calculate the heights of peaks in a power spectral density.

    This function calculates the heights of peaks in a power spectral density (PSD) based on the provided 
    peak values and peak base values.

    Parameters:
    - peaks (array-like): An array of peak values.
    - peaks_base (array-like): An array of base values corresponding to the peaks.

    Returns:
    - peak_height (array): An array of peak heights calculated as the difference between peak values and 
    peak base values.
    """
    peak_height = peaks - peaks_base
    return peak_height

def simulate_ito_combinations_and_collect_results(
              T_start = stable_temperature_solution_2_default,
						  noise_variance = 0,
						  dt = time_step_default,
						  num_steps = num_steps_default,
						  surface_thermal_capacity = surface_earth_thermal_capacity_in_years,
						  relaxation_time = relaxation_time_default,
						  stable_temperature_solution_1 = stable_temperature_solution_1_default,
						  unstable_temperature_solution = unstable_temperature_solution_default,
						  stable_temperature_solution_2 = stable_temperature_solution_2_default,
						  forcing_amplitude = forcing_amplitude_default,
						  forcing_angular_frequency = forcing_angular_frequency_default,
						  emission_model = emission_model_default,
              seed_value = 0):
    """
    Simulate temperature with various combinations of noise and forcing.

    This function simulates temperature evolution with different combinations of noise and forcing types
    using the Itô stochastic differential equation (SDE). It collects results for each combination.

    Parameters:
    - T_start (float): The initial temperature value at the start of the simulation.
    Default is 290.
    - noise_variance (float): The variance of the noise. Default is 0 (no noise).
    - dt (float): The time step size for the simulation in years. Default is 1 year.
    - num_steps (int): The number of time steps in the simulation. Default is 1000000.
    - surface_thermal_capacity (float): The thermal capacity of the Earth's surface in kg/(year^2 K).
    Default is the average Earth surface thermal capacity in kg/(m^2 K), 
    which corresponds to 0.31e9 J/(m^2 K).
    - relaxation_time (int): The relaxation time in years. Default is 13 years.
    - stable_temperature_solution_1 (float): The stable temperature solution 1 in Kelvin.
    Default is 280 K.
    - unstable_temperature_solution (float): The unstable temperature solution in Kelvin.
    Default is 285 K.
    - stable_temperature_solution_2 (float): The stable temperature solution 2 in Kelvin.
    Default is 290 K.
    - forcing_amplitude (float): The amplitude of periodic forcing. Default is 0.0005.
    - forcing_angular_frequency (float): The angular frequency of periodic forcing.
    Default is (2 * pi) / 1e5.
    - emission_model (str): The emission model to use ('linear' or 'black body'). Default is 'linear'.
    - seed_value (int): The seed value for the random number generator. Default is 0.

    Returns:
    - times (list): A list of arrays containing time values for each simulation.
    - temperatures (list): A list of arrays containing temperature values for each simulation.

    Notes:
    This function performs simulations for different combinations of noise and forcing types,
    collecting results for each combination. It can be used to explore various scenarios of temperature
    evolution with different parameters.
    """

    times = []
    temperatures = []

    for noise_status, forcing_type in [(False, 'constant'), (True, 'constant'), 
                                       (False, 'varying'), (True, 'varying')]:
      time, temperature = simulate_ito(T_start= T_start,
					                             noise_variance=noise_variance if noise_status else 0,
					                             dt = dt,
					                             num_steps = num_steps,
					                             num_simulations= 1,
					                             surface_thermal_capacity = surface_thermal_capacity,
					                             relaxation_time = relaxation_time,
					                             stable_temperature_solution_1 = stable_temperature_solution_1,
					                             unstable_temperature_solution = unstable_temperature_solution,
					                             stable_temperature_solution_2 = stable_temperature_solution_2,
					                             forcing_amplitude = forcing_amplitude,
					                             forcing_angular_frequency = forcing_angular_frequency,
					                             noise=noise_status,
					                             emission_model = emission_model,
					                             forcing=forcing_type,
                                       seed_value = seed_value)
      times.append(time)
      temperatures.append(temperature)

    return times, temperatures
