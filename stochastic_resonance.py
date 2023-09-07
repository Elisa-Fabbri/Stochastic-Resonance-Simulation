import numpy as np
import configparser
import aesthetics as aes

config = configparser.ConfigParser()
config.read('configuration.txt')

T1 = config['settings'].getfloat('T1')
T2 = config['settings'].getfloat('T2')
T3 = config['settings'].getfloat('T3')

C_j_per_m2_K = config['settings'].getfloat('C')
tau = config['settings'].getfloat('tau')
A = config['settings'].getfloat('A')
period = config['settings'].getfloat('period')

num_sec_in_a_year = 365.25*24*60*60
C_years = C_j_per_m2_K*np.power(num_sec_in_a_year,2)
omega = (2*np.pi)/period

num_steps_default = config['settings'].getint('num_steps')
num_simulations_default = config['settings'].getint('num_simulations')

def emitted_radiation(Temperature, model='linear', conversion_from_sec_to_year = num_sec_in_a_year):
    """
    This function computes the emitted radiation based on the selected model.
    """

    if model == 'linear':
        A = -339.647 * (conversion_from_sec_to_year ** 3)  # -339.647 W/m^2
        B = 2.218 * (conversion_from_sec_to_year ** 3)  # 2.218 W/(m^2 K)
        emitted_radiation = A + B * Temperature
    elif model == 'black body':
        Stefan_Boltzmann_constant = 5.67e-8 * (conversion_from_sec_to_year ** 3)  # W/(m^2 K^4)
        emitted_radiation = Stefan_Boltzmann_constant * (Temperature ** 4)
    else:
        raise ValueError("Invalid model selection. Choose 'linear' or 'black body'.")

    return emitted_radiation


def periodic_forcing(time, amplitude = A, angular_frequency = omega):
    """
    This function computes the periodic forcing applied to the system
    """
    return 1+amplitude*np.cos(angular_frequency*time)


def F(Temperature,
      surface_heat_capacity = C_years,
      relaxation_time = tau,
      stable_solution_1 = T1, unstable_solution = T2, stable_solution_2 = T3,
      model='linear',
      mu=1):

    """
    This function returns the value of F(T) given a certain value for the temperature using a specified emission model for the computation of beta.
    """

    beta = -((surface_heat_capacity / (relaxation_time * emitted_radiation(stable_solution_2, model))) * ((stable_solution_1 * unstable_solution * stable_solution_2) / ((stable_solution_1 - stable_solution_2) * (unstable_solution - stable_solution_2))))
    F_value = (emitted_radiation(Temperature, model) / surface_heat_capacity) * ((mu / (1 + beta * (1 - (Temperature / stable_solution_1)) * (1 - (Temperature / unstable_solution)) * (1 - (Temperature / stable_solution_2)))) - 1)

    return F_value

def emission_models_comparison(*temperatures):

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

def simulate_ito(T_start, t_start,
		 noise_variance = 0,
		 dt = 1,
		 num_steps = num_steps_default,
		 num_simulations = num_simulations_default,
		 surface_heat_capacity = C_years,
		 relaxation_time = tau,
                 stable_solution_1 = T1, unstable_solution = T2, stable_solution_2 = T3,
                 forcing_amplitude = A, forcing_angular_frequency = omega,
                 noise = True,
                 model = 'linear',
                 forcing = 'varying'):

    seed_value = 42
    np.random.seed(seed_value)

    sigma = np.sqrt(noise_variance)

    t = np.arange(t_start, t_start+num_steps*dt, dt) #len(t) = num_steps
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


    for i in aes.progress(range(num_steps-1)):
        Fi = F(T[:, i], surface_heat_capacity, relaxation_time, stable_solution_1, unstable_solution, stable_solution_2, model, forcing_values[i])
        dT = Fi*dt + sigma*W[:, i]
        T[:, i+1] = T[:, i] + dT
    with aes.green_text():
        print('finished!')

    return t, T

def calculate_evolution_towards_steady_states(temperatures):
    epsilon = 1.75
    T_start = np.array([temp - epsilon for temp in temperatures] + temperatures + [temp + epsilon for temp in temperatures])
    t_start = 0

    time, temperature = simulate_ito(T_start, t_start, num_steps = 100, num_simulations = len(T_start), noise = False, forcing = 'constant')

    return time, temperature

def find_peak_indices(frequencies, period):
    """
    This function returns the indices of the theoretically predicted peaks.
    """
    peaks_indices = np.abs(frequencies - (1/period)).argmin(axis = 1)
    return peaks_indices

def calculate_peaks(frequencies, PSD_mean, peaks_indices):
    """
    This function returns the values of the peaks
    """
    peaks = PSD_mean[np.arange(frequencies.shape[0]), peaks_indices]
    return peaks

def calculate_peaks_base(frequencies, PSD_mean, peaks_indices, num_neighbors = 2):
    """
    This function calculates the value of the base of the peaks.
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
    This function calculates the height of the peak in the power spectrum.
    """
    peak_height = peaks - peaks_base
    return peak_height

def simulate_ito_combinations_and_collect_results(initial_temperature, noise_variance):

    times = []
    temperatures = []

    for noise_status, forcing_type in [(False, 'constant'), (True, 'constant'), (False, 'varying'), (True, 'varying')]:
        time, temperature = simulate_ito(T_start= initial_temperature, t_start=0,
					 noise_variance=noise_variance if noise_status else 0, num_simulations=1,
					 noise=noise_status, forcing=forcing_type)
        times.append(time)
        temperatures.append(temperature)

    return times, temperatures
