import configparser
import os
import numpy as np
from scipy import signal

import stochastic_resonance
import aesthetics as aes

#---------------------------------------------------------------------

config = configparser.ConfigParser()
config.read('configuration.txt')

stable_solution_1 = config['settings'].getfloat('T1')
unstable_solution = config['settings'].getfloat('T2')
stable_solution_2 = config['settings'].getfloat('T3')

temperature_solutions = [stable_solution_1, unstable_solution, stable_solution_2]

C_j_per_m2_K = config['settings'].getfloat('C')
tau = config['settings'].getfloat('tau')
A = config['settings'].getfloat('A')
period = config['settings'].getfloat('period')

num_sec_in_a_year = 365.25*24*60*60

C_years = C_j_per_m2_K*np.power(num_sec_in_a_year,2)
omega = (2*np.pi)/period

num_steps = config['settings'].getint('num_steps')
num_simulations = config['settings'].getint('num_simulations')
dt = config['settings'].getfloat('time_step')

variance_start = config['settings'].getfloat('variance_start')
variance_end = config['settings'].getfloat('variance_end')
num_variances = config['settings'].getint('num_variances')

os.makedirs('data', exist_ok = True)

models_comparison_temperatures_destination = config['paths'].get('models_comparison_temperatures_destination')
emitted_radiation_values_destination = config['paths'].get('emitted_radiation_values_destination')
F_values_destination = config['paths'].get('F_values_destination')
evolution_towards_steady_states_time_destination = config['paths'].get('evolution_towards_steady_states_time_destination')
evolution_towards_steady_states_temperature_destination = config['paths'].get('evolution_towards_steady_states_temperature_destination')

time_destination = config['paths'].get('simulated_time')
temperature_destination = config['paths'].get('simulated_temperature')

frequencies_destination = config['paths'].get('simulated_frequencies')
PSD_destination = config['paths'].get('simulated_PSD')

peak_height_destination = config['paths'].get('simulated_peak_height')


#------Generation of data for the emission models comparison----------------------------------

models_comparison_temperatures, emitted_radiation_values, F_values = stochastic_resonance.emission_models_comparison(*temperature_solutions)

np.save(models_comparison_temperatures_destination, models_comparison_temperatures)
np.save(emitted_radiation_values_destination, emitted_radiation_values)
np.save(F_values_destination, F_values)

#----Generation of data for showing the evolution of temperature towards steady states----------

evolution_towards_steady_states_time, evolution_towards_steady_states_temperature = stochastic_resonance.calculate_evolution_towards_steady_states(temperature_solutions)

np.save(evolution_towards_steady_states_time_destination, evolution_towards_steady_states_time)
np.save(evolution_towards_steady_states_temperature_destination, evolution_towards_steady_states_temperature)

#-----------------------------------------------------------------------

V = np.linspace(variance_start, variance_end, num_variances)

Time = np.zeros((len(V), num_steps))
Temperature = np.zeros((len(V), num_simulations, num_steps))

for i, v in enumerate(V):
    print('\nSimulating the temperature evolution:  {0} of {1}  '.format((i+1), len(V)))
    time, simulated_temperature = stochastic_resonance.simulate_ito(T_start = stable_solution_2, t_start = 0,
					       			    noise_variance = v)
    Temperature[i, :, :] = simulated_temperature
    Time[i, :] = time

#----------------------------------------------------------------------

np.save(temperature_destination, Temperature)
np.save(time_destination, Time)

with aes.green_text():
    print('Data saved successfully!')

#-----------------------------------------------------------------------

Frequencies = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))
PSD_mean = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))

print('Computing the power spectra:')

for i in aes.progress(range(len(V))):
    psd = np.zeros((num_simulations, np.floor_divide(num_steps, 2) + 1))
    for j in range(num_simulations):
        frequencies, power_spectrum = signal.periodogram(Temperature[i, j, :], 1/dt)
        psd[j, :] = power_spectrum
    PSD_mean[i, :] = np.mean(psd, axis = 0)
    Frequencies[i, :] = frequencies

#-----------------------------------------------------------------------

np.save(frequencies_destination, Frequencies)
np.save(PSD_destination, PSD_mean)

with aes.green_text():
    print('Data saved successfully!')

#-------------------------------------------------------------------------

peaks_indices = stochastic_resonance.find_peak_indices(Frequencies, period)
peaks = stochastic_resonance.calculate_peaks(Frequencies, PSD_mean, peaks_indices)
peaks_base = stochastic_resonance.calculate_peaks_base(Frequencies, PSD_mean, peaks_indices)
peaks_height = stochastic_resonance.calculate_peak_height(peaks, peaks_base)

np.save(peak_height_destination, peaks_height)
