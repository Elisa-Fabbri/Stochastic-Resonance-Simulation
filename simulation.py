import configparser
import sys
from os import makedirs
import numpy as np
from scipy import signal

import stochastic_resonance
import aesthetics as aes

#---------------------------------------------------------------------

config = configparser.ConfigParser()
config.read(sys.argv[1])

stable_solution_1 = config['settings'].getfloat('stable_temperature_solution_1')
unstable_solution = config['settings'].getfloat('unstable_temperature_solution')
stable_solution_2 = config['settings'].getfloat('stable_temperature_solution_2')

temperature_solutions = [stable_solution_1, unstable_solution, stable_solution_2]

period = config['settings'].getfloat('forcing_period')

num_steps = config['settings'].getint('num_steps')
num_simulations = config['settings'].getint('num_simulations')
dt = config['settings'].getfloat('time_step')

variance_start = config['settings'].getfloat('variance_start')
variance_end = config['settings'].getfloat('variance_end')
num_variances = config['settings'].getint('num_variances')

makedirs('data', exist_ok = True)

temperatures_for_emission_models_comparison_destination = config['data_paths'].get('temperatures_for_emission_models_comparison')
emitted_radiation_for_emission_models_comparison_destination = config['data_paths'].get('emitted_radiation_for_emission_models_comparison')
F_for_emission_models_comparison_destination = config['data_paths'].get('F_for_emission_models_comparison')

times_for_evolution_towards_steady_states_destination = config['data_paths'].get('times_for_evolution_towards_steady_states')
temperatures_for_evolution_towards_steady_states_destination = config['data_paths'].get('temperatures_for_evolution_towards_steady_states')

times_destination = config['data_paths'].get('times')
temperatures_destination = config['data_paths'].get('temperatures')
frequencies_destination = config['data_paths'].get('frequencies')
averaged_PSD_destination = config['data_paths'].get('averaged_PSD')

peak_heights_in_PSD_destination = config['data_paths'].get('peak_heights_in_PSD')

times_combinations_destination = config['data_paths'].get('times_combinations')
temperatures_combinations_destination = config['data_paths'].get('temperatures_combinations')

#------Generation of data for the emission models comparison----------------------------------

models_comparison_temperatures, emitted_radiation_values, F_values = stochastic_resonance.emission_models_comparison(*temperature_solutions)

np.save(temperatures_for_emission_models_comparison_destination, models_comparison_temperatures)
np.save(emitted_radiation_for_emission_models_comparison_destination, emitted_radiation_values)
np.save(F_for_emission_models_comparison_destination, F_values)

#----Generation of data for showing the evolution of temperature towards steady states----------

evolution_towards_steady_states_time, evolution_towards_steady_states_temperature = stochastic_resonance.calculate_evolution_towards_steady_states(temperature_solutions)

np.save(times_for_evolution_towards_steady_states_destination, evolution_towards_steady_states_time)
np.save(temperatures_for_evolution_towards_steady_states_destination, evolution_towards_steady_states_temperature)

#-----------------------------------------------------------------------

V = np.linspace(variance_start, variance_end, num_variances)

Time = np.zeros((len(V), num_steps))
Temperature = np.zeros((len(V), num_simulations, num_steps))

for i, v in enumerate(V):
    print('\nSimulating the temperature evolution:  {0} of {1}  '.format((i+1), len(V)))
    time, simulated_temperature = stochastic_resonance.simulate_ito(T_start = stable_solution_2,
					       			    noise_variance = v)
    Temperature[i, :, :] = simulated_temperature
    Time[i, :] = time

#----------------------------------------------------------------------

np.save(temperatures_destination, Temperature)
np.save(times_destination, Time)

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
np.save(averaged_PSD_destination, PSD_mean)

with aes.green_text():
    print('Data saved successfully!')

#-------------------------------------------------------------------------

peaks_indices = stochastic_resonance.find_peak_indices(Frequencies, period)
peaks = stochastic_resonance.calculate_peaks(Frequencies, PSD_mean, peaks_indices)
peaks_base = stochastic_resonance.calculate_peaks_base(Frequencies, PSD_mean, peaks_indices)
peaks_height = stochastic_resonance.calculate_peak_height(peaks, peaks_base)

np.save(peak_heights_in_PSD_destination, peaks_height)

V_SR_index = np.argmax(peaks_height)
V_SR = V[V_SR_index]

print('The value of the variance for which the system shows the stochastic resonance mechanism is : {0}'.format(V_SR))

time_combinations, temperatures_combinations = stochastic_resonance.simulate_ito_combinations_and_collect_results(initial_temperature = stable_solution_2, noise_variance = V_SR)

np.save(times_combinations_destination, time_combinations)
np.save(temperatures_combinations_destination, temperatures_combinations)
