import configparser
import sys
from os import makedirs
import numpy as np
from scipy import signal

import stochastic_resonance as sr
import aesthetics as aes

#---------------------------------------------------------------------

config = configparser.ConfigParser()

try:
    config.read(sys.argv[1])
except IndexError:
    with aes.red_text():
        print('Error: You must specify a configuration file as an argument!')
        sys.exit()

stable_temperature_solution_1 = config['settings'].getfloat('stable_temperature_solution_1')
unstable_temperature_solution = config['settings'].getfloat('unstable_temperature_solution')
stable_temperature_solution_2 = config['settings'].getfloat('stable_temperature_solution_2')

temperature_solutions = [stable_temperature_solution_1, unstable_temperature_solution, stable_temperature_solution_2]

forcing_period = config['settings'].getfloat('forcing_period')

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

print('Calculating times and temperature values for comparing emission models...')

temperatures_values, emitted_radiation_values, F_values = sr.emission_models_comparison(*temperature_solutions)

np.save(temperatures_for_emission_models_comparison_destination, temperatures_values)
np.save(emitted_radiation_for_emission_models_comparison_destination, emitted_radiation_values)
np.save(F_for_emission_models_comparison_destination, F_values)

with aes.green_text():
    print('Results saved!')

#----Generation of data for showing the evolution of temperature towards steady states----------

print('Calculating temperature evolution data without noise or periodic forcing...')
evolution_towards_steady_states_time, evolution_towards_steady_states_temperature = sr.calculate_evolution_towards_steady_states(temperature_solutions)

np.save(times_for_evolution_towards_steady_states_destination, evolution_towards_steady_states_time)
np.save(temperatures_for_evolution_towards_steady_states_destination, evolution_towards_steady_states_temperature)

with aes.green_text():
    print('Results saved!')
#-----------------------------------------------------------------------

print('Simulating temperature evolution with noise and periodic forcing...')

V = np.linspace(variance_start, variance_end, num_variances)

Time = np.zeros((len(V), num_steps))
Temperature = np.zeros((len(V), num_simulations, num_steps))

for i, v in enumerate(V):
    print(f'Simulation {i + 1}/{len(V)}, Noise: {v:.3f}...')
    time, temperature = sr.simulate_ito(T_start = stable_temperature_solution_2,
					noise_variance = v)
    Temperature[i, :, :] = temperature
    Time[i, :] = time

with aes.green_text():
    print('Simulation completed.')

#----------------------------------------------------------------------

print('Saving results...')

np.save(temperatures_destination, Temperature)
np.save(times_destination, Time)

with aes.green_text():
    print('Data saved successfully!')

#-----------------------------------------------------------------------

Frequencies = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))
PSD_mean = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))

print('Computing power spectra...')

for i in aes.progress(range(len(V))):
    psd = np.zeros((num_simulations, np.floor_divide(num_steps, 2) + 1))
    for j in range(num_simulations):
        frequencies, power_spectrum = signal.periodogram(Temperature[i, j, :], 1/dt)
        psd[j, :] = power_spectrum
    PSD_mean[i, :] = np.mean(psd, axis = 0)
    Frequencies[i, :] = frequencies

with aes.green_text():
    print('Done!')

#-----------------------------------------------------------------------

print('Saving the results')

np.save(frequencies_destination, Frequencies)
np.save(averaged_PSD_destination, PSD_mean)

with aes.green_text():
    print('Data saved successfully!')

#-------------------------------------------------------------------------

print('Calculating peak heights in the power spectral density...')

peaks_indices = sr.find_peak_indices(Frequencies, forcing_period)
peaks = sr.calculate_peaks(Frequencies, PSD_mean, peaks_indices)
peaks_bases = sr.calculate_peaks_base(Frequencies, PSD_mean, peaks_indices)
peaks_heights = sr.calculate_peak_height(peaks, peaks_bases)

np.save(peak_heights_in_PSD_destination, peaks_heights)

with aes.green_text():
    print('Results saved!')

V_SR_index = np.argmax(peaks_heights)
V_SR = V[V_SR_index]

print(f'Stochastic resonance mechanism found at variance value: {V_SR:.3f}')

#-----------------------------------------------------------------------------

print('Simulating Temperature Combinations: Periodic Forcing, No Forcing, Noise, and No Noise...')

time_combinations, temperatures_combinations = sr.simulate_ito_combinations_and_collect_results(initial_temperature = stable_temperature_solution_2, noise_variance = V_SR)

np.save(times_combinations_destination, time_combinations)
np.save(temperatures_combinations_destination, temperatures_combinations)

with aes.green_text():
    print('Results saved!')


