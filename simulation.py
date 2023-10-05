"""
This module contains the main script for the simulation of the stochastic resonance mechanism for the
Earth's climate model.

The script reads a configuration file for the simulation parameters and saves the results in the
specified paths (all this files are contained in the data folder).

"""

import configparser
import sys
import os
import numpy as np
from scipy import signal

import stochastic_resonance as sr
import aesthetics as aes

#-------Reading parameters values and paths from the configuration file-----------

config = configparser.ConfigParser()

# Reading the configuration file; if it is not specified as an argument, 
# the 'configuration.txt' file is used as default:
config_file = sys.argv[1] if len(sys.argv) > 1 else 'configuration.txt'

if not os.path.isfile(config_file):
    with aes.red_text():
        if config_file == 'configuration.txt':
            print('Error: The default configuration file "configuration.txt" does not exist in the current folder!')
        else:
            print(f'Error: The specified configuration file "{config_file}" does not exist in the current folder!')
        sys.exit()

config.read(config_file)

stable_temperature_solution_1 = config['settings'].getfloat('stable_temperature_solution_1')
unstable_temperature_solution = config['settings'].getfloat('unstable_temperature_solution')
stable_temperature_solution_2 = config['settings'].getfloat('stable_temperature_solution_2')

surface_heat_capacity_j_per_m2_K = config['settings'].getfloat('surface_earth_thermal_capacity')

relaxation_time = config['settings'].getfloat('relaxation_time')
emission_model = config['settings'].get('emission_model')

num_sec_in_a_year = 365.25*24*60*60

C_years = surface_heat_capacity_j_per_m2_K * (num_sec_in_a_year ** 2)

forcing_amplitude = config['settings'].getfloat('forcing_amplitude')
forcing_period = config['settings'].getfloat('forcing_period')

forcing_angular_frequency = (2 * np.pi)/ forcing_period


num_steps = config['settings'].getint('num_steps')
num_simulations = config['settings'].getint('num_simulations')
time_step = config['settings'].getfloat('time_step')

variance_start = config['settings'].getfloat('variance_start')
variance_end = config['settings'].getfloat('variance_end')
num_variances = config['settings'].getint('num_variances')

if 'seed_value' in config['settings']:
    seed_value = config['settings'].getint('seed_value')
    print("The seed value used for this simulation is: ", seed_value)
else: 
    # Generate a random seed value between 0 and 1000000
    seed_value = np.random.randint(0, 1000000)
    with aes.orange_text():
        print('The seed value was not specified in the configuration file.')
    print('The randomly generated seed value used for this simulation is: ', seed_value)

os.makedirs('data', exist_ok = True)

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

#------Generation of data for the emission models comparison--------------

print('Calculating values for the emission models comparison...')

temperatures_values, emitted_radiation_values, F_values = sr.emission_models_comparison(
    										surface_thermal_capacity = C_years,
											relaxation_time = relaxation_time,
											stable_temperature_solution_1 = stable_temperature_solution_1,
											unstable_temperature_solution = unstable_temperature_solution,
											stable_temperature_solution_2 = stable_temperature_solution_2)

#Saving data
np.save(temperatures_for_emission_models_comparison_destination, temperatures_values)
np.save(emitted_radiation_for_emission_models_comparison_destination, emitted_radiation_values)
with aes.green_text():
    print('Emitted radiation values saved!')
np.save(F_for_emission_models_comparison_destination, F_values)
with aes.green_text():
    print('Rate of temperature change values saved!')

#----Generation of data for showing the evolution of temperature towards steady states---------

print('Calculating temperature evolution data without noise or periodic forcing...')
evolution_towards_steady_states_time, evolution_towards_steady_states_temperature = \
    sr.calculate_evolution_towards_steady_states(
        surface_thermal_capacity=C_years,
        relaxation_time=relaxation_time,
        stable_temperature_solution_1=stable_temperature_solution_1,
        unstable_temperature_solution=unstable_temperature_solution,
        stable_temperature_solution_2=stable_temperature_solution_2,
        forcing_amplitude=forcing_amplitude,
        forcing_angular_frequency=forcing_angular_frequency,
        emission_model=emission_model,
        seed_value = seed_value
    )

#Saving data

np.save(times_for_evolution_towards_steady_states_destination, 
        evolution_towards_steady_states_time)
np.save(temperatures_for_evolution_towards_steady_states_destination, 
        evolution_towards_steady_states_temperature)

with aes.green_text():
    print('Temperature values saved!')

#-----Simulation of temperature evolution for different noise variances---------

print('Simulating temperature evolution for different noise variances...')

V = np.linspace(variance_start, variance_end, num_variances)

Time = np.zeros((len(V), num_steps))
Temperature = np.zeros((len(V), num_simulations, num_steps))

for i, v in enumerate(V):
    print(f'Simulation {i + 1}/{len(V)}, Noise: {v:.3f}...')
    time, temperature = sr.simulate_ito(T_start = stable_temperature_solution_2,
					noise_variance = v,
					dt = time_step,
					num_steps = num_steps,
					num_simulations = num_simulations,
					surface_thermal_capacity = C_years,
					relaxation_time = relaxation_time,
					stable_temperature_solution_1 = stable_temperature_solution_1,
					unstable_temperature_solution = unstable_temperature_solution,
					stable_temperature_solution_2 = stable_temperature_solution_2,
					forcing_amplitude = forcing_amplitude,
					forcing_angular_frequency = forcing_angular_frequency,
					noise = True,
					emission_model = emission_model,
                    seed_value = seed_value)
    Temperature[i, :, :] = temperature
    Time[i, :] = time

with aes.green_text():
    print('Simulation completed.')

#Saving data

print('Saving results...')

np.save(temperatures_destination, Temperature)
np.save(times_destination, Time)

with aes.green_text():
    print('Time and temperature data saved successfully!')

#-----Calculation of the power spectral density-------

Frequencies = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))
PSD_mean = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))

print('Computing power spectra...')

for i in aes.progress(range(len(V))):
    psd = np.zeros((num_simulations, np.floor_divide(num_steps, 2) + 1))
    for j in range(num_simulations):
        frequencies, power_spectrum = signal.periodogram(Temperature[i, j, :], 1/time_step)
        psd[j, :] = power_spectrum
    PSD_mean[i, :] = np.mean(psd, axis = 0)
    Frequencies[i, :] = frequencies

with aes.green_text():
    print('Done!')

#-----------------------------------------------------------------------

print('Saving results...')

np.save(frequencies_destination, Frequencies)
np.save(averaged_PSD_destination, PSD_mean)

with aes.green_text():
    print('Frequency and PSD data saved successfully!')

#-----Calculation of the peak heights in the power spectral density-------

print('Calculating peak heights in the power spectral density...')

peaks_indices = sr.find_peak_indices(Frequencies, forcing_period)
peaks = sr.calculate_peaks(PSD_mean, peaks_indices)
peaks_bases = sr.calculate_peaks_base(PSD_mean, peaks_indices)
peaks_heights = sr.calculate_peak_height(peaks, peaks_bases)

np.save(peak_heights_in_PSD_destination, peaks_heights)

with aes.green_text():
    print('Peak heights values saved!')

V_SR_index = np.argmax(peaks_heights)
V_SR = V[V_SR_index]

print(f'Stochastic resonance mechanism found at variance value: {V_SR:.3f}')

#-----Simulation of temperature evolution with and without noise and periodic forcing----
# The variance value used is the one that maximizes the peak height in the PSD.

print('Simulating Temperature Combinations: Periodic Forcing, No Forcing, Noise, and No Noise...')

time_combinations, temperatures_combinations = sr.simulate_ito_combinations_and_collect_results(
    											T_start = stable_temperature_solution_2,
												noise_variance = V_SR,
												dt = time_step,
												num_steps = num_steps,
												stable_temperature_solution_1 = stable_temperature_solution_1,
												unstable_temperature_solution = unstable_temperature_solution,
												stable_temperature_solution_2 = stable_temperature_solution_2,
												forcing_amplitude = forcing_amplitude,
												forcing_angular_frequency = forcing_angular_frequency,
												emission_model = emission_model,
                                                seed_value = seed_value)

np.save(times_combinations_destination, time_combinations)
np.save(temperatures_combinations_destination, temperatures_combinations)

with aes.green_text():
    print('Time and temperature data saved!')


