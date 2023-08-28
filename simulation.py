import configparser
import os
import numpy as np
from scipy import signal

import stochastic_resonance

#---------------------------------------------------------------------

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

num_steps = config['settings'].getint('num_steps')
num_simulations = config['settings'].getint('num_simulations')
dt = config['settings'].getfloat('time_step')

variance_start = config['settings'].getfloat('variance_start')
variance_end = config['settings'].getfloat('variance_end')
num_variances = config['settings'].getint('num_variances')

os.makedirs('data', exist_ok = True)

time_destination = config['paths'].get('simulated_time')
temperature_destination = config['paths'].get('simulated_temperature')
frequencies_destination = config['paths'].get('simulated_frequencies')
PSD_destination = config['paths'].get('simulated_PSD')
SNR_destination = config['paths'].get('simulated_SNR')


V = np.linspace(variance_start, variance_end, num_variances)

#-----------------------------------------------------------------------

Time = np.zeros((len(V), num_steps))
Temperature = np.zeros((len(V), num_simulations, num_steps))

for i, v in enumerate(V):
    time, simulated_temperature = stochastic_resonance.simulate_ito(T_start = T3, t_start = 0,
					       			    noise_variance = v)
    Temperature[i, :, :] = simulated_temperature
    Time[i, :] = time
    print('Simulation {0} of {1} done!'.format(i+1, len(V)))

#----------------------------------------------------------------------

np.save(temperature_destination, Temperature)
np.save(time_destination, Time)

print('Data saved successfully!')

#-----------------------------------------------------------------------

Frequencies = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))
PSD_mean = np.zeros((len(V), np.floor_divide(num_steps, 2) +1))

for i in range(len(V)):
    psd = np.zeros((num_simulations, np.floor_divide(num_steps, 2) + 1))
    for j in range(num_simulations):
        frequencies, power_spectrum = signal.periodogram(Temperature[i, j, :], 1/dt)
        psd[j, :] = power_spectrum
    PSD_mean[i, :] = np.mean(psd, axis = 0)
    Frequencies[i, :] = frequencies

#-----------------------------------------------------------------------

np.save(frequencies_destination, Frequencies)
np.save(PSD_destination, PSD_mean)

print('Data saved successfully!')

#-------------------------------------------------------------------------

peaks_indices = stochastic_resonance.find_peak_indices(Frequencies, period)
peaks = stochastic_resonance.calculate_peaks(Frequencies, PSD_mean, peaks_indices)
peaks_base = stochastic_resonance.calculate_peaks_base(Frequencies, PSD_mean, peaks_indices)
SNR = stochastic_resonance.calculate_SNR(peaks, peaks_base)

np.save(SNR_destination, SNR)
