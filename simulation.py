import configparser
import os
import numpy as np
import stochastic_resonance

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

num_steps = 1000000
num_simulations = 10

V = np.linspace(0.01, 0.2, num = 2)
sigma = np.sqrt(V)

Time = np.zeros((len(V), num_steps))
Temperature = np.zeros((len(V), num_simulations, num_steps))

for i, v in enumerate(V):
    time, simulated_temperature = stochastic_resonance.simulate_ito(T_start = T3, t_start = 0,
					       			    noise_variance = v)
    Temperature[i, :, :] = simulated_temperature
    Time[i, :] = time
    print('Simulation {0} of {1} done!'.format(i+1, len(V)))

filename_temperature = 'simulated_temperature.npy'
filename_time = 'simulated_time.npy'

data_folder = 'data'
os.makedirs(data_folder, exist_ok=True)

save_file_path_temperature = os.path.join(data_folder, filename_temperature)
save_file_path_time = os.path.join(data_folder, filename_time)

np.save(save_file_path_temperature, Temperature)
np.save(save_file_path_time, Time)

print('Data saved successfully!')
