import configparser
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
