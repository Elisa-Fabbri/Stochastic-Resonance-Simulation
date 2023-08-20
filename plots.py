import numpy as np
import matplotlib.pyplot as plt
import os
import stochastic_resonance
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')

T1 = config['settings'].getfloat('T1')
T2 = config['settings'].getfloat('T2')
T3 = config['settings'].getfloat('T3')


def Plot_Emission_Models():

    """
    This function plots the two models for the emitted radiation
    """
    T_start = 270
    T_end = 300
    num_points = 300

    T = np.linspace(T_start, T_end, num_points)
    emitted_radiation_linear_values = [stochastic_resonance.emitted_radiation(Ti, model = 'linear') for Ti in T]
    emitted_radiation_black_body_values = [stochastic_resonance.emitted_radiation(Ti, model = 'black body') for Ti in T]

    f = plt.figure(figsize=(15, 10))
    plt.plot(T, emitted_radiation_linear_values, label='Linear emitted radiation')
    plt.plot(T, emitted_radiation_black_body_values, label='Black body emitted radiation')
    plt.grid(True)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Emitted radiation')
    #plt.legend()
    plt.title('Comparison between the two models for the emitted radiation')

    save_file_path = os.path.join('images', 'emitted_radiation_plot.png')
    plt.savefig(save_file_path)

    plt.show()


def Plot_F():

    """
    This function plots F(T) using the two different models for the emitted radiation.
    """
    T_start = 270
    T_end = 300
    num_points = 300

    T = np.linspace(T_start, T_end, num_points)
    F_linear_values = [stochastic_resonance.F(Ti, model = 'linear') for Ti in T]
    F_black_body_values = [stochastic_resonance.F(Ti, model = 'black body') for Ti in T]

    f = plt.figure(figsize=(15, 10))
    plt.plot(T, F_linear_values, label='Linear')
    plt.plot(T, F_black_body_values, label = 'Black body')
    plt.grid(True)
    plt.xlabel('Temperature (K)')
    plt.ylabel('F(T)')
    #plt.legend()
    plt.title('Comparison between F(T) using a linear and a black body model for beta')

    save_file_path = os.path.join('images', 'F_plot.png')
    plt.savefig(save_file_path)

    plt.show()

def Plot_evolution_towards_steady_states():
    """
    This function plots the evolution of the temperature towards the steady states
    """
    stable_solution_1 = T1
    unstable_solution = T2
    stable_solution_2 = T3
    epsilon = 2.5

    T_start = np.array([stable_solution_1-epsilon, stable_solution_1, stable_solution_1+epsilon,
			unstable_solution-epsilon, unstable_solution, unstable_solution+epsilon,
			stable_solution_2-epsilon, stable_solution_2, stable_solution_2+epsilon])
    t_start = 0

    time, simulated_temperature = stochastic_resonance.simulate_ito(T_start, t_start,
								    num_steps = 100, num_simulations = len(T_start),
								    noise = False, forcing = 'constant')
    f = plt.figure(figsize=(15, 10))
    for i in range(len(simulated_temperature)):
        plt.plot(simulated_temperature[i], time)

    plt.grid(True)
    plt.xlabel('Temperature (K)')
    plt.ylabel('time')
    #plt.legend()
    plt.title('Evolution of the temperature towards steady states')

    save_file_path = os.path.join('images', 'Evolution_of_the_temperature_towards_steady_states.png')
    plt.savefig(save_file_path)

    plt.show()


Plot_Emission_Models()
Plot_F()
Plot_evolution_towards_steady_states()
