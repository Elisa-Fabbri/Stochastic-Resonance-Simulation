import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import stochastic_resonance
import configparser


config = configparser.ConfigParser()
config.read('configuration.txt')

SNR_data = config['paths'].get('simulated_SNR')

os.makedirs('images', exist_ok = True)

radiation_destination = config['paths'].get('emitted_radiation_plot')
F_destination = config['paths'].get('F_plot')
evolution_towards_steady_states_destination = config['paths'].get('evolution_towards_steady_states')
temperature_evolution_plot_destination = config['paths'].get('temperature_evolution_plot')
SNR_destination = config['paths'].get('SNR_plot')

def emission_models_comparison_plot():

    """
    This function plots the two models for the emitted radiation
    """

    T_start = 270
    T_end = 300
    num_points = 300

    T = np.linspace(T_start, T_end, num = num_points)
    emitted_radiation_linear_values = [stochastic_resonance.emitted_radiation(Ti, model = 'linear') for Ti in T]
    emitted_radiation_black_body_values = [stochastic_resonance.emitted_radiation(Ti, model = 'black body') for Ti in T]

    f = plt.figure(figsize=(15, 10))
    plt.plot(T, emitted_radiation_linear_values, label='Linear emitted radiation')
    plt.plot(T, emitted_radiation_black_body_values, label='Black body emitted radiation')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.xlabel(r'Temperature $ \left[ K \right] $ ', fontsize=11)
    plt.ylabel(r'Emitted radiation $ \left[ \dot 10^{25} \dfrac{kg}{year^3} \right]$ ', fontsize=11)
    plt.legend()
    plt.title('Comparison between the two models for the emitted radiation', fontsize = 15, fontweight = 'bold')
    caption = """The graph illustrates the relationship between emitted radiation per unit of surface area from the Earth as a function of the temperature.
    \nThe blue curve represents the calculated radiation trend using a linear emission model, while the orange curve corresponds to a blackbody emission model."""
    plt.figtext(0.5, 0.01, caption , horizontalalignment='center', fontsize=11, linespacing = 0.8, style = 'italic' )

    plt.gca().get_yaxis().get_offset_text().set_visible(False)

    plt.savefig(radiation_destination)

def F_plot():

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
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.xlabel(r'Temperature $ \left[ K \right] $ ', fontsize = 11)
    plt.ylabel(r'$F(T)$ ' + r'$ \left[ \dfrac{K}{year} \right] $ ', fontsize = 11)
    plt.legend()
    plt.title('Comparison between $F(T)$ using a linear and a black body model for the emitted radiation', fontsize = 15, fontweight = 'bold')
    caption = """The graph shows the the rate of temperature change F(T), as a function of temperature.
    \nThe blue curve depicts the F(T) trend using a linear emission model, while the orange curve represents the F(T) trend using a blackbody emission model."""

    plt.figtext(0.5, 0.01, caption, horizontalalignment = 'center', fontsize = 11, linespacing = 0.8, style = 'italic')

    plt.savefig(F_destination)


def Plot_evolution_towards_steady_states():
    """
    This function plots the evolution of the temperature towards the steady states
    """
    stable_solution_1 = config['settings'].getfloat('T1')
    unstable_solution = config['settings'].getfloat('T2')
    stable_solution_2 = config['settings'].getfloat('T3')
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

    plt.savefig(evolution_towards_steady_states_destination)
    plt.show()

def plot_simulate_ito():
    """
    This function plots the evolution of the temperature w.r.t. the time with and without noise and periodic forcing
    """

    fig, axs = plt.subplots(2, 2, figsize = (30, 30))
    axs = axs.ravel()

    T3 = config['settings'].getfloat('T3')

    time1, temperature1 = stochastic_resonance.simulate_ito(T_start = T3, t_start = 0, num_simulations = 1, noise = False, forcing = 'constant')
    time2, temperature2 = stochastic_resonance.simulate_ito(T_start = T3, t_start = 0, noise_variance = 0.11 , num_simulations = 1, noise = True, forcing = 'constant')
    time3, temperature3 = stochastic_resonance.simulate_ito(T_start = T3, t_start = 0, num_simulations = 1, noise = False, forcing = 'varying')
    time4, temperature4 = stochastic_resonance.simulate_ito(T_start = T3, t_start = 0, noise_variance = 0.11 , num_simulations = 1, noise = True, forcing = 'varying')

    for i, (ax, time, temperature) in enumerate(zip(axs, [time1, time2, time3, time4], [temperature1, temperature2, temperature3, temperature4])):
        temperature_mean = np.mean(temperature, axis = 0)
        ax.plot(time, temperature_mean)
        ax.grid(True)
        ax.set_xlabel('time')
        ax.set_ylabel('temperature')
        ax.set_title(f'Plot {i+1}')

    plt.tight_layout()
    plt.savefig(temperature_evolution_plot_destination)
    plt.show()

def SNR_plot():
    """
    This function plots the Signal to Noise Ratio as a function of the noise variance.
    """
    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num = num_variances)
    SNR = np.load(SNR_data)

    f = plt.figure(figsize = (15, 10))
    plt.scatter(V, SNR)
    plt.xlabel('Noise variance')
    plt.ylabel('SNR')
    plt.title('')
    plt.grid(True)
    plt.savefig(SNR_destination)
    plt.show()

emission_models_comparison_plot()
F_plot()
Plot_evolution_towards_steady_states()
plot_simulate_ito()
SNR_plot()
