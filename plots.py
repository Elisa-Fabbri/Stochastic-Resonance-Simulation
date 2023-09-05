import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os
import stochastic_resonance
import configparser


config = configparser.ConfigParser()
config.read('configuration.txt')

models_comparison_temperatures_path = config['paths'].get('models_comparison_temperatures_destination')
emitted_radiation_values_path = config['paths'].get('emitted_radiation_values_destination')
F_values_path = config['paths'].get('F_values_destination')

peak_height_data = config['paths'].get('simulated_peak_height')

os.makedirs('images', exist_ok = True)

emission_models_comparison_plots_destination = config['paths'].get('emission_models_comparison_plots')
evolution_towards_steady_states_destination = config['paths'].get('evolution_towards_steady_states')
temperature_evolution_plot_destination = config['paths'].get('temperature_evolution_plot')
peak_height_destination = config['paths'].get('peak_height_plot')

def emission_models_comparison_plots():
    fig, axes = plt.subplots(1, 2, figsize = (20, 10))

    T = np.load(models_comparison_temperatures_path)

    #First plot
    emitted_radiation_values = np.load(emitted_radiation_values_path)
    emitted_radiation_linear_values = emitted_radiation_values[0]
    emitted_radiation_black_body_values = emitted_radiation_values[1]

    axes[0].plot(T, emitted_radiation_linear_values, label='Linear emitted radiation')
    axes[0].plot(T, emitted_radiation_black_body_values, label = 'Black body emitted radiation')

    axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    axes[0].set_xlabel(r'Temperature $ \left[ K \right] $ ', fontsize=11)
    axes[0].set_ylabel(r'Emitted radiation $ \left[ \dfrac{kg}{year^3} \right]$ ', fontsize=11)
    axes[0].set_title("Emitted radiation", fontsize = 13, fontweight = 'bold')
    axes[0].legend()

    axes[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))

    # Second plot
    F_values = np.load(F_values_path)
    F_linear_values = F_values[0]
    F_black_body_values = F_values[1]

    axes[1].plot(T, F_linear_values, label = r'Linear $F(T)$')
    axes[1].plot(T, F_black_body_values, label = r"Black body $F(T)$")

    axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    axes[1].set_xlabel(r'Temperature $ \left[ K \right] $ ', fontsize=11)
    axes[1].set_ylabel(r'$F(T)$ ' + r'$ \left[ \dfrac{K}{year} \right] $ ', fontsize = 11)
    axes[1].set_title("Computed F(T)", fontsize = 13, fontweight = 'bold')

    axes[1].legend()

    # Title
    title = "Linear and Black Body Emission Models Comparison"
    plt.suptitle(title, fontsize=16, fontweight = 'bold')

    # Caption
    caption = """The graphs compare the linear emission model with the black body emission model.
              \nThe left graph shows how Earth's emitted radiation varies with Earth's temperature, while the right graph illustrates the rate of temperature change, F(T), as a function of Earth's temperature. """

    fig.text(0.5, 0.01, caption , horizontalalignment='center', fontsize=11, linespacing = 0.8, style = 'italic' )

    plt.savefig(emission_models_comparison_plots_destination)


def evolution_towards_steady_states_plot():
    """
    This function plots the evolution of the temperature towards the steady states
    """
    stable_solution_1 = config['settings'].getfloat('T1')
    unstable_solution = config['settings'].getfloat('T2')
    stable_solution_2 = config['settings'].getfloat('T3')
    epsilon = 1.75

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

    plt.grid(True, linestyle = '--', linewidth = 0.5, color = 'gray', alpha = 0.7)
    plt.xlabel(r'Temperature $\left[ K \right]$', fontsize=11)
    plt.ylabel(r'time $\left[ year \right] $', fontsize = 11)
    #plt.legend()
    plt.title('Evolution of the temperature towards steady states', fontsize = 15, fontweight = 'bold')

    caption = f""" The graph illustrates the temporal evolution of the temperature without periodic forcing and noise. The different curves represent temperature trends for various initial values.
              \nThe temperature gradually converges towards the stable solutions {stable_solution_1} and {stable_solution_2}; while {unstable_solution} represents an unstable solution."""

    plt.figtext(0.5, 0.01, caption, horizontalalignment = 'center', fontsize = 10, linespacing = 0.8, style = 'italic')

    plt.savefig(evolution_towards_steady_states_destination)

def plot_simulate_ito():
    """
    This function plots the evolution of the temperature w.r.t. the time with and without noise and periodic forcing
    """

    fig, axs = plt.subplots(2, 2, figsize = (40, 30))
    axs = axs.ravel()

    stable_solution_1 = config['settings'].getfloat('T1')
    stable_solution_2 = config['settings'].getfloat('T3')

    temperatures = []
    times = []

    for noise_status, forcing_type in [(False, 'constant'), (True, 'constant'), (False, 'varying'), (True, 'varying')]:
        time, temperature = stochastic_resonance.simulate_ito(T_start=stable_solution_2, t_start=0, noise_variance=0.11 if noise_status else 0, num_simulations=1, noise=noise_status, forcing=forcing_type)
        times.append(time)
        temperatures.append(temperature)

    titles = ['without noise and without periodic forcing',
	      'with noise and without periodic forcing',
              'without noise and with periodic forcing',
              'with noise and with periodic forcing']

    for i, (ax, time, temperature, title) in enumerate(zip(axs, times, temperatures, titles)):
        temperature_mean = np.mean(temperature, axis = 0)
        ax.plot(time, temperature_mean)
        ax.grid(True, linestyle = '--', linewidth = 0.5, color = 'gray', alpha = 0.7)
        ax.set_xlabel(r'time $ \left[ \dot 10^{6} year \right] $', fontsize = 30)
        ax.set_ylabel(r'Temperature $ \left[ K \right] $', fontsize = 30)
        ax.set_ylim(stable_solution_1 - 5, stable_solution_2 + 5)
        ax.tick_params(axis='both', which='both', labelsize=22)
        ax.set_title(title, fontsize = 35, fontweight = 'bold')

        ax.xaxis.get_offset_text().set_visible(False)

    plt.tight_layout(pad = 4.0)
    fig.suptitle('Temperature evolution', fontsize = 50, fontweight = 'bold')
    plt.subplots_adjust(top=0.92)

    plt.savefig(temperature_evolution_plot_destination)


def peak_height_plot():
    """
    This function plots the height of the peak as a function of the noise variance.
    """
    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num = num_variances)
    peaks_height = np.load(peak_height_data)

    f = plt.figure(figsize = (15, 10))
    plt.scatter(V, peaks_height)
    plt.xlabel(r'Noise variance $\left[ \dfrac{K^2}{year} \right]$')
    plt.ylabel(r'Peak height $\left[ \dot 10^{6} K^2 \dot year \right]$')
    plt.title('Peak height as a function of the noise variance', fontsize = 15, fontweight = 'bold')
    plt.grid(True, linestyle = '--', linewidth = 0.5, color = 'gray', alpha = 0.7)
    caption = """The plot illustrates the height of the peak in the power spectrum computed for various noise variance values.
              \nThe quantity is determined by measuring the peak height in the power spectrum and subtracting the baseline height."""

    plt.gca().get_yaxis().get_offset_text().set_visible(False)
    plt.figtext(0.5, 0.01, caption, horizontalalignment = 'center', fontsize = 10, linespacing = 0.8, style = 'italic')
    plt.savefig(peak_height_destination)


emission_models_comparison_plots()
evolution_towards_steady_states_plot()
plot_simulate_ito()
peak_height_plot()
