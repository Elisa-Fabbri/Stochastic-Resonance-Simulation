import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from os import makedirs
import configparser
import sys

import aesthetics as aes

config = configparser.ConfigParser()

try:
    config.read(sys.argv[1])
except IndexError:
    with aes.red_text():
        print('Error: You must specify a configuration file as an argument!')
        sys.exit()

try:
    temperatures_for_emission_models_comparison_path = config['data_paths'].get('temperatures_for_emission_models_comparison')
    emitted_radiation_for_emission_models_comparison_path = config['data_paths'].get('emitted_radiation_for_emission_models_comparison')
    F_for_emission_models_comparison_path = config['data_paths'].get('F_for_emission_models_comparison')

    times_for_evolution_towards_steady_states_path = config['data_paths'].get('times_for_evolution_towards_steady_states')
    temperatures_for_evolution_towards_steady_states_path = config['data_paths'].get('temperatures_for_evolution_towards_steady_states')

    frequencies_path = config['data_paths'].get('frequencies')
    averaged_PSD_path = config['data_paths'].get('averaged_PSD')

    peak_heights_in_PSD_path = config['data_paths'].get('peak_heights_in_PSD')

    times_combinations_path = config['data_paths'].get('times_combinations')
    temperatures_combinations_path = config['data_paths'].get('temperatures_combinations')

except:
    with aes.red_text():
        print("An error occurred while reading data paths from the configuration file.")
    with aes.orange_text():
        print("Please make sure you have correctly specified the data paths in the configuration file.")
    sys.exit(1)


makedirs('images', exist_ok = True)

try:
    emission_models_comparison_plots_destination = config['image_paths'].get('emission_models_comparison_plots')
    temperatures_towards_steady_states_plot_destination = config['image_paths'].get('temperatures_towards_steady_states_plot')
    power_spectra_plots_destination = config['image_paths'].get('power_spectra_plots')
    peak_heights_plot_destination = config['image_paths'].get('peak_heights_plot')
    temperature_combinations_plots_destination = config['image_paths'].get('temperature_combinations_plots')
except:
    with aes.red_text():
        print("An error occurred while reading image paths from the configuration file.")
    with aes.orange_text():
        print("Please make sure you have correctly specified the image paths in the configuration file.")

def emission_models_comparison_plots():
    """
    Create comparison plots for emission models.

    This function generates two comparison plots for emission models: one for emitted radiation and another for computed F(T).
    The plots compare the linear emission model with the black body emission model.

    """

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    try:
        T = np.load(temperatures_for_emission_models_comparison_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing temperature data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    # First plot
    try:
        emitted_radiation_values = np.load(emitted_radiation_for_emission_models_comparison_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing the emitted radiation data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    emitted_radiation_linear_values = emitted_radiation_values[0]
    emitted_radiation_black_body_values = emitted_radiation_values[1]

    axes[0].plot(T, emitted_radiation_linear_values, label='Linear emitted radiation')
    axes[0].plot(T, emitted_radiation_black_body_values, label='Black body emitted radiation')

    axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    axes[0].set_xlabel(r'Temperature $ \left[ K \right] $ ', fontsize=11)
    axes[0].set_ylabel(r'Emitted radiation $ \left[ \dfrac{kg}{year^3} \right]$ ', fontsize=11)
    axes[0].set_title("Emitted radiation", fontsize=13, fontweight='bold')
    axes[0].legend()

    axes[0].yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))

    # Second plot
    try:
        F_values = np.load(F_for_emission_models_comparison_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing the rate of temperature change data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    F_linear_values = F_values[0]
    F_black_body_values = F_values[1]

    axes[1].plot(T, F_linear_values, label=r'Linear $F(T)$')
    axes[1].plot(T, F_black_body_values, label=r"Black body $F(T)$")

    axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    axes[1].set_xlabel(r'Temperature $ \left[ K \right] $ ', fontsize=11)
    axes[1].set_ylabel(r'$F(T)$ ' + r'$ \left[ \dfrac{K}{year} \right] $ ', fontsize=11)
    axes[1].set_title("Computed F(T)", fontsize=13, fontweight='bold')
    axes[1].legend()

    # Title
    title = "Linear and Black Body Emission Models Comparison"
    plt.suptitle(title, fontsize=16, fontweight='bold')

    # Caption
    caption = """The graphs compare the linear emission model with the black body emission model.
    The left graph shows how Earth's emitted radiation varies with Earth's temperature, while the right graph illustrates the rate of temperature change, F(T)."""
    fig.text(0.5, 0.01, caption, horizontalalignment='center', fontsize=11, linespacing=0.8, style='italic')

    plt.savefig(emission_models_comparison_plots_destination)


def evolution_towards_steady_states_plot():
    """
    Plot the evolution of temperature towards steady states.

    This function generates a plot illustrating the temporal evolution of temperature without periodic forcing and noise.
    Different curves represent temperature trends for various initial values as they converge towards the stable solutions.

    """
    try:
        time = np.load(times_for_evolution_towards_steady_states_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing time data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    try:
         simulated_temperature = np.load(temperatures_for_evolution_towards_steady_states_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing temperature data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    f = plt.figure(figsize=(15, 10))

    for i in range(len(simulated_temperature)):
        plt.plot(simulated_temperature[i], time)

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.xlabel(r'Temperature $\left[ K \right]$', fontsize=11)
    plt.ylabel(r'time $\left[ year \right] $', fontsize=11)
    plt.title('Evolution of the temperature towards steady states', fontsize=15, fontweight='bold')

    caption = """ The graph illustrates the temporal evolution of the temperature without periodic forcing and noise. 
    The different curves represent temperature trends for various initial values.
    The temperature gradually converges towards the stable solutions."""
    plt.figtext(0.5, 0.01, caption, horizontalalignment='center', fontsize=10, linespacing=0.8, style='italic')

    plt.savefig(temperatures_towards_steady_states_plot_destination)

def power_spectra_plots():
    """
    Plot power spectral density as a function of frequency for different noise variance values.

    This function generates a set of plots, each showing the computed power spectral density (PSD) as a function of frequency
    for different values of the noise variance.

    """

    try:
        frequencies = np.load(frequencies_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing frequency data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    try:
         PSD_mean = np.load(averaged_PSD_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing power spectral density data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()


    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num=num_variances)

    if frequencies.shape[0] != PSD_mean.shape[0]:
        raise ValueError("The dimensions of the two arrays do not align.")

    num_plots = frequencies.shape[0]
    num_cols = num_plots // 2
    num_rows = (num_plots // num_cols) + int(num_plots % num_cols != 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 10))

    for i in range(num_plots):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]

        ax.semilogy(frequencies[i], PSD_mean[i])
        ax.set_xlabel(r'Frequency $\left[ \frac{1}{year} \right]$', fontsize=11)
        ax.set_ylabel('PSD', fontsize=11)
        ax.set_title('Variance: {0}'.format(round(V[i], 3)), fontsize=13, fontweight='bold')
        ax.set_xlim(0, 2.5e-5)
        ax.set_ylim(1e-1, 1e8)
        ax.grid(True)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))

    title = 'Power Spectral Density as a function of the Frequency for different Noise Variance Values'
    plt.suptitle(title, fontsize=20, fontweight='bold')

    caption = 'The plots show the computed power spectral density for different values of the noise variance'
    fig.text(0.5, 0.01, caption, horizontalalignment='center', fontsize=12, linespacing=0.8, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(power_spectra_plots_destination)

def peak_height_plot():
    """
    Plot the height of the peak as a function of the noise variance.

    This function generates a scatter plot illustrating the height of the peak in the power spectrum computed for various
    noise variance values. The peak height is determined by measuring the peak height in the power spectrum and subtracting
    the baseline height.

    """

    variance_start = config['settings'].getfloat('variance_start')
    variance_end = config['settings'].getfloat('variance_end')
    num_variances = config['settings'].getint('num_variances')

    V = np.linspace(variance_start, variance_end, num=num_variances)

    try:
        peaks_height = np.load(peak_heights_in_PSD_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing peaks heights data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    f = plt.figure(figsize=(15, 10))
    plt.scatter(V, peaks_height)
    plt.xlabel(r'Noise variance $\left[ \frac{K^2}{year} \right]$')
    plt.ylabel(r'Peak height $\left[ K^2 \cdot year \right]$')
    plt.title('Peak height as a function of the noise variance', fontsize=15, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    caption = """The plot illustrates the height of the peak in the power spectrum computed for various noise variance values.
    The quantity is determined by measuring the peak height in the power spectrum and subtracting the baseline height."""

    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
    plt.figtext(0.5, 0.01, caption, horizontalalignment='center', fontsize=10, linespacing=0.8, style='italic')
    plt.savefig(peak_heights_plot_destination)

def plot_simulate_ito_combinations():
    """
    Plot the evolution of temperature with and without noise and periodic forcing.

    This function generates plots to visualize the evolution of temperature under different conditions:
    1. Without noise and without periodic forcing
    2. With noise and without periodic forcing
    3. Without noise and with periodic forcing
    4. With noise and with periodic forcing

    """
    fig, axs = plt.subplots(2, 2, figsize=(40, 30))
    axs = axs.ravel()

    stable_solution_1 = config['settings'].getfloat('stable_temperature_solution_1')
    stable_solution_2 = config['settings'].getfloat('stable_temperature_solution_2')

    try:
        times = np.load(times_combinations_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing time data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    try:
        temperatures = np.load(temperatures_combinations_path)
    except FileNotFoundError:
        with aes.red_text():
            print("An error occurred: The file containing temperature data does not exist.")
        print("To generate this file, please run 'simulation.py' first.")
        sys.exit(1)
    except TypeError:
        with aes.red_text():
            print("An error occurred while reading data paths from the configuration file.")
        with aes.orange_text():
            print("Please make sure you have correctly specified the data paths in the configuration file.")
        sys.exit()

    titles = [
        'Without noise and without periodic forcing',
        'With noise and without periodic forcing',
        'Without noise and with periodic forcing',
        'With noise and with periodic forcing'
    ]

    for i, (ax, time, temperature, title) in enumerate(zip(axs, times, temperatures, titles)):
        temperature_mean = np.mean(temperature, axis=0)
        ax.plot(time, temperature_mean)
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel(r'time $ \left[ \dot 10^{6} year \right] $', fontsize=30)
        ax.set_ylabel(r'Temperature $ \left[ K \right] $', fontsize=30)
        ax.set_ylim(stable_solution_1 - 5, stable_solution_2 + 5)
        ax.tick_params(axis='both', which='both', labelsize=22)
        ax.set_title(title, fontsize=35, fontweight='bold')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)

    plt.tight_layout(pad = 4.0)
    fig.suptitle('Temperature evolution', fontsize=50, fontweight='bold')
    plt.subplots_adjust(top=0.92)
    plt.savefig(temperature_combinations_plots_destination)


emission_models_comparison_plots()
evolution_towards_steady_states_plot()
power_spectra_plots()
peak_height_plot()
plot_simulate_ito_combinations()

with aes.green_text():
    print('Plots saved!')
