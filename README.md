# Stochastic-Resonance-Simulation

## General Introduction and Objective of the Repository

Stochastic resonance is a phenomenon observed in nonlinear systems, wherein a weak input signal, like a faint signal, can be enhanced and optimized through the 
presence of noise.
The three basic characteristics are (i) the presence of an energetic activation barrier (or a form of threshold), (ii) a weak coherent input (such as a periodic
signal), and (iii) a source of noise. Given these features, the responce of the system undergoes a resonance-like behaviour as a function of the noise level.

The concept of stochastic resonance was initially introduced by Italian physicists Roberto Benzi, Alfonso Sutera, and Angelo Vulpiani in 1981. In their pioneering 
work, they explored its potential applications in the field of climate dynamics, alongside Giorgio Parisi.

The aim of this repository is to investigate the stochastic resonance phenomenon through numerical simulations in the particular case of a climate model. 


## Structure of the Project

### Running the Simulation and Plotting Results

To execute the simulation and visualize the results, follow these steps:

1. **Configuration File:** Start by specifying simulation parameters and file paths in the `configuration.txt` file.

2. **Running the Simulation:** Execute the simulation by running `simulation.py` from the command line and specifying the desired configuration file: 'python simulation.py configuration.txt'.

This step generates data files and saves them in the `data` folder.

3. **Generating Plots:** After running the simulation, create plots to visualize the results using `plots.py` with the chosen configuration file: 'python plots.py configuration.txt'.

The plots are saved in the `images` folder.

### Creating a New Configuration File

To create a new configuration file for the simulation, follow these guidelines:

1. **Configuration Sections:**

   The configuration file should have three sections: `settings`, `data_paths`, and `image_paths`.

2. **`settings` Section:**

   In the `settings` section, specify the constants and simulation parameters:

   - `stable_temperature_solution_1`: The first stable temperature solution in Kelvin (average earth temperature during an ice age).
   - `unstable_temperature_solution`: The unstable temperature solution in Kelvin.
   - `stable_temperature_solution_2`: The second stable temperature solution in Kelvin (average earth temperature during an interglacial age).
   - `surface_earth_thermal_capacity`: Surface thermal capacity of the Earth in J/m²K.
   - `relaxation_time`: Relaxation time of the system in years.
   - `emission_model`: Emission model, choose between 'linear' and 'black body'.
   - `forcing_amplitude`: Amplitude of the applied periodic forcing.
   - `forcing_period`: Period of the applied periodic forcing.

   For stable temperature solutions, ensure that `stable_temperature_solution_1` < `unstable_temperature_solution` < `stable_temperature_solution_2`.

   Configure the simulation parameters:
   - `num_steps`: Number of time steps in the simulation.
   - `num_simulations`: Number of simulations for each noise intensity level.
   - `time_step`: Time step for each simulation in years.
   - `variance_start`: Starting variance for noise intensity.
   - `variance_end`: Ending variance for noise intensity.
   - `num_variances`: Number of variance levels to simulate.

3. **`data_paths` Section:**

   In the `data_paths` section of the configuration file, specify the file paths for data storage as follows:

   - `temperatures_for_emission_models_comparison`: Path to store temperature data used for emission models comparison.
   - `emitted_radiation_for_emission_models_comparison`: Path to store emitted radiation data used for emission models comparison.
   - `F_for_emission_models_comparison`: Path to store rate of temperature change used for emission models comparison.
   - `times_for_evolution_towards_steady_states`: Path to store time data for evolution towards steady states.
   - `temperatures_for_evolution_towards_steady_states`: Path to store temperature data for evolution towards steady states.
   - `times`: Path to store time values.
   - `temperatures`: Path to store temperature values.
   - `frequencies`: Path to store frequency data used for power spectra plots.
   - `averaged_PSD`: Path to store averaged power spectral density data.
   - `peak_heights_in_PSD`: Path to store peak heights data in power spectral density.
   - `times_combinations`: Path to store time data for combinations data.
   - `temperatures_combinations`: Path to store temperature data for combinations data.

4. **`image_paths` Section:**

   In the `image_paths` section of the configuration file, specify paths for storing generated images as follows:

   - `emission_models_comparison_plots`: Path to save plots for emission models comparison.
   - `temperatures_towards_steady_states_plot`: Path to save the plot showing temperature convergence towards stable solutions.
   - `power_spectra_plots`: Path to save power spectra plots.
   - `peak_heights_plot`: Path to save the plot displaying peak heights in power spectral density.
   - `temperature_combinations_plots`: Path to save plots showing temperature evolution for specific noise variance combinations.


Now you can create a custom configuration file by specifying values for these parameters and paths.

To run the simulation with your custom configuration file, follow the same steps described in the "Running the Simulation and Plotting Results" section, but replace 'configuration.txt' with the name of your custom configuration file.


### Project Files

The project consists of the following key files:

- **`configuration.txt`**: The configuration file where simulation parameters and file paths are specified.
- **`stochastic_resonance.py`**: Contains function definitions required for the simulation.
- **`testing.py`**: Contains unit tests for functions in `stochastic_resonance.py`.
- **`simulation.py`**: The main script for running simulations, reading parameters from the configuration file, and saving data to the `data` folder.
- **`plots.py`**: Contains functions for generating plots based on the data generated by `simulation.py`.
- **`aesthetics.py`**: Provides functions for enhancing user interaction, including a progress bar and text formatting.
