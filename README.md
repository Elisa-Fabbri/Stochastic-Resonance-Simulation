# Stochastic-Resonance-Simulation

## General Introduction and Objective of the Repository

Stochastic resonance is a phenomenon observed in nonlinear systems, wherein a weak input signal, like a faint signal, can be enhanced and optimized through the 
presence of noise.
The three basic characteristics are (i) the presence of an energetic activation barrier (or a form of threshold), (ii) a weak coherent input (such as a periodic
signal), and (iii) a source of noise. Given these features, the responce of the system undergoes a resonance-like behaviour as a function of the noise level.

The mechanism of stochastic resonance was first described in the early 1980s by Italian physicists Roberto Benzi, Alfonso Sutera, and Angelo Vulpiani, with the additional participation of Giorgio Parisi[^1]. They applied this concept to climatology with the aim of explaining how small variations in Earth's motions, known as the Milanković cycles, could lead to significant climate variations on Earth. Specifically, they sought to elucidate the transitions between glacial periods, characterized by colder temperatures and extensive ice cover, and interglacial periods, marked by warmer temperatures and reduced ice cover.

The goal of this repository is to use numerical simulations to investigate the stochastic resonance phenomenon within the context of climate dynamics, the field where it was first introduced.

## The stochastic resonance mechanism in climate change

### Observations of Past Climate and the Milankovitch Cycles

Over the past 800,000 years, Earth's climate has undergone cyclical shifts between glacial and interglacial periods, occurring approximately every 100,000 years. These climatic cycles follow a characteristic sawtooth pattern, featuring a rapid warming during transitions to interglacial periods, lasting about 10,000 years, followed by a gradual return to glacial conditions. This process leads to an average Earth temperature variation of approximately 10 degrees Celsius. In addition to this primary periodicity, two minor cyclic fluctuations with periods of about 41,000 and 26,000 years have been observed. Collectively, these three cycles are known as the "Milankovitch cycles," named in honor of the Serbian scientist who, as early as 1941, successfully explained their regularity using Earth's orbital parameters. Specifically, the 100,000-year cycle is associated with variations in Earth's orbital eccentricity (how much the orbit deviates from a perfect circle), the 41,000-year cycle relates to changes in the tilt of Earth's axis, and the 26,000-year cycle is influenced by the precession of the equinoxes.

These variations in orbital parameters, driven by the gravitational influence of other planets in our solar system, are responsible for fluctuations in the average solar radiation reaching our planet. It has been hypothesized that these Milankovitch cycles may have played a crucial role in initiating and concluding Earth's glacial epochs.

However, when considering the global scale, variations in solar irradiance are relatively limited and cannot, by themselves, account for the observed drastic temperature change of 10 degrees Celsius. Therefore, there is a need to identify an amplification mechanism capable of translating these modest solar variations into significant climatic shifts. This project focuses on the proposal put forth by Italian researchers Benzi, Parisi, Sutera, and Vulpiani in the early 1980s, introducing the concept of stochastic resonance as a potential amplification mechanism.

### A Global Energy Balance Model

The fundamental variable in this model is the Earth's average temperature, denoted as *T*. The average is taken over the entire Earth's surface and over a long meteorological time. The value of this variable is determined by the energy balance between incoming and outgoing energy within the system.

The incoming energy is provided by solar radiation, denoted as $R_{in}(T)$, while the outgoing energy $R_{out}(T)$ consists of two components: the radiation emitted by Earth, $\epsilon (T)$, and the fraction of incoming radiation directly reflected by the Earth's surface, $\alpha(T)R_{in}(T)$. For simplicity, this model disregards the contribution of the atmosphere.

In mathematical terms:

$$
C \frac{dT}{dt} = R_{in}(T) - R_{out}(T)
$$

Here, $C$ represents the thermal capacity of Earth [J/m² K]. The incoming radiation can be modeled as:

$$
R_{in}(T) = Q\mu
$$

Where $Q$ is a long-term average of incoming solar radiation, and the dimensionless parameter $\mu$ accounts for the small periodic variation in $Q$ caused by changes in Earth's orbital eccentricity (with a period of 100,000 years).

The outgoing radiation can be described as:

$$
R_{out}(T) = \alpha(T)R_{in}(T) + \epsilon(T)
$$

Here, $\alpha(T)$ represents the global average albedo, which is the fraction of incoming radiation directly reflected by Earth's surface. $\epsilon(T)$ denotes the infrared radiation emitted by Earth. This emitted radiation can be modeled using either Stefan-Boltzmann's law for a blackbody or a linear approximation.

Neglecting the role of albedo and the periodic modulation of incoming radiation, the system has a stable solution where $\frac{dT}{dt} = 0$, resulting in $Q = \epsilon(T)$. The temperature solution, denoted as $T^*$, is a mathematical fixed point. In other words, the system self-regulates to attain this temperature.

However, albedo is not constant; it varies between 0 and 1 and depends on Earth's temperature, mainly influenced by factors like ice cover and cloudiness. Both these factors contribute to an increase in the reflected radiation, i.e., an increase in albedo.

Qualitatively, the temperature dependence of albedo introduces feedback mechanisms in the system, leading to multiple fixed points at different temperatures. Within the temperature range of interest, feedback from ice and clouds results in the presence of two stable fixed points ($T1$ and $T3$) and an unstable fixed point ($T2$).

From a physical standpoint, only the stable fixed points are significant, as the system naturally moves away from an unstable fixed point, making it unobservable. The stable fixed points correspond to Earth's temperatures during glacial and interglacial states, and we assume $T3-T1 = 10 K$.

Now, consider the astronomical modulation of incoming radiation. If the incoming radiation were greatly reduced, there would be a single stable fixed point corresponding to the glacial state ($T1$). Conversely, if incoming radiation were significantly higher, there would be a single fixed point corresponding to the interglacial temperature ($T3$).

However, as mentioned earlier, the astronomical modulation is very small and alone insufficient to eliminate one of the two stable fixed points. The system maintains both stable fixed points but varies the degree of stability between them. The mechanism that allows the system to transition from one stable fixed point to the other, which is more stable, is the stochastic resonance.

### Introducing Meteorological Climate Variability and the Stochastic Resonance Mechanism

Up to this point, our model has overlooked the meteorological variability of the climate, which includes temperature fluctuations around the equilibrium temperature due to various factors. This temperature variability occurs on a much shorter timescale compared to what we have considered so far and can be viewed as a "fast" variable. We can treat this variability as noise directly affecting the slower variables.

The dynamics of the fast process need not be resolved in detail; instead, we can focus on its statistical properties. Moreover, the statistics of fast variables can be assumed to follow a Gaussian distribution since, when averaged over the timescale of slow processes, they result from many independent causes. This fast process can be modeled using a one-dimensional Wiener process.

A one-dimensional Wiener process, also known as Brownian motion, is characterized by:

- A continuous and random trajectory over time. Continuity means that the mapping $t \rightarrow W(t)$ is continuous.
- Independent increments between successive time points. In other words, given time points $0 = t_1 < t_2 < ... < t_N = T$, the increments $W(t_N) - W(t_{N-1}), ..., W(t_1) - W(t_0)$ are independent.
- Increments follow a normal (Gaussian) distribution with zero mean and variance proportional to the time interval. In mathematical terms, for $\Delta W = W(t) - W(s)$, we have $E[\Delta W] = 0$ and $V[\Delta W] = t - s$.

We can now revisit the previous energy balance equation and introduce a new term to model temperature variability. In this equation, denoting $F(T) = \frac{1}{C}(R_{in}(T) - R_{out}(T))$, we introduce the new term:

$$
\frac{dT}{dt} = F(T) + \sigma \frac{dW}{dt}
$$

Here, $\sigma^2 [K^2/year]$ represents the noise variance, and $W(t)$ is the standard Wiener process. This last equation is a stochastic differential equation (SDE) describing the evolution of temperature in our model.

It can be solved numerically using various procedures. In this project, the Forward Euler method was employed to simulate the temperature's evolution.

#### Forward Euler Discretization

Here's how the forward Euler numerical integration works:

- **Discretization**: The Forward Euler method approximates solutions of differential equations by breaking the continuous-time domain into discrete time steps ($\Delta t$).

- **Iterative Process**: It starts from an initial condition and iteratively updates the system's state at each time step.

- **Temperature Update**:
  - To account for the stochastic temperature variability (from the Wiener process), random numbers following a normal distribution (zero mean, variance equal to $\Delta t$) are generated at each time step.
  - The temperature change $dT$ is then found by adding the deterministic temperature change, $F(T)$, and the stochastic temperature change, $\sigma * W$.
  - The new temperature value $T(t_n)$ is then found by adding to the previuos temperature value $T(t_{n-1})$ che computed temperature change $dT$. 
  - The results is the new temperature value for the next time step.

- **Repeat**: The process repeats for the specified number of time steps, allowing us to simulate temperature evolution over time.

The Forward Euler discretization method allows us to simulate temperature evolution over time, considering both deterministic changes as per the energy balance equation and stochastic fluctuations captured by random numbers from the Wiener process.



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

## References

[^1]: R. Benzi, A. Sutera e A. Vulpiani, "The mechanism of stochastic resonance," J. Phys. A 14, L453 (1981).