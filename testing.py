import numpy as np
import stochastic_resonance as sr
import pytest
import random as rn

# Test the functions in stochastic_resonance.py

#Test emitted_radiation_function in the linear case 

steady_temperature_solutions_default_linear_test = [
    (sr.stable_temperature_solution_1_default, -339.647 + 2.218*280), # glacial temperature
    (sr.unstable_temperature_solution_default, -339.647 + 2.218*sr.unstable_temperature_solution_default), # ustable temperature solution
    (sr.stable_temperature_solution_2_default, -339.647 + 2.218*sr.stable_temperature_solution_2_default), # interglacial temperature
]

@pytest.mark.parametrize("steady_temperature, expected_radiation", steady_temperature_solutions_default_linear_test)
def test_emitted_radiation_linear(steady_temperature, expected_radiation):
    """
    Test the linear emitted radiation in W/m^2 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = 1
    THEN: the result should be the expected emitted radiation in W/m^2
    """
    
    expected_value = expected_radiation
    calculated_value = sr.emitted_radiation(steady_temperature, emission_model='linear', conversion_factor=1)
    
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

@pytest.mark.parametrize("steady_temperature, expected_radiation_W_m2", steady_temperature_solutions_default_linear_test)
def test_emitted_radiation_conversion(steady_temperature, expected_radiation_W_m2):
    """
    Test the linear emitted radiation in kg/m^3 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = num_sec_in_a_year
    THEN: the result should be the expected emitted radiation in kg/year^3
    """
    
    expected_value = expected_radiation_W_m2*(sr.num_sec_in_a_year**3)
    calculated_value = sr.emitted_radiation(steady_temperature, emission_model='linear', conversion_factor=sr.num_sec_in_a_year)
    
    assert calculated_value == pytest.approx(expected_value, rel=1)

#Test emitted_radiation_function in the black body case

steady_temperature_solutions_default_black_body_test = [
    (sr.stable_temperature_solution_1_default, 5.67e-8*sr.stable_temperature_solution_1_default**4), # glacial temperature
    (sr.unstable_temperature_solution_default, 5.67e-8*sr.unstable_temperature_solution_default**4), # ustable temperature solution
    (sr.stable_temperature_solution_2_default, 5.67e-8*sr.stable_temperature_solution_2_default**4), # interglacial temperature
]

@pytest.mark.parametrize("steady_temperature, expected_radiation", steady_temperature_solutions_default_black_body_test)
def test_emitted_radiation_black_body(steady_temperature, expected_radiation):
    """
    Test the black body emitted radiation in W/m^2 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = 1
    THEN: the result should be the expected emitted radiation in W/m^2
    """
    
    expected_value = expected_radiation
    calculated_value = sr.emitted_radiation(steady_temperature, emission_model='black body', conversion_factor=1)
    
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

@pytest.mark.parametrize("steady_temperature, expected_radiation_W_m2", steady_temperature_solutions_default_black_body_test)
def test_emitted_radiation_conversion_black_body(steady_temperature, expected_radiation_W_m2):
    """
    Test the black body emitted radiation in kg/m^3 for temperature solutions default parameters.

    GIVEN: a steady temperature and the expected emitted radiation in K
    WHEN: the emitted_radiation function is called with conversion_factor = num_sec_in_a_year
    THEN: the result should be the expected emitted radiation in kg/year^3
    """
    
    expected_value = expected_radiation_W_m2*(sr.num_sec_in_a_year**3)
    calculated_value = sr.emitted_radiation(steady_temperature, emission_model='black body', conversion_factor=sr.num_sec_in_a_year)
    
    assert calculated_value == pytest.approx(expected_value, rel=1)

# Test the emitted_radiation function for invalid emission_model selection

def test_invalid_emitted_radiation_model():
    """Test the emitted_radiation function for an invalid emission model selection.
    
    GIVEN: an invalid emission model selection
    WHEN: the emitted_radiation function is called
    THEN: a ValueError should be raised
    """
    with pytest.raises(ValueError, match='Invalid emission model selection. Choose "linear" or "black body".'):
        sr.emitted_radiation(0, emission_model='invalid', conversion_factor=1)

    with pytest.raises(ValueError, match='Invalid emission model selection. Choose "linear" or "black body".'):
        sr.emitted_radiation(0, emission_model='invalid', conversion_factor=sr.num_sec_in_a_year)

#Test the emitted_radiation for negative negative temperature value

def test_invalid_emitted_radiation_temperature():
    """Test the emitted_radiation function for a negative temperature value.
    
    GIVEN: a negative temperature value
    WHEN: the emitted_radiation function is called
    THEN: a ValueError should be raised
    """
    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='linear', conversion_factor=1)

    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='linear', conversion_factor=sr.num_sec_in_a_year)

    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='black body', conversion_factor=1)

    with pytest.raises(ValueError, match='Temperature in Kelvin must be non-negative.'):
        sr.emitted_radiation(-1, emission_model='black body', conversion_factor=sr.num_sec_in_a_year)


# Test periodic_forcing function

def test_periodic_forcing_max():
    """
    Test that the periodic forcing function returns the maximum value of the forcing amplitude 
    plus one when the time is equal to the forcing period.

    GIVEN: time equal to the forcing period and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be the maximum value of the forcing amplitude plus one
    """
    expected_value = 1 + sr.forcing_amplitude_default
    calculated_value = sr.periodic_forcing(sr.forcing_period_default, sr.forcing_amplitude_default)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

def test_periodic_forcing_min():
    """
    Test that the periodic forcing function returns one minus the value of the forcing amplitude 
    when the time is equal to the forcing period divided by two.

    GIVEN: time equal to the forcing period and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be one minus the value of the forcing amplitude
    """
    expected_value = 1 - sr.forcing_amplitude_default
    calculated_value = sr.periodic_forcing(sr.forcing_period_default/2, sr.forcing_amplitude_default)
    assert calculated_value == pytest.approx(expected_value, rel=1e-6)

def test_periodic_forcing_is_one():
    """
    Test that the periodic forcing function returns one when the time is equal to the forcing period divided by four.

    GIVEN: time equal to the forcing period divided by four and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be one
    """
    expected_value = 1
    calculated_value = sr.periodic_forcing(sr.forcing_period_default/4, sr.forcing_amplitude_default)
    assert sr.periodic_forcing(sr.forcing_period_default/4, sr.forcing_amplitude_default) == expected_value
    
def test_periodic_forcing_for_list():
    """
    Test that the periodic forcing function returns a list of values when the time is a list of values.

    GIVEN: time is a list of values and the forcing amplitude
    WHEN: the periodic_forcing function is called
    THEN: the result should be a list of values containig the ordered periodic forcing values
    """
    expected_value = [1 - sr.forcing_amplitude_default, 1, 1 + sr.forcing_amplitude_default]
    calculated_value = sr.periodic_forcing([sr.forcing_period_default/2, 
                                            sr.forcing_period_default/4, 
                                            sr.forcing_period_default], 
                                            sr.forcing_amplitude_default)
    assert np.array_equal(calculated_value, expected_value)

def test_periodic_forcing_no_amplitude():
    """ 
    Test that the periodic forcing function returns one when the forcing amplitude is zero.

    GIVEN: a list of values for the time and a forcing amplitude of zero
    WHEN: the periodic_forcing function is called
    THEN: the result should be a list of ones
    """
    expected_value = [1, 1, 1]
    calculated_value = sr.periodic_forcing([0, sr.forcing_period_default, 2], 0)
    assert np.array_equal(calculated_value, expected_value)


# Test calculate_rate_of_temperature_change function

steady_temperature_solutions = [
    (sr.stable_temperature_solution_1_default),
    (sr.unstable_temperature_solution_default),
    (sr.stable_temperature_solution_2_default)
]

@pytest.mark.parametrize("steady_temperature", steady_temperature_solutions)
def test_calculate_rate_of_temperature_change_is_zero(steady_temperature):
    """ 
    Test that the rate of temperature change is zero for the steady temperature solutions. 

    GIVEN: a steady temperature
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be zero
    """
    expected_value = 0
    calculated_value = sr.calculate_rate_of_temperature_change(steady_temperature)
    assert calculated_value == expected_value

def test_calculate_rate_of_temperature_change_stable_1():
    """
    Test the rate of temperature change near the first stable solution.

    GIVEN: a temperature value close to the first stable solution
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be negative if the input temperature is greater than the first stable solution,
            and positive if the input temperature is less than the first stable solution.
    """
    epsilon = 1  # small value to test near the stable solution

    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_1_default + epsilon) < 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default - epsilon) > 0

def test_calculate_rate_of_temperature_change_stable_2():
    """
    Test the rate of temperature change near the second stable solution.

    GIVEN: a temperature value close to the second stable solution
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be negative if the input temperature is greater than the second stable solution,
            and positive if the input temperature is less than the second stable solution.
    """
    epsilon = 1  # small value to test near the stable solution

    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default + epsilon) < 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default - epsilon) > 0

def test_calculate_rate_of_temperature_change_unstable():
    """
    Test the rate of temperature change near the unstable solution.

    GIVEN: a temperature value close to the unstable solution
    WHEN: the calculate_rate_of_temperature_change function is called
    THEN: the result should be positive if the input temperature is greater than the unstable solution,
          and negative if the input temperature is less than the unstable solution.
    """
    epsilon = 1  # small value to test near the unstable solution

    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default + epsilon) > 0
    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default - epsilon) < 0

# Test for emission_models_comparison_function

def test_emission_models_comparison_temperature_range():
    """
    Test that the temperature values returned by the emission_models_comparison function are within the correct range
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the temperature values returned should be within the correct range
    """

    calculated_values = sr.emission_models_comparison()
    assert (np.all(calculated_values[0] >= sr.stable_temperature_solution_1_default-10) and
           np.all(calculated_values[0] <= sr.stable_temperature_solution_2_default+10))

def test_emission_models_comparison_temperature_length():
    """
    Test that the temperature values returned by the emission_models_comparison function have the correct length
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the temperature values returned should have the correct length (num_points)
    """

    num_points = int((sr.stable_temperature_solution_2_default+10) - 
                     (sr.stable_temperature_solution_1_default-10))*10
    expected_value = num_points
    calculated_values = sr.emission_models_comparison()
    assert len(calculated_values[0]) == expected_value

def test_emission_models_comparison_emitted_radiation_shape():
    """
    Test that the emitted radiation values returned by the emission_models_comparison function have the correct shape
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the emitted radiation values returned should have the correct shape (2, num_points)
    """

    num_points = int((sr.stable_temperature_solution_2_default+10) - 
                    (sr.stable_temperature_solution_1_default-10))*10
    expected_value = (2, num_points)
    calculated_values = sr.emission_models_comparison()
    assert calculated_values[1].shape == expected_value

def test_emission_models_comparison_dT_dt_shape():
    """
    Test that the dT_dt values returned by the emission_models_comparison function have the correct shape
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the dT_dt values returned should have the correct shape (2, num_points)
    """

    num_points = int((sr.stable_temperature_solution_2_default+10) - 
                    (sr.stable_temperature_solution_1_default-10))*10
    expected_value = (2, num_points)
    calculated_values = sr.emission_models_comparison()
    assert calculated_values[2].shape == expected_value

def test_emission_models_comparison_temperature_type():
    """
    Test that the temperature values returned by the emission_models_comparison function are of type numpy.ndarray
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the temperature values returned should be of type numpy.ndarray
    """
    calculated_values = sr.emission_models_comparison()
    assert type(calculated_values[0]) == np.ndarray

def test_emission_models_comparison_emitted_radiation_type():
    """
    Test that the emitted radiation values returned by the emission_models_comparison function are of type numpy.ndarray
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the emitted radiation values returned should be of type numpy.ndarray
    """
    calculated_values = sr.emission_models_comparison()
    assert type(calculated_values[1]) == np.ndarray

def test_emission_models_comparison_dT_dt_type():
    """
    Test that the dT_dt values returned by the emission_models_comparison function are of type numpy.ndarray
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the dT_dt values returned should be of type numpy.ndarray
    """
    calculated_values = sr.emission_models_comparison()
    assert type(calculated_values[2]) == np.ndarray

def test_emission_models_comparison_temperature_values():
    """
    Test that the temperature values returned by the emission_models_comparison function are correct
    when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the temperature values returned should be correct
    """
    expected_value = np.linspace(sr.stable_temperature_solution_1_default-10 , 
                                 sr.stable_temperature_solution_2_default+10, 
                                 int(((sr.stable_temperature_solution_2_default+10) -
                                      (sr.stable_temperature_solution_1_default -10 ))*10))
    calculated_values = sr.emission_models_comparison()
    assert np.array_equal(calculated_values[0], expected_value)

def test_emission_models_comparison_emitted_radiation_values_type():
    """
    Test that the elements contained in the emitted radiation values returned by the 
    emission_models_comparison function are real when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the elements contained in the emitted radiation values returned should be real
    """
    calculated_values = sr.emission_models_comparison()
    assert np.isreal(calculated_values[1]).all()

def test_emission_models_comparison_dT_dt_values_type():
    """
    Test that the elements contained in the dT_dt values returned by the 
    emission_models_comparison function are real when the default parameters are used.

    GIVEN: the default parameters
    WHEN: the emission_models_comparison function is called
    THEN: the elements contained in the dT_dt values returned should be real
    """
    calculated_values = sr.emission_models_comparison()
    assert np.isreal(calculated_values[2]).all()


#Test for simulate_ito function

def test_simulate_ito_time_length():
    """
    Test if the time returned by the simulate_ito function have the correct length.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the time returned should have the correct length (num_steps)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    expected_value = 100
    assert len(calculated_values[0]) == expected_value

def test_simulate_ito_temperature_shape():
    """
    Test if the temperature returned by the simulate_ito function have the correct shape.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the temperature returned should have the correct shape (num_simulations, num_steps)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    expected_value = (sr.num_simulations_default, 100)
    assert calculated_values[1].shape == expected_value

def test_simulate_ito_time_type():
    """
    Test if the time returned by the simulate_ito function is of type numpy.ndarray.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the time returned should be of type numpy.ndarray
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    assert type(calculated_values[0]) == np.ndarray

def test_simulate_ito_temperature_type():
    """
    Test if the temperature returned by the simulate_ito function is of type numpy.ndarray.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito function is called
    THEN: the temperature returned should be of type numpy.ndarray
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito(
        num_steps=100,
        )
    assert type(calculated_values[1]) == np.ndarray


@pytest.mark.parametrize("T_start", steady_temperature_solutions)
def test_simulate_ito_no_forcing_no_noise(T_start):
    """
    Test if the temperature returned by the simulate_ito function is equal to the initial temperature
    when no forcing and no noise are applied and the initial temperature is a steady solution.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100), no forcing and no noise and
              the initial temperature is a steady solution
    WHEN: the simulate_ito function is called
    THEN: the temperature returned should be equal to the initial temperature
    """

    calculated_values = sr.simulate_ito(
        T_start=T_start,
        num_steps=100,
        noise=False,
        forcing='constant'
    )
    expected_value = T_start
    assert np.all(calculated_values[1] == expected_value)


# Test calculate_evolution_towards_steady_states function

def calculate_evolution_towards_steady_states_time_length():
    """
    Test if the time returned by the calculate_evolution_towards_steady_states function have the correct length.

    GIVEN: the default parameters
    WHEN: the calculate_evolution_towards_steady_states function is called
    THEN: the time returned should have the correct length (num_steps = 100)
    """

    calculated_values = sr.calculate_evolution_towards_steady_states()
    expected_value = 100
    assert len(calculated_values[0]) == expected_value

def calculate_evolution_towards_steady_states_temperature_shape():
    """
    Test if the temperature returned by the calculate_evolution_towards_steady_states function have the correct shape.

    GIVEN: the default parameters
    WHEN: the calculate_evolution_towards_steady_states function is called
    THEN: the temperature returned should have the correct shape (num_simulations = 9, num_steps = 100)
    """

    calculated_values = sr.calculate_evolution_towards_steady_states()
    
    T_start_length = 9
    expected_value = (T_start_length, 100)

    assert calculated_values[1].shape == expected_value

def calculate_evolution_towards_steady_states_time_type():
    """
    Test if the time returned by the calculate_evolution_towards_steady_states function is of type numpy.ndarray.

    GIVEN: the default parameters
    WHEN: the calculate_evolution_towards_steady_states function is called
    THEN: the time returned should be of type numpy.ndarray
    """
    calculated_values = sr.calculate_evolution_towards_steady_states()
    assert type(calculated_values[0]) == np.ndarray

def calculate_evolution_towards_steady_states_temperature_type():
    """
    Test if the temperature returned by the calculate_evolution_towards_steady_states function is of type numpy.ndarray.

    GIVEN: the default parameters
    WHEN: the calculate_evolution_towards_steady_states function is called
    THEN: the temperature returned should be of type numpy.ndarray
    """
    calculated_values = sr.calculate_evolution_towards_steady_states()
    assert type(calculated_values[1]) == np.ndarray

def test_calculate_evolution_towards_steady_states_temperature_convergence():
    """
    Test if the temperature returned by the calculate_evolution_towards_steady_states function converges to the
    correct steady state values.

    GIVEN: the default parameters
    WHEN: the calculate_evolution_towards_steady_states function is called
    THEN: the final temperatures should be near to the steady state values
    """

    calculated_values = sr.calculate_evolution_towards_steady_states()
    
    final_temperatures = calculated_values[1][:, -1]
    
    for final_temperature in final_temperatures:
        assert (final_temperature == pytest.approx(sr.stable_temperature_solution_1_default, rel=1) or
                final_temperature == pytest.approx(sr.unstable_temperature_solution_default, rel=1) or
                final_temperature == pytest.approx(sr.stable_temperature_solution_2_default, rel=1))

# Test find_peak_indices function

def test_find_peak_indices_same_peak_index():
    """
    This function tests the find_peak_indices function when the peak index is the same for all simulations (rows).
    
    GIVEN: a frequency array with equal rows and a period
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array of the same peak index for all simulations"""

    frequency = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    period = 1.0
    calculated_values = sr.find_peak_indices(frequency, period)
    assert np.array_equal(calculated_values, np.array([0, 0]))

def test_find_peak_indices_different_peak_index():
    """
    This function tests the find_peak_indices function when the peak index is different for all simulations (rows).
    
    GIVEN: a frequency array with different rows and a period
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array of different peak indices for all simulations"""

    frequency = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    period = 0.5
    calculated_values = sr.find_peak_indices(frequency, period)
    assert np.array_equal(calculated_values, np.array([1, 0]))

def test_find_peak_indices_approximate_peak_index():
    """
    This function tests the find_peak_indices function when the peak index is not exact.

    GIVEN: a period and a frequency array in which the value 1/period is not contained
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array of approximate peak indices for all simulations
    """
    
    frequencies = np.array([[3, 4], [0, 1]])
    period = 0.5
    calculated_values = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(calculated_values, np.array([0, 1]))

def test_find_peak_indices_multiple():
    """
    This function tests the find_peak_indices function when the desired frequency is contained more that once.

    GIVEN: a period and a frequency array in which the value 1/period is contained more than once
    WHEN: the find_peak_indices function is called
    THEN: the result should be an array containing just the first occurrence of 1/period for each row.
    """

    frequencies = np.array([[1,1,1], [1, 1, 1]])
    period = 1
    calculated_values = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(calculated_values, np.array([0, 0]))

# Test for calculate_peaks function

def test_calculate_peaks():
    """This function tests the calculate_peaks function."""

    PSD_mean = np.array([[10, 20, 30, 40, 50], [5, 15, 25, 35, 45]])
    peaks_indices = np.array([1, 3])

    expected_peaks = np.array([20, 35])
    calculated_peaks = sr.calculate_peaks(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks, expected_peaks)

# Test for calculate_peaks_base function

def calculate_peaks_base_normal_behaviour():
    """
    This function tests the calculate_peaks_base function when the peaks are not at the beginning or at the end of the
    frequency array and the number of neighbours is 2.

    GIVEN: a PSD_mean array, a peaks_indices array and a number of neighbours
    WHEN: the calculate_peaks_base function is called
    THEN: the result should be an array containing the mean of the values of the four neighbours of the peaks
        (two on the right side and two on the left side).
    """
    
    PSD_mean = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    peaks_indices = np.array([2, 6])

    expected_peaks_base = np.array([3, 8])

    calculated_peaks_base = sr.calculate_peaks_base(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks_base, expected_peaks_base)

def test_calculate_peaks_base_near_borders():
    """
    This function tests the calculate_peaks_base function when the peaks are at the beginning or at the end of the
    frequency array and the number of neighbours is 2.

    GIVEN: a PSD_mean array, a peaks_indices array and a number of neighbours
    WHEN: the calculate_peaks_base function is called
    THEN: the result should be an array containing the mean of the values of the existing neighbours of the peaks.
    """
    
    PSD_mean = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    peaks_indices = np.array([0, 8])

    expected_peaks_base = np.array([2.5, 8.5])

    calculated_peaks_base = sr.calculate_peaks_base(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks_base, expected_peaks_base)


# Test for calculate_peak_height function

def test_calculate_peak_height_empty():
    """
    This function tests the calculate_peak_height function when the peaks and the base values are empty arrays.

    GIVEN: empty arrays for peaks and base values
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an empty array
    """

    peaks = np.array([])
    peaks_base = np.array([])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([]))


def test_calculate_peak_height_one_element():
    """
    This function tests the calculate_peak_height function when the peaks and the base values have one element.

    GIVEN: peaks and base values with one element
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an array containing the difference between the two elements
    """

    peaks = np.array([5.0])
    peaks_base = np.array([2.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([3.0]))  

def test_calculate_peak_height_multiple_elements():
    """
    This function tests the calculate_peak_height function when the peaks and the base values have multiple elements.

    GIVEN: peaks and base values with multiple elements
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an array containing the difference between the corresponding elements
    """
    peaks = np.array([4.0, 7.0, 10.0, 6.0])
    peaks_base = np.array([2.0, 3.0, 8.0, 4.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    expected_heights = np.array([2.0, 4.0, 2.0, 2.0])
    assert np.array_equal(peak_height, expected_heights)   

def test_calculate_peak_height_zero():
    """
    This function tests the calculate_peak_height function when the peaks and the base values are all zeros

    GIVEN: peaks and base values with all zeros
    WHEN: the calculate_peak_height function is called
    THEN: the result should be an array containing zeros
    """
    peaks = np.array([0.0, 0.0, 0.0])
    peaks_base = np.array([0.0, 0.0, 0.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([0.0, 0.0, 0.0]))

# Test for simulate_ito_combinations_and_collect_results function

def test_simulate_ito_combinations_and_collect_results_time_length():
    """
    Test if the time returned by the simulate_ito_combinations_and_collect_results function is of length 4.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito_combinations_and_collect_results function is called
    THEN: the time returned should have the correct length (4)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito_combinations_and_collect_results(
        num_steps=100,
        )
    expected_value = 4
    assert len(calculated_values[0]) == expected_value

def test_simulate_ito_combinations_and_collect_results_temperature_length():
    """
    Test if the temperature returned by the simulate_ito_combinations_and_collect_results function is of length 4.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito_combinations_and_collect_results function is called
    THEN: the temperature returned should have the correct length (4)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito_combinations_and_collect_results(
        num_steps=100,
        )
    expected_value = 4
    assert len(calculated_values[1]) == expected_value

def test_simulate_ito_combinations_and_collect_results_time_type():
    """
    Test if the time returned by the simulate_ito_combinations_and_collect_results function is of type list.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito_combinations_and_collect_results function is called
    THEN: the time returned should be of type list
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito_combinations_and_collect_results(
        num_steps=100,
        )
    assert isinstance(calculated_values[0], list)

def test_simulate_ito_combinations_and_collect_results_temperature_type():
    """
    Test if the temperature returned by the simulate_ito_combinations_and_collect_results function is of type list.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito_combinations_and_collect_results function is called
    THEN: the temperature returned should be of type list
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito_combinations_and_collect_results(
        num_steps=100,
        )
    assert isinstance(calculated_values[1], list)

def test_simulate_ito_combinations_and_collect_results_time_sequences_length():
    """
    Test if the time sequences returned by the simulate_ito_combinations_and_collect_results function have the correct length.

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito_combinations_and_collect_results function is called
    THEN: the time sequences returned should have the correct length (num_steps = 100)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito_combinations_and_collect_results(
        num_steps=100,
        )
    expected_value = 100
    for time_sequence in calculated_values[0]:
        assert len(time_sequence) == expected_value

def test_simulate_ito_combinations_and_collect_results_temperature_sequences_shape():
    """
    Test if the temperature sequences returned by the simulate_ito_combinations_and_collect_results function have the correct shape

    GIVEN: the default parameters with reduced number of steps (num_steps = 100)
    WHEN: the simulate_ito_combinations_and_collect_results function is called
    THEN: the temperature sequences returned should have the correct shape (num_simulations = 1, num_steps = 100)
    """
    rn.seed(42)
    calculated_values = sr.simulate_ito_combinations_and_collect_results(
        num_steps=100,
        )
    expected_value = (1, 100)
    for temperature_sequence in calculated_values[1]:
        assert temperature_sequence.shape == expected_value