import numpy as np
import stochastic_resonance as sr
import pytest

# Test the functions in stochastic_resonance.py

#Test emitted_radiation_function in the linear case 

steady_temperature_solutions_default_linear_test = [
    (280, -339.647 + 2.218*280), # glacial temperature
    (285, -339.647 + 2.218*285), # ustable temperature solution
    (290, -339.647 + 2.218*290), # interglacial temperature
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
    (280, 5.67e-8*280**4), # glacial temperature
    (285, 5.67e-8*285**4), # ustable temperature solution
    (290, 5.67e-8*290**4), # interglacial temperature
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

def test_periodic_forcing_no_amplitude():
    """ Test the periodic forcing function when the amplitude is zero. """
    amplitude = 0
    assert sr.periodic_forcing(1, amplitude) == 1

def test_periodic_forcing_for_time_0():
    """ Test the periodic forcing function for time = 0. """
    assert sr.periodic_forcing(time = 0, amplitude = 1) == 2
    assert sr.periodic_forcing(time = 0) == 1 + sr.forcing_amplitude_default

# Test calculate_rate_of_temperature_change function

def test_calculate_rate_of_temperature_change_for_steady_states_solution():
    """ Test the rate of temperature change function for steady state solutions. """
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_1_default) == 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default) == 0
    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default) == 0

def test_sign_calculate_rate_of_temperature_change():
    """ Test the sign of the rate of temperature change function. """
    epsilon = 1
    # Temperature values near the first stable solution should converge to it.
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_1_default + epsilon) < 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_1_default - epsilon) > 0
    # Temperature values near the unstable solution should diverge from it.
    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default + epsilon) > 0
    assert sr.calculate_rate_of_temperature_change(sr.unstable_temperature_solution_default - epsilon) < 0
    #Temperature values near the second stable solution should converge to it.
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default + epsilon) < 0
    assert sr.calculate_rate_of_temperature_change(sr.stable_temperature_solution_2_default - epsilon) > 0

# Test for emission_models_comparison_function

#Test for simulate_ito function

def test_simulate_ito_no_noise_no_forcing():
    """ Test the simulate_ito function for no noise and no forcing."""

    noise = False  
    forcing = 'constant' 
    num_steps = 100
    num_simulations = 1

    T_start = sr.stable_temperature_solution_1_default

    t, T = sr.simulate_ito(
        T_start=T_start,
        num_steps= num_steps,
        num_simulations=num_simulations,
        noise=noise,
        forcing=forcing
        )
    assert np.all(T == T_start)

    T_start = sr.stable_temperature_solution_2_default

    t, T = sr.simulate_ito(
        T_start=T_start,
        num_steps=num_steps,
        num_simulations=num_simulations,
        noise=noise,
        forcing=forcing
        )
    assert np.all(T == T_start)

    T_start = sr.unstable_temperature_solution_default

    t, T = sr.simulate_ito(
        T_start=T_start,
        num_steps=num_steps,
        num_simulations=num_simulations,
        noise=noise,
        forcing=forcing
        )
    assert np.all(T == T_start)

def test_simulate_ito_no_noise_with_forcing():
    """Test the simulate_ito function for no noise and forcing."""

    noise = False  
    forcing = 'varying' 
    num_steps = 100
    num_simulations = 1
    max_value_forcing = 1 + sr.forcing_amplitude_default
    min_value_forcing = 1 - sr.forcing_amplitude_default

    T_start = sr.stable_temperature_solution_1_default

    t, T = sr.simulate_ito(
        T_start=T_start,
        num_steps= num_steps,
        num_simulations=num_simulations,
        noise=noise,
        forcing=forcing
        )
    assert np.all((T >= T_start - min_value_forcing) & (T <= T_start + max_value_forcing))

    T_start = sr.stable_temperature_solution_2_default

    t, T = sr.simulate_ito(
        T_start=T_start,
        num_steps=num_steps,
        num_simulations=num_simulations,
        noise=False,
        forcing=forcing
        )
    assert np.all((T >= T_start - min_value_forcing) & (T <= T_start + max_value_forcing))

    T_start = sr.unstable_temperature_solution_default

# Test calculate_evolution_towards_steady_states function

# Test find_peak_indices function

def test_find_peak_indices():
    """This function tests the find_peak_indices function."""

    frequency = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9]])
    period = 1.0
    peak_indices = sr.find_peak_indices(frequency, period)
    assert np.array_equal(peak_indices, np.array([0, 0]))

    frequencies = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]])
    peak_indices = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(peak_indices, np.array([4, 4]))

    period = 2.0
    peak_indices = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(peak_indices, np.array([4, 3]))

    frequencies = np.array([[0.5, 0.1, 0.5],[0.5, 0.5, 0.5]])
    peak_indices = sr.find_peak_indices(frequencies, period)
    assert np.array_equal(peak_indices, np.array([0, 0]))

# Test for calculate_peaks function

def test_calculate_peaks():
    """This function tests the calculate_peaks function."""

    PSD_mean = np.array([[10, 20, 30, 40, 50], [5, 15, 25, 35, 45]])
    peaks_indices = np.array([1, 3])

    expected_peaks = np.array([20, 35])
    calculated_peaks = sr.calculate_peaks(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks, expected_peaks)

# Test for calculate_peaks_base function


def test_calculate_peaks_base():
    PSD_mean = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9],
                         [2, 3, 4, 5, 6, 7, 8, 9, 10]])
    peaks_indices = np.array([2, 6])

    # Expected result based on the sample data and default num_neighbors (2)
    expected_peaks_base = np.array([3, 8])

    calculated_peaks_base = sr.calculate_peaks_base(PSD_mean, peaks_indices)

    assert np.array_equal(calculated_peaks_base, expected_peaks_base)

    peaks_indices = np.array([0, 8])
    expected_peaks_base = np.array([2.5, 8.5])

    calculated_peaks_base = sr.calculate_peaks_base(PSD_mean, peaks_indices)    
    assert np.array_equal(calculated_peaks_base, expected_peaks_base) 

# Test for calculate_peak_height function

def test_calculate_peak_height():
    # Case 1: Peaks and base values are both empty arrays
    peaks = np.array([])
    peaks_base = np.array([])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([]))

    # Case 2: Peaks and base values have one element
    peaks = np.array([5.0])
    peaks_base = np.array([2.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([3.0]))

    # Case 3: Peaks and base values have multiple elements
    peaks = np.array([4.0, 7.0, 10.0, 6.0])
    peaks_base = np.array([2.0, 3.0, 8.0, 4.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    expected_heights = np.array([2.0, 4.0, 2.0, 2.0])
    assert np.array_equal(peak_height, expected_heights)

    # Case 4: Peaks and base values are both zero
    peaks = np.array([0.0, 0.0, 0.0])
    peaks_base = np.array([0.0, 0.0, 0.0])
    peak_height = sr.calculate_peak_height(peaks, peaks_base)
    assert np.array_equal(peak_height, np.array([0.0, 0.0, 0.0]))

# Test for simulate_ito_combinations_and_collect_results function