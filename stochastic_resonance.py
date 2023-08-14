import numpy as np


def emitted_radiation(Temperature, model='linear'):
	"""
	This function computes the emitted radiation based on the selected model.
	"""

	number_of_seconds_in_a_year = 365 * 24 * 60 * 60

	if model == 'linear':
		A = -339.647 * (number_of_seconds_in_a_year ** 3)  # -339.647 W/m^2
		B = 2.218 * (number_of_seconds_in_a_year ** 3)  # 2.218 W/(m^2 K)
		Emitted_radiation = A + B * Temperature
	elif model == 'black_body':
		Stefan_Boltzmann_constant = 5.67e-8 * (number_of_seconds_in_a_year ** 3)  # W/(m^2 K^4)
		Emitted_radiation = Stefan_Boltzmann_constant * (Temperature ** 4)
	else:
		raise ValueError("Invalid model selection. Choose 'linear' or 'black_body'.")

	return Emitted_radiation


def periodic_forcing(A, omega, time):
	"""
	This function computes the periodic forcing applied to the system
	"""
	return 1+A*np.cos(omega*time)


def F(Temperature, C, tau, T3, T1, T2, model='linear', mu=1):
	"""
	This function returns the value of F(T) given a certain value for the temperature using a specified emission model for the computation of beta.
	"""
	if model == 'linear':
		emitted_radiation_func = linear_emitted_radiation
	elif model == 'black_body':
		emitted_radiation_func = black_body_emitted_radiation
	else:
		raise ValueError("Invalid emission model selection. Choose 'linear' or 'black_body'.")

	beta = -((C / (tau * emitted_radiation_func(T3))) * ((T1 * T2 * T3) / ((T1 - T3) * (T2 - T3))))
	F_value = (emitted_radiation_func(Temperature) / C) * ((mu / (1 + beta * (1 - (Temperature / T1)) * (1 - (Temperature / T2)) * (1 - (Temperature / T3)))) - 1)

	return F_value

