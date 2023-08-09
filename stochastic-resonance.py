import numpy as np

number_of_seconds_in_a_year = 365.25*24*60*60

def linear_emitted_radiation(Temperature):

	"""
	This function computes the emitted radiation using a linear model.
	"""
	A = -339.647*(number_of_seconds_in_a_year**3) #-339.647 W/m^2
	B = 2.218*(number_of_seconds_in_a_year**3) #2.218 W/(m^2 K)
	Emitted_radiation_linear = A + B*Temperature
	return Emitted_radiation_linear

def black_body_emitted_radiation(Temperature):
	"""
	This function computes the emitted radiation using the black body model
	"""
	Stefan_Boltzmann_constant = 5.67e-8*(number_of_seconds_in_a_year**3) #W/(m^2 K^4)
	return Stefan_Boltzmann_constant*(Temperature**4)
