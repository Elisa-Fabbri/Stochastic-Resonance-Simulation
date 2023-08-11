import numpy as np

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

def periodic_forcing(A, omega, time):
	"""
	This function computes the periodic forcing applied to the system
	"""
	return 1+A*np.cos(omega*time)

def F_linear(Temperature, C, tau, T3, T1, T2, mu = 1):
	"""
	This function returns the value of F(T) given a certain value for the temperature using a linear model for the computation of beta.
	"""
	beta_linear = -((C/(tau*linear_emitted_radiation(T3)))*((T1*T2*T3) / ((T1 - T3)*(T2 - T3))))
	F_linear = (linear_emitted_radiation(Temperature)/C)*((mu/(1 + beta_linear*(1-(Temperature/T1))*(1-(Temperature/T2))*(1-(Temperature/T3))))-1)
	return F_linear

def F_black_body(Temperature, C, tau, T3, T1, T2, mu = 1):
	"""
	This function returns the value of F(T) given a certain value for the temperature using the black body radiation model for the computation of beta.
	"""
	beta_black_body = -((C/(tau*black_body_emitted_radiation(T3)))*((T1*T2*T3) / ((T1 - T3)*(T2 - T3))))
	F_black_body = (black_body_emitted_radiation(Temperature)/C)*((mu/(1 + beta_black_body*(1-(Temperature/T1))*(1-(Temperature/T2))*(1-(Temperature/T3))))-1)
	return F_black_body
