import numpy as np

number_of_seconds_in_a_year = 365.25*24*60*60

#Steady solutions:
T1 = 280 #K
T2 = 285 #K
T3 = 290 #K

#Constant of the model

C = 0.31e9*(number_of_seconds_in_a_year**2) #-> 0.31e9 J/(m^2 K)
tau = 13 #years
A = 0.0005
omega = (2*np.pi)/(1e5) #1/years


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

def periodic_forcing(A, omega, time):
	"""
	This function computes the periodic forcing applied to the system
	"""
	return 1+A*np.cos(omega*time)


beta_linear = -((C/(tau*linear_emitted_radiation(T3)))*((T1*T2*T3) / ((T1 - T3)*(T2 - T3))))
beta_black_body = -((C/(tau*black_body_emitted_radiation(T3)))*((T1*T2*T3) / ((T1 - T3)*(T2 - T3))))

#print(beta_linear, '\n', beta_black_body)

# F(T)

def F_linear(Temperature, mu = 1):
	"""
	This function returns the value of F(T) given a certain value for the temperature using a linear model for the computation of beta.
	"""
	F_linear = (linear_emitted_radiation(Temperature)/C)*((mu/(1 + beta_linear*(1-(Temperature/T1))*(1-(Temperature/T2))*(1-(Temperature/T3))))-1)
	return F_linear

def F_black_body(Temperature, mu = 1):
	"""
	This function returns the value of F(T) given a certain value for the temperature using the black body radiation model for the computation of beta.
	"""
	F_black_body = (black_body_emitted_radiation(Temperature)/C)*((mu/(1 + beta_black_body*(1-(Temperature/T1))*(1-(Temperature/T2))*(1-(Temperature/T3))))-1)
	return F_black_body
