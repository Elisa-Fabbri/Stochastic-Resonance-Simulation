import numpy as np
import matplotlib.pyplot as plt
import os
import stochastic_resonance

def Plot_Emission_Models():

	"""
	This function plots the two models for the emitted radiation
	"""
	T_start = 270
	T_end = 300
	num_points = 300

	T = np.linspace(T_start, T_end, num_points)
	emitted_radiation_linear_values = [stochastic_resonance.linear_emitted_radiation(Ti) for Ti in T]
	emitted_radiation_black_body_values = [stochastic_resonance.black_body_emitted_radiation(Ti) for Ti in T]

	f = plt.figure(figsize=(15, 10))
	plt.plot(T, emitted_radiation_linear_values, label='Linear emitted radiation')
	plt.plot(T, emitted_radiation_black_body_values, label='Black body emitted radiation')
	plt.grid(True)
	plt.xlabel('Temperature (K)')
	plt.ylabel('Emitted radiation')
	#plt.legend()
	plt.title('Comparison between the two models for the emitted radiation')

	save_file_path = os.path.join('images', 'emitted_radiation_plot.png')
	plt.savefig(save_file_path)

	plt.show()


def Plot_F():

	"""
	This function plots F(T) using the two different models for the emitted radiation.
	"""
	T_start = 270
	T_end = 300
	num_points = 300

	T = np.linspace(T_start, T_end, num_points)
	F_linear_values = [stochastic_resonance.F_linear(Ti) for Ti in T]
	F_black_body_values = [stochastic_resonance.F_black_body(Ti) for Ti in T]

	f = plt.figure(figsize=(15, 10))
	plt.plot(T, F_linear_values, label='Linear')
	plt.plot(T, F_black_body_values, label = 'Black body')
	plt.grid(True)
	plt.xlabel('Temperature (K)')
	plt.ylabel('F(T)')
	#plt.legend()
	plt.title('Comparison between F(T) using a linear and a black body model for beta')

	save_file_path = os.path.join('images', 'F_plot.png')
	plt.savefig(save_file_path)

	plt.show()

Plot_Emission_Models()
Plot_F()



