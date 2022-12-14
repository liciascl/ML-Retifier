###################################################################################################
### Licia Sales
### out-17-2022
### Universal Rectifier Analysis
###################################################################################################
###################################################################################################



import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack

###################################################################################################

class analysis:
	def __init__(self):

		self.diode = None
		self.frequency = 2.45@u_GHz
		self.amplitude = 100@u_mV

		self.temperature = 25
		self.nominal_temperature = 25

		self.CP = 0
		self.LS = 0

		self.IS= 0
		self.RS =0
		self.N=0
		self.CJ0=0
		self.M = 0
		self.VJ=0
		self.EG=0
		self.IBV=0
		self.BV=0
		self.XTI=0

		self.RL=0
		self.CL=0
		self.df = pd.read_csv('diodes_3K.csv')
		
		local_data = { 'Vg' : [], 'Vo' : [], 'Ig' : [] , 'Zin' : [], 'Vl' : [], 'Pin' : [], 'Pout' : [], 'PCE' : [] , 'Temperature' : [], 'Time' : []}

	def choose_diode(self):
		linhas = len(self.df)
		for valor in range(linhas):
			self.CP = self.df['CP'][valor]
			self.LS = self.df['LS'][valor]
			self.IS= self.df['IS'][valor]
			self.RS = self.df['RS'][valor]
			self.N=self.df['N'][valor]
			self.CJ0=self.df['CJ0'][valor]
			self.M = self.df['M'][valor]
			self.VJ=self.df['VJ'][valor]
			self.EG=self.df['EG'][valor]
			self.IBV=self.df['IBV'][valor]
			self.BV=self.df['BV'][valor]
			self.XTI=self.df['XTI'][valor]

			self.RL=self.df['RL'][valor]
			self.CL=self.df['CL'][valor]
			self.circuit_maker()

			self.diode=valor
		
			
		
	def circuit_maker(self):
		self.circuit = Circuit("Diode {} Rectifier".format(self.diode))
		self.circuit.model('Diodo','D',IS=self.IS, RS=self.RS, N=self.N, CJ0=self.CJ0, M=self.M, VJ=self.VJ, EG=self.EG, IBV=self.IBV, BV=self.BV, XTI=self.XTI)
		# Componentes
		self.circuit.L('s',1,2,self.LS)
		self.circuit.Diode('1',2,3,model='Diodo')
		self.circuit.C('p',2,3,self.CP)
		self.circuit.R('load',3,self.circuit.gnd,self.RL)
		self.circuit.C('load',3,self.circuit.gnd,self.CL)
		self.source = self.circuit.SinusoidalVoltageSource('input', 1, self.circuit.gnd, amplitude=self.amplitude, frequency=self.frequency)	
		print("Netlist: \n\n", self.circuit)
		print(self.simulation_transient())
		return self.circuit
		
	def simulation_transient(self):
		print("Variando a temperatura")
		local_data = { 'Vg' : [], 'Vo' : [], 'Ig' : [] , 'Zin' : [], 'Vl' : [], 'Pin' : [], 'Pout' : [], 'PCE' : [] , 'Temperature' : [], 'Time' : []}
		for temperatura in range (-20, 50,5):
			simulator = self.circuit.simulator(temperature = temperatura, nominal_temperature = 25)		
			analysis = simulator.transient(step_time=self.source.period/200, end_time=self.source.period*5000)
			# selecionando os ??ltimos 10 per??odos
			local_data['Vg'] = np.array(analysis['1'][-2000:])
			local_data['Vo'] = np.array(analysis['3'][-2000:])
			local_data['Ig'] = np.array(-analysis.Vinput[-2000:])
			local_data['Time']= np.array(analysis.time[-2000:])
			local_data['Temperature'] = np.append(local_data['Temperature'],temperatura)

			# FFT para extrair imped??ncia
			Vg_f = scipy.fftpack.fft(local_data['Vg'])
			Ig_f = scipy.fftpack.fft(local_data['Ig'])
			Y_f = Vg_f/Ig_f
			local_data['Zin'] = np.append(local_data['Zin'],Y_f[10])

			# tens??o m??dia de sa??da
			Vl = np.mean(local_data['Vo'], dtype= np.float32)
			local_data['Vl'] = np.append(local_data['Vl'], Vl)
			# pot??ncia m??dia de entrada
			Pin = np.mean(local_data['Vg']*local_data['Ig'], dtype= np.float32)
			local_data['Pin'] = np.append(local_data['Pin'], Pin)
			# pot??ncia m??dia de sa??da
			Pout = float((Vl**2)/self.RL)
			local_data['Pout'] = np.append(local_data['Pout'], Pout)
			# efici??ncia
			PCE = Pout/Pin
			local_data['PCE'] = np.append(local_data['PCE'], PCE)
			#print("Eficiencia em ", PCE)
	
		
		plt.plot(local_data['Temperature'], local_data['PCE'], label = "Diodo_{}".format(self.diode))
		plt.title("Efici??ncia do Diodo em rela????o a Temperatura")
		plt.legend()
		plt.xlabel("Temperatura")
		plt.ylabel("Eficiencia do circuito")
		plt.savefig("output_png/diodo_{}.png".format(self.diode))
		
		#self.generate_csv()
		print("  ##########  SALVANDO NUM CSV ##################    ")
		df = pd.DataFrame.from_dict(local_data, orient='index').T.to_csv('output_csv/output_parameters_{}.csv'.format(self.diode), index=True)
	


	def generate_csv(self):
		print("  ##########  SALVANDO NUM CSV ##################    ")
		df = pd.DataFrame.from_dict(self.data, orient='index').T.to_csv('output_csv/output_parameters_{}.csv'.format(self.diode), index=True)
	
if __name__ == '__main__':
		diode = analysis()
		diode.choose_diode()

		
		
		
		
		

