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

		self.diode = 0
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
		self.df = pd.read_csv('diodes_100K.csv')
		
		self.data = { 'Vg' : [], 'Vo' : [], 'Ig' : [] , 'Zin' : [], 'Vl' : [], 'Pin' : [], 'Pout' : [], 'PCE' : [] , 'Temperature' : [], 'Time' : []}

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

		for temperatura in range (-20, 50,5):
			simulator = self.circuit.simulator(temperature = temperatura, nominal_temperature = 25)		
			analysis = simulator.transient(step_time=self.source.period/200, end_time=self.source.period*5000)
			# selecionando os últimos 10 períodos
			self.data['Vg'] = np.array(analysis['1'][-2000:])
			self.data['Vo'] = np.array(analysis['3'][-2000:])
			self.data['Ig'] = np.array(-analysis.Vinput[-2000:])
			self.data['Time']= np.array(analysis.time[-2000:])
			self.data['Temperature'] = np.append(self.data['Temperature'],temperatura)

			# FFT para extrair impedância
			Vg_f = scipy.fftpack.fft(self.data['Vg'])
			Ig_f = scipy.fftpack.fft(self.data['Ig'])
			Y_f = Vg_f/Ig_f
			self.data['Zin'] = np.append(self.data['Zin'],Y_f[10])

			# tensão média de saída
			Vl = np.mean(self.data['Vo'], dtype= np.float32)
			self.data['Vl'] = np.append(self.data['Vl'], Vl)
			# potência média de entrada
			Pin = np.mean(self.data['Vg']*self.data['Ig'], dtype= np.float32)
			self.data['Pin'] = np.append(self.data['Pin'], Pin)
			# potência média de saída
			Pout = float((Vl**2)/self.RL)
			self.data['Pout'] = np.append(self.data['Pout'], Pout)
			# eficiência
			PCE = Pout/Pin
			self.data['PCE'] = np.append(self.data['PCE'], PCE)
			#print("Eficiencia em ", PCE)
	
		
		plt.plot(self.data['Temperature'], self.data['PCE'], label = "Eficiência")
		plt.title("Eficiência do Diodo em relação a Temperatura")
		plt.xlabel("Temperatura")
		plt.ylabel("Eficiencia do circuito")
		plt.savefig("output_png/diodo_{}.png".format(self.diode))
		
		self.generate_csv()



	def generate_csv(self):
		print("  ##########  SALVANDO NUM CSV ##################    ")
		df = pd.DataFrame.from_dict(self.data, orient='index').T.to_csv('output_csv/output_parameters_{}.csv'.format(self.diode), index=True)
	
if __name__ == '__main__':
		diode = analysis()
		diode.choose_diode()

		
		
		
		
		

