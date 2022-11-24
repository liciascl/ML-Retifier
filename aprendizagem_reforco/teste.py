
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack

class brute_force():
	def __init__(self):
		self.temperature = 25
		self.nominal_temperature = 25
		self.diode = 0
		#Parametros do circuito

		self.list_diodes =[]
		self.data = { 'Diode' : [], 'Vg' : [], 'Vo' : [], 'Ig' : [] , 'Zin' : [], 'Vl' : [], 'Pin' : [], 'Pout' : [], 'PCE' : [] , 'Temperature' : [], 'Time' : []}
		self.model = {'Diode' : [], 'CP' : [], 'LS' : [], 'IS' : [] , 'RS' : [] , 'N' : [], 'CJ0' : [], 'M' : [], 'VJ' : [] , 'EG' : [], 'IBV' : [],  'BV' : [], 'XTI' : [], 'RL' : [] , 'CL' : []}
	def parameters(self):
		
		self.CP = (np.random.uniform(0.05e-12 ,  0.25e-12))
		self.LS = (np.random.uniform(0.8e-9 ,  20e-9))

		self.IS = (np.random.uniform(1e-8 ,  1e-4))
		self.RS = (np.random.uniform( 5 ,  30))
		self.N = (np.random.uniform(1.05 ,   1.05))
		self.CJ0 = (np.random.uniform(0.1e-12,  0.2e-12))
		self.M = (np.random.uniform(0.3 , 0.5))
		self.VJ = (np.random.uniform(0.3 ,  0.65))
		self.EG = (np.random.uniform(0.69 ,  0.69))
		self.IBV = (np.random.uniform(0.1e-3 ,  0.1e-3))
		self.BV = (np.random.uniform(2 ,  2))
		self.XTI = (np.random.uniform(2 ,  2))

		self.RL = (np.random.uniform(100,  50000))
		self.CL = (np.random.uniform(1e-12 , 100e-12))

		self.model['CP'].append(self.CP)
		self.model['LS'].append(self.LS)
		self.model['IS'].append(self.IS)
		self.model['RS'].append(self.RS)
		self.model['N'].append(self.N)
		self.model['CJ0'].append(self.CJ0)
		self.model['M'].append(self.M)
		self.model['VJ'].append(self.VJ)
		self.model['EG'].append(self.EG)
		self.model['IBV'].append(self.IBV)
		self.model['BV'].append(self.BV)
		self.model['XTI'].append(self.XTI)
		self.model['RL'].append(self.RL)
		self.model['CL'].append(self.CL)
		
				
	def gerenerate(self, value):
		print("Gerando dados ...")
		for diode in range(0,value):
			self.parameters()
			self.diode = diode
			self.model['Diode'].append(self.diode)
			self.circuit_build()
			print(self.simulation())
			self.target_result()
			
	def circuit_build(self):
		self.circuit = Circuit("Diode {} Rectifier".format(self.diode))
		self.circuit.model('Diodo','D',IS=self.IS, RS=self.RS, N=self.N, CJ0=self.CJ0, M=self.M, VJ=self.VJ, EG=self.EG, IBV=self.IBV, BV=self.BV, XTI=self.XTI)
		# Componentes
		self.circuit.L('s',1,2,self.LS)
		self.circuit.Diode('1',2,3,model='Diodo')
		self.circuit.C('p',2,3,self.CP)
		self.circuit.R('load',3,self.circuit.gnd,self.RL)
		self.circuit.C('load',3,self.circuit.gnd,self.CL)
		self.source = self.circuit.SinusoidalVoltageSource('input', 1, self.circuit.gnd, amplitude=100@u_mV, frequency=2.45@u_GHz)	
		print("Netlist: \n\n", self.circuit)
		
		return self.circuit
		
		
	def simulation(self):		
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
			print("Eficiencia em ", PCE)
			
		return self.data
		
		
	def target_result(self):
		df = pd.DataFrame.from_dict(self.data, orient='index')
		for dado in self.data['PCE']:
			if dado >= 0.5:
				plt.plot(self.data['Temperature'], self.data['PCE'], label = "Eficiência")
				plt.title("Eficiência do Diodo em relação a Temperatura")
				plt.xlabel("Temperatura")
				plt.ylabel("Eficiencia do circuito")
				plt.savefig("output_png/diodo_{}.png".format(self.diode))
		
		
		new_data = {'Input_data' : self.model[self.diode:], 'Output_data' : self.data['Temperature':'PCE']} 
		print(new_data)
		#df.to_csv('output_csv/output_parameters_{}.csv'.format(self.diode), index=True)
				
		
if __name__ == '__main__':
	diode = brute_force()
	print(diode.gerenerate(2))		
		
			

		
