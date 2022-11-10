
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack

class brute_force(self):
	def __init__(self):
		self.temperature = 25
		self.nominal_temperature = 25
		self.diode = 0
		#Parametros do circuito

		self.list_diodes =[]
        ######### MINIMAL VALUES ###########
		self.CP_min = 0.05e-12
		self.LS_min = 0.8e-9

		self.IS_min = 1e-8
		self.RS_min = 5
		self.N_min = 1.05
		self.CJ0_min=0.1e-12
		self.M_min = 0.3
		self.VJ_min=0.3
		self.EG_min=0.69
		self.IBV_min=0.1e-3
		self.BV_min=2
		self.XTI_min=2

		self.RL_min=100
		self.CL_min=1e-12
		######### MAX VALUES ###########

		self.CP_max = 0.25e-12
		self.LS_max = 20e-9

		self.IS_max = 1e-4
		self.RS_max = 30
		self.N_max = 1.05
		self.CJ0_max=0.2e-12
		self.M_max = 0.5
		self.VJ_max = 0.65
		self.EG_max=0.69
		self.IBV_max=0.1e-3
		self.BV_max=2
		self.XTI_max=2

		self.RL_max=50000
		self.CL_max=100e-12

		############ VALUES ###################

		self.model = { 'CP' : [], 'LS' : [], 'IS' : [] , 'RS' : [] , 'N' : [], 'CJ0' : [], 'M' : [], 'VJ' : [] , 'EG' : [], 'IBV' : [],  'BV' : [], 'XTI' : [], 'RL' : [] , 'CL' : []}
		

	def parameters(self):
		
		self.model['CP'].append(random.uniform(self.CP_min ,  self.CP_max))
		self.model['LS'].append(random.uniform(self.LS_min ,  self.LS_max))

		self.model['IS'].append(random.uniform(self.IS_min ,  self.IS_max))
		self.model['RS'].append(random.uniform(self.RS_min ,  self.RS_max))
		self.model['N'].append(random.uniform(self.N_min ,  self.N_max))
		self.model['CJ0'].append(random.uniform(self.CJ0_min ,  self.CJ0_max))
		self.model['M'].append(random.uniform(self.M_min ,  self.M_max))
		self.model['VJ'].append(random.uniform(self.VJ_min ,  self.VJ_max))
		self.model['EG'].append(random.uniform(self.EG_min ,  self.EG_max))
		self.model['IBV'].append(random.uniform(self.IBV_min ,  self.IBV_max))
		self.model['BV'].append(random.uniform(self.BV_min ,  self.BV_max))
		self.model['XTI'].append(random.uniform(self.XTI_min ,  self.XTI_max))

		self.model['RL'].append(random.uniform(self.RL_min ,  self.RL_max))
		self.model['CL'].append(random.uniform(self.CL_min ,  self.CL_max))


		return self.model
		
		
		
	def gerenerate(self, value):
		print("Gerando dados ...")
		for diode in range(0,value):
			model = self.parameters()
			self.diode = diode
			self.circuit_build()
			print(self.simulation())
			
			
	def circuit_build(self):
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
if __name__ == '__main__':
	diode = brute_force()
	print(diode.gerenerate(2))		
		
			

		
