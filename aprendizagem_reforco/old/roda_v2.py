#!/usr/bin/env python3
# *-* coding: utf-8 *-*
#-------Lícia Sales-------
#11/2022


import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fftpack

import tensorflow as tf
from tensorflow import keras


class State:
	def __init__(self,n_simulation):
		self.n_simulation=n_simulation
		self.counter_simulation = 1
		self.score = 0
		self.eficiency = 0.1
		self.state = 1
		self.RL_now = np.random.uniform(100, 50000)
		self.RL = np.random.uniform(100, 50000)
		
        # set hyperparameters
		self.max_episodes = 2
		self.max_actions = 99
		self.discount = 0.93
		self.exploration_rate = 1.0
		self.exploration_decay = 1.0/20000

		# nn_model parameters
		self.in_units = 16

		# construct nn model
		self._nn_model()

		# save nn model
		self.saver = tf.train.Saver()		

		
		
		
	def run(self):
		print("Vamos rodar {} simulações".format(self.n_simulation))

		while(self.n_simulation >= self.counter_simulation):
				
			if self.eficiency > 0.85:
				print("Hora de avaliar o modelo e plotar os resultados")
				self.evaluate_model()
			else:
				print("Resistencia em {} eficiência em {}".format(self.RL, self.eficiency))
				print("Pegando novo parametro ")
				#self.RL = self.choose_parameters(self.RL,self.eficiency)
				self.train()				
				print("Valor previsto em {}".format(self.RL))
				print("Nova Eficiencia em ", self.run_simulation())
				print("Saldo atual", self.rewards())
			
			self.counter_simulation = self.counter_simulation + 1
	

	def choose_parameters(self, RL_anteiror, Eficiencia_anterior):
		# Escolhendo o valor da resistência 
		model_predictions = model.predict(self.RL)

	
		return self.RL
			
	def _nn_model(self):
		print("Entrou no dnn")	
		# Next, we build a very simple model.
		model = keras.Sequential([
			keras.layers.Dense(10, input_shape=(1,), activation='tanh'),
			keras.layers.Dense(1)
			])

		optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
		model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              metrics=['MSE'])
		model.fit(self.RL, self.eficiency, epochs=1000)
		
	def train(self):




		
	def run_simulation(self):
		self.circuit = Circuit("Simulação Com RL valendo {} Ohms".format(self.RL))
		# Componentes
		self.circuit.model('Diodo','D',IS=5e-6, RS=20, N=1.05, CJ0=0.14e-12, M=0.4, VJ=0.34, EG=0.69, IBV=0.1e-3, BV=2, XTI=2)	
		self.circuit.L('s',1,2,0.8@u_nH)
		self.circuit.Diode('1',2,3,model="Diodo")
		self.circuit.C('p',2,3,0.16@u_pF)
		self.circuit.R('load',3,self.circuit.gnd,self.RL)
		self.circuit.C('load',3,self.circuit.gnd,100@u_pF)
		self.source = self.circuit.SinusoidalVoltageSource('input', 1, self.circuit.gnd, amplitude=100@u_mV, frequency=2.45@u_GHz)	
		print("Circuito: \n\n", self.circuit)
		local_data = { 'Simulation' : [], 'Zin' : [], 'Vl' : [], 'Pin' : [], 'Pout' : [], 'PCE' : [] , 'Temperature' : []}		
		for temperatura in range (-20, 50,5):
			simulator = self.circuit.simulator(temperature = temperatura, nominal_temperature = 25)		
			analysis = simulator.transient(step_time=self.source.period/200, end_time=self.source.period*5000)
			# selecionando os últimos 10 períodos
			local_data['Vg'] = np.array(analysis['1'][-2000:])
			local_data['Vo'] = np.array(analysis['3'][-2000:])
			local_data['Ig'] = np.array(-analysis.Vinput[-2000:])
			local_data['Time']= np.array(analysis.time[-2000:])
			local_data['Temperature'] = np.append(local_data['Temperature'],temperatura)

			# FFT para extrair impedância
			Vg_f = scipy.fftpack.fft(local_data['Vg'])
			Ig_f = scipy.fftpack.fft(local_data['Ig'])
			Y_f = Vg_f/Ig_f
			local_data['Zin'] = np.append(local_data['Zin'],Y_f[10])

			# tensão média de saída
			Vl = np.mean(local_data['Vo'], dtype= np.float32)
			local_data['Vl'] = np.append(local_data['Vl'], Vl)
			# potência média de entrada
			Pin = np.mean(local_data['Vg']*local_data['Ig'], dtype= np.float32)
			local_data['Pin'] = np.append(local_data['Pin'], Pin)
			# potência média de saída
			Pout = float((Vl**2)/self.RL)
			local_data['Pout'] = np.append(local_data['Pout'], Pout)
			# eficiência
			PCE = Pout/Pin
			local_data['PCE'] = np.append(local_data['PCE'], PCE)
			#print("Eficiencia em {} a {} graus".format(PCE,temperatura))

		self.eficiency=local_data['PCE'].mean()		
		print("Eficiencia media",self.eficiency)
		
		
		return self.eficiency
		
		
	def rewards(self):
	
		if self.eficiency <= 0.1:
			self.score -=10
			
		if self.eficiency > 0.1 and self.eficiency <= 0.5:
			self.score +=1
			
		if self.eficiency > 0.5 and self.eficiency <= 0.8:
			self.score +=2
			
		if self.eficiency > 0.8:
			self.score += 10
			
		
		return self.score

	def evaluate_model(self):
		filename = 'Deep_q_learning.png'
		plt.plot(x,model_predictions)
		plt.plot(self.RL,self.eficiency)
		
		plt.plot(x, self.scores)
		plt.savefig(filename)
		plt.show()
	






if __name__ == "__main__":
	teste = State(3000)#rodando 3K simulacoes
	teste.run()


