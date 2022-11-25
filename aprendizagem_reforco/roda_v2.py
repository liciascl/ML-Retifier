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

from collections import deque
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

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
		self.out_units = 1
		self.hidden_units = 1

		# construct nn model
		self._nn_model()

		# save nn model
		self.saver = tf.train.Saver()		

		
		
		
	def run(self):
		print("Vamos rodar {} simulações".format(self.n_simulation))
#		print("Escolhendo os parametros, RL em ", self.choose_parameters())
#		print("Rodando a simulação numero {}".format(self.counter_simulation))
#		print("Eficiencia em ", self.run_simulation())
		while(self.n_simulation >= self.counter_simulation):
				
			if self.eficiency > 0.85:
				print("Hora de avaliar o modelo e plotar os resultados")
				self.evaluate_model()
			else:
				print("Resistencia em {} eficiência em {}".format(self.RL, self.eficiency))
				print("Pegando novo parametro ")
				#self.RL = self.predict_new_parameter()
				self.train()				
				print("Valor previsto em {}".format(self.RL))
				print("Nova Eficiencia em ", self.run_simulation())
				print("Saldo atual", self.rewards())
			
			self.counter_simulation = self.counter_simulation + 1
	

	def choose_parameters(self):
		# Escolhendo o valor da resistência aleatoriamente
		self.RL_now = random.uniform(100, 50000)
	
		return self.RL
			
	def _nn_model(self):
		print("Entrou no dnn")	
		self.a0 = tf.placeholder(tf.float32, shape=[1, self.in_units]) # input layer
		self.y = tf.placeholder(tf.float32, shape=[1, self.out_units]) # ouput layer
		
		# from input layer to hidden layer
		self.w1 = tf.Variable(tf.zeros([self.in_units, self.hidden_units], dtype=tf.float32)) # weight
		self.b1 = tf.Variable(tf.random_uniform([self.hidden_units], 0, 0.01, dtype=tf.float32)) # bias
		self.a1 = tf.nn.relu(tf.matmul(self.a0, self.w1) + self.b1) # the ouput of hidden layer
		
		# from hidden layer to output layer
		self.w2 = tf.Variable(tf.zeros([self.hidden_units, self.out_units], dtype=tf.float32)) # weight
		self.b2 = tf.Variable(tf.random_uniform([self.out_units], 0, 0.01, dtype=tf.float32)) # bias
		
		# Q-value and Action
		self.a2 = tf.matmul(self.a1, self.w2) + self.b2 # the predicted_y (Q-value) of four actions
		self.action = tf.argmax(self.a2, 1) # the agent would take the action which has maximum Q-value

		# loss function
		self.loss = tf.reduce_sum(tf.square(self.a2-self.y))
		
		# upate model
		self.update_model =  tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(self.loss)			
		print("saiu do dnn")

	def train(self):
		# get hyper parameters
		max_episodes = self.max_episodes
		max_actions = self.max_actions
		discount = self.discount
		exploration_rate = self.exploration_rate
		exploration_decay = self.exploration_decay
		print("Modo treinamento ativado")
		# start training
		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables()) # initialize tf variables

			for i in range(max_episodes):
				self.RL_now = tf.convert_to_tensor(self.RL_now, dtype=tf.float32)
				self.RL_now, pred_Q = sess.run([self.RL_now, self.a2],feed_dict={self.a0:np.eye(16)[self.score:self.score+1]})
				print(self.RL_now,pred_Q)
				# if explorating, then taking a random action instead
				random_action = random.uniform(100, 50000)
			
				self.RL_now =pred_Q[0][0]
				self.RL = self.RL_now
				print("Valor de RL ", self.RL)
				print("Valor previsto em ", self.RL)
				print("Nova Eficiencia em ", self.run_simulation())
				print("Saldo atual", self.rewards())


				# update
				update_Q = pred_Q
				update_Q = self.score + discount*np.max(1/self.eficiency)

#				sess.run([self.update_model],
#						 feed_dict={self.a0:np.identity(16)[self.score:self.score+1],self.y:update_Q})




		
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

		x = [i+1 for i in range(self.n_simulation)]
		plt.plot(x, self.scores)
		plt.savefig(filename)
		#plt.show()
	






if __name__ == "__main__":
	teste = State(3000)#rodando 3K simulacoes
	teste.run()


