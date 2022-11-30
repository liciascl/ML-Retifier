import os
import re
import datetime
import numpy as np
from subprocess import check_output



class Simulador:
    
	def __init__(self, params):
		self.params = params

		self.action_space_length = self.params
		self.observation_space_size = 15     # number of features

		self.iteration = 0
		self.episode = 0
		self.sequence = ['strash']
		self.RL= float('inf')

		self.best_known_lut_6 = (float('inf'), float('inf'), -1, -1)
		self.best_known_levels = (float('inf'), float('inf'), -1, -1)
		self.best_known_lut_6_meets_constraint = (float('inf'), float('inf'), -1, -1)
		self.mapa={'Eficiency' : [], 'RL' : []}
		# logging
		self.log = None


	def log(message):
		print('[Simulação {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


	def reset(self):
		"""
		resets the environment and returns the state
		"""
		self.iteration = 0
		self.episode += 1
		self.lut_6, self.levels = float('inf'), float('inf')
		self.sequence = ['strash']
		self.episode_dir = os.path.join(self.params['playground_dir'], str(self.episode))
		if not os.path.exists(self.episode_dir):
			os.makedirs(self.episode_dir)

		# logging
		log_file = os.path.join(self.episode_dir, 'log.csv')
		if self.log:
			self.log.close()
		self.log = open(log_file, 'w')
		self.log.write('iteration, optimization, RL, best RL \n')

		state, _ = self._run()

		# logging
		self.log.write(', '.join(str(self.iteration), self.sequence[-1], str(int(self.RL)), + '\n'))
		self.log.flush()

		return state

	def step(self, optimization):
		"""
		accepts optimization index and returns (new state, reward, done, info)
		"""
		self.sequence.append(self.params['optimizations'][optimization])
		new_state, reward = self._run()

		# logging
		if self.RL < self.best_known_RL[0]:
			self.best_known_lut_6 = (int(self.RL), self.episode, self.iteration)
		self.log.flush()

		return new_state, reward, self.iteration == self.params['iterations'], None

	def _run(self):
		self.circuit = Circuit("Simulação Com RL valendo {} Ohms".format(self.RL))
		# Componentes
		self.circuit.model('Diodo','D',IS=5e-6, RS=20, N=1.05, CJ0=0.14e-12, M=0.4, VJ=0.34, EG=0.69, IBV=0.1e-3, BV=2, XTI=2)	
		self.circuit.L('s',1,2,0.8@u_nH)
		self.circuit.Diode('1',2,3,model="Diodo")
		self.circuit.C('p',2,3,0.16@u_pF)
		self.circuit.R('load',3,self.circuit.gnd, RL)
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

		eficiency=local_data['PCE'].mean()		
		print("Eficiencia media", eficiency)

		try:
			
			# get reward
			RL = self._get_metrics(eficiency)
			reward = self._get_reward(RL)
			self.RL = RL
			# get new state of the circuit
			state = self._get_state(eficiency)
			return state, reward
		except Exception as e:
			print(e)
			return None, None

	def _get_reward(self, RL):
		constraint_met = True
		optimization_improvement = 0    # (-1, 0, 1) <=> (worse, same, improvement)
		constraint_improvement = 0      # (-1, 0, 1) <=> (worse, same, improvement)

		# check optimizing parameter
		if RL < self.RL:
			optimization_improvement = 1
		elif RL == self.RL:
			optimization_improvement = 0
		else:
			optimization_improvement = -1

		   # now calculate the reward
		return self._reward_table(constraint_met, optimization_improvement)

	def _reward_table(self, constraint_met, optimization_improvement):
		return {
			True: {
			0: {
				1: 3,
				0: 0,
				-1: -1
			}
			},
			False: {
			1: {
				1: 3,
				0: 2,
				-1: 1
			},
			0: {
				1: 2,
				0: 0,
				-1: -2
			},
			-1: {
				1: -1,
				0: -2,
				-1: -3
			}
			}
		}[constraint_met][optimization_improvement]

	def _get_state(self, eficiency):
		self.mapa['Eficiency'].append(eficiency)
		self.mapa['RL'].append(self.RL)
		return 
