import tensorflow as tf
import random
import numpy as np
import operator
from functools import reduce 
import gym
from gym import spaces

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

def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class RandomCircuits(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self):

        super(RandomCircuits, self).__init__()
        self.circuit = None
        self.target = 0.5
        self.steps = 0
        self.max_steps = 1
        self.num_params = 1
        self.num_q = 0
        self.num_d = 0
        self.RL = 0
        self.eficiency = 0
        self.action_space = spaces.Discrete(2)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=10000, high=50000,
                                            shape=(1,), dtype=np.uint32)
        self.info = dict()

    def get_error(self):
        return -abs(self.eficiency - self.target)**2

    # Return State, Reward, Done
    def step(self, action):
        done = False
        if self.steps > self.max_steps:
            done = True
        print(action)
        self.RL=action
        error = self.get_error()
        if -error < 0.1:
            done = True
        self.steps += 1
        observation = self.run_simulation(self.RL)
        return observation, error, done, self.info 

    def run_simulation(self, RL):
        self.circuit = Circuit("Simulação Com RL valendo {} Ohms".format(RL))
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

        self.eficiency=local_data['PCE'].mean()		
        print("Eficiencia media",self.eficiency)


        return self.eficiency


    def reset(self):
        self.steps = 0
        self.eficiency = 0
        self.RL = np.random.uniform(40000, 50000)
        self.target = 0.5
        return 

    def close(self):
        pass

    def render(self):
        pass


'''

if __name__ == "__main__":
    env = RandomCircuits()
    print(env.observation_space)
    state = env.reset()
    iterations = 1
    for i in range(iterations):
        done = False
        while not done:
            state, reward, done info = env.step(np.random.uniform(0, np.pi))
            print(state, reward, done)
            for i in range(5):
                print("SHEET")
                print(state[:,:,i])

from random_circuit_env import RandomCircuits
env = RandomCircuits(10, 10, 100, 100)
from stable_baselines3.common.env_checker import check_env
check_env(env)

'''
