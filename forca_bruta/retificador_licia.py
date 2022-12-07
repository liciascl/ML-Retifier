# -*- coding: utf-8 -*-
"""
Created on 24 Octuber 2022

@author: Lícia Sales
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
import pandas as pd

spice_library = SpiceLibrary("./netlist/")

circuit = Circuit('Retificador')

circuit.model('Diodo','D',IS=5e-6, RS=20, N=1.05, CJ0=0.14e-12, M=0.4, VJ=0.34, EG=0.69, IBV=0.1e-3, BV=2, XTI=2)
data = { 'Vg' : [], 'Vo' : [], 'Ig' : [] , 'Zin' : [], 'Vl' : [], 'Pin' : [], 'Pout' : [], 'PCE' : [] , 'Temperature' : []}


# Componentes
RL=10337.230826262235

circuit.L('s',1,2,0.8@u_nH)
circuit.Diode('1',2,3,model='Diodo')
circuit.C('p',2,3,0.16@u_pF)
circuit.R('load',3,circuit.gnd,RL)
circuit.C('load',3,circuit.gnd,100@u_pF)
print("Netlist: \n\n", circuit)

source = circuit.SinusoidalVoltageSource('input', 1, circuit.gnd, amplitude=100@u_mV, frequency=2.45@u_GHz)
ti = -20
tf = 60

for temperatura in range (ti, tf,5):
	simulator = circuit.simulator(temperature=25, nominal_temperature=temperatura)
	analysis = simulator.transient(step_time=source.period/200, end_time=source.period*5000)
	# selecionando os últimos 10 períodos
	data['Vg'] = np.array(analysis['1'][-2000:])
	data['Vo'] = np.array(analysis['3'][-2000:])
	data['Ig'] = np.array(-analysis.Vinput[-2000:])
	data['Time']= np.array(analysis.time[-2000:])
	data['Temperature'] = np.append(data['Temperature'],temperatura)

	# FFT para extrair impedância
	Vg_f = scipy.fftpack.fft(data['Vg'])
	Ig_f = scipy.fftpack.fft(data['Ig'])
	Y_f = Vg_f/Ig_f
	data['Zin'] = np.append(data['Zin'],Y_f[10])

	# tensão média de saída
	Vl = np.mean(data['Vo'], dtype= np.float32)
	data['Vl'] = np.append(data['Vl'], Vl)
	# potência média de entrada
	Pin = np.mean(data['Vg']*data['Ig'], dtype= np.float32)
	data['Pin'] = np.append(data['Pin'], Pin)
	# potência média de saída
	Pout = float((Vl**2)/RL)
	data['Pout'] = np.append(data['Pout'], Pout)
	# eficiência
	PCE = Pout/Pin
	data['PCE'] = np.append(data['PCE'], PCE)
	
	# plotando os últimos 10 períodos
print(data['Pout'])
plt.plot(data['Temperature'], data['PCE'], label = "Eficiência")
plt.title("Eficiência do Diodo em relação a Temperatura")
plt.legend()
plt.xlabel("Temperatura")
plt.ylabel("Eficiencia do circuito")
plt.savefig("diode_curve.png")	
plt.show()	
