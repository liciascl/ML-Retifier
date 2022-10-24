# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:44:53 2022

@author: Renan
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack

from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *



circuit = Circuit('Retificador')

circuit.model('Diodo','D',IS=5e-6, RS=20, N=1.05, CJ0=0.14e-12, M=0.4, VJ=0.34, EG=0.69, IBV=0.1e-3, BV=2, XTI=2)


# Componentes
RL=10@u_kΩ

circuit.L('s',1,2,0.8@u_nH)
circuit.Diode('1',2,3,model="Diodo")
circuit.C('p',2,3,0.16@u_pF)
circuit.R('load',3,circuit.gnd,RL)
circuit.C('load',3,circuit.gnd,100@u_pF)


source = circuit.SinusoidalVoltageSource('input', 1, circuit.gnd, amplitude=100@u_mV, frequency=2.45@u_GHz)


simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.transient(step_time=source.period/200, end_time=source.period*5000)

print("Netlist: \n\n", circuit)

# selecionando os últimos 10 períodos
Vg = analysis['1'][-2000:]
Vo = analysis['3'][-2000:]
Ig = -analysis.Vinput[-2000:]

# plotando os últimos 10 períodos
plt.plot(Vg)
plt.plot(Vo)
plt.plot(Ig)

# FFT para extrair impedância
Vg_f = scipy.fftpack.fft(Vg)
Ig_f = scipy.fftpack.fft(Ig)
Y_f = Vg_f/Ig_f

Zin = Y_f[10]
print("Zin:" , Zin)

# tensão média de saída
Vl = float(np.mean(Vo))
print("Vl:" , Vl)

# potência média de entrada
Pin = float(np.mean(Vg*Ig))
print("Pin:" , Pin)

# potência média de saída
Pout = float((Vl**2)/RL)
print("Pout:" , Pout)

# eficiência
PCE = Pout/Pin
print("PCE:" , PCE)