###################################################################################################
### Licia Sales
### out-17-2022
### Universal Rectifier Analysis
###################################################################################################
###################################################################################################

import matplotlib.pyplot as plt
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
spice_library = SpiceLibrary("./netlist/")

###################################################################################################

class analysis:
	def __init__(self):
		self.circuit = Circuit("12V/2.4MHz Rectifier")
		self.list_diodes =['1N4148','1N4735A', '1PS10SB82', '1PS181', '1PS300', '1PS79SB30', '1PS79SB31', '1PS88SB82', 'BAS101', 'BAS101S', 'BAS116H', 'BAS116', 'BAS16H', 'BAS16J', 'BAS16L', 'BAS16VV', 'BAS216', 'BAS21J', 'BAS21', 'BAS21VD', 'BAS28', 'BAS32L', 'BAS40-07', 'BAS40-07V', 'BAS40H', 'BAS40XY', 'BAS416', 'BAS45A', 'BAS516', 'BAS521', 'BAS56', 'BAS70-07', 'BAS70-07V', 'BAS70H', 'BAS70VV', 'BAS716', 'BAT160A', 'BAT160C', 'BAT160S', 'DIODE', 'DIODE', 'BAT54XY', 'BAT754L', 'BAV103', 'BAV199', 'BAW56', 'BAW56S', 'BAW56T', 'BAW56W', 'BAW62', 'BZA418A', 'BZA820A', 'BZA856A', 'BZA856AVL', 'BZA862A', 'BZA862AVL', 'BZA868A', 'BZA956A', 'BZA956AVL', 'BZA962A', 'BZA962AVL', 'BZA968A', 'BZB84-B10', 'BZB84-B16', 'BZB84-B18', 'BZB84-B20', 'BZB84-B22', 'BZB84-B24', 'BZB84-B27', 'BZB84-B30', 'BZB84-B33', 'BZB84-B36', 'BZB84-B39', 'BZB84-B43', 'BZB84-B47', 'BZB84-B51', 'BZB84-B56', 'BZB84-B5V1', 'BZB84-B5V6', 'BZB84-B62', 'BZB84-B68', 'BZB84-B6V2', 'BZB84-B6V8', 'BZB84-B75', 'BZB84-B7V5', 'BZB84-B8V2', 'BZB84-B9V1', 'BZV49-C12', 'BZV85-C33', 'BZV85-C3V6', 'BZV90-C30', 'BZX100A', 'BZX284-B6V2', 'BZX284-C7V5', 'BZX284-C8V2', 'BZX384-B10', 'BZX384-B11', 'BZX384-B12', 'BZX384-B13', 'BZX384-B15', 'BZX384-B16', 'BZX384-B18', 'BZX384-B20', 'BZX384-B22', 'BZX384-B24', 'BZX384-B27', 'BZX384-B2V4', 'BZX384-B2V7', 'BZX384-B30', 'BZX384-B33', 'BZX384-B36', 'BZX384-B39', 'BZX384-B3V0', 'BZX384-B3V3', 'BZX384-B3V6', 'BZX384-B3V9', 'BZX384-B43', 'BZX384-B47', 'BZX384-B4V3', 'BZX384-B4V7', 'BZX384-B51', 'BZX384-B56', 'BZX384-B5V1', 'BZX384-B5V6', 'BZX384-B62', 'BZX384-B68', 'BZX384-B6V2', 'BZX384-B6V8', 'BZX384-B75', 'BZX384-B7V5', 'BZX384-B8V2', 'BZX384-B9V1', 'BZX585-B10', 'BZX585-B11', 'BZX585-B12', 'BZX585-B13', 'BZX585-B15', 'BZX585-B16', 'BZX585-B18', 'BZX585-B20', 'BZX585-B22', 'BZX585-B24', 'BZX585-B27', 'BZX585-B2V4', 'BZX585-B2V7', 'BZX585-B30', 'BZX585-B33', 'BZX585-B36', 'BZX585-B39', 'BZX585-B3V0', 'BZX585-B3V3', 'BZX585-B3V6', 'BZX585-B3V9', 'BZX585-B43', 'BZX585-B47', 'BZX585-B4V3', 'BZX585-B4V7', 'BZX585-B51', 'BZX585-B56', 'BZX585-B5V1', 'BZX585-B5V6', 'BZX585-B62', 'BZX585-B68', 'BZX585-B6V2', 'BZX585-B6V8', 'BZX585-B75', 'BZX585-B7V5', 'BZX585-B8V2', 'BZX585-B9V1', 'BZX79-C6V8', 'BZX84-C10', 'BZX84-C11', 'BZX84-C12', 'BZX84-C13', 'BZX84-C15', 'BZX84-C16', 'BZX84-C18', 'BZX84-C20', 'BZX84-C22', 'BZX84-C24', 'BZX84-C27', 'BZX84-C2V4', 'BZX84-C2V7', 'BZX84-C30', 'BZX84-C33', 'BZX84-C36', 'BZX84-C39', 'BZX84-C3V0', 'BZX84-C3V3', 'BZX84-C3V6', 'BZX84-C3V9', 'BZX84-C43', 'BZX84-C47', 'BZX84-C4V3', 'BZX84-C4V7', 'BZX84-C51', 'BZX84-C56', 'BZX84-C5V1', 'BZX84-C5V6', 'BZX84-C62', 'BZX84-C68', 'BZX84-C6V2', 'BZX84-C6V8', 'BZX84-C75', 'BZX84-C7V5', 'BZX84-C8V2', 'BZX84-C9V1', 'BZX84J-B10', 'BZX84J-B11', 'BZX84J-B12', 'BZX84J-B13', 'BZX84J-B15', 'BZX84J-B16', 'BZX84J-B18', 'BZX84J-B20', 'BZX84J-B22', 'BZX84J-B24', 'BZX84J-B27', 'BZX84J-B2V4', 'BZX84J-B2V7', 'BZX84J-B30', 'BZX84J-B33', 'BZX84J-B36', 'BZX84J-B39', 'BZX84J-B3V0', 'BZX84J-B3V3', 'BZX84J-B3V6', 'BZX84J-B3V9', 'BZX84J-B43', 'BZX84J-B47', 'BZX84J-B4V3', 'BZX84J-B4V7', 'BZX84J-B51', 'BZX84J-B56', 'BZX84J-B5V1', 'BZX84J-B5V6', 'BZX84J-B62', 'BZX84J-B68', 'BZX84J-B6V2', 'BZX84J-B6V8', 'BZX84J-B75', 'BZX84J-B7V5', 'BZX84J-B8V2', 'BZX84J-B9V1', 'BZX884-B10', 'BZX884-B11', 'BZX884-B12', 'BZX884-B13', 'BZX884-B15', 'BZX884-B16', 'BZX884-B18', 'BZX884-B20', 'BZX884-B22', 'BZX884-B24', 'BZX884-B27', 'BZX884-B2V4', 'BZX884-B2V7', 'BZX884-B30', 'BZX884-B33', 'BZX884-B36', 'BZX884-B39', 'BZX884-B3V0', 'BZX884-B3V3', 'BZX884-B3V6', 'BZX884-B3V9', 'BZX884-B43', 'BZX884-B47', 'BZX884-B4V3', 'BZX884-B4V7', 'BZX884-B51', 'BZX884-B56', 'BZX884-B5V1', 'BZX884-B5V6', 'BZX884-B62', 'BZX884-B68', 'BZX884-B6V2', 'BZX884-B6V8', 'BZX884-B75', 'BZX884-B7V5', 'BZX884-B8V2', 'BZX884-B9V1', 'NZX10A', 'NZX10C', 'NZX11A', 'NZX11C', 'NZX11D', 'NZX12A', 'NZX12B', 'NZX12C', 'NZX12D', 'NZX14A', 'NZX14C', 'NZX15B', 'NZX16A', 'NZX16B', 'NZX16C', 'NZX18B', 'NZX20A', 'NZX20B', 'NZX24B', 'NZX27A', 'NZX27B', 'NZX27C', 'NZX30A', 'NZX36C', 'PDZ10B', 'PDZ11B', 'PDZ4.7B', 'PDZ5.6B', 'PDZ6.2B', 'PDZ9.1B', 'PESD12VL1BA', 'PESD12VL2BT', 'PESD12VS1UB', 'PESD12VS1UL', 'PESD12VS2UAT', 'PESD12VS2UQ', 'PESD12VS2UT', 'PESD12VS4UD', 'PESD12VS5UD', 'PESD15VL1BA', 'PESD15VL2BT', 'PESD15VS1UB', 'PESD15VS1UL', 'PESD15VS2UAT', 'PESD15VS2UQ', 'PESD15VS4UD', 'PESD15VS5UD', 'PESD24VL1BA', 'PESD24VL2BT', 'PESD24VS1UB', 'PESD24VS1UL', 'PESD24VS2UAT', 'PESD24VS2UQ', 'PESD24VS2UT', 'PESD24VS4UD', 'PESD24VS5UD', 'PESD3V3L1BA', 'PESD3V3L2BT', 'PESD3V3L2UM', 'PESD3V3L4UG', 'PESD3V3L4UW', 'PESD3V3L5UF', 'PESD3V3L5UV', 'PESD3V3L5UY', 'PESD3V3S1UB', 'PESD3V3S1UL', 'PESD3V3S2UAT', 'PESD3V3S2UQ', 'PESD3V3S2UT', 'PESD3V3S4UD', 'PESD3V3S5UD', 'PESD3V3V4UG', 'PESD3V3V4UW', 'PESD5V0L1BA', 'PESD5V0L1UA', 'PESD5V0L1UB', 'PESD5V0L1UL', 'PESD5V0L2BT', 'PESD5V0L4UG', 'PESD5V0L6UAS', 'PESD5V0L6US', 'PESD5V0L7BAS', 'PESD5V0L7BS', 'PESD5V0S1UA', 'PESD5V0S1UB', 'PESD5V0S1UL', 'PESD5V0S2UAT', 'PESD5V0S2UQ', 'PESD5V0S4UD', 'PESD5V0S5UD', 'PESD5V0V4UF', 'PESD5V0V4UG', 'PESD5V0V4UW', 'PLVA2656A', 'PLVA650A', 'PMBD6050', 'PMBD6100', 'PMBD7000', 'PMBD7100', 'PMEG1020EA', 'PMEG1020EH', 'PMEG1020EJ', 'PMEG1020EV', 'PMEG1030EH', 'PMEG1030EJ', 'PMEG2005AEA', 'PMEG2005AEL', 'PMEG2005AEV', 'PMEG2005EH', 'PMEG2005EJ', 'PMEG2005ET', 'PMEG2010AEB', 'PMEG2010AEH', 'PMEG2010AEJ', 'PMEG2010AET', 'PMEG2010BEA', 'PMEG2010BER', 'PMEG2010BEV', 'PMEG2010EH', 'PMEG2010EJ', 'PMEG2010EPA', 'PMEG2010ET', 'PMEG2015EA', 'PMEG2015EH', 'PMEG2015EJ', 'PMEG2020AEA', 'PMEG2020EH', 'PMEG2020EJ', 'PMEG3002AEB', 'PMEG3002AEL', 'PMEG3005AEA', 'PMEG3005AEV', 'PMEG3005EB', 'PMEG3005EH', 'PMEG3005EJ', 'PMEG3005EL', 'PMEG3005ET', 'PMEG3010BEA', 'PMEG3010BEP', 'PMEG3010BER', 'PMEG3010BEV', 'PMEG3010EB', 'PMEG3010EH', 'PMEG3010EJ', 'PMEG3010EP', 'PMEG3010ER', 'PMEG3010ET', 'PMEG3015EH', 'PMEG3015EJ', 'PMEG3015EV', 'PMEG3020BEP', 'PMEG3020BER', 'PMEG3020CEP', 'PMEG3020DEP', 'PMEG3020EH', 'PMEG3020EJ', 'PMEG3020EPA', 'PMEG3020ER', 'PMEG3030BEP', 'PMEG3050BEP', 'PMEG4002EL', 'PMEG4005AEA', 'PMEG4005AEV', 'PMEG4005EH', 'PMEG4005EJ', 'PMEG4005ET', 'PMEG4010BEA', 'PMEG4010BEV', 'PMEG4010CEJ', 'PMEG4010EH', 'PMEG4010EJ', 'PMEG4010ER', 'PMEG4010ET', 'PMEG4020EP', 'PMEG4020ER', 'PMEG4030EP', 'PMEG4030ER', 'PMEG4050EP', 'PMEG6002EB', 'PMEG6002TV', 'PMEG6010AED', 'PMEG6010CEJ', 'PMEG6010EP', 'PMEG6010ER', 'PMEG6020EPA', 'PMEG6020EP', 'PMEG6020ER', 'PMEG6030EP', 'PMLL4148L', 'PTVS14VP1UP', 'PTVS20VS1UR', 'PTVS22VP1UP', 'PTVS22VS1UR', 'PTVS26VS1UR', 'PTVS28VS1UR ', 'PTVS33VS1UR', 'PTVS36VS1UR', 'PTVS40VS1UR', 'PTVS43VS1UR', 'PTVS45VP1UP', 'PTVS45VS1UR', 'PTVS48VS1UR', 'PTVS54VS1UR', 'PTVS60VS1UR', 'PTVS7V0P1UP', 'PZU10B1A', 'PZU10B1', 'PZU10B2A', 'PZU10B2L', 'PZU10B2', 'PZU10B3A', 'PZU10B3', 'PZU10BA', 'PZU10BL', 'PZU10B', 'PZU10DB2', 'PZU5.1B1A', 'PZU5.1B1', 'PZU5.1B2A', 'PZU5.1B2L', 'PZU5.1B2', 'PZU5.1B3A', 'PZU5.1B3', 'PZU5.1BA', 'PZU5.1BL', 'PZU5.1B', 'PZU5.1DB2', 'PZU5.6B1A', 'PZU5.6B1', 'PZU5.6B2A', 'PZU5.6B2L', 'PZU5.6B2', 'PZU5.6B3A', 'PZU5.6B3', 'PZU5.6BA', 'PZU5.6BL', 'PZU5.6B', 'PZU5.6DB2', 'PZU6.2B1A', 'PZU6.2B1', 'PZU6.2B2A', 'PZU6.2B2L', 'PZU6.2B2', 'PZU6.2B3A', 'PZU6.2B3', 'PZU6.2BA', 'PZU6.2BL', 'PZU6.2B', 'PZU6.2DB2', 'PZU6.8B1A', 'PZU6.8B1', 'PZU6.8B2A', 'PZU6.8B2L', 'PZU6.8B2', 'PZU6.8B3A', 'PZU6.8B3', 'PZU6.8BA', 'PZU6.8BL', 'PZU6.8B', 'PZU6.8DB2', 'PZU7.5B1A', 'PZU7.5B1', 'PZU7.5B2A', 'PZU7.5B2L', 'PZU7.5B2', 'PZU7.5B3A', 'PZU7.5B3', 'PZU7.5BA', 'PZU7.5BL', 'PZU7.5B', 'PZU7.5DB2', 'PZU8.2B1A', 'PZU8.2B1', 'PZU8.2B2A', 'PZU8.2B2L', 'PZU8.2B2', 'PZU8.2B3A', 'PZU8.2B3', 'PZU8.2BA', 'PZU8.2BL', 'PZU8.2B', 'PZU8.2DB2', 'PZU9.1B1A', 'PZU9.1B1', 'PZU9.1B2A', 'PZU9.1B2L', 'PZU9.1B2', 'PZU9.1B3A', 'PZU9.1B3', 'PZU9.1BA', 'PZU9.1BL', 'PZU9.1B', 'PZU9.1DB2', 'RB521S30', 'RB751S40', 'RB751V40']
		self.frequency = 2.45@u_GHz
		self.amplitude = 100@u_mV
		self.capacitor = 100@u_pF
		self.RL=10@u_kΩ
		self.temperature = 25
		self.nominal_temperature = 25
		self.last_diode = self.list_diodes[-1]
		self.diode = self.list_diodes[0]
		self.i=0
	
		
	def choose_diode(self):
		if (self.diode != self.last_diode):
			self.diode = self.list_diodes[self.i]
		return self.diode
		
	def circuit_maker(self):
		self.circuit.include(spice_library[self.diode])
		# Componentes
		self.circuit.L('s',1,2,0.8@u_nH)
		self.circuit.Diode('1',2,3,model = self.diode)
		self.circuit.C('p',2,3,self.capacitor)
		self.circuit.R('load',3,self.circuit.gnd,self.RL)
		self.circuit.C('load',3,self.circuit.gnd,self.capacitor)
		self.source = self.circuit.SinusoidalVoltageSource('input', 1, self.circuit.gnd, amplitude=self.amplitude, frequency=self.frequency)	
		return self.circuit
		
	def simulation_transient(self):
		simulator = self.circuit.simulator(temperature = self.temperature, nominal_temperature = self.nominal_temperature)		
		analysis = simulator.transient(step_time=self.source.period/200, end_time=self.source.period*5000)
		fig, ax = plt.subplots(figsize=(10, 10))
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
		ax.legend(('input', 'output'), loc=(1.1,0.5))
		plt.tight_layout()
		fig.savefig("./output/Simulacao_transiente_diodo_{}".format(self.diode),dpi=300)
		plt.close(fig)				

	
if __name__ == '__main__':
		diode = analysis()
		print(diode.choose_diode())
		print(diode.circuit_maker())
		diode.simulation_transient()
		
		
		
		
		

