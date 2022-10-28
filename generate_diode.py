###################################################################################################
### Licia Sales
### out-27-2022
###Generate Diode Parameters
###################################################################################################
###################################################################################################

import random
import pandas as pd 


class generate:
	def __init__(self):
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



	def gerenerate_all(self, value):
		print("Gerando dados ...")
		for diode in range(0,value):
			self.parameters()


		print("Dados gerados")
			

			
		print("  ##########  SALVANDO NUM CSV ##################    ")


		df = pd.DataFrame.from_dict(self.model, orient='index').T.to_csv('diodes_120.csv', index=True)

		print("         DONE        ")
if __name__ == '__main__':
	diode = generate()
	print(diode.gerenerate_all(120))
	
