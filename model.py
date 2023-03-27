import numpy as np
#from docplex.mp.advmodel import AdvModel
from docplex.mp.model import Model
import cplex


# content of test_sample.py
def func(x):
    return x + 1


"""
SOCi(t)	État de charge de la flotte des véhicules électriques à l’instant t	kWh
Pch,ni(t)	Puissance de charge à l’instant (t)	kW
Pdis,ni(t)	Puissance de décharge à l’instant (t)	kW
delta_ch,i(t)	1, Si le profil d’utilisateur i est en charge
0, Sinon	Bin
delta_dis,i(t)	1, Si le profil d’utilisateur i est en décharge
0, Sinon	Bin
Rborne,ni	Ratio de borne de type n utilisé par le profil d’utilisateur i	%
"""

"""
-graph, tableau: Profile des utilisateurs  
-graph: profile avant vs P avec V2B 
-tableau: # bornes par type 
Par année: 
-tableau: valeurs des coûts 
"""


"""
		[self.add_range(lb=self.params['SOCmin'], expr='SOC__{n}{i}'.format(n=n,i=i), ub=self.params['SOCmax']) for n in self.ens['N'] for i in self.ens['I']]
"""



# import cplex._internal._procedural as proc
#print(proc.__file__)


# https://ibmdecisionoptimization.github.io/docplex-doc/mp/docplex.mp.model.html
# https://notebook.community/IBMDecisionOptimization/docplex-examples/examples/mp/jupyter/tutorials/Beyond_Linear_Programming
class ModelShaving(Model):
#class Model(AdvModel):
#class Model(cplex.Cplex):
	
		
	def __init__(self, *args, **kwargs):
		self.params = kwargs.pop('params',None)
		if self.params:
			self.ens = {"utilsateur":np.shape(self.params['Rut'])[0], # i
			"borne":np.shape(self.params['Pch_max_n'])[0], # n
			 "instant":np.shape(self.params['Pb'])[0]} # t
			self.ens['I'] = range(1,self.ens['utilsateur'])
			self.ens['N'] = range(1,self.ens['borne'])
			self.ens['T'] = range(0,self.ens['instant'])
			print(self.ens)
			
			#x = np.array([model.continuous_var_list(J,name='x_%d'%(i)) for i in range(I)])
			#s = np.array([model.binary_var_list(J,name='s_%d'%(i)) for i in range(I)])
			#y = np.array([model.binary_var_list(J,name='y_%d'%(i)) for i in range(I)])

		
			#self.objective.set_sense(c.objective.sense.minimize)
			
		super(ModelShaving, self).__init__(*args, **kwargs)
		

	def problem_variables(self):
		#self.SOC__i_t = np.array([self.continuous_var_list(self.ens['instant'],name='SOC_%d'%(i)) for i in range(self.ens['utilsateur'])])
		#self.SOC__n_i_t = {(n,i): self.continuous_var(name='SOC__{n}{i}'.format(n=n,i=i)) for n in self.ens['N'] for i in self.ens['I']}
		#self.SOC__n_i_t = {(n,i): self.continuous_var_list(self.ens['instant'], name='SOC__{n}{i}'.format(n=n,i=i)) for n in self.ens['N'] for i in self.ens['I']}
		
		#continuous_var_cube(keys1, keys2, keys3, lb=None, ub=None, name=None, key_format=None)
		
		self.delta_ch__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_ch_')
		self.delta_dis__i_t = self.binary_var_matrix(self.ens['I'], self.ens['T'], name='delta_dis_')
		return self
		
		
	def problem_constrainsts(self):
		
		#[self.add_range(lb=self.params['SOCmin'], expr='SOC__{n}{i}'.format(n=n,i=i), ub=self.params['SOCmax']) for n in self.ens['N'] for i in self.ens['I']]
		#[self.add_range(lb=self.params['SOCmin'], expr=self.SOC__n_i_t[n,i], ub=self.params['SOCmax']) for n in self.ens['N'] for i in self.ens['I']];
		#[self.le_constraint(self.delta_ch__i_t[i,t] + self.delta_dis__i_t[i,t], 1.0, name='ctr_ch_dis') for i in self.ens['I'] for t in self.ens['T']]
		[self.add_constraint(self.delta_ch__i_t[i,t] + self.delta_dis__i_t[i,t] == 1.0) for i in self.ens['I'] for t in self.ens['T']]
		



#%%
