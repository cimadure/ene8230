from model import ModelShaving
from docplex.mp.advmodel import AdvModel
import numpy as np
import pandas as pd
import sys
import cplex

print('Load')

print('Load Pb and other data')
filename = "ConsommationUniversite.pickle"
df = pd.read_pickle(filename)

year = df['year'].unique()
print(year)

an = df[df['year'] == year[0]]
Pb = an['Power Clipped [kW]'].to_numpy(copy=True)

Pb = Pb[:10]

print(Pb.shape)
#print(df.info)
#[69504 rows x 14 columns]>

print('Set problem parameters')


params = {
#Variables	Descriptions	Unités
'SOCmin':20.0, #	État de charge minimum des véhicules électriques	kWh
'SOCmax':80.0, #	État de charge maximum des véhicules électriques	kWh
'Pch_min': 5.0	,#Puissance de charge minimum 	kW
'Pch_max_n' : np.array([[7.5],[50.0]]) ,#Puissance de charge maximum en utilisant les bornes de type n	kW
'Pdis_min': 5.0	,#Puissance de décharge minimum	kW
'Pdis_max_n' : np.array([[5.0],[40.0]])	,#Puissance de décharge maximum en utilisant les bornes de type n	kW
'beta_ch':0.90	,#Efficacité de charge	%
'beta_dis':0.90	,#Efficacité de décharge	%
'NEVs':1000	,#Nombre de véhicules électriques	Qté
'Rut' : np.array([[0.25],[0.25],[0.25], [0.25]])	,#Ratio d’utilisateurs avec le profil i	%

'delta_t' : 15/60 , # 15mins
'MAX_OPTIM' : 1e8, # Facteur majorant du programme d'optimisation

#Puissance appelée par le bâtiment appelée à l’instant t	kW
'Pb' : Pb	

}

# start modeling...
# messages passed to the results channel will be upcased
# any function that takes a string and returns a string
# could be passed in
	#screen_output = mdl.set_results_stream(sys.stdout, lambda a: a.upper())
    # pass in None to delete the duct
    #mdl.set_results_stream(None)

    # sys.stdout is the default output stream for log and results
    # so these lines may be omitted



#with ModelShaving(params=params) as mdl:
if True:


	mdl = ModelShaving(params=params)
	
	#mdl = cplex.Cplex()
	#mdl = AdvModel()
	#mdl.params = params
#	mdl.ens = {"utilsateur":np.shape(mdl.params['Rut'])[0], # i
#	"borne":np.shape(mdl.params['Pch_max_n'])[0], # n
#	 "instant":np.shape(mdl.params['Pb'])[0]} # t
#	mdl.ens['I'] = range(1,mdl.ens['utilsateur'])
#	mdl.ens['N'] = range(1,mdl.ens['borne'])
#	mdl.ens['T'] = range(0,mdl.ens['instant'])

	#mdl.delta_ch__i_t = mdl.binary_var_matrix(mdl.ens['I'], mdl.ens['T'], name='delta_ch_')
	#mdl.delta_dis__i_t = mdl.binary_var_matrix(mdl.ens['I'], mdl.ens['T'], name='delta_dis_')
	
	
#
	mdl.x = np.array([mdl.continuous_var_list( mdl.ens['T'],name='x__%d_%d'%(n,i)) for n in mdl.ens['N'] for i in mdl.ens['I']])
	
	#mdl.s = np.array([mdl.binary_var_list(mdl.ens['T'],name='s_%d'%(i)) for i in mdl.ens['I']])
	
	mdl.problem_variables();
	
	mdl.problem_constrainsts();	

#	[mdl.add_constraint(mdl.delta_ch__i_t[i,t] + mdl.delta_dis__i_t[i,t] == 1.0) for i in mdl.ens['I'] for t in mdl.ens['T']]
	print(mdl.print_information())
		
	#mdl.set_results_stream(sys.stdout)
	#mdl.set_log_stream(sys.stdout)
	#mdl.objective.set_sense(mdl.objective.sense.minimize)

	mdl.solve()
    #mdl.write('V2B.lp')	
	#mdl.objective.set_sense(mdl.objective.sense.minimize)
	
	# Set an overall node limit
	#mdl.parameters.mip.limits.nodes.set(5000)
#print(vars(mdl))  

#print(mdl.print_information())
 

#'delta_ch__i_t'

#print("Solution status = ", mdl.solution.get_status())

#print("Obj", mdl.solution.get_objective_value())

#eps = c.parameters.mip.tolerances.integrality.get()
#used = c.solution.get_values("use0", "use" + str(n_machines - 1))
#for i in range(n_machines):#
#	if used[i] > eps:
#		print("E", i, "is used for ", end=' ')
#		print(c.solution.get_values(i))
print("--------------------------------------------")
print(mdl.print_solution(print_zeros=True))
print('end')

#print(mdl.get_values('delta_ch__i_t'))

#print(mdl.x)
#print(vars(mdl.solution))


#%%

#print(vars(mdl))


#%%
